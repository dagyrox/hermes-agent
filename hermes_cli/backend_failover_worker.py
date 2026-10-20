"""Subprocess supervisor for bounded Kanban backend failover chains."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable

from hermes_cli.backend_failover import (
    BackendLeg,
    EXIT_AMBIGUOUS,
    EXIT_ELIGIBLE,
    EXIT_REFUSAL,
    supervise_chain,
)

TERMINAL_STATUSES = frozenset({"done", "blocked", "review", "changes_requested"})
_PROTOCOL_EXIT = 70
_CLAUDE_REFUSAL = re.compile(r"content policy|safety policy|cannot assist|refus", re.IGNORECASE)
_CLAUDE_MINIMUM_VERSIONS = {
    "claude-opus-5-5": (2, 1, 280),
    "claude-sonnet-5-5": (2, 1, 284),
}

_BASE_ENV_KEYS = frozenset({
    "HOME", "USER", "LOGNAME", "PATH", "SHELL", "TMPDIR", "TMP", "TEMP",
    "LANG", "LANGUAGE", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY",
    "ALL_PROXY", "NO_PROXY", "SSH_AUTH_SOCK",
})
_API_ENV_KEYS = frozenset({
    "HERMES_HOME", "HERMES_PROFILE", "HERMES_TENANT", "PYTHONPATH",
    "TERMINAL_CWD", "TERMINAL_TIMEOUT", "TERMINAL_MAX_FOREGROUND_TIMEOUT",
})
_KANBAN_ENV_KEYS = frozenset({
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_BRANCH",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
})
_LIFECYCLE_TOOLS = (
    "kanban_show",
    "kanban_comment",
    "kanban_heartbeat",
    "kanban_complete",
    "kanban_block",
)
_PROFILE_TOOLS = ("skills_list", "skill_view")
_ORCHESTRATOR_TOOLS = ("kanban_list", "kanban_create", "kanban_link")
_CLAUDE_CONTEXT_CHAR_LIMIT = 600_000
_REQUEST_ENVELOPE_CHARS = 512
_NATIVE_IMAGE_ENVELOPE_CHARS = 256
_EMPTY_TOOLSET_PIN = "__backend_failover_no_explicit_tools__"
_CLAUDE_NATIVE_TOOL_EQUIVALENTS = frozenset({
    "execute_code",
    "patch",
    "process",
    "read_file",
    "search_files",
    "terminal",
    "web_extract",
    "web_search",
    "write_file",
})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=int)
    parser.add_argument("--skill", action="append", default=[])
    parser.add_argument("--required-toolset", action="append", default=[])
    parser.add_argument("--required-context-chars", type=int, default=0)
    parser.add_argument("--requires-native-image", action="store_true")
    parser.add_argument("--native-image-bytes", type=int, default=0)
    parser.add_argument("--native-image-path", action="append", default=[])
    parser.add_argument("--chain-json")
    return parser.parse_args(argv)


def _resolve_hermes_argv() -> list[str]:
    from hermes_cli.kanban_db import _resolve_hermes_argv as resolve

    return resolve()


def _load_chain(profile: str, raw_chain: str | None = None) -> tuple[BackendLeg, ...]:
    from hermes_cli.backend_failover import load_profile_chain, resolve_role_chain

    if raw_chain:
        parsed = json.loads(raw_chain)
        return resolve_role_chain(
            {"enabled": True, "roles": {profile: parsed}},
            profile,
        )
    return load_profile_chain(profile, os.environ.get("HERMES_HOME"))


def _validate_dispatch_context(
    args: argparse.Namespace,
    *,
    profile_resolver: Callable[[str], str] | None = None,
) -> Path:
    """Resolve and verify the dispatcher-owned assignee identity without writes."""

    resolver = profile_resolver
    if resolver is None:
        from hermes_cli.profiles import resolve_profile_env

        resolver = resolve_profile_env
    if os.environ.get("HERMES_KANBAN_TASK") != args.task:
        raise RuntimeError("task argument does not match dispatcher-owned task")
    if os.environ.get("HERMES_PROFILE") != args.profile:
        raise RuntimeError("profile argument does not match dispatcher-owned profile")
    run_id = (os.environ.get("HERMES_KANBAN_RUN_ID") or "").strip()
    if not run_id.isdigit():
        raise RuntimeError("dispatcher-owned run identity is missing or invalid")
    configured_home = (os.environ.get("HERMES_HOME") or "").strip()
    if not configured_home:
        raise RuntimeError("dispatcher-owned profile home is missing")

    expected = Path(resolver(args.profile)).expanduser()
    if not expected.is_dir():
        raise RuntimeError(f"assignee profile home does not exist: {expected}")
    actual = Path(configured_home).expanduser()
    if not actual.is_dir():
        raise RuntimeError(f"dispatcher profile home does not exist: {actual}")
    if actual.resolve() != expected.resolve():
        raise RuntimeError("dispatcher profile home does not match the assignee profile")
    soul_path = expected / "SOUL.md"
    try:
        soul = soul_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"assignee profile identity is missing: {soul_path}") from exc
    if not soul:
        raise RuntimeError(f"assignee profile identity is empty: {soul_path}")

    # Environment identity is only a claim.  Bind it to the dispatcher's
    # existing board row before creating a ledger, MCP config, or prompt file.
    # Open SQLite read-only so a typo cannot manufacture a fresh empty board or
    # run schema migrations as a side effect of validation.
    db_raw = (os.environ.get("HERMES_KANBAN_DB") or "").strip()
    if not db_raw:
        raise RuntimeError("dispatcher-owned board database is missing")
    db_path = Path(db_raw).expanduser()
    if not db_path.is_file():
        raise RuntimeError(f"dispatcher-owned board database does not exist: {db_path}")
    claim_lock = (os.environ.get("HERMES_KANBAN_CLAIM_LOCK") or "").strip()
    workspace = (os.environ.get("HERMES_KANBAN_WORKSPACE") or "").strip()
    if not claim_lock:
        raise RuntimeError("dispatcher-owned claim identity is missing")
    if not workspace:
        raise RuntimeError("dispatcher-owned workspace identity is missing")

    from hermes_cli.profiles import normalize_profile_name

    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            task = conn.execute(
                "SELECT id, assignee, status, workspace_path, claim_lock, current_run_id "
                "FROM tasks WHERE id = ?",
                (args.task,),
            ).fetchone()
            run = conn.execute(
                "SELECT id, task_id, profile, status, claim_lock FROM task_runs WHERE id = ?",
                (int(run_id),),
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RuntimeError("dispatcher-owned board database could not be validated") from exc

    if task is None:
        raise RuntimeError("dispatcher-owned task does not exist")
    if task["status"] != "running":
        raise RuntimeError("dispatcher-owned task is not running")
    if task["current_run_id"] != int(run_id) or run is None:
        raise RuntimeError("dispatcher-owned run does not match the live task")
    if run["task_id"] != args.task or run["status"] != "running":
        raise RuntimeError("dispatcher-owned run is not active for this task")
    expected_profile = normalize_profile_name(args.profile)
    if normalize_profile_name(task["assignee"] or "") != expected_profile:
        raise RuntimeError("dispatcher-owned task assignee does not match the profile")
    if normalize_profile_name(run["profile"] or "") != expected_profile:
        raise RuntimeError("dispatcher-owned run profile does not match the assignee")
    if task["claim_lock"] != claim_lock or run["claim_lock"] != claim_lock:
        raise RuntimeError("dispatcher-owned claim does not match the live task/run")
    persisted_workspace = str(task["workspace_path"] or "").strip()
    if not persisted_workspace or (
        Path(persisted_workspace).expanduser().resolve()
        != Path(workspace).expanduser().resolve()
    ):
        raise RuntimeError("dispatcher-owned workspace does not match the live task")
    return expected.resolve()


def _task_status(task_id: str) -> str:
    from hermes_cli import kanban_db as kb

    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    expected = int(raw_run_id) if raw_run_id.isdigit() else None
    with kb.connect_closing(board=os.environ.get("HERMES_KANBAN_BOARD")) as conn:
        task = kb.get_task(conn, task_id)
        if task is None:
            return "missing"
        # Successful lifecycle writes clear current_run_id atomically with the
        # terminal transition. Observe that terminal state before applying the
        # stale-run guard, otherwise this supervisor misclassifies its own
        # completed child as "superseded" and reports a protocol failure.
        if task.status in TERMINAL_STATUSES:
            return task.status
        if expected is not None and task.current_run_id != expected:
            return "superseded"
        return task.status


def _comment(task_id: str, body: str) -> bool:
    from hermes_cli import kanban_db as kb

    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    expected = int(raw_run_id) if raw_run_id.isdigit() else None
    with kb.connect_closing(board=os.environ.get("HERMES_KANBAN_BOARD")) as conn:
        task = kb.get_task(conn, task_id)
        if task is None or task.status != "running":
            return False
        if expected is not None and task.current_run_id != expected:
            return False
        kb.add_comment(conn, task_id, os.environ.get("HERMES_PROFILE", "backend-supervisor"), body)
    return True


def _block_task(task_id: str, reason: str) -> bool:
    from hermes_cli import kanban_db as kb

    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    expected = int(raw_run_id) if raw_run_id.isdigit() else None
    with kb.connect_closing(board=os.environ.get("HERMES_KANBAN_BOARD")) as conn:
        return kb.block_task(
            conn,
            task_id,
            reason=reason,
            kind="needs_input",
            expected_run_id=expected,
        )


class _OwnershipLost(RuntimeError):
    """The dispatcher task/run no longer belongs to this supervisor."""


def _claude_environment(base: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in base.items() if key in _BASE_ENV_KEYS}
    # Subscription authentication remains in HOME/.claude.  Deliberately do
    # not manufacture API credentials or inherit an Anthropic-compatible URL.
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("USER", os.environ.get("USER", "root"))
    return env


def _api_environment(base: dict[str, str]) -> dict[str, str]:
    """Return runtime state plus exact dispatcher identity, never ambient secrets."""

    return {
        key: value
        for key, value in base.items()
        if key in _BASE_ENV_KEYS
        or key in _API_ENV_KEYS
        or key in _KANBAN_ENV_KEYS
    }


def _claude_cli_compatible(model: str, env: dict[str, str]) -> bool:
    """Fail closed when the installed CLI predates a requested model alias."""
    minimum = _CLAUDE_MINIMUM_VERSIONS.get(model)
    if minimum is None:
        return False
    try:
        probe = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", probe.stdout or probe.stderr or "")
    return bool(
        probe.returncode == 0
        and match
        and tuple(int(part) for part in match.groups()) >= minimum
    )


def _leg_available(leg: BackendLeg, profile_home: Path | None = None) -> bool:
    """Preflight local auth/runtime availability without making a model call."""
    if leg.kind == "claude_code":
        if not shutil.which("claude"):
            return False
        claude_env = _claude_environment(dict(os.environ))
        if not _claude_cli_compatible(leg.model, claude_env):
            return False
        try:
            probe = subprocess.run(
                ["claude", "auth", "status", "--json"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                env=claude_env,
            )
            if probe.returncode != 0:
                return False
            payload = json.loads(probe.stdout or "{}")
            return bool(payload.get("loggedIn") or payload.get("logged_in"))
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return False

    try:
        from agent.secret_scope import (
            build_profile_secret_scope,
            is_multiplex_active,
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.runtime_provider import resolve_runtime_provider

        if profile_home is None:
            raise RuntimeError("API availability requires a validated assignee profile home")
        previous_multiplex = is_multiplex_active()
        home_token = set_hermes_home_override(profile_home)
        scope_token = set_secret_scope(build_profile_secret_scope(profile_home))
        set_multiplex_active(True)
        try:
            resolved = resolve_runtime_provider(
                requested=leg.provider,
                target_model=leg.model,
            )
        finally:
            set_multiplex_active(previous_multiplex)
            reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)
        return bool(resolved.get("base_url") and resolved.get("api_key"))
    except Exception:
        return False


def _resolved_required_tool_schemas(
    required_toolsets: list[str] | tuple[str, ...],
) -> tuple[set[str] | None, int]:
    """Resolve the profile-visible schemas actually offered to an API child.

    A toolset name is configuration, not execution evidence.  The registry's
    requirement checks and dynamic schema assembly are the authoritative view.
    Resolution errors fail closed instead of silently treating an allowlist as
    proof that a backend can execute it.
    """

    try:
        get_tool_definitions = importlib.import_module("model_tools").get_tool_definitions
        definitions = get_tool_definitions(
            enabled_toolsets=list(required_toolsets),
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        names = {
            str(tool.get("function", {}).get("name") or "")
            for tool in definitions
            if isinstance(tool, dict)
        }
        names.discard("")
        return names, len(json.dumps(definitions, sort_keys=True, separators=(",", ":")))
    except Exception:
        return None, 0


def _claude_bridged_tool_names(leg: BackendLeg) -> set[str]:
    names = {*_PROFILE_TOOLS, *_LIFECYCLE_TOOLS}
    if leg.capability_mode == "orchestrator":
        names.update(_ORCHESTRATOR_TOOLS)
    if leg.capability_mode == "full":
        names.update(_CLAUDE_NATIVE_TOOL_EQUIVALENTS)
    else:
        names.update({"read_file", "search_files", "web_extract", "web_search"})
    return names


def _capability_mismatches(leg: BackendLeg, args: argparse.Namespace) -> tuple[str, ...]:
    """Compare this task's effective tool/context contract to one route leg."""
    required_toolsets = [
        str(name) for name in (getattr(args, "required_toolset", ()) or ())
    ]
    required_tools, _schema_chars = _resolved_required_tool_schemas(required_toolsets)
    if required_tools is None:
        return ("tool_schema_evidence",)
    if leg.kind == "api":
        from agent.models_dev import get_model_capabilities

        capabilities = get_model_capabilities(leg.provider or "", leg.model)
        if capabilities is None:
            return ("capability_evidence",)
        missing: set[str] = set()
        if not (
            getattr(capabilities, "context_window_observed", False)
            and getattr(capabilities, "max_output_tokens_observed", False)
            and isinstance(capabilities.context_window, int)
            and not isinstance(capabilities.context_window, bool)
            and capabilities.context_window > 0
            and isinstance(capabilities.max_output_tokens, int)
            and not isinstance(capabilities.max_output_tokens, bool)
            and capabilities.max_output_tokens > 0
        ):
            missing.add("context_limit_evidence")
        if required_tools and not capabilities.supports_tools:
            missing.add("tools")
        if (
            bool(getattr(args, "requires_native_image", False))
            and not capabilities.supports_vision
        ):
            missing.add("native_image_input")
        if bool(getattr(args, "requires_native_image", False)) and len(
            getattr(args, "native_image_path", ()) or ()
        ) != 1:
            missing.add("native_image_transport")
        # required_context_chars covers the frozen effective request buckets.
        # Reserve the model's maximum output budget as providers enforce the
        # combined input+output window.  Four chars/token is deliberately rough
        # and conservative; exact transport construction remains authoritative.
        input_tokens = (int(getattr(args, "required_context_chars", 0) or 0) + 3) // 4
        if input_tokens + int(capabilities.max_output_tokens) > int(capabilities.context_window):
            missing.add("context_window")
        return tuple(sorted(missing))
    supported = _claude_bridged_tool_names(leg)
    missing = {f"tool:{name}" for name in required_tools - supported}
    if bool(getattr(args, "requires_native_image", False)):
        # The Kanban Claude path does not currently transfer native media parts.
        # Claude's image-capable Read tool is not equivalent to receiving the
        # frozen turn's image payload.
        missing.add("native_image_input")
    if int(getattr(args, "required_context_chars", 0) or 0) > _CLAUDE_CONTEXT_CHAR_LIMIT:
        missing.add("context_window")
    # Reasoning mode is frozen and validated at config load. Lifecycle is the
    # strict MCP surface; native image payloads remain ineligible on this path.
    return tuple(sorted(missing))


def _effective_context_chars(
    task_id: str,
    *,
    declared_chars: int,
    system_prompt_path: Path | None,
    required_toolsets: list[str] | tuple[str, ...] = (),
    native_image_bytes: int = 0,
) -> int:
    """Measure the durable handoff plus loaded prompt, not only title/body."""

    from hermes_cli import kanban_db as kb

    with kb.connect_closing(board=os.environ.get("HERMES_KANBAN_BOARD")) as conn:
        worker_context = kb.build_worker_context(conn, task_id)
    system_chars = 0
    if system_prompt_path is not None:
        system_chars = len(system_prompt_path.read_text(encoding="utf-8"))
    _tool_names, schema_chars = _resolved_required_tool_schemas(required_toolsets)
    task_prompt_chars = len(f"work kanban task {task_id}")
    image_bytes = max(0, int(native_image_bytes or 0))
    # CLI/API transports serialize native image bytes into a base64 request
    # part. Count that expansion plus fixed message/envelope framing.
    native_media_chars = (
        4 * ((image_bytes + 2) // 3) + _NATIVE_IMAGE_ENVELOPE_CHARS
        if image_bytes
        else 0
    )
    return max(
        int(declared_chars or 0),
        len(worker_context)
        + system_chars
        + task_prompt_chars
        + schema_chars
        + native_media_chars
        + _REQUEST_ENVELOPE_CHARS,
    )


def _terminate_child_tree(proc: subprocess.Popen[str]) -> None:
    """Boundedly stop a supervised runtime and any MCP/tool descendants."""
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)  # windows-footgun: ok — POSIX-only branch
        else:
            proc.terminate()
        proc.wait(timeout=5)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only branch
        else:
            proc.kill()
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run_child(
    command: list[str],
    *,
    cwd: str | None,
    env: dict[str, str],
    capture_output: bool,
    timeout: float | None,
    ownership_check: Any = None,
) -> subprocess.CompletedProcess[str]:
    """Run one backend in its own process group and forward cancellation."""
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
        start_new_session=os.name != "nt",
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        ),
    )
    previous_handlers: dict[int, Any] = {}

    def _forward(signum: int, _frame: Any) -> None:
        _terminate_child_tree(proc)
        raise SystemExit(128 + signum)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _forward)
    started = time.monotonic()
    try:
        while True:
            if ownership_check is not None and not ownership_check():
                _terminate_child_tree(proc)
                raise _OwnershipLost("dispatcher ownership changed")
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                _terminate_child_tree(proc)
                raise subprocess.TimeoutExpired(command, timeout or 0)
            poll_timeout = 1.0 if remaining is None else min(1.0, remaining)
            try:
                stdout, stderr = proc.communicate(timeout=poll_timeout)
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)


def _leg_command(
    leg: BackendLeg,
    args: argparse.Namespace,
    base_env: dict[str, str],
    mcp_path: Path | None,
    system_prompt_path: Path | None,
) -> tuple[list[str], dict[str, str]]:
    prompt = f"work kanban task {args.task}"
    if leg.kind == "api":
        cmd = [
            *_resolve_hermes_argv(),
            "-p",
            args.profile,
            "--cli",
            "--accept-hooks",
            "-m",
            leg.model,
            "--provider",
            str(leg.provider),
        ]
        if leg.reasoning_effort:
            cmd.extend(["--reasoning-effort", leg.reasoning_effort])
        for skill in getattr(args, "skill", ()):
            cmd.extend(["--skills", skill])
        required_toolsets = list(getattr(args, "required_toolset", ()) or ())
        cmd.extend([
            "--toolsets",
            ",".join(required_toolsets) if required_toolsets else _EMPTY_TOOLSET_PIN,
        ])
        native_images = list(getattr(args, "native_image_path", ()) or ())
        # Quiet mode propagates classified pre-tool provider failures as exit
        # 75 so the supervisor can advance rather than treating CLI exit 0 as
        # a successful task execution.
        cmd.extend(["chat", "-q", prompt])
        if len(native_images) == 1:
            cmd.extend(["--image", native_images[0]])
        cmd.append("-Q")
        env = _api_environment(base_env)
        env["HERMES_BACKEND_FAILOVER_CHILD"] = "1"
        return cmd, env

    if mcp_path is None or system_prompt_path is None:
        raise RuntimeError("Claude Code leg requires MCP and system prompt files")
    builtins = ["Read", "Glob", "Grep"]
    if "web" in set(getattr(args, "required_toolset", ()) or ()):
        builtins.extend(["WebFetch", "WebSearch"])
    if leg.capability_mode == "full":
        # Claude's restricted sandbox confines file and shell execution to the
        # assigned workspace. The child environment omits Kanban identity and
        # credentials; lifecycle mutation remains available only through the
        # task-fenced MCP server.
        builtins.extend(["Edit", "Write", "Bash"])
    allowed = ["Read", "Glob", "Grep"]
    if "web" in set(getattr(args, "required_toolset", ()) or ()):
        allowed.extend(["WebFetch", "WebSearch"])
    if leg.capability_mode == "full":
        allowed.extend(["Edit", "Write", "Bash"])
    allowed.extend(f"mcp__hermes-tools__{name}" for name in _LIFECYCLE_TOOLS)
    allowed.extend(f"mcp__hermes-tools__{name}" for name in _PROFILE_TOOLS)
    if leg.capability_mode == "orchestrator":
        allowed.extend(f"mcp__hermes-tools__{name}" for name in _ORCHESTRATOR_TOOLS)
    command = [
        "claude",
        "-p",
        "--model",
        leg.model,
        "--max-turns",
        "90",
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--verbose",
        "--append-system-prompt-file",
        str(system_prompt_path),
        "--setting-sources",
        "",
        "--mcp-config",
        str(mcp_path),
        "--restricted",
        "--strict-mcp-config",
        "--tools",
        ",".join(builtins),
        "--permission-mode",
        "auto" if leg.capability_mode == "full" else "dontAsk",
        "--permission-prompts",
        "none",
        "--allowedTools",
        *allowed,
    ]
    if leg.reasoning_effort:
        effort = "medium" if leg.reasoning_effort == "balanced" else leg.reasoning_effort
        command.extend(["--effort", effort])
    command.extend([
        "--",
        (
            f"You are the dispatcher-owned Kanban worker for task {args.task}. "
            "Call mcp__hermes-tools__kanban_show first; the card, parent handoffs, and comments "
            "are the durable cross-runtime transcript. Work only in the assigned workspace, "
            "preserve the profile's authority and approval boundaries, and finish exactly once "
            "with kanban_complete or kanban_block. Do not retry or switch providers yourself."
        ),
    ])
    return command, _claude_environment(base_env)


def _mcp_config(profile_home: Path, task_id: str, profile: str) -> Path:
    mcp_env = {
        key: value
        for key, value in os.environ.items()
        if key in _KANBAN_ENV_KEYS or key in {"HERMES_HOME", "HERMES_PROFILE", "PYTHONPATH"}
    }
    exposed: list[str] = [*_LIFECYCLE_TOOLS, *_PROFILE_TOOLS]
    if profile == "hyperion":
        exposed.extend(_ORCHESTRATOR_TOOLS)
    mcp_env["HERMES_MCP_EXPOSED_TOOLS"] = ",".join(exposed)
    mcp_env["HERMES_MCP_ASSIGNED_TASK"] = task_id
    mcp_env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parents[1]))
    payload = {
        "mcpServers": {
            "hermes-tools": {
                "command": sys.executable,
                "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                "env": mcp_env,
            }
        }
    }
    scratch = profile_home / "cache" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="backend-failover-mcp-", suffix=".json", dir=scratch)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return Path(name)


def _system_prompt(profile_home: Path, required_skills: list[str]) -> Path:
    from agent.prompt_builder import (
        build_context_files_prompt,
        build_skills_system_prompt,
        load_soul_md,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home_token = set_hermes_home_override(profile_home)
    try:
        soul = load_soul_md()
    finally:
        reset_hermes_home_override(home_token)
    if not soul:
        raise RuntimeError(f"missing profile identity: {profile_home / 'SOUL.md'}")
    context = build_context_files_prompt(
        cwd=os.environ.get("HERMES_KANBAN_WORKSPACE") or os.getcwd(),
        skip_soul=True,
    )
    skills = build_skills_system_prompt(
        available_tools={*_LIFECYCLE_TOOLS, *_PROFILE_TOOLS, *_ORCHESTRATOR_TOOLS},
    )
    scratch = profile_home / "cache" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="backend-failover-system-", suffix=".md", dir=scratch)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(soul)
        for memory_name in ("USER.md", "MEMORY.md"):
            memory_path = profile_home / memory_name
            try:
                memory_text = memory_path.read_text(encoding="utf-8").strip()
            except OSError:
                memory_text = ""
            if memory_text:
                handle.write(f"\n\n# {memory_name}\n\n{memory_text}\n")
        if skills.strip():
            handle.write("\n\n" + skills.strip() + "\n")
        if required_skills:
            handle.write(
                "\nRequired task skills (load each with skill_view before work): "
                + ", ".join(required_skills)
                + "\n"
            )
        if context.strip():
            handle.write("\n\n" + context.strip() + "\n")
    return Path(name)


def _ledger_path(task_id: str) -> Path | None:
    db_path = os.environ.get("HERMES_KANBAN_DB", "").strip()
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    if not db_path or not run_id.isdigit():
        return None
    safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
    root = Path(db_path).resolve().parent / "backend-failover-ledgers"
    return root / f"{safe_task}-run-{run_id}.json"


def _load_ledger(path: Path | None, task_id: str) -> dict[str, Any]:
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "")
    empty = {"task_id": task_id, "run_id": run_id, "attempts": []}
    if path is None or not path.exists():
        return empty
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_id") != task_id or str(payload.get("run_id")) != run_id:
        raise RuntimeError("backend failover ledger identity mismatch")
    if not isinstance(payload.get("attempts"), list):
        raise RuntimeError("backend failover ledger attempts are invalid")
    return payload


def _write_ledger(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _classify_claude_failure(returncode: int, stdout: str, stderr: str) -> int:
    if returncode == 0:
        return 0
    # Refusal is authoritative even when the transport emits a later capacity
    # error or the stream itself is malformed.
    if _CLAUDE_REFUSAL.search(stderr):
        return EXIT_REFUSAL

    events: list[dict[str, Any]] = []
    malformed = False
    known_event_types = {
        "system",
        "assistant",
        "user",
        "result",
        "rate_limit_event",
    }
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed = True
            continue
        if not isinstance(event, dict):
            malformed = True
            continue
        event_type = event.get("type")
        if event_type not in known_event_types:
            malformed = True
        events.append(event)

    if any(int(event.get("safety_stops", 0) or 0) > 0 for event in events):
        return EXIT_REFUSAL
    if any(event.get("permission_denials") for event in events):
        return EXIT_REFUSAL

    # Assistant/user events represent current-attempt output, tool activity, or
    # approval/tool results. They dominate a later availability event. Inspect
    # the whole object so nested tool_use records cannot evade the gate.
    if any(event.get("type") in {"assistant", "user"} for event in events):
        return EXIT_AMBIGUOUS
    if any('"type":"tool_use"' in json.dumps(event, separators=(",", ":")) for event in events):
        return EXIT_AMBIGUOUS
    if malformed or not events:
        return EXIT_AMBIGUOUS

    eligible_errors = {
        "authentication_error",
        "billing_error",
        "model_not_found",
        "overloaded_error",
        "permission_error",
        "rate_limit_error",
    }
    eligible_statuses = {401, 403, 404, 408, 429, 500, 502, 503, 504}
    for event in events:
        if event.get("error") in eligible_errors:
            return EXIT_ELIGIBLE
        status = event.get("api_error_status")
        if isinstance(status, int) and status in eligible_statuses:
            return EXIT_ELIGIBLE
    # Claude Code has no stable machine-readable taxonomy guaranteeing that a
    # remaining nonzero process exit happened before tool execution. Unknown
    # failures therefore stop for reconciliation rather than being guessed.
    return EXIT_AMBIGUOUS


def _run_leg(
    leg: BackendLeg,
    index: int,
    args: argparse.Namespace,
    mcp_path: Path | None,
    system_prompt_path: Path | None,
    timeout_seconds: int | None = None,
) -> int:
    mismatches = _capability_mismatches(leg, args)
    if mismatches:
        if not _comment(
            args.task,
            f"Backend route leg {index + 1}/3 refused by capability parity gate: "
            + ", ".join(mismatches),
        ):
            return 0
        return EXIT_ELIGIBLE
    profile_home = getattr(args, "profile_home", None)
    available = (
        _leg_available(leg, profile_home)
        if profile_home is not None
        else _leg_available(leg)
    )
    if not available:
        return EXIT_ELIGIBLE
    if not _comment(
        args.task,
        f"Backend route attempt {index + 1}/3: kind={leg.kind}, "
        f"provider={leg.provider or 'subscription'}, model={leg.model}.",
    ):
        return 0
    cmd, env = _leg_command(leg, args, dict(os.environ), mcp_path, system_prompt_path)
    child_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else (args.timeout_seconds if args.timeout_seconds > 0 else None)
    )
    try:
        completed = _run_child(
            cmd,
            cwd=os.environ.get("HERMES_KANBAN_WORKSPACE") or None,
            env=env,
            capture_output=leg.kind == "claude_code",
            timeout=child_timeout,
            ownership_check=lambda: _task_status(args.task) == "running",
        )
    except _OwnershipLost:
        return 0
    except subprocess.TimeoutExpired:
        return EXIT_AMBIGUOUS
    except OSError:
        return EXIT_ELIGIBLE

    status = _task_status(args.task)
    if status in TERMINAL_STATUSES:
        return 0
    if leg.kind == "claude_code":
        if completed.returncode == 0:
            return EXIT_AMBIGUOUS
        return _classify_claude_failure(
            completed.returncode,
            completed.stdout or "",
            completed.stderr or "",
        )
    return completed.returncode or _PROTOCOL_EXIT


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    profile_home = _validate_dispatch_context(args)
    args.profile_home = profile_home
    legs = (
        _load_chain(args.profile, args.chain_json)
        if args.chain_json
        else _load_chain(args.profile)
    )
    if len(legs) != 3:
        raise RuntimeError("backend failover supervisor requires one validated three-leg chain")

    ledger_path = _ledger_path(args.task)
    try:
        ledger = _load_ledger(ledger_path, args.task)
    except Exception:
        if _task_status(args.task) == "running":
            _block_task(
                args.task,
                "Backend failover attempt ledger is unreadable; automatic replay is denied.",
            )
        return _PROTOCOL_EXIT
    attempts = ledger["attempts"]
    if any(attempt.get("outcome") == "started" for attempt in attempts):
        if _task_status(args.task) == "running":
            _block_task(
                args.task,
                "Backend failover supervisor restarted after an unfinished attempt; automatic replay is denied.",
            )
        return EXIT_AMBIGUOUS
    for attempt in attempts:
        prior_returncode = int(attempt.get("returncode", _PROTOCOL_EXIT))
        if prior_returncode != EXIT_ELIGIBLE:
            if _task_status(args.task) == "running":
                _block_task(
                    args.task,
                    "Backend failover recovered a non-replayable prior outcome; operator reconciliation is required.",
                )
            return prior_returncode
    resume_index = len(attempts)

    needs_claude = any(leg.kind == "claude_code" for leg in legs)
    mcp_path = _mcp_config(profile_home, args.task, args.profile) if needs_claude else None
    system_path = (
        _system_prompt(profile_home, list(getattr(args, "skill", ()) or ()))
        if needs_claude
        else None
    )
    if needs_claude:
        args.required_context_chars = _effective_context_chars(
            args.task,
            declared_chars=args.required_context_chars,
            system_prompt_path=system_path,
            required_toolsets=list(getattr(args, "required_toolset", ()) or ()),
            native_image_bytes=int(getattr(args, "native_image_bytes", 0) or 0),
        )
    deadline = (
        time.monotonic() + args.timeout_seconds
        if args.timeout_seconds > 0
        else None
    )

    def run_leg(leg: BackendLeg, index: int) -> int:
        if index < resume_index:
            return EXIT_ELIGIBLE
        remaining = None if deadline is None else int(deadline - time.monotonic())
        if remaining is not None and remaining <= 0:
            return EXIT_AMBIGUOUS
        record = {
            "index": index,
            "kind": leg.kind,
            "provider": leg.provider,
            "model": leg.model,
            "outcome": "started",
            "handoff": {
                "task_id": args.task,
                "run_id": os.environ.get("HERMES_KANBAN_RUN_ID", ""),
                "workspace": os.environ.get("HERMES_KANBAN_WORKSPACE", ""),
                "branch": os.environ.get("HERMES_KANBAN_BRANCH", ""),
                "skills": list(getattr(args, "skill", ()) or ()),
                "transcript_source": "kanban_card_parent_handoffs_and_comments",
                "approval_state": "preserved_on_kanban_task",
                "requires_native_image": bool(args.requires_native_image),
                "native_image_bytes": max(0, int(args.native_image_bytes or 0)),
            },
        }
        attempts.append(record)
        _write_ledger(ledger_path, ledger)
        returncode = _run_leg(leg, index, args, mcp_path, system_path, remaining)
        record["outcome"] = "finished"
        record["returncode"] = returncode
        if returncode == EXIT_ELIGIBLE:
            record["degraded_source"] = {
                "kind": leg.kind,
                "provider": leg.provider or "claude-code-subscription",
                "model": leg.model,
            }
            if not _comment(
                args.task,
                f"Backend degraded before tool execution: kind={leg.kind}, "
                f"provider={leg.provider or 'claude-code-subscription'}, model={leg.model}; "
                "continuing to the next frozen route leg with the Kanban card as the durable handoff.",
            ):
                returncode = 0
                record["returncode"] = 0
        _write_ledger(ledger_path, ledger)
        return returncode

    try:
        outcome = supervise_chain(
            legs,
            run_leg=run_leg,
        )
    except Exception:
        if _task_status(args.task) == "running":
            _block_task(
                args.task,
                "Backend failover supervisor failed before a verified terminal transition; automatic replay is denied.",
            )
        return _PROTOCOL_EXIT
    finally:
        if mcp_path:
            mcp_path.unlink(missing_ok=True)
        if system_path:
            system_path.unlink(missing_ok=True)

    if outcome.returncode == EXIT_AMBIGUOUS and _task_status(args.task) == "running":
        _block_task(
            args.task,
            "Backend failover stopped for reconciliation: a prior attempt may have executed tools or timed out; automatic replay is denied.",
        )
    elif outcome.exhausted and _task_status(args.task) == "running":
        _block_task(
            args.task,
            "Backend exhaustion: all three canonical route legs failed with an eligible availability/capacity error.",
        )
    elif outcome.returncode != 0 and _task_status(args.task) == "running":
        if outcome.returncode == EXIT_REFUSAL:
            reason = (
                "Backend route stopped on a safety/refusal boundary; automatic "
                "fallback was denied and human review is required."
            )
        else:
            reason = (
                f"Backend route terminated with non-replayable exit code "
                f"{outcome.returncode}; human review is required."
            )
        _block_task(args.task, reason)
    return outcome.returncode


if __name__ == "__main__":
    raise SystemExit(run())
