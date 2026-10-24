"""Opt-in, pre-context Kanban worker admission (no recovery or auth repair).

The assignee's kanban.worker_preflight config is default-off. Admission probes
the exact direct worker or frozen supervisor primary before expensive context.
HTTP requests retain the 15-second/16-token ceiling; Codex OAuth Responses and
managed native Claude use a bounded 45-second diagnostic without a fake token
cap. Other transports and unavailable tool/capability evidence fail closed.

Activation is a separate, approved live-config operation. Turning the setting
off preserves legacy startup; no credential, provider or recovery changes are
performed here. Startup evidence uses the existing current task/run lifecycle.
"""

from __future__ import annotations

import re
import os
from pathlib import Path
import subprocess
import json
import logging
import signal
import sys
import time
import tempfile
import threading
from urllib.parse import urlsplit

_STAGES = frozenset({
    "dispatch",
    "workspace",
    "identity",
    "credential",
    "transport",
    "tools",
    "operations",
    "probe",
})
_CODES = frozenset({
    "capability_mismatch",
    "workspace_invalid",
    "config_invalid",
    "model_unresolved",
    "credential_missing",
    "credential_rejected",
    "transport_unsupported",
    "tool_unavailable",
    "operation_unavailable",
    "probe_failed",
    "probe_timeout",
    "probe_transient",
    "startup_passed",
})
_MODES = frozenset({
    "claude_native",
    "chat_completions",
    "anthropic_messages",
    "codex_responses",
    "codex_app_server",
    "bedrock_converse",
    "acp",
    "other",
})
_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,127}\Z")
_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_HOST = re.compile(r"[A-Za-z0-9.:-]{1,253}\Z")


def sanitize_outcome(payload: object) -> dict:
    """Allow only bounded diagnostic metadata, never exception/response text."""
    raw = payload if isinstance(payload, dict) else {}
    safe = {}
    for key, allowed in (
        ("stage", _STAGES),
        ("code", _CODES),
        ("api_mode", _MODES),
        ("outcome", {"passed", "failed"}),
        ("route", {"direct", "supervisor"}),
    ):
        value = raw.get(key)
        if isinstance(value, str) and value in allowed:
            safe[key] = value
    if "stage" not in safe or "code" not in safe:
        safe.update(stage="probe", code="probe_failed")
    for key, pattern in (("profile", _ID), ("provider", _ID), ("model", _MODEL)):
        value = raw.get(key)
        if isinstance(value, str) and pattern.fullmatch(value):
            safe[key] = value
    host = raw.get("base_url_host")
    if isinstance(host, str) and _HOST.fullmatch(host):
        # Input is already a hostname; never retain a URL's path/userinfo.
        if urlsplit("//" + host).hostname == host.lower():
            safe["base_url_host"] = host.lower()
    for key in ("toolsets", "missing"):
        value = raw.get(key)
        if (
            isinstance(value, list)
            and len(value) <= 64
            and all(isinstance(item, str) and _NAME.fullmatch(item) for item in value)
        ):
            safe[key] = value
    for key, low, high in (("elapsed_ms", 0, 600000), ("http_status", 100, 599)):
        value = raw.get(key)
        if type(value) is int and low <= value <= high:
            safe[key] = value
    return safe


class WorkerPreflightBlocked(RuntimeError):
    """A classified admission failure, not a retryable spawn exception."""

    def __init__(self, code: str, *, stage: str = "dispatch", **evidence):
        self.payload = sanitize_outcome({
            "stage": stage,
            "code": code,
            "outcome": "failed",
            **evidence,
        })
        super().__init__(self.payload["code"])


def dispatch_admission(profile_home: str | None) -> dict | None:
    """Read opt-in from the resolved assignee only; never inherit admission."""
    if not profile_home:
        return None
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.config import load_config

    token = set_hermes_home_override(profile_home)
    try:
        try:
            raw = (load_config().get("kanban") or {}).get("worker_preflight", {})
        except Exception:
            # Legacy profiles with unreadable config have not opted in.
            return None
    finally:
        reset_hermes_home_override(token)
    if not isinstance(raw, dict) or type(raw.get("enabled", False)) is not bool:
        raise WorkerPreflightBlocked("config_invalid")
    if not raw.get("enabled", False):
        return None
    if set(raw) - {
        "enabled",
        "timeout_seconds",
        "native_timeout_seconds",
        "max_output_tokens",
        "required_tools",
    }:
        raise WorkerPreflightBlocked("config_invalid")
    settings = {"enabled": True}
    for key, ceiling in (
        ("timeout_seconds", 15),
        ("native_timeout_seconds", 45),
        ("max_output_tokens", 16),
    ):
        value = raw.get(key, ceiling)
        if type(value) is not int:
            raise WorkerPreflightBlocked("config_invalid")
        settings[key] = max(1, min(value, ceiling))
    names = raw.get("required_tools", ["terminal"])
    if (
        not isinstance(names, list)
        or len(names) > 64
        or not all(isinstance(name, str) and _NAME.fullmatch(name) for name in names)
    ):
        raise WorkerPreflightBlocked("config_invalid")
    settings["required_tools"] = names
    return settings


def dispatch_checks(workspace: str, *, branch: str | None, worktree: bool) -> None:
    """Verify workspace identity without creating it or falling back to cwd."""
    if not workspace or not os.path.isabs(workspace) or not Path(workspace).is_dir():
        raise WorkerPreflightBlocked("workspace_invalid", stage="workspace")
    canonical = os.path.realpath(workspace)
    if worktree or branch:
        try:
            top = subprocess.check_output(
                ["git", "-C", workspace, "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                timeout=3,
                text=True,
            ).strip()
            actual = subprocess.check_output(
                ["git", "-C", workspace, "branch", "--show-current"],
                stderr=subprocess.DEVNULL,
                timeout=3,
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            raise WorkerPreflightBlocked(
                "workspace_invalid", stage="workspace"
            ) from None
        if os.path.realpath(top) != canonical or not branch or actual != branch:
            raise WorkerPreflightBlocked("workspace_invalid", stage="workspace")
    if os.path.realpath(workspace) != canonical:
        raise WorkerPreflightBlocked("workspace_invalid", stage="workspace")


def _check_tools(args, settings: dict) -> None:
    from hermes_cli.config import load_config
    from model_tools import get_tool_definitions
    from toolsets import resolve_toolset, validate_toolset

    pinned = [
        name.strip()
        for name in (getattr(args, "toolsets", None) or "").split(",")
        if name.strip()
    ]
    try:
        disabled = (load_config().get("agent") or {}).get("disabled_toolsets") or []
        definitions = get_tool_definitions(
            enabled_toolsets=pinned,
            disabled_toolsets=disabled,
            quiet_mode=True,
            skip_tool_search_assembly=True,
        )
        available = {tool["function"]["name"] for tool in definitions}
        missing = [
            name
            for name in pinned
            if not validate_toolset(name) or not set(resolve_toolset(name)) & available
        ]
        missing.extend(
            name for name in settings["required_tools"] if name not in available
        )
    except Exception:
        raise WorkerPreflightBlocked("tool_unavailable", stage="tools") from None
    if missing:
        raise WorkerPreflightBlocked(
            "tool_unavailable", stage="tools", missing=sorted(set(missing))
        )
    _check_operations(available)


def _check_operations(available) -> None:
    from tools.registry import registry

    missing = []
    for name in ("kanban_complete", "kanban_block", "kanban_heartbeat"):
        entry = registry.get_entry(name)
        if name not in available or entry is None or not callable(entry.handler):
            missing.append(name)
    if missing:
        raise WorkerPreflightBlocked(
            "operation_unavailable", stage="operations", missing=missing
        )


def _check_native_tools(primary, args, settings):
    from importlib.util import find_spec
    from hermes_cli.backend_failover_worker import _claude_bridged_tool_names

    supported = _claude_bridged_tool_names(primary)
    missing = sorted(set(settings["required_tools"]) - supported)
    if missing:
        raise WorkerPreflightBlocked("tool_unavailable", stage="tools", missing=missing)
    _check_operations(supported)
    if find_spec("agent.transports.hermes_tools_mcp_server") is None:
        raise WorkerPreflightBlocked("operation_unavailable", stage="operations")


def _check_primary_capability(primary, args):
    from hermes_cli.backend_failover_worker import _capability_mismatches

    mismatches = _capability_mismatches(primary, args)
    tools = [name[5:] for name in mismatches if name.startswith("tool:")]
    if tools:
        raise WorkerPreflightBlocked("tool_unavailable", stage="tools", missing=tools)
    if mismatches:
        raise WorkerPreflightBlocked(
            "capability_mismatch", stage="tools", missing=list(mismatches)
        )


def _terminate_probe(proc, *, immediate=False) -> None:
    """Only terminate this owned group, including a descendant holding stdout."""
    kill_group = getattr(os, "killpg", None)
    signals = (
        (getattr(signal, "SIGKILL", signal.SIGTERM),)
        if immediate
        else (signal.SIGTERM, getattr(signal, "SIGKILL", signal.SIGTERM))
    )
    for sig in signals:
        try:
            if os.name == "posix" and kill_group is not None:
                kill_group(proc.pid, sig)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            continue
        # Even when the leader exited, SIGKILL any descendant ignoring TERM.
        if sig == signal.SIGTERM and os.name == "posix" and kill_group is not None:
            try:
                kill_group(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            except ProcessLookupError:
                pass
        return


def _capture_bounded(command, *, cwd, timeout, env=None, limit=4096):
    """Bound retained bytes as well as wall time; suppress all raw diagnostics."""
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
    )
    deadline = time.monotonic() + timeout
    output = bytearray()
    done, overflow = threading.Event(), threading.Event()

    def read_output():
        try:
            while True:
                chunk = proc.stdout.read1(min(4096, limit + 1 - len(output)))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > limit:
                    overflow.set()
                    break
        finally:
            done.set()

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    previous = {}

    def cancelled(signum, frame):
        # The outer owner has a one-second TERM grace. Escalate immediately so
        # it cannot kill this interpreter before its detached native group.
        _terminate_probe(proc, immediate=True)
        raise SystemExit(128 + signum)

    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, cancelled)
    try:
        if not done.wait(max(0, deadline - time.monotonic())):
            return None, b"", "probe_timeout"
        if overflow.is_set():
            return None, b"", "probe_failed"
        try:
            proc.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            return None, b"", "probe_timeout"
        return proc.returncode, bytes(output), None
    finally:
        _terminate_probe(proc)
        reader.join(timeout=1)
        proc.stdout.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def bounded_probe(
    command: list[str], *, cwd: str, timeout: int, env: dict | None = None
) -> dict:
    """One owned subprocess bounds imports, credential resolution and I/O."""
    rc, output, failure = _capture_bounded(command, cwd=cwd, timeout=timeout, env=env)
    if failure:
        return {"stage": "probe", "code": failure, "outcome": "failed"}
    try:
        if rc != 0:
            raise ValueError("invalid probe result")
        raw = json.loads(output)
        if not isinstance(raw, dict):
            raise ValueError("invalid probe result")
        result = sanitize_outcome(raw)
        if result.get("outcome") not in {"passed", "failed"}:
            raise ValueError("invalid probe outcome")
        return result
    except (ValueError, TypeError):
        return {"stage": "probe", "code": "probe_failed", "outcome": "failed"}


def run_supervisor_preflight(args, primary, profile_home: Path) -> None:
    """Admit the frozen primary before ledger, MCP, memory or task context."""
    from types import SimpleNamespace

    run_worker_preflight(
        SimpleNamespace(
            toolsets=",".join(args.required_toolset),
            provider=primary.provider,
            model=primary.model,
            supervisor_args=args,
        ),
        primary=primary,
    )


def run_worker_preflight(args, *, primary=None) -> None:
    """Run before CLI/context construction; failed admission never retries."""
    if (
        not os.environ.get("HERMES_KANBAN_TASK")
        or os.environ.get("HERMES_KANBAN_PREFLIGHT") != "1"
    ):
        return
    from hermes_constants import get_hermes_home, get_default_hermes_root
    from hermes_cli import kanban_db as kb

    try:
        run_id = int(os.environ["HERMES_KANBAN_RUN_ID"])
    except (KeyError, ValueError):
        raise SystemExit(2) from None
    task_id = os.environ["HERMES_KANBAN_TASK"]
    profile = os.environ.get("HERMES_PROFILE", "")
    started = time.monotonic()
    try:
        with kb.connect_closing() as conn:
            row = conn.execute(
                "SELECT assignee, current_run_id, status, claim_lock FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            if (
                row is None
                or row["current_run_id"] != run_id
                or row["status"] != "running"
                or row["assignee"] != profile
                or row["claim_lock"] != os.environ.get("HERMES_KANBAN_CLAIM_LOCK")
            ):
                raise SystemExit(2)
        expected_home = (
            get_default_hermes_root()
            if profile == "default"
            else get_default_hermes_root() / "profiles" / profile
        )
        if (
            not _ID.fullmatch(profile)
            or expected_home.resolve() != get_hermes_home().resolve()
        ):
            raise WorkerPreflightBlocked("config_invalid", stage="identity")
        settings = dispatch_admission(str(get_hermes_home()))
        if settings is None:
            # Admission changed after spawn; never silently bypass it.
            raise WorkerPreflightBlocked("config_invalid")
        workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
        dispatch_checks(
            workspace, branch=os.environ.get("HERMES_KANBAN_BRANCH"), worktree=False
        )
        if os.path.realpath(os.getcwd()) != os.path.realpath(
            workspace
        ) or os.path.realpath(os.environ.get("TERMINAL_CWD", "")) != os.path.realpath(
            workspace
        ):
            raise WorkerPreflightBlocked("workspace_invalid", stage="workspace")
        native_claude = primary is not None and primary.kind == "claude_code"
        if primary is not None:
            _check_primary_capability(primary, args.supervisor_args)
        if native_claude:
            _check_native_tools(primary, args.supervisor_args, settings)
        else:
            _check_tools(args, settings)
        command = [
            sys.executable,
            "-P",
            "-m",
            "hermes_cli.worker_preflight",
            "--probe",
            "--timeout",
            str(settings["timeout_seconds"]),
            "--native-timeout",
            str(settings["native_timeout_seconds"]),
            "--max-tokens",
            str(settings["max_output_tokens"]),
        ]
        for attr, flag in (("provider", "--provider"), ("model", "--model")):
            if getattr(args, attr, None):
                command.extend([flag, getattr(args, attr)])
        probe_env = None
        if primary is not None and not native_claude:
            from hermes_cli.backend_failover_worker import _leg_command

            _, probe_env = _leg_command(
                primary, args.supervisor_args, dict(os.environ), None, None
            )
        if native_claude:
            command.append("--native-claude")
            scratch = get_hermes_home() / "cache" / "scratch"
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="preflight-", dir=scratch) as cwd:
                result = bounded_probe(
                    command, cwd=cwd, timeout=settings["native_timeout_seconds"] + 2
                )
        else:
            result = bounded_probe(
                command,
                cwd=str(get_hermes_home()),
                timeout=settings["native_timeout_seconds"] + 2,
                env=probe_env,
            )
    except WorkerPreflightBlocked as exc:
        result = exc.payload
    except Exception:
        # Diagnostics must not expose a credential-bearing resolver exception.
        result = {"stage": "probe", "code": "probe_failed", "outcome": "failed"}
    result = sanitize_outcome({
        **result,
        "profile": profile,
        "route": "supervisor" if primary is not None else "direct",
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    })
    passed = (
        result.get("outcome") == "passed" and result.get("code") == "startup_passed"
    )
    result["outcome"] = "passed" if passed else "failed"
    kind = (
        "transient"
        if result["code"] in {"probe_timeout", "probe_transient"}
        else "capability"
    )
    with kb.connect_closing() as conn:
        recorded = kb.record_preflight_outcome(
            conn, task_id, run_id=run_id, passed=passed, payload=result, kind=kind
        )
    print(json.dumps(result, sort_keys=True), file=sys.stderr)
    if not recorded or not passed:
        raise SystemExit(2)
    os.environ.pop("HERMES_KANBAN_PREFLIGHT", None)


def _probe(args) -> dict:
    """Use the actual worker resolver and SDK transport, never an agent."""
    started = time.monotonic()
    result = {"stage": "identity", "code": "model_unresolved", "outcome": "failed"}
    try:
        from hermes_cli.config import load_config
        from hermes_cli.runtime_provider import (
            resolve_requested_provider,
            resolve_runtime_provider,
        )
        from hermes_cli.provider_access import load_provider_access, trusted_profile
        from hermes_cli.model_normalize import (
            _AGGREGATOR_PROVIDERS,
            normalize_model_for_provider,
        )
        from hermes_cli.auth import has_usable_secret, LMSTUDIO_NOAUTH_PLACEHOLDER

        cfg = load_config().get("model") or {}
        model = args.model or (cfg.get("default") if isinstance(cfg, dict) else cfg)
        requested = resolve_requested_provider(args.provider)
        result.update(provider=requested, model=model)
        result.update(stage="credential", code="credential_missing")
        runtime = resolve_runtime_provider(
            requested=requested,
            target_model=model,
            provider_access_snapshot=load_provider_access()
            if trusted_profile()
            else {},
        )
        provider = runtime.get("provider", requested)
        mode = runtime.get("api_mode", "other")
        runtime_model = runtime.get("model")
        if runtime_model and (
            runtime.get("provider_access_mode") == "api"
            or not model
            or model in {provider, runtime.get("name")}
        ):
            model = runtime_model
        if not model:
            result.update(stage="identity", code="model_unresolved")
            return sanitize_outcome(result)
        if provider not in _AGGREGATOR_PROVIDERS:
            model = normalize_model_for_provider(model, provider)
        result.update(
            provider=provider,
            model=model,
            api_mode=mode,
            base_url_host=urlsplit(runtime.get("base_url", "")).hostname,
        )
        key = runtime.get("api_key")
        if callable(key):
            result.update(stage="transport", code="transport_unsupported")
            return sanitize_outcome(result)
        if not has_usable_secret(key) or key in {
            "no-key-required",
            "no-key",
            LMSTUDIO_NOAUTH_PLACEHOLDER,
        }:
            return sanitize_outcome(result)
        if (
            mode not in {"chat_completions", "anthropic_messages", "codex_responses"}
            or (mode == "codex_responses" and provider != "openai-codex")
        ) or runtime.get("command"):
            result.update(stage="transport", code="transport_unsupported")
            return sanitize_outcome(result)
        result.update(stage="probe", code="probe_failed")
        ceiling = (
            getattr(args, "native_timeout", 45)
            if mode == "codex_responses"
            else args.timeout
        )
        timeout = max(0.1, ceiling - (time.monotonic() - started) - 0.5)
        if mode == "codex_responses":
            result.update(_codex_probe(runtime, model, key, timeout))
            return sanitize_outcome(result)
        elif mode == "chat_completions":
            from openai import OpenAI
            from agent.auxiliary_client import auxiliary_max_tokens_param

            with OpenAI(
                api_key=key,
                base_url=runtime["base_url"],
                timeout=timeout,
                max_retries=0,
            ) as client:
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Reply with OK."}],
                    **auxiliary_max_tokens_param(args.max_tokens, model=model),
                )
        else:
            from agent.anthropic_adapter import build_anthropic_client

            with build_anthropic_client(
                key, runtime["base_url"], timeout=timeout
            ) as client:
                client.messages.create(
                    model=model,
                    max_tokens=args.max_tokens,
                    messages=[{"role": "user", "content": "Reply with OK."}],
                )
        result.update(code="startup_passed", outcome="passed")
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if type(status) is int:
            result["http_status"] = status
            result["code"] = (
                "credential_rejected"
                if status in {401, 403}
                else "probe_transient"
                if status == 429 or status >= 500
                else "probe_failed"
            )
        elif result["stage"] == "probe":
            # Both supported SDKs expose these stable exception classes. Do not
            # import an optional, unused SDK while translating another's error.
            if isinstance(exc, TimeoutError) or type(exc).__name__ == "APITimeoutError":
                result["code"] = "probe_timeout"
            elif (
                isinstance(exc, ConnectionError)
                or type(exc).__name__ == "APIConnectionError"
            ):
                result["code"] = "probe_transient"
    return sanitize_outcome(result)


def _codex_probe(runtime: dict, model: str, key: str, timeout: float) -> dict:
    """Use the worker's actual Responses wire; close on first output event."""
    from openai import OpenAI
    from agent.auxiliary_client import _codex_cloudflare_headers
    from agent.transports.codex import ResponsesApiTransport

    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        model,
        messages=[{"role": "user", "content": "Reply with OK."}],
        tools=None,
        instructions="Reply with OK.",
        is_codex_backend=True,
        timeout=timeout,
    )
    deadline = time.monotonic() + timeout
    with OpenAI(
        api_key=key,
        base_url=runtime["base_url"],
        timeout=timeout,
        max_retries=0,
        default_headers=_codex_cloudflare_headers(key),
    ) as client:
        with client.responses.create(**kwargs, stream=True) as stream:
            for index, event in enumerate(stream):
                if index >= 256 or time.monotonic() >= deadline:
                    return {"code": "probe_timeout", "outcome": "failed"}
                if event.type in {"response.failed", "error", "response.incomplete"}:
                    error = getattr(getattr(event, "response", None), "error", None)
                    code = str(
                        getattr(error, "code", None) or getattr(event, "code", "")
                    )
                    transient = any(
                        part in code
                        for part in ("rate_limit", "usage_limit", "server_error")
                    )
                    return {
                        "code": "probe_transient" if transient else "probe_failed",
                        "outcome": "failed",
                    }
                if event.type in {
                    "response.output_item.added",
                    "response.output_text.delta",
                    "response.completed",
                }:
                    return {"code": "startup_passed", "outcome": "passed"}
    return {"code": "probe_failed", "outcome": "failed"}


def _native_claude_probe(args) -> dict:
    """Use native subscription auth, never the Anthropic HTTP API."""
    import shutil
    from hermes_cli.backend_failover_worker import (
        _claude_cli_compatible,
        _claude_environment,
    )
    from hermes_cli.provider_access import claude_source

    result = dict(
        stage="transport",
        code="transport_unsupported",
        outcome="failed",
        provider="claude-code",
        api_mode="claude_native",
        model=args.model,
    )
    deadline = time.monotonic() + args.native_timeout
    if not shutil.which("claude"):
        return result
    try:
        if claude_source()[1] == "none":
            return {**result, "stage": "credential", "code": "credential_missing"}
        env = _claude_environment(dict(os.environ))
        if not _claude_cli_compatible(args.model, env, runner=_native_check):
            return result
        status = _native_check(
            ["claude", "auth", "status", "--json"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=max(0.1, min(5, deadline - time.monotonic())),
            check=False,
        )
        logged_in = json.loads(status.stdout) if len(status.stdout) <= 4096 else {}
        if status.returncode or not (
            logged_in.get("loggedIn") or logged_in.get("logged_in")
        ):
            return {**result, "stage": "credential", "code": "credential_missing"}
    except (ValueError, OSError):
        return {**result, "stage": "credential", "code": "credential_missing"}
    except subprocess.TimeoutExpired:
        return {**result, "stage": "probe", "code": "probe_timeout"}
    command = [
        "claude",
        "-p",
        "--model",
        args.model,
        "--max-turns",
        "1",
        "--tools",
        "",
        "--no-session-persistence",
        "--setting-sources",
        "",
        "--safe-mode",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--output-format",
        "stream-json",
        "--verbose",
        "--",
        "Reply with OK.",
    ]
    result.update(stage="probe", code="probe_failed")
    rc, output, failure = _capture_bounded(
        command,
        cwd=os.getcwd(),
        env=env,
        timeout=max(0.1, deadline - time.monotonic()),
        limit=65536,
    )
    if failure:
        return {**result, "code": failure}
    try:
        events = [json.loads(line) for line in output.splitlines() if line.strip()]
        if not events or not all(isinstance(event, dict) for event in events):
            return result
        for event in events:
            status = event.get("api_error_status")
            if event.get("error") in {
                "authentication_error",
                "permission_error",
            } or status in {401, 403}:
                return {**result, "code": "credential_rejected"}
            if (
                event.get("error") in {"rate_limit_error", "overloaded_error"}
                or type(status) is int
                and (status in {408, 429} or status >= 500)
            ):
                return {**result, "code": "probe_transient"}
        results = [event for event in events if event.get("type") == "result"]
        if (
            rc == 0
            and len(results) == 1
            and results[0].get("subtype") == "success"
            and results[0].get("is_error") is False
            and not any(event.get("is_error") or event.get("error") for event in events)
        ):
            result.update(code="startup_passed", outcome="passed")
    except (ValueError, TypeError):
        pass
    return result


def _native_check(command, **kwargs):
    """Apply the same byte/process bounds to CLI version and auth status."""
    rc, output, failure = _capture_bounded(
        command,
        cwd=os.getcwd(),
        env=kwargs["env"],
        timeout=kwargs.get("timeout", 5),
        limit=4096,
    )
    if failure == "probe_timeout":
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout", 5))
    if failure:
        raise OSError("native check failed")
    return subprocess.CompletedProcess(
        command, rc, output.decode("utf-8", errors="replace"), ""
    )


def main() -> int:
    # Save the result channel before imports can emit a key/response on stdio.
    output_fd = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    logging.disable(logging.CRITICAL)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", required=True)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--native-timeout", type=int, default=45)
    parser.add_argument("--native-claude", action="store_true")
    args = parser.parse_args()
    args.timeout = max(1, min(15, args.timeout))
    args.native_timeout = max(1, min(45, args.native_timeout))
    args.max_tokens = max(1, min(16, args.max_tokens))
    try:
        result = _native_claude_probe(args) if args.native_claude else _probe(args)
    except Exception:
        result = {"stage": "probe", "code": "probe_failed", "outcome": "failed"}
    with os.fdopen(output_fd, "w") as output:
        output.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
