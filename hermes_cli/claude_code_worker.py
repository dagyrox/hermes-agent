"""Dispatcher-owned Kanban worker backed by the real Claude Code CLI.

This is deliberately a narrow worker-lane adapter, not an Anthropic provider alias.
Claude Code owns the model loop; Hermes owns the task claim, scoped MCP lifecycle
tools, heartbeats, logs, timeout/cancellation supervision, and terminal outcome.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any

from agent.redact import redact_sensitive_text
from agent.delegation_context import (
    DELEGATED_CHILD_ENV_MARKER,
    KANBAN_ENV_KEYS,
    delegated_child_subprocess_env,
)
from tools.environments.local import hermes_subprocess_env


TERMINAL_STATUSES = {"done", "blocked", "review", "changes_requested"}
CLAIM_LOST_STATUSES = {"missing", "superseded"}
CLAIM_LOST_EXIT_CODE = 74
PROTOCOL_VIOLATION_EXIT_CODE = 70
TIMEOUT_EXIT_CODE = 124
TERMINATION_GRACE_SECONDS = 5
LIFECYCLE_TOOLS = (
    "kanban_complete", "kanban_block", "kanban_request_review",
    "kanban_request_changes", "kanban_comment", "kanban_heartbeat", "kanban_show",
)
READ_ONLY_DENIED_TOOLS = (
    "Bash", "Edit", "Write", "NotebookEdit", "Task", "TaskOutput",
    "KillShell", "SlashCommand", "WebFetch", "WebSearch",
)
KANBAN_LOCATION_KEYS = (
    "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_PROFILE",
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", default="opus")
    parser.add_argument("--binary", default="claude")
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--read-only", action="store_true")
    return parser.parse_args(argv)


def _call_kanban_tool(name: str, arguments: dict[str, Any]) -> str:
    from model_tools import handle_function_call
    return str(handle_function_call(name, arguments))


def _task_status(task_id: str) -> str:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    raw_run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    if not raw_run_id.isdigit():
        return "missing"
    run_id = int(raw_run_id)
    with kbc.connect() as conn:
        return kb.goal_run_status(conn, task_id, expected_run_id=run_id) or "missing"


def _tool_failed(result: str) -> bool:
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return True
    if not isinstance(payload, dict):
        return True
    return bool(payload.get("error")) or payload.get("ok") is False


def _terminate_process_group(proc: subprocess.Popen[str], *, force: bool = False) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        from hermes_cli._subprocess_compat import windows_hide_flags

        command = ["taskkill", "/PID", str(proc.pid), "/T"]
        if force:
            command.append("/F")
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TERMINATION_GRACE_SECONDS,
                creationflags=windows_hide_flags(),
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            (proc.kill if force else proc.terminate)()
        return
    sig = getattr(signal, "SIGKILL", signal.SIGTERM) if force else signal.SIGTERM
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)  # windows-footgun: ok — POSIX branch only


def _terminate_with_grace(proc: subprocess.Popen[str]) -> None:
    _terminate_process_group(proc)
    try:
        proc.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc, force=True)


def _heartbeat_loop(
    task_id: str,
    stop: threading.Event,
    claim_lost: threading.Event,
    process_ref: list[subprocess.Popen[str] | None],
) -> None:
    while not stop.wait(60):
        try:
            result = _call_kanban_tool(
                "kanban_heartbeat",
                {"task_id": task_id, "note": "Claude Code CLI worker is active."},
            )
        except Exception:
            result = '{"error":"heartbeat failed"}'
        if _tool_failed(result):
            claim_lost.set()
            proc = process_ref[0]
            if proc is not None:
                _terminate_with_grace(proc)
            return


def _mcp_config(profile_home: Path) -> Path:
    env = {
        key: os.environ[key]
        for key in (*KANBAN_ENV_KEYS, *KANBAN_LOCATION_KEYS)
        if os.environ.get(key)
    }
    env["HERMES_HOME"] = str(profile_home)
    env["HERMES_MCP_EXPOSED_TOOLS"] = ",".join(LIFECYCLE_TOOLS)
    env["HERMES_MCP_ASSIGNED_TASK"] = os.environ["HERMES_KANBAN_TASK"]
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    # The MCP endpoint is the explicitly supervised owner callback. The Claude
    # executor itself remains a fenced descendant and cannot mutate the board.
    # Claude itself stays fenced, but MCP config env is merged over the Claude
    # process environment. An explicit empty override clears the inherited marker
    # for this supervised owner callback; omitting the key would inherit the fence.
    env[DELEGATED_CHILD_ENV_MARKER] = ""
    payload = {
        "mcpServers": {
            "hermes-tools": {
                "command": sys.executable,
                "args": ["-m", "agent.transports.hermes_tools_mcp_server"],
                "env": env,
            }
        }
    }
    scratch = profile_home / "cache" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix="claude-kanban-mcp-", suffix=".json", dir=scratch)
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        path.unlink(missing_ok=True)
        raise
    return path


def _system_prompt_file(profile_home: Path) -> Path:
    from agent.prompt_builder import build_context_files_prompt, load_soul_md

    soul = load_soul_md(None, home_override=profile_home)
    if not soul:
        raise RuntimeError(f"Claude Code lane requires a valid SOUL.md in {profile_home}")
    scratch = profile_home / "cache" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix="claude-kanban-system-", suffix=".md", dir=scratch)
    path = Path(raw_path)
    try:
        os.fchmod(fd, 0o600)
        workspace = os.environ.get("HERMES_KANBAN_WORKSPACE") or os.getcwd()
        context_files = build_context_files_prompt(cwd=workspace, skip_soul=True)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(soul)
            if context_files.strip():
                handle.write("\n\n" + context_files.strip() + "\n")
    except Exception:
        with contextlib.suppress(OSError):
            os.close(fd)
        path.unlink(missing_ok=True)
        raise
    return path


def _prompt(task_id: str, read_only: bool) -> str:
    boundary = (
        "This lane is read-only: do not edit/write files, implement fixes, commit, push, "
        "merge, deploy, or approve your own work."
        if read_only else
        "Respect the task's role boundary. A review task must remain non-mutating and may not build, merge, or deploy."
    )
    return f"""You are running as an actual Claude Code CLI Kanban worker for task {task_id}.
{boundary}

Required protocol:
1. Call mcp__hermes-tools__kanban_show first and treat its worker_context as ground truth.
2. Work only in HERMES_KANBAN_WORKSPACE. Preserve exact repo/card/run/work-packet provenance.
3. Use mcp__hermes-tools__kanban_heartbeat during long work and durable comments for findings.
4. Finish exactly once with the correct lifecycle tool: kanban_complete, kanban_request_review,
   kanban_request_changes, or kanban_block. Do not merely print a verdict and exit.
5. Never expose credentials or copy secret-bearing environment values into output.
6. There is no model/provider fallback. If Claude capacity or required access is unavailable,
   report the precise safe blocker through kanban_block.
"""


def _claude_command(
    args: argparse.Namespace, mcp_path: Path, system_prompt_path: Path, prompt: str,
) -> list[str]:
    allowed = [
        "Read", "Glob", "Grep",
        "mcp__hermes-tools__kanban_show", "mcp__hermes-tools__kanban_comment",
        "mcp__hermes-tools__kanban_heartbeat", "mcp__hermes-tools__kanban_complete",
        "mcp__hermes-tools__kanban_block", "mcp__hermes-tools__kanban_request_review",
        "mcp__hermes-tools__kanban_request_changes",
    ]

    cmd = [
        args.binary, "-p", "--model", args.model, "--max-turns", str(args.max_turns),
        "--no-session-persistence", "--output-format", "json",
        "--append-system-prompt-file", str(system_prompt_path),
        "--setting-sources", "",
        "--mcp-config", str(mcp_path), "--strict-mcp-config",
        "--permission-mode", "dontAsk",
        "--allowedTools", *allowed,
    ]
    if args.read_only:
        cmd += ["--disallowedTools", *READ_ONLY_DENIED_TOOLS]
    cmd += ["--", prompt]
    return cmd


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    if os.environ.get("HERMES_KANBAN_TASK") != args.task:
        raise RuntimeError("task argument does not match dispatcher-owned HERMES_KANBAN_TASK")
    if os.environ.get("HERMES_PROFILE") != args.profile:
        raise RuntimeError("profile argument does not match dispatcher-owned HERMES_PROFILE")
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID", "").strip()
    if not run_id.isdigit():
        raise RuntimeError("dispatcher did not provide a valid HERMES_KANBAN_RUN_ID")
    if not args.read_only:
        raise RuntimeError("Claude Code Kanban lanes are review-only and require --read-only")
    profile_home = Path(os.environ.get("HERMES_HOME", "")).expanduser().resolve()
    if not profile_home.is_dir():
        raise RuntimeError("dispatcher did not provide a valid profile HERMES_HOME")

    initial_heartbeat = _call_kanban_tool(
        "kanban_heartbeat", {"task_id": args.task, "note": "Starting Claude Code CLI worker."})
    if _tool_failed(initial_heartbeat):
        raise RuntimeError("Claude Code worker does not own the claimed Kanban run")
    stop = threading.Event()
    claim_lost = threading.Event()
    process_ref: list[subprocess.Popen[str] | None] = [None]
    heartbeat = threading.Thread(
        target=_heartbeat_loop,
        args=(args.task, stop, claim_lost, process_ref),
        daemon=True,
    )
    heartbeat.start()
    mcp_path: Path | None = None
    system_prompt_path: Path | None = None
    proc: subprocess.Popen[str] | None = None
    cancel_requested = threading.Event()

    def cancel(_signum, _frame) -> None:
        cancel_requested.set()
        if proc:
            threading.Thread(target=_terminate_with_grace, args=(proc,), daemon=True).start()

    previous = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        try:
            mcp_path = _mcp_config(profile_home)
            system_prompt_path = _system_prompt_file(profile_home)
        except Exception as exc:
            with contextlib.suppress(Exception):
                _call_kanban_tool(
                    "kanban_comment",
                    {"task_id": args.task, "body": f"Claude Code lane setup failed: {type(exc).__name__}."},
                )
            return 78
        env = hermes_subprocess_env(inherit_credentials=False)
        # Claude Code subscription auth lives in HOME/.claude, not Hermes provider credentials.
        env["HOME"] = str(Path.home())
        env["USER"] = os.environ.get("USER", "root")
        env = delegated_child_subprocess_env(env)
        env["HERMES_KANBAN_WORKSPACE"] = os.environ.get("HERMES_KANBAN_WORKSPACE", "")
        try:
            proc = subprocess.Popen(
                _claude_command(args, mcp_path, system_prompt_path, _prompt(args.task, args.read_only)),
                cwd=os.environ.get("HERMES_KANBAN_WORKSPACE") or None,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            _call_kanban_tool(
                "kanban_comment",
                {"task_id": args.task, "body": f"Claude Code CLI failed to start: {type(exc).__name__}."},
            )
            return 127
        process_ref[0] = proc
        if cancel_requested.is_set():
            _terminate_process_group(proc)
        try:
            stdout, stderr = proc.communicate(timeout=max(1, args.timeout_seconds))
        except subprocess.TimeoutExpired:
            _terminate_with_grace(proc)
            try:
                stdout, stderr = proc.communicate(timeout=TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            status = _task_status(args.task)
            if status in TERMINAL_STATUSES:
                return 0
            if status in CLAIM_LOST_STATUSES:
                return CLAIM_LOST_EXIT_CODE
            with contextlib.suppress(Exception):
                _call_kanban_tool(
                    "kanban_comment",
                    {"task_id": args.task, "body": "Claude Code CLI exceeded its dispatcher-owned wall-clock budget."},
                )
            return TIMEOUT_EXIT_CODE
        except BaseException:
            _terminate_with_grace(proc)
            raise
        safe_out = redact_sensitive_text(stdout, force=True)
        # Claude's prose belongs on the durable board via MCP lifecycle tools, not in
        # process logs. Emit only non-content execution metadata so a model response
        # can never copy an unrecognised secret pattern into the worker log.
        metadata: dict[str, Any] = {}
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(safe_out)
            if isinstance(parsed, dict):
                metadata = {
                    key: parsed.get(key)
                    for key in ("subtype", "terminal_reason", "num_turns", "duration_ms")
                    if parsed.get(key) is not None
                }
                model_usage = parsed.get("modelUsage")
                if isinstance(model_usage, dict):
                    metadata["models"] = sorted(str(key) for key in model_usage)
        if stderr:
            metadata["stderr_present"] = True
        print(f"CLAUDE_CODE_WORKER_EXIT: rc={proc.returncode} metadata={json.dumps(metadata, sort_keys=True)}")
        status = _task_status(args.task)
        if status in TERMINAL_STATUSES:
            return 0
        if status in CLAIM_LOST_STATUSES:
            print("CLAUDE_CODE_WORKER_CLAIM_LOST", file=sys.stderr)
            return CLAIM_LOST_EXIT_CODE
        if claim_lost.is_set():
            print("CLAUDE_CODE_WORKER_CLAIM_LOST", file=sys.stderr)
            return CLAIM_LOST_EXIT_CODE
        if cancel_requested.is_set():
            return 130
        if proc.returncode != 0:
            with contextlib.suppress(Exception):
                _call_kanban_tool(
                    "kanban_comment",
                    {"task_id": args.task, "body": f"Claude Code CLI exited nonzero (rc={proc.returncode}); dispatcher retry policy applies."},
                )
            return 69
        print(
            "CLAUDE_CODE_WORKER_PROTOCOL_VIOLATION: CLI exited successfully without a terminal Kanban handoff.",
            file=sys.stderr,
        )
        return PROTOCOL_VIOLATION_EXIT_CODE
    finally:
        stop.set()
        heartbeat.join(timeout=2)
        if mcp_path is not None:
            mcp_path.unlink(missing_ok=True)
        if system_prompt_path is not None:
            system_prompt_path.unlink(missing_ok=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    from hermes_cli.quiet_single_query import exit_single_query

    exit_single_query(run())