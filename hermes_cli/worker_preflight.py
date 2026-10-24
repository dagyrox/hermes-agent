"""Opt-in, pre-context Kanban worker admission (no recovery or auth repair).

The assignee's kanban.worker_preflight config is default-off. An admitted
worker probes its own resolved HTTP chat-completions or Anthropic Messages
transport once, capped at 15 seconds and 16 output tokens. Keyless endpoints,
native subscription transports and supervised failover chains are unsupported
and block explicitly. MCP toolsets still discovering also fail closed.

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
    if set(raw) - {"enabled", "timeout_seconds", "max_output_tokens", "required_tools"}:
        raise WorkerPreflightBlocked("config_invalid")
    settings = {"enabled": True}
    for key, ceiling in (("timeout_seconds", 15), ("max_output_tokens", 16)):
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
    from tools.registry import registry
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
    missing = []
    for name in ("kanban_complete", "kanban_block", "kanban_heartbeat"):
        entry = registry.get_entry(name)
        if name not in available or entry is None or not callable(entry.handler):
            missing.append(name)
    if missing:
        raise WorkerPreflightBlocked(
            "operation_unavailable", stage="operations", missing=missing
        )


def bounded_probe(command: list[str], *, cwd: str, timeout: int) -> dict:
    """One owned subprocess bounds imports, credential resolution and I/O."""
    kill_group = getattr(os, "killpg", None)
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=os.name == "posix",
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix" and kill_group is not None:
            kill_group(proc.pid, signal.SIGTERM)
        else:
            proc.kill()
        try:
            proc.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            if os.name == "posix" and kill_group is not None:
                kill_group(proc.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
            else:
                proc.kill()
            proc.communicate()
        return {"stage": "probe", "code": "probe_timeout", "outcome": "failed"}
    try:
        if len(output) > 4096 or proc.returncode != 0:
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


def run_worker_preflight(args) -> None:
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
        _check_tools(args, settings)
        command = [
            sys.executable,
            "-P",
            "-m",
            "hermes_cli.worker_preflight",
            "--probe",
            "--timeout",
            str(settings["timeout_seconds"]),
            "--max-tokens",
            str(settings["max_output_tokens"]),
        ]
        for attr, flag in (("provider", "--provider"), ("model", "--model")):
            if getattr(args, attr, None):
                command.extend([flag, getattr(args, attr)])
        result = bounded_probe(
            command, cwd=str(get_hermes_home()), timeout=settings["timeout_seconds"]
        )
    except WorkerPreflightBlocked as exc:
        result = exc.payload
    except Exception:
        # Diagnostics must not expose a credential-bearing resolver exception.
        result = {"stage": "probe", "code": "probe_failed", "outcome": "failed"}
    result = sanitize_outcome({
        **result,
        "profile": profile,
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
        if mode not in {"chat_completions", "anthropic_messages"} or runtime.get(
            "command"
        ):
            result.update(stage="transport", code="transport_unsupported")
            return sanitize_outcome(result)
        result.update(stage="probe", code="probe_failed")
        timeout = max(1, args.timeout - (time.monotonic() - started) - 0.5)
        if mode == "chat_completions":
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
    args = parser.parse_args()
    args.timeout = max(1, min(15, args.timeout))
    args.max_tokens = max(1, min(16, args.max_tokens))
    result = _probe(args)
    with os.fdopen(output_fd, "w") as output:
        output.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
