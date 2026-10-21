"""Read-only live probes for canonical backend routes.

The smoke runner is deliberately isolated from dispatcher, session, and
delegation identity.  A probe can verify an already-authorized account, but it
cannot inherit ownership of the Kanban task that launched it or receive any
model-call tools.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Mapping, Sequence

_SMOKE_ENV_ALLOWLIST = frozenset({
    "HOME",
    "USER",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
})
_SMOKE_PROMPT = "Reply with exactly BACKEND_SMOKE_OK and no other text. Do not use tools."
_NO_TOOLS_TOOLSET = "__backend_smoke_no_tools__"
_CLAUDE_PROVIDER = "claude-code-cli"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _resolve_hermes_argv() -> list[str]:
    from hermes_cli.kanban_db import _resolve_hermes_argv as resolve

    return resolve()


def sanitized_smoke_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Construct an allowlisted environment with no inherited agent identity.

    Provider credentials are intentionally not copied from the parent process.
    API routes resolve credentials from the selected Hermes profile, while the
    Claude CLI uses its existing subscription state under HOME.
    """

    return {
        key: value
        for key, value in base.items()
        if key in _SMOKE_ENV_ALLOWLIST and value
    }


def build_smoke_command(*, profile: str, provider: str, model: str) -> list[str]:
    """Build a one-turn, no-tools probe for one canonical backend route."""

    from hermes_cli.backend_failover import _CANONICAL_ROUTES

    profile_key = str(profile or "default").strip().lower() or "default"
    requested = (
        ("claude_code", None, model)
        if provider == _CLAUDE_PROVIDER
        else ("api", provider, model)
    )
    if requested not in _CANONICAL_ROUTES.get(profile_key, ()):
        raise ValueError(
            f"{provider}/{model} is not a canonical backend route for {profile_key}"
        )

    if provider == _CLAUDE_PROVIDER:
        return [
            "claude",
            "-p",
            "--model",
            model,
            "--max-turns",
            "1",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--safe-mode",
            "--strict-mcp-config",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--tools",
            "",
            "--",
            _SMOKE_PROMPT,
        ]

    return [
        *_resolve_hermes_argv(),
        "-p",
        profile,
        "--cli",
        "--ignore-rules",
        "--toolsets",
        _NO_TOOLS_TOOLSET,
        "-m",
        model,
        "--provider",
        provider,
        "chat",
        "--max-turns",
        "1",
        "-q",
        _SMOKE_PROMPT,
    ]


def smoke_succeeded(provider: str, stdout: str, stderr: str) -> bool:
    """Validate an exact no-tools response instead of substring matching."""

    if stderr.strip():
        return False
    if provider != _CLAUDE_PROVIDER:
        plain = _ANSI_ESCAPE.sub("", stdout).replace("\r", "")
        lines = [line.strip() for line in plain.splitlines()]
        exact_responses = [line for line in lines if line == "BACKEND_SMOKE_OK"]
        no_tools = any(
            line.startswith("Messages:") and "0 tool calls" in line
            for line in lines
        )
        return len(exact_responses) == 1 and no_tools

    events: list[object] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            return False
    compact = json.dumps(events, separators=(",", ":"))
    if not events or '"type":"tool_use"' in compact:
        return False
    results = [
        event.get("result")
        for event in events
        if isinstance(event, dict) and event.get("type") == "result"
    ]
    return results == ["BACKEND_SMOKE_OK"]


def run_smoke_command(
    command: Sequence[str],
    *,
    base_env: Mapping[str, str] | None = None,
    timeout: float = 90,
) -> subprocess.CompletedProcess[str]:
    """Run a smoke child with no inherited task/session/delegation context."""

    return subprocess.run(
        list(command),
        env=sanitized_smoke_environment(base_env or os.environ),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one read-only, no-tools backend model smoke probe."
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=90)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    command = build_smoke_command(
        profile=args.profile,
        provider=args.provider,
        model=args.model,
    )
    try:
        completed = run_smoke_command(command, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired:
        print(json.dumps({
            "ok": False,
            "provider": args.provider,
            "model": args.model,
            "reason": "timeout",
        }, sort_keys=True))
        return 124

    ok = completed.returncode == 0 and smoke_succeeded(
        args.provider,
        completed.stdout,
        completed.stderr,
    )
    print(json.dumps({
        "ok": ok,
        "provider": args.provider,
        "model": args.model,
        "returncode": completed.returncode,
    }, sort_keys=True))
    return 0 if ok else (completed.returncode or 1)


if __name__ == "__main__":
    raise SystemExit(main())
