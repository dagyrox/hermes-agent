"""Interactive native Claude entry point sharing the worker credential builder.

Launch with ``python -m hermes_cli.native_claude --profile athena``. The parent
never exports secrets; only the selected child receives its credential scope.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from hermes_constants import get_default_hermes_root, reset_hermes_home_override, set_hermes_home_override
from hermes_cli.backend_failover import ConfigError
from hermes_cli.provider_access import (
    NATIVE_ENV_KEYS,
    claude_environment,
    claude_source,
    load_provider_access,
    native_home,
    provider_access_status,
    trusted_profile,
)


def launch_command(args: argparse.Namespace, base: dict[str, str]) -> tuple[list[str], dict[str, str]]:
    if not trusted_profile():
        raise ConfigError("Native launch requires a trusted internal profile")
    mode = load_provider_access()["anthropic"]
    if args.login:
        # Recovery is operator-driven; no token is seeded, revoked or copied.
        env = {key: value for key, value in base.items() if key in NATIVE_ENV_KEYS}
        env["HOME"] = str(native_home())
        return [args.claude_bin, "auth", "login", "--claudeai"], env
    if mode["mode"] == "api":
        from hermes_cli.provider_access import root_api_key
        from hermes_constants import get_hermes_home

        if args.native_args:
            raise ConfigError("Native-only CLI arguments cannot be replayed on the API route; use an interactive launch or --query")
        root_api_key("anthropic")  # fail before launching; never place the key in env
        command = [shutil.which("hermes") or "hermes", "-p", args.profile, "--cli",
                   "--provider", "anthropic", "-m", mode["api_model"]]
        if args.query:
            command.extend(["chat", "-q", args.query])
        env = {key: value for key, value in base.items() if key in NATIVE_ENV_KEYS}
        env["HOME"] = str(native_home())
        env["HERMES_HOME"] = str(get_hermes_home())
        env["HERMES_BACKEND_FAILOVER_CHILD"] = "1"
        env["HERMES_PROVIDER_ACCESS_SNAPSHOT"] = json.dumps({"anthropic": mode})
        return command, env
    if claude_source()[1] == "none":
        raise ConfigError("Missing native Claude authentication: run claude setup-token / claude auth login under the canonical HOME")
    command = [args.claude_bin, "--setting-sources", "", "--model", args.model]
    if args.name:
        command.extend(["--name", args.name])
    if args.query:
        command.extend(["-p", "--max-turns", "1", "--tools", "", "--no-session-persistence",
                        "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}', "--", args.query])
    else:
        command.extend(args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args)
    return command, claude_environment(base)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--name")
    parser.add_argument("--query")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--login", action="store_true")
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Profile identity is path-derived; reject traversal before constructing it.
    if args.profile != "default" and (Path(args.profile).name != args.profile or args.profile in {".", ".."}):
        parser.error("invalid profile name")
    root = get_default_hermes_root()
    home = root if args.profile == "default" else root / "profiles" / args.profile
    token = set_hermes_home_override(home)
    try:
        if args.status:
            print(json.dumps(provider_access_status(), indent=2, sort_keys=True))
            return 0
        command, env = launch_command(args, dict(os.environ))
        return subprocess.run(command, env=env, check=False).returncode
    except (ConfigError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    finally:
        reset_hermes_home_override(token)


if __name__ == "__main__":
    raise SystemExit(main())
