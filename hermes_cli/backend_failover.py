"""Validated cross-runtime backend chains for dispatcher-owned Kanban workers.

The normal Hermes fallback chain remains the fast in-process path.  This module
owns only the coarser task-boundary route needed when a chain contains an
external runtime such as the authenticated Claude Code CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import sys
from typing import Any, Callable, Iterable, Sequence


# Child-to-supervisor protocol.  Values are sysexits-compatible but internal;
# they are never presented as user-facing configuration.
EXIT_ELIGIBLE = 75
EXIT_REFUSAL = 77
EXIT_AMBIGUOUS = 78

_ELIGIBLE_REASONS = frozenset({
    "auth",
    "auth_permanent",
    "billing",
    "rate_limit",
    "upstream_rate_limit",
    "overloaded",
    "server_error",
    "timeout",
    "model_not_found",
    "format_error",
})
_REFUSAL_REASONS = frozenset({"content_policy_blocked", "safety_refusal", "refusal"})
_EXCLUDED_LANES = frozenset({"sentinel", "forge", "guest", "guestwhatsapp", "gf", "orion"})
_FORBIDDEN_API_PROVIDERS = frozenset({"openai", "anthropic"})
_ROLE_CLAUDE_MODES = {
    "athena": "read_only",
    "apollo": "read_only",
    "hyperion": "orchestrator",
}
_CANONICAL_ROUTES = {
    "default": (("api", "openai-codex", "gpt-6.1-sol"), ("claude_code", None, "claude-sonnet-5-5"), ("api", "deepseek", "deepseek-flash")),
    "hyperion": (("api", "openai-codex", "gpt-6-luna"), ("claude_code", None, "claude-sonnet-5-5"), ("api", "deepseek", "deepseek-flash")),
    "athena": (("claude_code", None, "claude-opus-5-5"), ("api", "openai-codex", "gpt-6.1-sol"), ("api", "deepseek", "deepseek-flash")),
    "talos": (("api", "openai-codex", "gpt-6.1-sol"), ("claude_code", None, "claude-sonnet-5-5"), ("api", "deepseek", "deepseek-flash")),
    "apollo": (("claude_code", None, "claude-sonnet-5-5"), ("api", "openai-codex", "gpt-6.1-sol"), ("api", "deepseek", "deepseek-flash")),
}
_TOP_LEVEL_KEYS = frozenset({"enabled", "max_attempts_per_leg", "roles"})
_LEG_KEYS = frozenset({"kind", "provider", "model", "reasoning_effort", "capability_mode"})


class ConfigError(ValueError):
    """A configured backend chain violates a fail-closed invariant."""


@dataclass(frozen=True)
class BackendLeg:
    kind: str
    model: str
    provider: str | None = None
    reasoning_effort: str | None = None
    capability_mode: str = "full"


@dataclass(frozen=True)
class SupervisorOutcome:
    returncode: int
    attempted: int
    exhausted: bool


def _has_tool_activity(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool" or message.get("tool_calls"):
            return True
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in {"tool_use", "tool_result"}
            for part in content
        ):
            return True
    return False


def classify_failed_result(result: dict[str, Any]) -> int:
    """Map one failed Hermes turn to the cross-runtime supervisor protocol.

    Any observed tool activity is conservatively ambiguous.  Replaying the task
    on another runtime could duplicate an external write, so the supervisor must
    stop for reconciliation rather than guessing whether the tool was read-only.
    """

    if _has_tool_activity(result.get("messages")):
        return EXIT_AMBIGUOUS
    reason = str(result.get("failure_reason") or "").strip().lower()
    if reason in _REFUSAL_REASONS:
        return EXIT_REFUSAL
    if result.get("should_fallback") is True and reason in _ELIGIBLE_REASONS:
        return EXIT_ELIGIBLE
    return 1


def resolve_role_chain(config: dict[str, Any] | None, profile: str) -> tuple[BackendLeg, ...]:
    """Validate and return one profile's exact three-leg canonical chain."""

    cfg = config or {}
    unknown_top = set(cfg) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ConfigError(
            "agent_backend_failover contains unsupported keys: "
            + ", ".join(sorted(unknown_top))
        )
    if not cfg.get("enabled"):
        return ()
    if cfg.get("max_attempts_per_leg", 1) != 1:
        raise ConfigError("max_attempts_per_leg must be exactly 1")
    profile_key = str(profile or "default").strip().lower() or "default"
    if profile_key not in _CANONICAL_ROUTES and profile_key not in _EXCLUDED_LANES:
        raise ConfigError(f"unknown backend failover role {profile_key!r}")
    roles = cfg.get("roles")
    if not isinstance(roles, dict):
        raise ConfigError("agent_backend_failover.roles must be a mapping")
    raw_chain = roles.get(profile_key)
    if raw_chain is None:
        return ()
    if profile_key in _EXCLUDED_LANES:
        raise ConfigError(f"excluded lane {profile_key!r} cannot have a backend chain")
    if not isinstance(raw_chain, list) or len(raw_chain) != 3:
        raise ConfigError(f"{profile_key} must define exactly three backend legs")

    chain: list[BackendLeg] = []
    for index, raw in enumerate(raw_chain):
        if not isinstance(raw, dict):
            raise ConfigError(f"{profile_key} leg {index} must be a mapping")
        unknown_leg = set(raw) - _LEG_KEYS
        if unknown_leg:
            raise ConfigError(
                f"{profile_key} leg {index} contains unsupported keys: "
                + ", ".join(sorted(unknown_leg))
            )
        if any(key in raw for key in ("api_key", "token", "secret", "password")):
            raise ConfigError(f"{profile_key} leg {index} contains an inline secret")
        kind = str(raw.get("kind") or "").strip().lower()
        model = str(raw.get("model") or "").strip()
        provider = str(raw.get("provider") or "").strip().lower() or None
        effort = str(raw.get("reasoning_effort") or "").strip() or None
        capability_mode = str(
            raw.get("capability_mode")
            or (_ROLE_CLAUDE_MODES.get(profile_key) if kind == "claude_code" else "full")
            or "full"
        ).strip().lower()
        if kind not in {"api", "claude_code"} or not model:
            raise ConfigError(f"{profile_key} leg {index} has invalid kind/model")
        if kind == "api":
            if not provider:
                raise ConfigError(f"{profile_key} API leg {index} requires provider")
            if provider in _FORBIDDEN_API_PROVIDERS:
                raise ConfigError(f"direct API provider {provider!r} is forbidden")
        elif provider:
            raise ConfigError("Claude Code subscription legs must not declare an API provider")
        if capability_mode not in {"full", "read_only", "orchestrator"}:
            raise ConfigError(f"{profile_key} leg {index} has invalid capability_mode")
        if kind == "claude_code" and profile_key in _ROLE_CLAUDE_MODES:
            required_mode = _ROLE_CLAUDE_MODES[profile_key]
            if capability_mode != required_mode:
                raise ConfigError(
                    f"{profile_key} Claude Code leg must use {required_mode} capability mode"
                )
        leg = BackendLeg(kind, model, provider, effort, capability_mode)
        chain.append(leg)

    final = chain[-1]
    if (final.kind, final.provider, final.model) != ("api", "deepseek", "deepseek-flash"):
        raise ConfigError("final leg must be deepseek/deepseek-flash")
    provider_identities = [
        "claude-code" if leg.kind == "claude_code" else str(leg.provider)
        for leg in chain
    ]
    if len(set(provider_identities)) != len(provider_identities):
        raise ConfigError(
            f"{profile_key} must use three distinct providers; got "
            + ", ".join(provider_identities)
        )
    actual_route = tuple((leg.kind, leg.provider, leg.model) for leg in chain)
    if actual_route != _CANONICAL_ROUTES[profile_key]:
        raise ConfigError(f"{profile_key} does not match its frozen route")
    return tuple(chain)


def validate_backend_failover_config(config: dict[str, Any] | None) -> None:
    """Validate every staged role even while activation remains disabled."""

    cfg = config or {}
    unknown_top = set(cfg) - _TOP_LEVEL_KEYS
    if unknown_top:
        raise ConfigError(
            "agent_backend_failover contains unsupported keys: "
            + ", ".join(sorted(unknown_top))
        )
    roles = cfg.get("roles", {})
    if not isinstance(roles, dict):
        raise ConfigError("agent_backend_failover.roles must be a mapping")
    if cfg.get("max_attempts_per_leg", 1) != 1:
        raise ConfigError("max_attempts_per_leg must be exactly 1")
    enabled_cfg = dict(cfg)
    enabled_cfg["enabled"] = True
    for role in roles:
        resolve_role_chain(enabled_cfg, str(role))


def supervise_chain(
    legs: Sequence[Any],
    *,
    run_leg: Callable[[Any, int], int],
) -> SupervisorOutcome:
    """Run each configured leg at most once; stop on success or unsafe failure."""

    last_rc = 1
    attempted = 0
    for index, leg in enumerate(legs):
        attempted += 1
        last_rc = int(run_leg(leg, index))
        if last_rc == 0:
            return SupervisorOutcome(0, attempted, False)
        if last_rc != EXIT_ELIGIBLE:
            return SupervisorOutcome(last_rc, attempted, False)
    return SupervisorOutcome(last_rc, attempted, True)


def worker_argv(
    task: Any,
    profile: str,
    chain: Sequence[BackendLeg] | None = None,
    required_toolsets: Sequence[str] | None = None,
) -> list[str]:
    configured_timeout = getattr(task, "max_runtime_seconds", None)
    timeout = max(0, int(configured_timeout)) if configured_timeout is not None else 0
    argv = [
        sys.executable,
        "-m",
        "hermes_cli.backend_failover_worker",
        "--profile",
        profile,
        "--task",
        str(task.id),
        "--timeout-seconds",
        str(timeout),
    ]
    if chain:
        argv.extend([
            "--chain-json",
            json.dumps(chain_as_dicts(chain), separators=(",", ":")),
        ])
    for toolset in required_toolsets or ():
        argv.extend(["--required-toolset", str(toolset)])
    context_chars = len(str(getattr(task, "title", "") or "")) + len(
        str(getattr(task, "body", "") or "")
    )
    argv.extend(["--required-context-chars", str(context_chars)])
    for skill in getattr(task, "skills", None) or ():
        if skill:
            argv.extend(["--skill", str(skill)])
    return argv


def load_profile_chain(profile: str, hermes_home: str | None) -> tuple[BackendLeg, ...]:
    """Load and validate a role chain from the dispatcher's authoritative config.

    ``hermes_home`` is captured before the dispatcher switches environment to
    the assignee profile, keeping one role-to-chain map authoritative.
    """

    if not hermes_home:
        return ()
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.config import load_config

    token = set_hermes_home_override(hermes_home)
    try:
        config = load_config()
    finally:
        reset_hermes_home_override(token)
    raw = config.get("agent_backend_failover")
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ConfigError("agent_backend_failover must be a mapping")
    return resolve_role_chain(raw, profile)


def chain_as_dicts(legs: Iterable[BackendLeg]) -> list[dict[str, str]]:
    """Secret-free shape for diagnostics and deterministic tests."""

    rows: list[dict[str, str]] = []
    for leg in legs:
        row = {"kind": leg.kind, "model": leg.model}
        if leg.provider:
            row["provider"] = leg.provider
        if leg.reasoning_effort:
            row["reasoning_effort"] = leg.reasoning_effort
        if leg.kind == "claude_code":
            row["capability_mode"] = leg.capability_mode
        rows.append(row)
    return rows


def build_status(
    config: dict[str, Any] | None,
    *,
    availability_check: Callable[[str, BackendLeg], bool],
) -> dict[str, Any]:
    """Return secret-free configured/available/active route diagnostics."""

    cfg = config or {}
    roles_raw = cfg.get("roles")
    roles: dict[str, Any] = roles_raw if isinstance(roles_raw, dict) else {}
    activation_requested = bool(cfg.get("enabled"))
    result: dict[str, Any] = {
        "configured": bool(roles),
        # ``enabled`` is requested policy, not runtime readback from a live
        # dispatcher. Keep ``active`` as the compatibility spelling while
        # refusing to claim that activation has been observed.
        "active": activation_requested,
        "activation_requested": activation_requested,
        "live_activated": False,
        "roles": {},
    }
    for profile in sorted(roles):
        chain = resolve_role_chain({**cfg, "enabled": True}, profile)
        leg_rows = []
        for index, leg in enumerate(chain):
            row: dict[str, Any] = chain_as_dicts((leg,))[0]
            row.update({
                "position": index + 1,
                "available": bool(availability_check(profile, leg)),
                "active": bool(cfg.get("enabled")) and index == 0,
            })
            leg_rows.append(row)
        result["roles"][profile] = {
            "configured": True,
            "active": activation_requested,
            "activation_requested": activation_requested,
            "live_activated": False,
            "available": all(row["available"] for row in leg_rows),
            "chain": leg_rows,
        }
    return result
