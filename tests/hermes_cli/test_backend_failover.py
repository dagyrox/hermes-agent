from __future__ import annotations

from types import SimpleNamespace

import pytest


def _chain(*legs):
    return {"enabled": True, "max_attempts_per_leg": 1, "roles": {"talos": list(legs)}}


def test_canonical_chain_requires_exactly_three_legs_and_flash_final():
    from hermes_cli.backend_failover import ConfigError, resolve_role_chain

    valid = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    chain = resolve_role_chain(valid, "talos")
    assert [(leg.kind, leg.provider, leg.model) for leg in chain] == [
        ("api", "openai-codex", "gpt-6.1-sol"),
        ("claude_code", None, "claude-sonnet-5-5"),
        ("api", "deepseek", "deepseek-flash"),
    ]

    with pytest.raises(ConfigError, match="exactly three"):
        resolve_role_chain(_chain(*valid["roles"]["talos"][:2]), "talos")
    invalid_tail = _chain(*valid["roles"]["talos"])
    invalid_tail["roles"]["talos"][-1] = {
        "kind": "api", "provider": "deepseek", "model": "deepseek-pro"
    }
    with pytest.raises(ConfigError, match="final leg"):
        resolve_role_chain(invalid_tail, "talos")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_direct_api_providers_are_rejected(provider):
    from hermes_cli.backend_failover import ConfigError, resolve_role_chain

    cfg = _chain(
        {"kind": "api", "provider": provider, "model": "forbidden"},
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    with pytest.raises(ConfigError, match="direct API provider"):
        resolve_role_chain(cfg, "talos")


def test_rejects_more_than_one_attempt_per_leg():
    from hermes_cli.backend_failover import ConfigError, resolve_role_chain

    cfg = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    cfg["max_attempts_per_leg"] = 2
    with pytest.raises(ConfigError, match="exactly 1"):
        resolve_role_chain(cfg, "talos")


def test_rejects_duplicate_providers_even_when_models_differ():
    from hermes_cli.backend_failover import ConfigError, resolve_role_chain

    cfg = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6-sol"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    with pytest.raises(ConfigError, match="three distinct providers"):
        resolve_role_chain(cfg, "talos")


def test_excluded_lanes_cannot_be_configured():
    from hermes_cli.backend_failover import ConfigError, resolve_role_chain

    cfg = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6-sol"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    cfg["roles"]["sentinel"] = cfg["roles"].pop("talos")
    with pytest.raises(ConfigError, match="excluded lane"):
        resolve_role_chain(cfg, "sentinel")


@pytest.mark.parametrize("profile", ["forge", "guest", "guestwhatsapp", "gf", "orion"])
def test_other_excluded_lanes_cannot_receive_routes(profile):
    from hermes_cli.backend_failover import ConfigError, resolve_role_chain

    cfg = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    cfg["roles"][profile] = cfg["roles"].pop("talos")
    with pytest.raises(ConfigError, match="excluded lane"):
        resolve_role_chain(cfg, profile)


def test_excluded_lane_without_route_remains_unaffected():
    from hermes_cli.backend_failover import resolve_role_chain

    cfg = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    assert resolve_role_chain(cfg, "sentinel") == ()


def test_failure_exit_classification_refuses_policy_hopping_and_replay():
    from hermes_cli.backend_failover import (
        EXIT_AMBIGUOUS,
        EXIT_ELIGIBLE,
        EXIT_REFUSAL,
        classify_failed_result,
    )

    assert classify_failed_result({"failed": True, "failure_reason": "auth", "should_fallback": True}) == EXIT_ELIGIBLE
    assert classify_failed_result({"failed": True, "failure_reason": "format_error", "should_fallback": True}) == EXIT_ELIGIBLE
    assert classify_failed_result({"failed": True, "failure_reason": "rate_limit"}) != EXIT_ELIGIBLE
    assert classify_failed_result({
        "failed": True, "should_fallback": True,
        "failure_reason": "auth",
        "messages": [{"role": "assistant", "tool_calls": [{"id": "call_1"}]}],
    }) == EXIT_AMBIGUOUS
    assert classify_failed_result({
        "failed": True, "failure_reason": "content_policy_blocked", "should_fallback": True
    }) == EXIT_REFUSAL


def test_canonical_role_route_is_frozen():
    from hermes_cli.backend_failover import ConfigError, resolve_role_chain

    wrong_model = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    with pytest.raises(ConfigError, match="frozen route"):
        resolve_role_chain(wrong_model, "talos")

    unsupported_key = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5", "unexpected": True},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    with pytest.raises(ConfigError, match="unsupported keys"):
        resolve_role_chain(unsupported_key, "talos")


def test_disabled_staged_routes_are_validated():
    from hermes_cli.backend_failover import ConfigError, validate_backend_failover_config

    cfg = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "wrong"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    cfg["enabled"] = False
    with pytest.raises(ConfigError, match="frozen route"):
        validate_backend_failover_config(cfg)


def test_capability_parity_fails_closed_when_runtime_contract_is_narrowed(monkeypatch):
    from hermes_cli import backend_failover as failover

    monkeypatch.setattr(
        failover,
        "_API_CAPABILITIES",
        getattr(failover, "_API_CAPABILITIES") - {"image_input"},
    )
    with pytest.raises(failover.ConfigError, match="image_input"):
        failover.resolve_role_chain(_chain(
            {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
            {"kind": "claude_code", "model": "claude-sonnet-5-5"},
            {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
        ), "talos")


def test_supervisor_uses_each_leg_once_and_never_adds_a_fourth():
    from hermes_cli.backend_failover import EXIT_ELIGIBLE, supervise_chain

    legs = (SimpleNamespace(model="one"), SimpleNamespace(model="two"), SimpleNamespace(model="three"))
    attempts = []

    def run_leg(leg, index):
        attempts.append((leg.model, index))
        return EXIT_ELIGIBLE

    outcome = supervise_chain(legs, run_leg=run_leg)
    assert outcome.exhausted is True
    assert outcome.returncode == EXIT_ELIGIBLE
    assert attempts == [("one", 0), ("two", 1), ("three", 2)]


def test_supervisor_stops_on_refusal_and_ambiguous_side_effect():
    from hermes_cli.backend_failover import EXIT_AMBIGUOUS, EXIT_REFUSAL, supervise_chain

    legs = (SimpleNamespace(model="one"), SimpleNamespace(model="two"), SimpleNamespace(model="three"))
    refusal_attempts = []
    refusal = supervise_chain(
        legs,
        run_leg=lambda leg, index: refusal_attempts.append(index) or EXIT_REFUSAL,
    )
    assert refusal.returncode == EXIT_REFUSAL
    assert refusal_attempts == [0]

    ambiguous_attempts = []
    ambiguous = supervise_chain(
        legs,
        run_leg=lambda leg, index: ambiguous_attempts.append(index) or EXIT_AMBIGUOUS,
    )
    assert ambiguous.returncode == EXIT_AMBIGUOUS
    assert ambiguous_attempts == [0]


def test_dispatcher_argv_routes_enabled_profile_through_supervisor():
    from hermes_cli.backend_failover import worker_argv

    task = SimpleNamespace(id="t_probe", max_runtime_seconds=600)
    argv = worker_argv(task, "talos", None, ["terminal", "vision"])
    assert argv[:3] == [pytest.importorskip("sys").executable, "-m", "hermes_cli.backend_failover_worker"]
    assert argv[argv.index("--task") + 1] == "t_probe"
    assert argv[argv.index("--profile") + 1] == "talos"
    assert argv[argv.index("--timeout-seconds") + 1] == "600"
    assert argv.count("--required-toolset") == 2
    assert argv[argv.index("--required-context-chars") + 1] == "0"


def test_supervised_child_disables_nested_provider_fallback(monkeypatch):
    from hermes_cli.fallback_config import get_fallback_chain

    config = {
        "fallback_providers": [
            {"provider": "deepseek", "model": "deepseek-flash"}
        ]
    }
    monkeypatch.setenv("HERMES_BACKEND_FAILOVER_CHILD", "1")
    assert get_fallback_chain(config) == []


def test_status_distinguishes_requested_activation_from_live_activation():
    from hermes_cli.backend_failover import build_status

    cfg = _chain(
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    )
    status = build_status(
        cfg,
        availability_check=lambda _profile, leg: leg.provider != "deepseek",
    )
    assert status["configured"] is True
    assert status["active"] is True
    assert status["activation_requested"] is True
    assert status["live_activated"] is False
    assert status["roles"]["talos"]["available"] is False
    assert status["roles"]["talos"]["activation_requested"] is True
    assert status["roles"]["talos"]["live_activated"] is False
    assert [leg["active"] for leg in status["roles"]["talos"]["chain"]] == [
        True,
        False,
        False,
    ]
