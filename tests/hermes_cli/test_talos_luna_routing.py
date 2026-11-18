"""Owner-approved frozen routing migration, not a general model catalog snapshot."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def canonical_config():
    path = Path(__file__).resolve().parents[2] / "docs/agentic-operations/backend-failover.canonical.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["agent_backend_failover"]


def test_talos_luna_preserves_subscription_fallbacks_and_capabilities(canonical_config):
    from hermes_cli.backend_failover import chain_as_dicts, resolve_role_chain, validate_backend_failover_config
    from hermes_cli.backend_failover_worker import _load_chain

    config = deepcopy(canonical_config)
    # Express the requested contract even before the checked-in map is migrated.
    config["roles"]["talos"][0]["model"] = "gpt-6-luna"
    validate_backend_failover_config(config)  # Disabled staged maps must also validate.
    config["enabled"] = True
    chain = resolve_role_chain(config, "talos")
    assert [(leg.kind, leg.provider, leg.model, leg.capability_mode) for leg in chain] == [
        ("api", "openai-codex", "gpt-6-luna", "full"),
        ("claude_code", None, "claude-sonnet-5-5", "full"),
        ("api", "deepseek", "deepseek-flash", "full"),
    ]
    assert [leg.reasoning_effort for leg in chain] == ["high", "high", None]
    assert all(leg.access_mode == "subscription" for leg in chain)
    assert _load_chain("talos", json.dumps(chain_as_dicts(chain))) == chain


@pytest.mark.parametrize("enabled", [False, True])
def test_previous_talos_sol_route_is_rejected_even_when_staged(canonical_config, enabled):
    from hermes_cli.backend_failover import ConfigError, validate_backend_failover_config

    config = deepcopy(canonical_config)
    config["enabled"] = enabled
    config["roles"]["talos"][0]["model"] = "gpt-6.1-sol"
    with pytest.raises(ConfigError, match="talos does not match its frozen route"):
        validate_backend_failover_config(config)


def test_checked_in_map_resolves_all_roles_without_changing_other_contracts(canonical_config):
    from hermes_cli.backend_failover import resolve_role_chain, validate_backend_failover_config

    validate_backend_failover_config(canonical_config)
    config = {**canonical_config, "enabled": True}
    expected = {
        "default": [("api", "openai-codex", "gpt-6.1-sol"), ("claude_code", None, "claude-sonnet-5-5")],
        "hyperion": [("api", "openai-codex", "gpt-6-luna"), ("claude_code", None, "claude-sonnet-5-5")],
        "athena": [("claude_code", None, "claude-opus-5-5"), ("api", "openai-codex", "gpt-6.1-sol")],
        "talos": [("api", "openai-codex", "gpt-6-luna"), ("claude_code", None, "claude-sonnet-5-5")],
        "apollo": [("claude_code", None, "claude-sonnet-5-5"), ("api", "openai-codex", "gpt-6.1-sol")],
    }
    assert set(config["roles"]) == set(expected)
    for profile, first_two in expected.items():
        chain = resolve_role_chain(config, profile)
        assert [(leg.kind, leg.provider, leg.model) for leg in chain] == [
            *first_two, ("api", "deepseek", "deepseek-flash"),
        ]
        claude = next(leg for leg in chain if leg.kind == "claude_code")
        assert claude.capability_mode == {
            "athena": "read_only", "apollo": "read_only", "hyperion": "orchestrator",
        }.get(profile, "full")


def test_talos_smoke_admits_luna_and_subscription_fallbacks_not_old_sol(monkeypatch):
    from hermes_cli import backend_smoke

    monkeypatch.setattr(backend_smoke, "_resolve_hermes_argv", lambda: ["/opt/hermes"])
    for provider, model in [
        ("openai-codex", "gpt-6-luna"),
        ("claude-code-cli", "claude-sonnet-5-5"),
        ("deepseek", "deepseek-flash"),
    ]:
        command = backend_smoke.build_smoke_command(profile="talos", provider=provider, model=model)
        flag = "--model" if provider == "claude-code-cli" else "-m"
        assert command[command.index(flag) + 1] == model
    with pytest.raises(ValueError, match="not a canonical backend route"):
        backend_smoke.build_smoke_command(
            profile="talos", provider="openai-codex", model="gpt-6.1-sol",
        )


def test_activation_config_commands_preserve_all_unrelated_settings(canonical_config, tmp_path, monkeypatch):
    from hermes_cli.config import set_config_value
    from hermes_cli.backend_failover import validate_backend_failover_config

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    old = {
        "agent_backend_failover": deepcopy(canonical_config),
        "model": {"provider": "openai-codex", "default": "gpt-6.1-sol"},
        "terminal": {"cwd": "/synthetic/unchanged"},
    }
    old["agent_backend_failover"]["enabled"] = True
    old["agent_backend_failover"]["roles"]["talos"][0]["model"] = "gpt-6.1-sol"
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump(old), encoding="utf-8")
    set_config_value("agent_backend_failover.roles.talos.0.model", "gpt-6-luna")
    migrated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = deepcopy(old)
    expected["agent_backend_failover"]["roles"]["talos"][0]["model"] = "gpt-6-luna"
    assert migrated == expected
    validate_backend_failover_config(migrated["agent_backend_failover"])

    profile = home / "profiles" / "talos"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    previous = {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
        "platform_toolsets": {"cli": ["terminal", "kanban"]},
    }
    profile_config = profile / "config.yaml"
    profile_config.write_text(yaml.safe_dump(previous), encoding="utf-8")
    set_config_value("model.provider", "openai-codex")
    set_config_value("model.default", "gpt-6-luna")
    expected_profile = deepcopy(previous)
    expected_profile["model"]["default"] = "gpt-6-luna"
    assert yaml.safe_load(profile_config.read_text(encoding="utf-8")) == expected_profile
