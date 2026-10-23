"""Bounded behavior contracts for internal vendor access policy (no network)."""
from pathlib import Path

import pytest
import yaml


def _root(tmp_path, monkeypatch, profile="talos"):
    from hermes_cli import provider_access
    monkeypatch.setattr(provider_access, "native_home", lambda: tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    root.mkdir()
    home = root if profile == "default" else root / "profiles" / profile
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (root / "config.yaml").write_text(yaml.safe_dump({"profiles": {"shared_credentials": {
        "enabled": True, "trusted_profiles": ["talos"], "providers": ["openai-codex"]
    }}}))
    return root, home


def test_native_child_uses_only_canonical_source_and_protects_it(tmp_path, monkeypatch):
    from hermes_cli.provider_access import claude_environment
    from agent.file_safety import get_read_block_error
    from tools.credential_files import register_credential_file

    root, _ = _root(tmp_path, monkeypatch)
    token = root / "claude" / "subscription.token"
    token.parent.mkdir()
    token.write_text("sk-ant-oat01-test-only")
    token.chmod(0o600)
    child = claude_environment({"PATH": "/bin", "HOME": "/wrong", "ANTHROPIC_API_KEY": "wrong",
                                "CLAUDE_CODE_OAUTH_TOKEN": "stale", "HERMES_KANBAN_TASK": "parent"})
    assert child["HOME"] == str(tmp_path)
    assert child["CLAUDE_CODE_OAUTH_TOKEN"] == token.read_text()
    assert "ANTHROPIC_API_KEY" not in child and "HERMES_KANBAN_TASK" not in child
    from argparse import Namespace
    from hermes_cli.native_claude import launch_command
    from hermes_cli.provider_access import provider_access_status
    args = Namespace(profile="talos", claude_bin="claude", login=False, model="opus",
                     name="Talos", query=None, native_args=[])
    command, interactive_env = launch_command(args, {"HOME": "/wrong", "PATH": "/bin"})
    assert command[0] == "claude" and interactive_env == child
    status = provider_access_status()["anthropic"]
    assert status["source_path"] == str(token) and status["credential_present"]
    assert status["readiness"] == "configured-unverified" and not status["available"]
    assert get_read_block_error(str(token))
    monkeypatch.setenv("HERMES_HOME", str(root))
    assert not register_credential_file("claude/subscription.token")


def test_explicit_mode_changes_only_primary_and_snapshot_survives_revert(tmp_path, monkeypatch):
    import json
    from hermes_cli.backend_failover import load_profile_chain, worker_argv
    from hermes_cli.backend_failover_worker import _load_chain
    from types import SimpleNamespace

    root, _ = _root(tmp_path, monkeypatch)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    cfg["agent_backend_failover"] = {"enabled": True, "roles": {"talos": [
        {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
        {"kind": "claude_code", "model": "claude-sonnet-5-5"},
        {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
    ]}}
    cfg["provider_access"] = {"openai": {"mode": "api", "api_provider": "openai-direct", "api_model": "explicit-model"}}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    api = load_profile_chain("talos", str(root))
    assert api[0].provider == "openai-direct" and api[0].model == "explicit-model"
    assert all(leg.provider not in {"openai", "openai-direct", "anthropic"} for leg in api[1:])
    argv = worker_argv(SimpleNamespace(id="probe"), "talos", api)
    frozen = argv[argv.index("--chain-json") + 1]
    cfg["provider_access"]["openai"]["mode"] = "subscription"
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    assert load_profile_chain("talos", str(root))[0].provider == "openai-codex"
    assert _load_chain("talos", frozen) == api
    tampered = json.loads(frozen)
    tampered[1] = {"kind": "api", "provider": "anthropic", "model": "paid"}
    with pytest.raises(ValueError):
        _load_chain("talos", json.dumps(tampered))


def test_runtime_api_is_explicit_root_only_and_never_oauth(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.provider_access import validate_provider_access
    from hermes_cli.auth import AuthError

    root, home = _root(tmp_path, monkeypatch)
    (home / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-api03-stale-lane\n")
    with pytest.raises(AuthError, match="native Claude"):
        resolve_runtime_provider(requested="anthropic")
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    cfg["provider_access"] = {"anthropic": {"mode": "api", "api_provider": "anthropic", "api_model": "explicit-model"}}
    (root / "config.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        resolve_runtime_provider(requested="anthropic")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-oat01-not-an-api-key\n")
    with pytest.raises(ValueError, match="API key"):
        resolve_runtime_provider(requested="anthropic")
    (root / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-api03-test-only\n")
    resolved = resolve_runtime_provider(requested="anthropic")
    assert resolved["api_key"] == "sk-ant-api03-test-only"
    assert resolved["model"] == "explicit-model"
    assert resolved["source"] == str(root / ".env")
    from argparse import Namespace
    from hermes_cli.native_claude import launch_command
    args = Namespace(profile="talos", claude_bin="claude", login=False, model="opus",
                     name="Talos", query=None, native_args=[])
    command, api_env = launch_command(args, {"CLAUDE_CODE_OAUTH_TOKEN": "stale", "HERMES_KANBAN_TASK": "parent"})
    assert command[command.index("--provider") + 1] == "anthropic"
    assert command[command.index("-m") + 1] == "explicit-model"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in api_env and "HERMES_KANBAN_TASK" not in api_env
    for invalid in ({"openai": {"mode": "api"}}, {"other": {}}, {"openai": {"mode": "typo"}}):
        with pytest.raises(ValueError):
            validate_provider_access(invalid)
    (root / "config.yaml").write_text("provider_access: [invalid")
    with pytest.raises(ValueError, match="authoritative"):
        resolve_runtime_provider(requested="anthropic")


def test_guest_never_reads_central_native_or_api_source(tmp_path, monkeypatch):
    from hermes_cli.provider_access import claude_environment, root_api_key
    from hermes_cli.backend_failover import load_profile_chain

    root, _ = _root(tmp_path, monkeypatch, profile="guest")
    # Policy reads are fine; touching the secret sources is not.
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        assert path.name == "config.yaml", "guest attempted central secret access"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    with pytest.raises(ValueError, match="trusted"):
        claude_environment({})
    with pytest.raises(ValueError, match="trusted"):
        root_api_key("openai")
    assert load_profile_chain("guest", str(root)) == ()
    from agent import anthropic_adapter
    monkeypatch.setattr(anthropic_adapter, "_read_claude_code_credentials_from_file",
                        lambda: pytest.fail("raw provider attempted native store read"))
    assert anthropic_adapter.read_claude_code_credentials() is None
    with pytest.raises(RuntimeError, match="native CLI"):
        anthropic_adapter._write_claude_code_credentials("test-only", "test-only", 1)
