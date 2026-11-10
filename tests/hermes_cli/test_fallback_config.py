"""Tests for hermes_cli/fallback_config.py — fallback chain + API-key resolution."""

import pytest
import yaml

from hermes_cli.fallback_config import resolve_entry_api_key, resolve_fallback_runtime


SAMPLE_RUNTIME = {
    "api_key": "fb-key",
    "base_url": "https://fallback.invalid",
    "provider": "deepseek",
    "requested_provider": "deepseek",
    "api_mode": "chat_completions",
    "command": None,
    "args": ["--serve"],
    "credential_pool": "pool-1",
    "max_output_tokens": 4096,
    "provider_access_mode": "api",
}

# The exact keys hermes_cli/oneshot.py and gateway/run.py hand to AIAgent from
# a fallback entry. Both lanes must agree on this shape.
FALLBACK_RUNTIME_KEYS = {
    "api_key",
    "base_url",
    "provider",
    "requested_provider",
    "api_mode",
    "command",
    "args",
    "credential_pool",
    "model",
}


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"

    def test_key_env_resolves_from_environment(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"

    def test_api_key_env_alias(self, monkeypatch):
        monkeypatch.setenv("FB_ALIAS_KEY", "alias-key")
        assert resolve_entry_api_key({"api_key_env": "FB_ALIAS_KEY"}) == "alias-key"

    def test_unset_env_var_returns_none(self, monkeypatch):
        monkeypatch.delenv("FB_MISSING", raising=False)
        # None (not "") lets resolve_runtime_provider fall through to the
        # provider's standard credential resolution.
        assert resolve_entry_api_key({"key_env": "FB_MISSING"}) is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("FB_EMPTY", "   ")
        assert resolve_entry_api_key({"key_env": "FB_EMPTY"}) is None

    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None

    def test_non_dict_returns_none(self):
        assert resolve_entry_api_key(None) is None
        assert resolve_entry_api_key("nope") is None  # type: ignore[arg-type]

    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"api_key": "   ", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "env-key"


def _write_config(tmp_path, config: dict):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _chain_config():
    return {
        "model": {"default": "gpt-6.1-sol", "provider": "openai-codex"},
        "fallback_providers": [
            {"provider": "openai-codex", "model": "gpt-6-sol"},
            {"provider": "deepseek", "model": "deepseek-flash"},
        ],
    }


class TestResolveFallbackRuntime:
    def test_skips_unresolvable_entry_and_returns_agent_shape(
        self, monkeypatch, tmp_path
    ):
        cfg_path = _write_config(tmp_path, _chain_config())
        seen = []

        def fake_resolve(*, requested=None, explicit_api_key=None,
                         explicit_base_url=None, target_model=None, **_kwargs):
            seen.append(requested)
            if requested == "openai-codex":
                raise RuntimeError("codex credentials unavailable")
            return dict(SAMPLE_RUNTIME)

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )

        runtime = resolve_fallback_runtime(cfg_path=cfg_path)

        assert seen == ["openai-codex", "deepseek"]
        assert set(runtime) == FALLBACK_RUNTIME_KEYS
        # The config entry's literal model wins over the runtime's own model —
        # that is the model the operator asked the fallback to run.
        assert runtime["model"] == "deepseek-flash"
        assert runtime["provider"] == "deepseek"
        assert runtime["api_key"] == "fb-key"
        assert runtime["args"] == ["--serve"]

    def test_returns_none_when_every_entry_fails(self, monkeypatch, tmp_path):
        cfg_path = _write_config(tmp_path, _chain_config())

        def fake_resolve(**_kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )
        assert resolve_fallback_runtime(cfg_path=cfg_path) is None

    def test_returns_none_when_no_chain_configured(self, monkeypatch, tmp_path):
        cfg_path = _write_config(tmp_path, {"model": {"default": "x"}})
        called = []
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **kw: called.append(kw) or dict(SAMPLE_RUNTIME),
        )
        assert resolve_fallback_runtime(cfg_path=cfg_path) is None
        assert called == []

    def test_returns_none_when_config_file_missing(self, tmp_path):
        assert resolve_fallback_runtime(cfg_path=tmp_path / "absent.yaml") is None

    def test_entry_api_key_env_is_forwarded(self, monkeypatch, tmp_path):
        config = _chain_config()
        config["fallback_providers"] = [
            {"provider": "custom", "model": "local", "key_env": "FB_INLINE_KEY"}
        ]
        cfg_path = _write_config(tmp_path, config)
        captured = {}

        def fake_resolve(*, requested=None, explicit_api_key=None,
                         explicit_base_url=None, target_model=None, **_kwargs):
            captured["api_key"] = explicit_api_key
            captured["requested"] = requested
            return dict(SAMPLE_RUNTIME)

        monkeypatch.setenv("FB_INLINE_KEY", "from-env")
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", fake_resolve
        )

        runtime = resolve_fallback_runtime(cfg_path=cfg_path)

        assert captured == {"api_key": "from-env", "requested": "custom"}
        assert runtime["model"] == "local"
