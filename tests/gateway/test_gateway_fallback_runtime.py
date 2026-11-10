"""Gateway fallback resolution delegates to the shared chain walker.

The gateway lane is the reference implementation for provider failover on
quota/auth death. It must keep returning the same runtime dict shape after
the walker moved into ``hermes_cli.fallback_config`` (and became shared with
``hermes -z`` oneshot).
"""

from unittest.mock import patch

import yaml

import gateway.run as gateway_run
from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, AuthError


AGENT_RUNTIME_KEYS = {
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

CONFIG = {
    "model": {"default": "gpt-6.1-sol", "provider": "openai-codex"},
    "fallback_providers": [
        {"provider": "openai-codex", "model": "gpt-6-sol"},
        {"provider": "deepseek", "model": "deepseek-flash"},
    ],
}


def _fallback_runtime(provider: str) -> dict:
    return {
        "api_key": f"{provider}-key",
        "base_url": f"https://{provider}.invalid",
        "provider": provider,
        "requested_provider": provider,
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "model": "deepseek-flash",
        "provider_access_mode": "api",
    }


def _write_config(tmp_path, config=CONFIG):
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return tmp_path / "config.yaml"


def _resolver(primary_error):
    def resolve_impl(*, requested=None, explicit_api_key=None, **_kwargs):
        if requested in (None, "openai-codex"):
            raise primary_error
        return _fallback_runtime(requested)

    return resolve_impl


def test_try_resolve_fallback_provider_shape_unchanged(tmp_path, monkeypatch):
    _write_config(tmp_path)
    monkeypatch.delenv("HERMES_BACKEND_FAILOVER_CHILD", raising=False)

    with patch("gateway.run._hermes_home", tmp_path), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _resolver(AuthError("unused", provider="openai-codex")),
    ):
        runtime = gateway_run._try_resolve_fallback_provider()

    assert set(runtime) == AGENT_RUNTIME_KEYS
    assert runtime["provider"] == "deepseek"
    assert runtime["model"] == "deepseek-flash"
    assert runtime["api_key"] == "deepseek-key"
    assert runtime["args"] == []


def test_try_resolve_fallback_provider_none_without_config(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_BACKEND_FAILOVER_CHILD", raising=False)
    with patch("gateway.run._hermes_home", tmp_path):
        assert gateway_run._try_resolve_fallback_provider() is None


def test_resolve_runtime_agent_kwargs_fails_over_on_quota_exhaustion(
    tmp_path, monkeypatch
):
    """Quota-capped primary during resolution must still produce a runtime."""
    _write_config(tmp_path)
    monkeypatch.delenv("HERMES_BACKEND_FAILOVER_CHILD", raising=False)

    quota_exc = AuthError(
        "Codex provider quota exhausted (429); retry after 49960s.",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
        relogin_required=False,
    )

    with patch("gateway.run._hermes_home", tmp_path), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        _resolver(quota_exc),
    ):
        kwargs = gateway_run._resolve_runtime_agent_kwargs()

    assert kwargs["provider"] == "deepseek"
    assert kwargs["model"] == "deepseek-flash"


def test_resolve_runtime_agent_kwargs_raises_when_every_entry_fails(
    tmp_path, monkeypatch
):
    _write_config(tmp_path)
    monkeypatch.delenv("HERMES_BACKEND_FAILOVER_CHILD", raising=False)

    quota_exc = AuthError(
        "Codex provider quota exhausted (429); retry after 49960s.",
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
        relogin_required=False,
    )

    with patch("gateway.run._hermes_home", tmp_path), patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **_kwargs: (_ for _ in ()).throw(quota_exc),
    ):
        try:
            gateway_run._resolve_runtime_agent_kwargs()
        except RuntimeError as exc:
            assert "quota exhausted" in str(exc)
        else:  # pragma: no cover - explicit failure better than silent pass
            raise AssertionError("expected RuntimeError for exhausted chain")
