"""Tests for ``hermes -z`` failover when the primary provider cannot resolve.

A quota-capped (429) or genuinely broken primary raises ``AuthError`` inside
``resolve_runtime_provider()`` — before ``AIAgent`` (and therefore its
in-process ``fallback_model`` chain) exists. Without the oneshot-level walk,
the whole run dies even though the operator configured a fallback chain.
"""

import logging
import sys
import types

import pytest
import yaml

import hermes_cli.oneshot as oneshot_mod
from hermes_cli.auth import CODEX_RATE_LIMITED_CODE, AuthError
from hermes_constants import get_hermes_home


FALLBACK_CONFIG = {
    "model": {"default": "gpt-6.1-sol", "provider": "openai-codex"},
    "fallback_providers": [
        {"provider": "openai-codex", "model": "gpt-6-sol"},
        {"provider": "deepseek", "model": "deepseek-flash"},
    ],
}

RATE_LIMITED_MESSAGE = (
    "Codex provider quota exhausted (429); retry after 49960s. "
    "Credentials are still valid."
)


class _CapturingAgent:
    """Stand-in for AIAgent that records the constructor kwargs."""

    captured: dict = {}

    def __init__(self, **kwargs):
        type(self).captured = kwargs
        self.suppress_status_output = False
        self.stream_delta_callback = object()
        self.tool_gen_callback = object()

    def run_conversation(self, _prompt, **_kwargs):
        return {"final_response": "PONG"}

    def shutdown_memory_provider(self, messages=None):
        pass

    def close(self):
        pass


def _runtime(provider: str, model: str) -> dict:
    return {
        "api_key": f"{provider}-key",
        "base_url": f"https://{provider}.invalid",
        "provider": provider,
        "requested_provider": provider,
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": f"{provider}-pool",
        "model": model,
    }


def _install(monkeypatch, resolve_impl, config=FALLBACK_CONFIG) -> None:
    """Point oneshot at a temp config home and fake provider resolution."""
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    monkeypatch.delenv("HERMES_BACKEND_FAILOVER_CHILD", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", resolve_impl
    )
    monkeypatch.setattr(oneshot_mod, "_create_session_db_for_oneshot", lambda: None)
    monkeypatch.setitem(
        sys.modules, "run_agent", types.SimpleNamespace(AIAgent=_CapturingAgent)
    )
    _CapturingAgent.captured = {}


def _rate_limited() -> AuthError:
    return AuthError(
        RATE_LIMITED_MESSAGE,
        provider="openai-codex",
        code=CODEX_RATE_LIMITED_CODE,
        relogin_required=False,
    )


def _relogin_required() -> AuthError:
    return AuthError(
        "No Codex credentials stored. Run `hermes auth` to authenticate.",
        provider="openai-codex",
        code="codex_auth_missing",
        relogin_required=True,
    )


def test_rate_limited_primary_fails_over_to_first_resolvable_entry(monkeypatch):
    calls = []

    def resolve_impl(
        *, requested=None, explicit_api_key=None, explicit_base_url=None,
        target_model=None, **_kwargs,
    ):
        calls.append(requested)
        if requested in (None, "openai-codex"):
            raise _rate_limited()
        return _runtime(requested, "deepseek-flash")

    _install(monkeypatch, resolve_impl)
    notes = []

    response, _result = oneshot_mod._run_agent(
        "Reply with exactly: PONG",
        use_config_toolsets=False,
        on_fallback=notes.append,
    )

    assert response == "PONG"
    # Effective model + credentials come from the fallback entry, not the
    # primary that failed to resolve.
    assert _CapturingAgent.captured["model"] == "deepseek-flash"
    assert _CapturingAgent.captured["provider"] == "deepseek"
    assert _CapturingAgent.captured["api_key"] == "deepseek-key"
    assert _CapturingAgent.captured["credential_pool"] == "deepseek-pool"
    # The unresolvable first fallback entry was skipped, not fatal.
    assert calls == [None, "openai-codex", "deepseek"]
    assert len(notes) == 1
    assert "quota/rate limited" in notes[0]
    assert "deepseek/deepseek-flash" in notes[0]
    assert notes[0].startswith("hermes -z: primary provider unavailable")


def test_relogin_required_primary_fails_over(monkeypatch):
    def resolve_impl(
        *, requested=None, explicit_api_key=None, explicit_base_url=None,
        target_model=None, **_kwargs,
    ):
        if requested in (None, "openai-codex"):
            raise _relogin_required()
        return _runtime(requested, "deepseek-flash")

    _install(monkeypatch, resolve_impl)
    notes = []

    response, _result = oneshot_mod._run_agent(
        "hi", use_config_toolsets=False, on_fallback=notes.append
    )

    assert response == "PONG"
    assert _CapturingAgent.captured["model"] == "deepseek-flash"
    assert len(notes) == 1
    assert "auth failure" in notes[0]
    assert "quota" not in notes[0]


def test_every_entry_failing_reraises_primary_auth_error(monkeypatch):
    def resolve_impl(**_kwargs):
        raise _rate_limited()

    _install(monkeypatch, resolve_impl)
    notes = []

    with pytest.raises(AuthError) as excinfo:
        oneshot_mod._run_agent(
            "hi", use_config_toolsets=False, on_fallback=notes.append
        )

    # The operator sees the real primary cause, not a generic wrapper, and
    # nothing claims a fallback was used.
    assert str(excinfo.value) == RATE_LIMITED_MESSAGE
    assert excinfo.value.code == CODEX_RATE_LIMITED_CODE
    assert notes == []


def test_no_fallback_configured_reraises_primary_auth_error(monkeypatch):
    def resolve_impl(**_kwargs):
        raise _rate_limited()

    _install(monkeypatch, resolve_impl, config={"model": {"default": "gpt-6.1-sol"}})

    with pytest.raises(AuthError):
        oneshot_mod._run_agent("hi", use_config_toolsets=False)


def test_run_oneshot_reports_fallback_on_real_stderr(monkeypatch, capsys):
    def resolve_impl(
        *, requested=None, explicit_api_key=None, explicit_base_url=None,
        target_model=None, **_kwargs,
    ):
        if requested in (None, "openai-codex"):
            raise _rate_limited()
        return _runtime(requested, "deepseek-flash")

    _install(monkeypatch, resolve_impl)

    try:
        rc = oneshot_mod.run_oneshot("Reply with exactly: PONG")
    finally:
        # run_oneshot disables logging for the duration of the process; leave
        # the process-global logger usable for the rest of the suite.
        logging.disable(logging.NOTSET)

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "PONG"
    assert "using fallback deepseek/deepseek-flash" in captured.err


def test_run_oneshot_failure_keeps_primary_cause_and_exit_code(monkeypatch, capsys):
    def resolve_impl(**_kwargs):
        raise _rate_limited()

    _install(monkeypatch, resolve_impl)

    try:
        rc = oneshot_mod.run_oneshot("hi")
    finally:
        logging.disable(logging.NOTSET)

    captured = capsys.readouterr()
    assert rc == 1
    assert RATE_LIMITED_MESSAGE in captured.err
    assert "hermes -z: agent failed:" in captured.err
    assert "Traceback" not in captured.err
