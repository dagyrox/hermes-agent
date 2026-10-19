from __future__ import annotations

import json

from agent import credential_pool
from hermes_cli import auth


def _codex_state(access: str, refresh: str) -> dict:
    return {"tokens": {"access_token": access, "refresh_token": refresh}}


def test_shared_write_through_verifies_persisted_state(tmp_path, monkeypatch):
    root_auth = tmp_path / "root" / "auth.json"
    root_auth.parent.mkdir(parents=True)
    root_auth.write_text(json.dumps({"version": 1, "providers": {}}), encoding="utf-8")
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda provider_id=None: root_auth)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    state = _codex_state("rotated-access", "rotated-refresh")
    assert credential_pool._write_through_provider_state_to_global_root(
        "openai-codex", state
    ) is True
    persisted = json.loads(root_auth.read_text(encoding="utf-8"))
    assert persisted["providers"]["openai-codex"] == state


def test_shared_write_through_detects_silent_noop(tmp_path, monkeypatch, caplog):
    root_auth = tmp_path / "root" / "auth.json"
    root_auth.parent.mkdir(parents=True)
    root_auth.write_text(json.dumps({"version": 1, "providers": {}}), encoding="utf-8")
    monkeypatch.setattr(auth, "_global_auth_file_path", lambda provider_id=None: root_auth)
    monkeypatch.setattr(auth, "_persist_provider_state_to_store", lambda *args, **kwargs: root_auth)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    state = _codex_state("rotated-access", "rotated-refresh")
    assert credential_pool._write_through_provider_state_to_global_root(
        "openai-codex", state
    ) is False
    assert "verification failed" in caplog.text
    assert "rotated-access" not in caplog.text
    assert "rotated-refresh" not in caplog.text
