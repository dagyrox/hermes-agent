from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from hermes_cli import auth


def _configure(tmp_path: Path, monkeypatch, profile_name: str, *, enabled: bool = True):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / profile_name
    profile.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-process-home"))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    (root / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "profiles": {
                    "shared_credentials": {
                        "enabled": enabled,
                        "trusted_profiles": ["talos"],
                        "providers": ["openai-codex", "deepseek"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return root, profile


def _write_store(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trusted_profile_inherits_only_allowlisted_root_credentials(tmp_path, monkeypatch):
    root, _profile = _configure(tmp_path, monkeypatch, "talos")
    _write_store(
        root / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "codex-access", "refresh_token": "codex-refresh"}
                },
                "xai-oauth": {
                    "tokens": {"access_token": "xai-access", "refresh_token": "xai-refresh"}
                },
            },
            "credential_pool": {
                "deepseek": [{"id": "manual:api_key", "api_key": "deepseek-key"}],
                "openrouter": [{"id": "manual:api_key", "api_key": "openrouter-key"}],
            },
        },
    )

    codex = auth.get_provider_auth_state("openai-codex")
    assert codex is not None
    assert codex["tokens"]["access_token"] == "codex-access"
    deepseek_pool = cast(Any, auth.read_credential_pool("deepseek"))
    assert deepseek_pool[0]["api_key"] == "deepseek-key"
    assert auth.get_provider_auth_state("xai-oauth") is None
    assert auth.read_credential_pool("openrouter") == []

    status = auth.get_auth_status("deepseek")
    assert status["inherited"] is True
    assert status["source_store"] == "root"
    assert status["shadowed_by_local"] is False
    assert "api_key" not in status


def test_untrusted_profile_cannot_inherit_valid_root_store(tmp_path, monkeypatch):
    root, _profile = _configure(tmp_path, monkeypatch, "guestwhatsapp")
    _write_store(
        root / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "codex-access", "refresh_token": "codex-refresh"}
                }
            },
            "credential_pool": {
                "deepseek": [{"id": "manual:api_key", "api_key": "deepseek-key"}]
            },
        },
    )

    assert auth.get_provider_auth_state("openai-codex") is None
    assert auth.read_credential_pool("deepseek") == []


def test_local_provider_and_pool_override_do_not_require_sharing(tmp_path, monkeypatch):
    _root, profile = _configure(tmp_path, monkeypatch, "guestwhatsapp", enabled=False)
    _write_store(
        profile / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "local-access", "refresh_token": "local-refresh"}
                }
            },
            "credential_pool": {
                "deepseek": [{"id": "manual:api_key", "api_key": "local-key"}]
            },
        },
    )

    local_codex = auth.get_provider_auth_state("openai-codex")
    assert local_codex is not None
    assert local_codex["tokens"]["access_token"] == "local-access"
    local_deepseek = cast(Any, auth.read_credential_pool("deepseek"))
    assert local_deepseek[0]["api_key"] == "local-key"
    status = auth.get_auth_status("openai-codex")
    assert status["inherited"] is False
    assert status["source_store"] == str(profile / "auth.json")
    assert status["shadowed_by_local"] is True


def test_corrupt_root_and_corrupt_local_store_both_refuse_inheritance(tmp_path, monkeypatch):
    root, profile = _configure(tmp_path, monkeypatch, "talos")
    (root / "auth.json").write_text("{broken", encoding="utf-8")
    assert auth.get_provider_auth_state("openai-codex") is None
    assert not (root / "auth.json.corrupt").exists()

    _write_store(
        root / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "root-access", "refresh_token": "root-refresh"}
                }
            },
        },
    )
    (profile / "auth.json").write_text("{broken-local", encoding="utf-8")
    assert auth.get_provider_auth_state("openai-codex") is None
    assert (profile / "auth.json.corrupt").exists()
    assert auth.get_provider_auth_state("openai-codex") is None


def test_malformed_root_provider_state_is_not_inherited(tmp_path, monkeypatch):
    root, _profile = _configure(tmp_path, monkeypatch, "talos")
    _write_store(
        root / "auth.json",
        {
            "version": 1,
            "providers": {"openai-codex": {"tokens": {"refresh_token": "refresh-only"}}},
        },
    )

    assert auth.get_provider_auth_state("openai-codex") is None
