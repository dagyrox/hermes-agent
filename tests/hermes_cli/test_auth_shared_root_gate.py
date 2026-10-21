from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from agent.credential_pool import load_pool
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
                "deepseek": [
                    {
                        "id": "manual:api_key",
                        "api_key": "deepseek-key",
                        "base_url": "https://api.deepseek.com/v1",
                    }
                ],
                "openrouter": [{"id": "manual:api_key", "api_key": "openrouter-key"}],
            },
        },
    )

    codex = auth.get_provider_auth_state("openai-codex")
    assert codex is not None
    assert codex["tokens"]["access_token"] == "codex-access"
    deepseek_pool = cast(Any, auth.read_credential_pool("deepseek"))
    assert deepseek_pool[0]["api_key"] == "deepseek-key"
    assert deepseek_pool[0]["base_url"] == "https://api.deepseek.com/v1"
    assert auth.get_provider_auth_state("xai-oauth") is None
    assert auth.read_credential_pool("openrouter") == []

    status = auth.get_auth_status("deepseek")
    assert status["inherited"] is True
    assert status["source_store"] == "root"
    assert status["shadowed_by_local"] is False
    assert "api_key" not in status


def test_xai_root_tokens_are_not_read_when_provider_is_not_allowlisted(
    tmp_path, monkeypatch
):
    root, _profile = _configure(tmp_path, monkeypatch, "talos")
    _write_store(
        root / "auth.json",
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "xai-root-access",
                        "refresh_token": "xai-root-refresh",
                    }
                }
            },
        },
    )

    with pytest.raises(auth.AuthError, match="No xAI OAuth credentials stored"):
        auth._read_xai_oauth_tokens()


def test_xai_save_from_excluded_profile_does_not_mutate_root(tmp_path, monkeypatch):
    root, _profile = _configure(tmp_path, monkeypatch, "guestwhatsapp")
    root_auth = root / "auth.json"
    _write_store(
        root_auth,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                }
            },
        },
    )
    before = root_auth.read_bytes()

    auth._save_xai_oauth_tokens(
        {"access_token": "profile-access", "refresh_token": "profile-refresh"}
    )

    assert root_auth.read_bytes() == before


def test_inherited_pool_entry_is_not_materialized_as_local_shadow(tmp_path, monkeypatch):
    root, profile = _configure(tmp_path, monkeypatch, "talos")
    _write_store(
        root / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                }
            },
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "root-codex",
                        "label": "root",
                        "source": "device_code",
                        "auth_type": "oauth",
                        "priority": 0,
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                ]
            },
        },
    )

    pool = load_pool("openai-codex")
    assert pool.entries()[0].extra.get("shared_root_source") is True
    pool._persist()

    profile_store = json.loads((profile / "auth.json").read_text(encoding="utf-8"))
    assert not profile_store.get("providers", {}).get("openai-codex")
    assert not profile_store.get("credential_pool", {}).get("openai-codex")


def test_inherited_singleton_seed_is_not_materialized_as_local_shadow(
    tmp_path, monkeypatch
):
    root, profile = _configure(tmp_path, monkeypatch, "talos")
    _write_store(
        root / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                }
            },
        },
    )

    pool = load_pool("openai-codex")

    assert pool.entries()[0].extra.get("shared_root_source") is True
    profile_store = json.loads((profile / "auth.json").read_text(encoding="utf-8"))
    assert not profile_store.get("providers", {}).get("openai-codex")
    assert not profile_store.get("credential_pool", {}).get("openai-codex")


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


def test_runtime_refresh_root_lock_timeout_records_failed_write_through(
    tmp_path, monkeypatch
):
    root, profile = _configure(tmp_path, monkeypatch, "talos")
    root_auth = root / "auth.json"
    _write_store(
        root_auth,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                }
            },
        },
    )
    real_lock = auth._auth_store_lock

    @contextmanager
    def timeout_root_lock(*args, **kwargs):
        if kwargs.get("target_path") == root_auth:
            raise TimeoutError("root lock unavailable")
        with real_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(auth, "_auth_store_lock", timeout_root_lock)

    with pytest.raises(TimeoutError, match="root lock unavailable"):
        auth.resolve_codex_runtime_credentials(force_refresh=True)

    profile_store = json.loads((profile / "auth.json").read_text(encoding="utf-8"))
    assert profile_store["credential_inheritance_status"]["openai-codex"][
        "write_through_ok"
    ] is False
    assert not profile_store.get("providers", {}).get("openai-codex")


def test_runtime_refresh_root_replace_failure_preserves_root_and_records_failure(
    tmp_path, monkeypatch
):
    root, profile = _configure(tmp_path, monkeypatch, "talos")
    root_auth = root / "auth.json"
    _write_store(
        root_auth,
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": "root-access",
                        "refresh_token": "root-refresh",
                    }
                }
            },
        },
    )
    before = root_auth.read_bytes()
    unaffected = {
        root / "config.yaml": (root / "config.yaml").read_bytes(),
        root / "memory" / "MEMORY.md": b"durable-memory\n",
        root / "sessions" / "session.json": b'{"id":"session"}\n',
    }
    for path, content in unaffected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
    unaffected_hashes = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in unaffected
    }
    monkeypatch.setattr(
        auth,
        "refresh_codex_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
        },
    )
    real_save = auth._save_auth_store

    def fail_root_save(store, target_path=None):
        if target_path == root_auth:
            raise OSError("replace failed")
        return real_save(store, target_path=target_path)

    monkeypatch.setattr(auth, "_save_auth_store", fail_root_save)

    with pytest.raises(OSError, match="replace failed"):
        auth.resolve_codex_runtime_credentials(force_refresh=True)

    assert root_auth.read_bytes() == before
    profile_store = json.loads((profile / "auth.json").read_text(encoding="utf-8"))
    assert profile_store["credential_inheritance_status"]["openai-codex"][
        "write_through_ok"
    ] is False
    assert not profile_store.get("providers", {}).get("openai-codex")
    assert {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in unaffected
    } == unaffected_hashes


def test_auth_store_crash_before_replace_preserves_old_file_and_cleans_temp(
    tmp_path, monkeypatch
):
    auth_file = tmp_path / "auth.json"
    original = {"version": 1, "providers": {"openai-codex": {"label": "old"}}}
    _write_store(auth_file, original)
    monkeypatch.setattr(auth, "_auth_file_path", lambda: auth_file)

    def crash_before_replace(_source, _destination):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(auth, "atomic_replace", crash_before_replace)

    with pytest.raises(OSError, match="simulated crash before replace"):
        auth._save_auth_store(
            {"version": 1, "providers": {"openai-codex": {"label": "new"}}}
        )

    assert json.loads(auth_file.read_text(encoding="utf-8")) == original
    assert list(tmp_path.glob("auth.json.tmp.*")) == []
