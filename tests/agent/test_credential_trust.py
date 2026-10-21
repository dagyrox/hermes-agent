from __future__ import annotations

from pathlib import Path

import yaml

from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def _write_policy(root: Path, payload: object) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_trusted_profile_may_inherit_only_explicit_provider(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "talos"
    _write_policy(
        root,
        {
            "profiles": {
                "shared_credentials": {
                    "enabled": True,
                    "trusted_profiles": ["talos"],
                    "providers": ["openai-codex", "deepseek"],
                }
            }
        },
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile))

    from agent.credential_trust import (
        describe_credential_trust,
        is_shared_inheritance_allowed,
    )

    assert is_shared_inheritance_allowed("openai-codex") is True
    assert is_shared_inheritance_allowed("deepseek") is True
    assert is_shared_inheritance_allowed("xai-oauth") is False
    assert is_shared_inheritance_allowed("anthropic") is False
    assert describe_credential_trust("openai-codex") == {
        "profile_name": "talos",
        "is_root": False,
        "trusted": True,
        "provider_allowed": True,
        "inherited": True,
        "source_store": "root",
        "resolved_path": str(root / "auth.json"),
    }


def test_unlisted_and_non_profile_homes_are_denied(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    _write_policy(
        root,
        {
            "profiles": {
                "shared_credentials": {
                    "enabled": True,
                    "trusted_profiles": ["talos"],
                    "providers": ["deepseek"],
                }
            }
        },
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    from agent.credential_trust import is_shared_inheritance_allowed

    guest_token = set_hermes_home_override(root / "profiles" / "guestwhatsapp")
    try:
        assert is_shared_inheritance_allowed("deepseek") is False
    finally:
        reset_hermes_home_override(guest_token)

    custom_token = set_hermes_home_override(tmp_path / "unrelated")
    try:
        assert is_shared_inheritance_allowed("deepseek") is False
    finally:
        reset_hermes_home_override(custom_token)


def test_missing_malformed_and_profile_local_policy_fail_closed(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "talos"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile))

    from agent.credential_trust import is_shared_inheritance_allowed

    assert is_shared_inheritance_allowed("deepseek") is False

    (root / "config.yaml").write_text("profiles: [not-a-mapping]\n", encoding="utf-8")
    assert is_shared_inheritance_allowed("deepseek") is False

    (root / "config.yaml").unlink()
    _write_policy(
        profile,
        {
            "profiles": {
                "shared_credentials": {
                    "enabled": True,
                    "trusted_profiles": ["talos"],
                    "providers": ["deepseek"],
                }
            }
        },
    )
    assert is_shared_inheritance_allowed("deepseek") is False


def test_inheritance_denied_without_cross_process_locking(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile = root / "profiles" / "talos"
    _write_policy(
        root,
        {
            "profiles": {
                "shared_credentials": {
                    "enabled": True,
                    "trusted_profiles": ["talos"],
                    "providers": ["deepseek"],
                }
            }
        },
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(profile))

    import agent.credential_trust as trust

    monkeypatch.setattr(trust, "_HAS_CROSS_PROCESS_LOCKING", False)
    assert trust.is_shared_inheritance_allowed("deepseek") is False
