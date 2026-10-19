"""Fail-closed policy for profile inheritance of root credential stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from hermes_constants import get_default_hermes_root, get_hermes_home

try:
    import fcntl as _fcntl  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - platform dependent
    _fcntl = None
try:
    import msvcrt as _msvcrt  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - platform dependent
    _msvcrt = None

_HAS_CROSS_PROCESS_LOCKING = _fcntl is not None or _msvcrt is not None
_NEVER_INHERIT = frozenset({"anthropic"})


@dataclass(frozen=True)
class SharedCredentialPolicy:
    enabled: bool = False
    trusted_profiles: frozenset[str] = frozenset()
    providers: frozenset[str] = frozenset()


def resolve_profile_identity() -> tuple[str, bool]:
    """Return ``(profile_name, is_root)`` from the resolved Hermes home path."""
    home = get_hermes_home()
    root = get_default_hermes_root()
    try:
        home_resolved = home.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except Exception:
        home_resolved = home
        root_resolved = root
    if home_resolved == root_resolved:
        return "default", True
    if home_resolved.parent.name != "profiles" or home_resolved.parent.parent != root_resolved:
        return "", False
    name = home_resolved.name.strip()
    return (name, False) if name else ("", False)


def load_shared_credential_policy() -> SharedCredentialPolicy:
    """Read the root-only sharing policy; malformed input is an all-deny policy."""
    path = get_default_hermes_root() / "config.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        profiles = raw.get("profiles") if isinstance(raw, dict) else None
        shared = profiles.get("shared_credentials") if isinstance(profiles, dict) else None
        if not isinstance(shared, dict) or shared.get("enabled") is not True:
            return SharedCredentialPolicy()
        trusted = shared.get("trusted_profiles")
        providers = shared.get("providers")
        if not isinstance(trusted, list) or not isinstance(providers, list):
            return SharedCredentialPolicy()
        if not all(isinstance(item, str) and item.strip() for item in trusted + providers):
            return SharedCredentialPolicy()
        return SharedCredentialPolicy(
            enabled=True,
            trusted_profiles=frozenset(item.strip() for item in trusted),
            providers=frozenset(item.strip().lower() for item in providers),
        )
    except Exception:
        return SharedCredentialPolicy()


def is_shared_inheritance_allowed(provider: str) -> bool:
    """Return whether this profile may inherit ``provider`` from the root store."""
    provider_id = (provider or "").strip().lower()
    if not provider_id or provider_id in _NEVER_INHERIT or not _HAS_CROSS_PROCESS_LOCKING:
        return False
    profile_name, is_root = resolve_profile_identity()
    if is_root or not profile_name:
        return False
    policy = load_shared_credential_policy()
    return (
        policy.enabled
        and profile_name in policy.trusted_profiles
        and provider_id in policy.providers
    )


def describe_credential_trust(provider: str) -> dict[str, object]:
    """Return metadata-only inheritance diagnostics (never credential material)."""
    provider_id = (provider or "").strip().lower()
    profile_name, is_root = resolve_profile_identity()
    policy = load_shared_credential_policy()
    trusted = bool(profile_name and profile_name in policy.trusted_profiles)
    provider_allowed = provider_id in policy.providers and provider_id not in _NEVER_INHERIT
    inherited = is_shared_inheritance_allowed(provider_id)
    return {
        "profile_name": profile_name,
        "is_root": is_root,
        "trusted": trusted,
        "provider_allowed": provider_allowed,
        "inherited": inherited,
        "source_store": "root" if inherited else "profile",
        "resolved_path": str(
            (get_default_hermes_root() if inherited else get_hermes_home()) / "auth.json"
        ),
    }
