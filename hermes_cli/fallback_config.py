"""Helpers for reading the effective fallback provider chain from config."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def resolve_entry_api_key(entry: dict[str, Any] | None) -> str | None:
    """API key for one fallback entry: inline ``api_key``, else ``key_env``.

    Mirrors the custom-provider convention (``key_env`` names the env var
    holding the key; ``api_key_env`` accepted as an alias). Returns None when
    neither yields a non-empty value, letting ``resolve_runtime_provider``
    fall through to the provider's standard credential resolution.
    """
    if not isinstance(entry, dict):
        return None
    inline = str(entry.get("api_key") or "").strip()
    if inline:
        return inline
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "").strip()
    if key_env:
        return os.getenv(key_env, "").strip() or None
    return None


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the effective fallback chain merged across old and new config keys.

    ``fallback_providers`` remains the primary source of truth and keeps its
    order. Legacy ``fallback_model`` entries are appended afterwards unless
    they target the same provider/model/base_url route as an earlier entry.
    The returned list always contains fresh dict copies.
    """

    # A dispatcher-owned backend supervisor is already traversing the exact
    # canonical chain. Nested in-process fallback would skip external-runtime
    # legs, duplicate attempts, and make provenance nondeterministic.
    if os.environ.get("HERMES_BACKEND_FAILOVER_CHILD") == "1":
        return []

    config = config or {}
    from hermes_cli.provider_access import load_provider_access, managed_deployment, trusted_profile

    managed = managed_deployment() and trusted_profile()
    modes = load_provider_access() if managed else {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if managed and identity[0] in {"openai", "openai-direct", "anthropic"}:
                # Vendor API access is an explicit primary only, never an
                # automatic subscription-to-paid escalation.
                continue
            if managed and identity[0] == "openai-codex" and modes["openai"]["mode"] == "api":
                continue
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    return chain


def _load_config_for_fallback(cfg_path: str | Path | None) -> dict[str, Any]:
    """Read the raw ``config.yaml`` that owns the fallback chain.

    Mirrors the gateway's original read: raw YAML, no managed overlay, empty
    dict when the file is missing or unparseable.
    """
    if cfg_path is None:
        from hermes_constants import get_hermes_home

        path = get_hermes_home() / "config.yaml"
    else:
        path = Path(cfg_path)
    if not path.exists():
        return {}
    import yaml

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw if isinstance(raw, dict) else {}


def resolve_fallback_runtime(cfg_path: str | Path | None = None) -> dict[str, Any] | None:
    """Resolve the first working entry of the configured fallback chain.

    Walks ``fallback_providers`` / ``fallback_model`` in config order and
    returns the resolved runtime of the first entry whose credentials resolve.
    An entry that raises (missing credentials, disabled provider, bad custom
    block) is skipped so one broken entry cannot disable the whole chain.

    ``None`` means "no fallback available": either none is configured, the
    config could not be read, or every entry failed to resolve. Callers keep
    their original failure and surface it — this helper never masks it.

    The returned dict is the subset of ``resolve_runtime_provider()`` output
    that ``AIAgent`` consumes, plus the config entry's literal ``model``.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider

    try:
        chain = get_fallback_chain(_load_config_for_fallback(cfg_path))
        if not chain:
            return None
        for entry in chain:
            try:
                runtime = resolve_runtime_provider(
                    requested=entry.get("provider"),
                    explicit_base_url=entry.get("base_url"),
                    explicit_api_key=resolve_entry_api_key(entry),
                )
            except Exception as fb_exc:  # noqa: BLE001 - one entry must not kill the chain
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
                continue
            # Log the literal `provider` key from config, not the resolved
            # runtime category — an Ollama fallback resolves through the
            # OpenAI-compatible path and would otherwise be logged as
            # "openrouter", contradicting the operator's config (#32790).
            logger.info(
                "Fallback provider resolved: %s model=%s",
                entry.get("provider") or runtime.get("provider"),
                entry.get("model"),
            )
            return {
                "api_key": runtime.get("api_key"),
                "base_url": runtime.get("base_url"),
                "provider": runtime.get("provider"),
                "requested_provider": runtime.get("requested_provider"),
                "api_mode": runtime.get("api_mode"),
                "command": runtime.get("command"),
                "args": list(runtime.get("args") or []),
                "credential_pool": runtime.get("credential_pool"),
                "model": entry.get("model"),
            }
    except Exception:  # noqa: BLE001 - fail closed: caller surfaces its own error
        logger.debug("Fallback chain resolution failed", exc_info=True)
    return None
