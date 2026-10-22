"""Root-owned vendor access policy and native Claude child authentication.

No credential is exported into the supervisor. Native Claude alone owns its
subscription lifecycle; Hermes never refreshes or writes its source.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import yaml

from agent.credential_trust import load_shared_credential_policy, resolve_profile_identity
from hermes_constants import get_default_hermes_root
from hermes_cli.backend_failover import ConfigError

# Shared by the native worker bridge and interactive launcher. In particular,
# no API keys, profile identity, Kanban ownership, or Claude config override.
NATIVE_ENV_KEYS = frozenset({
    "USER", "LOGNAME", "PATH", "SHELL", "TMPDIR", "TMP", "TEMP", "LANG",
    "LANGUAGE", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY",
    "ALL_PROXY", "NO_PROXY", "SSH_AUTH_SOCK",
})
EXCLUDED_PROFILES = frozenset({"guest", "guestwhatsapp", "gf", "orion", "sentinel", "forge"})
API_PROVIDERS = {"anthropic": "anthropic", "openai": "openai-direct"}
API_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}


def native_home() -> Path:
    # HOME is input from the launching lane, not ownership of the native store.
    getuid = getattr(os, "getuid", None)
    if os.name == "posix" and getuid is not None:
        import pwd

        return Path(pwd.getpwuid(getuid()).pw_dir)
    return Path.home()


def validate_provider_access(raw: object) -> dict[str, dict[str, str]]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict) or set(raw) - set(API_PROVIDERS):
        raise ConfigError("provider_access must contain only anthropic/openai mappings")
    result = {}
    for vendor, provider in API_PROVIDERS.items():
        block = raw.get(vendor, {})
        if not isinstance(block, dict) or set(block) - {"mode", "api_provider", "api_model"}:
            raise ConfigError(f"Invalid provider_access.{vendor} settings")
        mode = block.get("mode", "subscription")
        if mode not in ("subscription", "api"):
            raise ConfigError(f"provider_access.{vendor}.mode must be subscription or api")
        if "api_provider" in block and block["api_provider"] != provider:
            raise ConfigError(f"provider_access.{vendor}.api_provider must be {provider}")
        model = block.get("api_model", "")
        if not isinstance(model, str) or ("api_model" in block and not model.strip()):
            raise ConfigError(f"provider_access.{vendor}.api_model must be an explicit model id")
        if mode == "api" and (block.get("api_provider") != provider or not model.strip()):
            raise ConfigError(f"provider_access.{vendor} api mode requires api_provider and explicit api_model")
        result[vendor] = {"mode": mode}
        if "api_provider" in block:
            result[vendor]["api_provider"] = provider
        if model:
            result[vendor]["api_model"] = model.strip()
    return result


def _root_config() -> dict:
    """Do not reuse the inheritance loader's all-deny result as an opt-out.

    Malformed managed policy must not re-enable legacy ambient/API resolution.
    """
    path = get_default_hermes_root() / "config.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigError("Cannot load authoritative root provider_access policy") from exc
    if raw is not None and not isinstance(raw, dict):
        raise ConfigError("Root config must be a mapping")
    return raw or {}


def load_provider_access() -> dict[str, dict[str, str]]:
    """Read root policy only; absence defaults to subscription."""
    return validate_provider_access(_root_config().get("provider_access"))


def managed_deployment() -> bool:
    profiles = _root_config().get("profiles", {})
    if not isinstance(profiles, dict):
        raise ConfigError("Root profiles policy must be a mapping")
    shared = profiles.get("shared_credentials", {})
    if not isinstance(shared, dict) or not isinstance(shared.get("enabled", False), bool):
        raise ConfigError("Root shared_credentials policy must declare a boolean enabled setting")
    return shared.get("enabled") is True


def trusted_profile(profile: str | None = None) -> bool:
    name, is_root = resolve_profile_identity()
    name = profile if profile is not None else name
    if name in EXCLUDED_PROFILES:
        return False
    if name == "default" and (is_root or profile is not None):
        return True
    policy = load_shared_credential_policy()
    return policy.enabled and name in policy.trusted_profiles


def claude_source() -> tuple[Path, str, str]:
    """Return path, shape and child-only token; deny before reading any secret."""
    if not trusted_profile():
        raise ConfigError("Native Claude central authentication requires a trusted internal profile")
    path = get_default_hermes_root() / "claude" / "subscription.token"
    try:
        # No symlink following and a bounded read; reject permissive storage.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        fd = None
    except OSError as exc:
        raise ConfigError("Cannot open canonical Claude subscription source") from exc
    if fd is not None:
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            getuid = getattr(os, "getuid", None)
            if getuid is None or not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != getuid():
                raise ConfigError("Canonical Claude subscription source must be owner-only mode 0600")
            token = handle.read(16385).strip()
        if len(token) > 16384 or not token.startswith("sk-ant-oat") or any(c.isspace() for c in token):
            raise ConfigError("Invalid canonical Claude source; run claude setup-token")
        return path, "setup_token", token
    path = native_home() / ".claude" / ".credentials.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        oauth = data.get("claudeAiOauth") or {}
        if oauth.get("accessToken") and oauth.get("refreshToken") and not str(oauth["accessToken"]).startswith("sk-ant-api"):
            return path, "home_store", ""
    except (OSError, ValueError, AttributeError):
        pass
    return path, "none", ""


def claude_environment(base: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in base.items() if key in NATIVE_ENV_KEYS}
    env["HOME"] = str(native_home())
    _, shape, token = claude_source()
    if shape == "setup_token":
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


def root_api_key(vendor: str) -> str:
    if not trusted_profile():
        raise ConfigError("Central API authentication requires a trusted internal profile")
    from dotenv import dotenv_values

    path = get_default_hermes_root() / ".env"
    key = str(dotenv_values(path, interpolate=False).get(API_ENV[vendor]) or "").strip()
    if not key:
        raise ConfigError(f"Missing root {API_ENV[vendor]} in {path}; configure a separate API key (no automatic subscription escalation)")
    valid = key.startswith("sk-ant-api") if vendor == "anthropic" else (
        key.startswith("sk-") and not key.startswith("sk-ant-")
    )
    if not valid or any(c.isspace() for c in key):
        raise ConfigError(f"Root {API_ENV[vendor]} must contain a true vendor API key, never a subscription token")
    return key


def managed_runtime(requested: str, *, snapshot: dict | None = None) -> dict | None:
    """Enforce managed vendor boundaries before pools, explicit keys, or ambient env.

    A supervisor passes a frozen per-leg policy; ordinary processes snapshot
    root policy during initial resolution. Other providers remain untouched.
    """
    vendor = {"anthropic": "anthropic", "openai": "openai", "openai-direct": "openai", "custom:openai-direct": "openai", "openai-codex": "openai"}.get(requested)
    if vendor is None:
        return None
    if not managed_deployment() or not trusted_profile():
        return None
    modes = validate_provider_access(snapshot) if snapshot is not None else load_provider_access()
    mode = modes[vendor]
    if mode["mode"] == "api":
        key = root_api_key(vendor)
        return {
            "provider": "anthropic" if vendor == "anthropic" else "custom",
            "requested_provider": API_PROVIDERS[vendor],
            "api_mode": "anthropic_messages" if vendor == "anthropic" else "chat_completions",
            "base_url": "https://api.anthropic.com" if vendor == "anthropic" else "https://api.openai.com/v1",
            "api_key": key,
            "model": mode["api_model"],
            "source": str(get_default_hermes_root() / ".env"),
            "provider_access_mode": "api",
        }
    if vendor == "anthropic":
        from hermes_cli.auth import AuthError
        raise AuthError("Managed Anthropic subscription requires native Claude Code; run claude setup-token / claude auth login, not the raw anthropic provider",
                        provider="anthropic", code="native_claude_required", relogin_required=True)
    if requested != "openai-codex":
        raise ConfigError("Managed OpenAI subscription requires openai-codex; explicitly select provider_access.openai.mode api for paid API")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from hermes_cli.auth import resolve_codex_runtime_credentials

    home_token = set_hermes_home_override(get_default_hermes_root())
    try:
        creds = resolve_codex_runtime_credentials()
    finally:
        reset_hermes_home_override(home_token)
    if not creds.get("api_key") or str(creds["api_key"]).startswith("sk-"):
        raise ConfigError("OpenAI subscription requires a root OAuth credential, never an API key")
    return {
        **creds,
        "provider": "openai-codex", "api_mode": "codex_responses",
        "requested_provider": requested,
        "source": str(get_default_hermes_root() / "auth.json"),
    }


def provider_access_status() -> dict:
    """Credential presence is not evidence of server acceptance or funding."""
    if not trusted_profile():
        return {}
    result = {}
    for vendor, mode in load_provider_access().items():
        source = get_default_hermes_root() / (".env" if mode["mode"] == "api" else "auth.json")
        if vendor == "anthropic" and mode["mode"] == "subscription":
            source = get_default_hermes_root() / "claude" / "subscription.token"
        shape, present = "none", False
        error = None
        try:
            if mode["mode"] == "api":
                present = bool(root_api_key(vendor))
                shape = "api_key" if present else "none"
            elif vendor == "anthropic":
                source, shape, _ = claude_source()
                present = shape != "none"
            else:
                data = json.loads(source.read_text(encoding="utf-8"))
                tokens = ((data.get("providers") or {}).get("openai-codex") or {}).get("tokens") or {}
                present = bool(tokens.get("access_token") and tokens.get("refresh_token"))
                entries = (data.get("credential_pool") or {}).get("openai-codex") or []
                if isinstance(entries, list):
                    present = present or any(
                        isinstance(entry, dict) and entry.get("access_token") and entry.get("refresh_token")
                        and not str(entry["access_token"]).startswith("sk-")
                        for entry in entries
                    )
                shape = "home_store" if present else "none"
        except (OSError, ValueError, AttributeError):
            error = "missing-or-invalid-credential; operator authentication required"
        result[vendor] = {
            **mode, "default_mode": "subscription", "source_path": str(source),
            "credential_present": present, "credential_shape": shape,
            "available": False, "readiness": "configured-unverified" if present else "missing-auth",
            "applies_to": "new_runs", "live_sessions_unchanged": True,
            "api_funded": "unverified", "server_usability": "unverified",
        }
        if error:
            result[vendor]["status"] = error
    return result
