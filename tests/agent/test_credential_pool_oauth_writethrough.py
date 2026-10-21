"""Regression tests for credential-pool OAuth refresh write-through to root.

Companion to ``tests/hermes_cli/test_xai_oauth_writethrough.py``. That file
covers the *non-pool* xAI refresh path (``_save_xai_oauth_tokens``). These
cover the **credential-pool** refresh path
(``CredentialPool._sync_device_code_entry_to_auth_store``): when a profile
that has no own ``providers.<id>`` block refreshes — via the pool — a rotating
OAuth grant it resolved from the global-root fallback, the rotated chain must
be written back to the global root too. Otherwise root keeps a revoked refresh
token and every other profile reading root's stale grant dies with
``refresh_token_reused`` / ``invalid_grant`` once its access token expires
(issue #48415, the Codex/xAI analog of #43589).

The tests drive the real ``_sync_device_code_entry_to_auth_store`` against
real on-disk auth stores (profile + root under ``tmp_path``) rather than
mocking the save boundary, so they exercise the actual atomic write path.
"""

import base64
import json
import multiprocessing
import os
import threading
import time

import pytest

from agent import credential_pool as CP
from agent.credential_pool import (
    AUTH_TYPE_OAUTH,
    CredentialPool,
    PooledCredential,
)
from hermes_cli import auth as A


def _write_store(path, store):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")


def _read_store(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(provider: str, *, id: str, access_token: str, refresh_token: str):
    return PooledCredential(
        provider=provider,
        id=id,
        label="cred",
        auth_type=AUTH_TYPE_OAUTH,
        priority=0,
        source="device_code",
        access_token=access_token,
        refresh_token=refresh_token,
    )


def _jwt_with_exp(exp: int) -> str:
    def _part(payload):
        return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

    return f"{_part({'alg': 'none'})}.{_part({'exp': exp})}.sig"


def _multiprocess_refresh_worker(
    root_home,
    profile_name,
    start_barrier,
    finish_barrier,
    consume_lock,
    consumed,
    post_count,
    results,
):
    os.environ["HERMES_HOME"] = os.path.join(
        root_home, "profiles", profile_name
    )
    os.environ["HERMES_PROFILE"] = profile_name
    from hermes_constants import set_hermes_home_override

    set_hermes_home_override(os.environ["HERMES_HOME"])

    def fake_refresh(_access_token, refresh_token, **_kwargs):
        with consume_lock:
            if refresh_token in consumed:
                raise RuntimeError("refresh token reused")
            consumed[refresh_token] = True
            post_count.value += 1
        return {
            "access_token": _jwt_with_exp(int(time.time()) + 3600),
            "refresh_token": "rotated-refresh",
            "last_refresh": "2020-01-02T00:00:00Z",
        }

    A.refresh_codex_oauth_pure = fake_refresh
    pool = CP.load_pool("openai-codex")
    entry = pool.entries()[0]
    start_barrier.wait(timeout=10)
    refreshed = pool._refresh_entry(entry, force=False)
    finish_barrier.wait(timeout=10)
    converged = CP.load_pool("openai-codex").entries()[0]
    results.put(
        bool(
            refreshed
            and converged.access_token == refreshed.access_token
            and converged.refresh_token == refreshed.refresh_token
        )
    )


def _multiprocess_runtime_refresh_worker(
    root_home,
    profile_name,
    start_barrier,
    finish_barrier,
    consume_lock,
    consumed,
    post_count,
    results,
):
    os.environ["HERMES_HOME"] = os.path.join(
        root_home, "profiles", profile_name
    )
    os.environ["HERMES_PROFILE"] = profile_name
    from hermes_constants import set_hermes_home_override

    set_hermes_home_override(os.environ["HERMES_HOME"])

    def fake_refresh(_access_token, refresh_token, **_kwargs):
        with consume_lock:
            if refresh_token in consumed:
                raise A.AuthError(
                    "refresh token reused",
                    provider="openai-codex",
                    code="refresh_token_reused",
                    relogin_required=True,
                )
            consumed[refresh_token] = True
            post_count.value += 1
        return {
            "access_token": _jwt_with_exp(int(time.time()) + 3600),
            "refresh_token": "rotated-runtime-refresh",
            "last_refresh": "2020-01-02T00:00:00Z",
        }

    A.refresh_codex_oauth_pure = fake_refresh
    A._import_codex_cli_tokens = lambda: None
    start_barrier.wait(timeout=10)
    try:
        resolved = A.resolve_codex_runtime_credentials(
            refresh_if_expiring=True,
            refresh_skew_seconds=0,
        )
        ok = bool(resolved.get("api_key"))
    except Exception:
        ok = False
    finish_barrier.wait(timeout=10)
    from pathlib import Path

    profile_store = _read_store(
        Path(root_home) / "profiles" / profile_name / "auth.json"
    )
    results.put(
        bool(
            ok
            and not profile_store.get("providers", {}).get("openai-codex")
        )
    )


def _terminal_refresh_worker(
    profile_path,
    root_path,
    root_lock_held,
    profile_lock_held,
    results,
):
    from pathlib import Path

    profile_path = Path(profile_path)
    root_path = Path(root_path)
    A._auth_file_path = lambda: profile_path
    A._global_auth_file_path = lambda provider_id=None: root_path
    A.AUTH_LOCK_TIMEOUT_SECONDS = 1.0
    A._is_terminal_xai_oauth_refresh_error = lambda _exc: True

    def terminal_refresh(*_args, **_kwargs):
        root_lock_held.set()
        if not profile_lock_held.wait(timeout=5):
            raise RuntimeError("profile contender did not acquire its lock")
        raise A.AuthError(
            "revoked",
            provider="xai-oauth",
            code="invalid_grant",
            relogin_required=True,
        )

    A.refresh_xai_oauth_pure = terminal_refresh
    entry = _entry(
        "xai-oauth",
        id="shared-xai",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    entry.extra["shared_root_source"] = True
    try:
        result = CredentialPool("xai-oauth", [entry])._refresh_entry(
            entry, force=True
        )
        results.put(("refresh", result is None, None))
    except Exception as exc:
        results.put(("refresh", False, type(exc).__name__))


def _profile_then_root_lock_worker(
    profile_path,
    root_path,
    root_lock_held,
    profile_lock_held,
    results,
):
    from pathlib import Path

    profile_path = Path(profile_path)
    root_path = Path(root_path)
    A._auth_file_path = lambda: profile_path
    A._global_auth_file_path = lambda provider_id=None: root_path
    A.AUTH_LOCK_TIMEOUT_SECONDS = 1.0
    if not root_lock_held.wait(timeout=5):
        results.put(("writer", False, "root lock was not acquired"))
        return
    try:
        with A._auth_store_lock():
            profile_lock_held.set()
            with A._auth_store_lock(target_path=root_path):
                store = A._load_auth_store(root_path)
                ok = isinstance(store.get("providers"), dict)
        results.put(("writer", ok, None))
    except Exception as exc:
        results.put(("writer", False, type(exc).__name__))


@pytest.fixture
def profile_and_root(tmp_path, monkeypatch):
    """Wire a profile auth store + a distinct global-root auth store on disk.

    The pytest seat belt in ``_write_through_provider_state_to_global_root``
    only refuses the *real* user's ``$HOME/.hermes/auth.json``; a tmp_path
    root is allowed, so point HOME away from the tmp root to keep the guard
    from tripping on these fixtures.
    """
    profile_path = tmp_path / "profiles" / "work" / "auth.json"
    root_path = tmp_path / "root" / "auth.json"

    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda provider_id=None: root_path)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))
    return profile_path, root_path


@pytest.mark.parametrize(
    "provider",
    ["openai-codex", "xai-oauth"],
)
def test_pool_refresh_writes_through_to_root_when_profile_reads_root(
    profile_and_root, provider
):
    """A profile reading root's grant must push rotated tokens back to root."""
    profile_path, root_path = profile_and_root
    # Profile has NO own provider block (reads root via fallback).
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )

    pool = CredentialPool(provider, [])
    pool._sync_device_code_entry_to_auth_store(
        _entry(provider, id="e1", access_token="new-access", refresh_token="new-refresh")
    )

    # Profile got the rotated chain (existing behavior).
    profile = _read_store(profile_path)
    assert (
        profile["providers"][provider]["tokens"]["refresh_token"] == "new-refresh"
    )

    # AND the global root no longer holds the revoked refresh token (#48415).
    root = _read_store(root_path)
    assert root["providers"][provider]["tokens"]["access_token"] == "new-access"
    assert root["providers"][provider]["tokens"]["refresh_token"] == "new-refresh"


@pytest.mark.parametrize(
    "provider",
    ["openai-codex", "xai-oauth"],
)
def test_pool_refresh_does_not_touch_root_when_profile_shadows(
    profile_and_root, provider
):
    """A profile that genuinely shadows root must NOT clobber the root grant."""
    profile_path, root_path = profile_and_root
    # Profile has its OWN provider block: it shadows root legitimately.
    _write_store(
        profile_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "profile-old",
                        "refresh_token": "profile-old-refresh",
                    }
                }
            },
        },
    )
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "root-untouched",
                        "refresh_token": "root-untouched-refresh",
                    }
                }
            },
        },
    )

    pool = CredentialPool(provider, [])
    pool._sync_device_code_entry_to_auth_store(
        _entry(
            provider,
            id="e2",
            access_token="profile-new",
            refresh_token="profile-new-refresh",
        )
    )

    profile = _read_store(profile_path)
    assert (
        profile["providers"][provider]["tokens"]["refresh_token"]
        == "profile-new-refresh"
    )

    # Root keeps its own grant — write-through must not run when the profile
    # owns the block.
    root = _read_store(root_path)
    assert (
        root["providers"][provider]["tokens"]["refresh_token"]
        == "root-untouched-refresh"
    )


def test_write_through_helper_is_noop_in_classic_mode(monkeypatch, tmp_path):
    """When profile == root (classic mode), the helper must be a no-op.

    ``_global_auth_file_path`` returns None in classic mode; the profile save
    already wrote to root, so a second write would be redundant (and the
    helper has nothing to target).
    """
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    # Must not raise and must not attempt any write.
    CP._write_through_provider_state_to_global_root(
        "openai-codex", {"tokens": {"access_token": "a", "refresh_token": "r"}}
    )


def test_global_write_through_preserves_concurrent_root_update(
    profile_and_root, monkeypatch
):
    """A stale profile write-through must not erase a concurrent root login."""
    _profile_path, root_path = profile_and_root
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {"access_token": "old-xai", "refresh_token": "old-r"}
                }
            },
            "credential_pool": {
                "anthropic": [{"id": "anthropic-existing"}],
                "openrouter": [{"id": "openrouter-existing"}],
            },
        },
    )

    helper_loaded = threading.Event()
    helper_has_target_lock = threading.Event()
    allow_helper_save = threading.Event()
    writer_started = threading.Event()
    writer_done = threading.Event()
    real_auth_load = A._load_auth_store

    def paused_helper_load(path=None):
        store = real_auth_load(path)
        if threading.current_thread().name == "profile-write-through":
            target_holder = A._auth_lock_holder_for(root_path)
            if getattr(target_holder, "depth", 0) > 0:
                helper_has_target_lock.set()
            helper_loaded.set()
            assert allow_helper_save.wait(timeout=5)
        return store

    monkeypatch.setattr(A, "_load_auth_store", paused_helper_load)
    # The pre-fix implementation imported the loader directly; patch both
    # bindings so reverting the safe helper still exercises the stale ordering.
    monkeypatch.setattr(CP, "_load_auth_store", paused_helper_load)

    def profile_write_through():
        CP._write_through_provider_state_to_global_root(
            "xai-oauth",
            {"tokens": {"access_token": "new-xai", "refresh_token": "new-r"}},
        )

    def concurrent_codex_login():
        writer_started.set()
        with A._auth_store_lock(target_path=root_path):
            store = A._load_auth_store(root_path)
            A._store_provider_state(
                store,
                "openai-codex",
                {"tokens": {"access_token": "codex-a", "refresh_token": "codex-r"}},
                set_active=False,
            )
            pool = store.setdefault("credential_pool", {})
            pool["openai-codex"] = [{"id": "codex-login"}]
            A._save_auth_store(store, target_path=root_path)
        writer_done.set()

    helper = threading.Thread(target=profile_write_through, name="profile-write-through")
    helper.start()
    assert helper_loaded.wait(timeout=5)

    writer = threading.Thread(target=concurrent_codex_login, name="concurrent-login")
    writer.start()
    assert writer_started.wait(timeout=5)
    # A fixed helper already owns the target lock, so the writer will merge
    # after release. A reverted unlocked helper must first let the competing
    # login finish; only then do we release its stale save. This makes the
    # losing pre-fix ordering deterministic rather than scheduler-dependent.
    if not helper_has_target_lock.is_set():
        assert writer_done.wait(timeout=5)
    allow_helper_save.set()
    helper.join(timeout=5)
    writer.join(timeout=5)
    assert not helper.is_alive()
    assert not writer.is_alive()

    root = _read_store(root_path)
    assert root["providers"]["xai-oauth"]["tokens"]["refresh_token"] == "new-r"
    assert root["providers"]["openai-codex"]["tokens"]["refresh_token"] == "codex-r"
    assert root["credential_pool"]["openai-codex"] == [{"id": "codex-login"}]
    assert root["credential_pool"]["anthropic"] == [{"id": "anthropic-existing"}]
    assert root["credential_pool"]["openrouter"] == [{"id": "openrouter-existing"}]


def test_codex_pool_refresh_holds_auth_store_lock_across_post(monkeypatch, tmp_path):
    """The Codex OAuth pool refresh must POST under the cross-process auth lock.

    Codex refresh tokens are single-use. If two Hermes processes both read the
    same on-disk token and both POST it, the loser gets ``refresh_token_reused``.
    Serializing the sync -> refresh POST -> write-back sequence through the
    shared ``_auth_store_lock`` closes that window: a second process blocks on
    the flock and, once inside, adopts the rotated token instead of re-POSTing.

    This asserts the invariant directly — that ``refresh_codex_oauth_pure`` is
    only ever called while the auth-store lock is held — rather than snapshotting
    any token value.
    """
    provider = "openai-codex"
    profile_path = tmp_path / "auth.json"
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None)
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

    lock_held: dict = {"during_post": None}
    real_lock = A._auth_store_lock

    depth = {"n": 0}

    import contextlib

    @contextlib.contextmanager
    def tracking_lock(*args, **kwargs):
        depth["n"] += 1
        try:
            with real_lock(*args, **kwargs):
                yield
        finally:
            depth["n"] -= 1

    monkeypatch.setattr(A, "_auth_store_lock", tracking_lock)
    # credential_pool imported _auth_store_lock by name; patch that binding too.
    monkeypatch.setattr(CP, "_auth_store_lock", tracking_lock)

    def fake_refresh(access_token, refresh_token, **kwargs):
        # The POST to the token endpoint must happen with the lock held.
        lock_held["during_post"] = depth["n"] > 0
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "last_refresh": "2020-01-02T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)

    entry = _entry(
        provider,
        id="codex-1",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    pool = CredentialPool(provider, [entry])

    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert refreshed.access_token == "rotated-access"
    assert refreshed.refresh_token == "rotated-refresh"
    # The invariant: the single-use token POST ran inside the auth-store lock.
    assert lock_held["during_post"] is True


def test_inherited_codex_refresh_holds_root_lock_across_post(monkeypatch, tmp_path):
    provider = "openai-codex"
    profile_path = tmp_path / "profiles" / "talos" / "auth.json"
    root_path = tmp_path / "auth.json"
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    }
                }
            },
        },
    )
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(
        A,
        "_global_auth_file_path",
        lambda provider_id=None: root_path,
    )
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

    observed_targets = []
    real_lock = A._auth_store_lock

    import contextlib

    @contextlib.contextmanager
    def tracking_lock(*args, **kwargs):
        observed_targets.append(kwargs.get("target_path"))
        with real_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(A, "_auth_store_lock", tracking_lock)
    monkeypatch.setattr(CP, "_auth_store_lock", tracking_lock)

    def fake_refresh(access_token, refresh_token, **kwargs):
        assert observed_targets[-1] == root_path
        return {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "last_refresh": "2020-01-02T00:00:00Z",
        }

    monkeypatch.setattr(A, "refresh_codex_oauth_pure", fake_refresh)
    entry = _entry(
        provider,
        id="shared-codex",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    entry.extra["shared_root_source"] = True
    pool = CredentialPool(provider, [entry])

    refreshed = pool._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert root_path in observed_targets
    root = _read_store(root_path)
    assert root["providers"][provider]["tokens"]["refresh_token"] == "rotated-refresh"
    profile = _read_store(profile_path)
    assert not profile.get("providers", {}).get(provider)
    assert not profile.get("credential_pool", {}).get(provider)
    assert profile["credential_inheritance_status"][provider]["write_through_ok"] is True


def test_inherited_pool_refresh_does_not_invert_root_and_profile_locks(
    monkeypatch, tmp_path
):
    provider = "openai-codex"
    profile_path = tmp_path / "profiles" / "talos" / "auth.json"
    root_path = tmp_path / "auth.json"
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    }
                }
            },
        },
    )
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(
        A,
        "_global_auth_file_path",
        lambda provider_id=None: root_path,
    )
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

    stack = []
    inversions = []
    real_lock = A._auth_store_lock

    import contextlib

    @contextlib.contextmanager
    def tracking_lock(*args, **kwargs):
        target = kwargs.get("target_path")
        if target is None and root_path in stack:
            inversions.append((tuple(stack), target))
        stack.append(target)
        try:
            with real_lock(*args, **kwargs):
                yield
        finally:
            stack.pop()

    monkeypatch.setattr(A, "_auth_store_lock", tracking_lock)
    monkeypatch.setattr(CP, "_auth_store_lock", tracking_lock)
    monkeypatch.setattr(
        A,
        "refresh_codex_oauth_pure",
        lambda *_args, **_kwargs: {
            "access_token": "rotated-access",
            "refresh_token": "rotated-refresh",
            "last_refresh": "2020-01-02T00:00:00Z",
        },
    )
    entry = _entry(
        provider,
        id="shared-codex",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    entry.extra["shared_root_source"] = True

    refreshed = CredentialPool(provider, [entry])._refresh_entry(entry, force=True)

    assert refreshed is not None
    assert inversions == []


@pytest.mark.parametrize(
    ("provider", "terminal"),
    [("openai-codex", False), ("xai-oauth", True)],
)
def test_inherited_refresh_failure_does_not_take_profile_lock_or_create_shadow(
    monkeypatch, tmp_path, provider, terminal
):
    profile_path = tmp_path / "profiles" / "talos" / "auth.json"
    root_path = tmp_path / "auth.json"
    _write_store(profile_path, {"version": 1, "providers": {}})
    root_store = {
        "version": 1,
        "providers": {
            provider: {
                "tokens": {
                    "access_token": "stale-access",
                    "refresh_token": "stale-refresh",
                }
            }
        },
    }
    if terminal:
        root_store["credential_pool"] = {
            provider: [
                {
                    "id": f"shared-{provider}",
                    "provider": provider,
                    "auth_type": "oauth",
                    "source": "device_code",
                    "access_token": "stale-access",
                    "refresh_token": "stale-refresh",
                }
            ]
        }
    _write_store(root_path, root_store)
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(
        A,
        "_global_auth_file_path",
        lambda provider_id=None: root_path,
    )
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-root"))

    stack = []
    inversions = []
    real_lock = A._auth_store_lock

    import contextlib

    @contextlib.contextmanager
    def tracking_lock(*args, **kwargs):
        target = kwargs.get("target_path")
        if target is None and root_path in stack:
            inversions.append((tuple(stack), target))
        stack.append(target)
        try:
            with real_lock(*args, **kwargs):
                yield
        finally:
            stack.pop()

    monkeypatch.setattr(A, "_auth_store_lock", tracking_lock)
    monkeypatch.setattr(CP, "_auth_store_lock", tracking_lock)
    error = A.AuthError(
        "refresh failed",
        provider=provider,
        code="invalid_grant" if terminal else "temporarily_unavailable",
        relogin_required=terminal,
    )
    if provider == "openai-codex":
        monkeypatch.setattr(
            A,
            "refresh_codex_oauth_pure",
            lambda *_a, **_k: (_ for _ in ()).throw(error),
        )
        monkeypatch.setattr(
            A,
            "_is_terminal_codex_oauth_refresh_error",
            lambda _exc: terminal,
        )
    else:
        monkeypatch.setattr(
            A,
            "refresh_xai_oauth_pure",
            lambda *_a, **_k: (_ for _ in ()).throw(error),
        )
        monkeypatch.setattr(
            A,
            "_is_terminal_xai_oauth_refresh_error",
            lambda _exc: terminal,
        )

    entry = _entry(
        provider,
        id=f"shared-{provider}",
        access_token="stale-access",
        refresh_token="stale-refresh",
    )
    entry.extra["shared_root_source"] = True
    pool = CredentialPool(provider, [entry])
    source_path_calls = 0

    def source_path(_entry):
        nonlocal source_path_calls
        source_path_calls += 1
        return root_path if source_path_calls == 1 else None

    monkeypatch.setattr(pool, "_shared_root_auth_path", source_path)

    assert pool._refresh_entry(entry, force=True) is None

    assert source_path_calls >= 2
    assert inversions == []
    profile = _read_store(profile_path)
    assert not profile.get("providers", {}).get(provider)
    assert not profile.get("credential_pool", {}).get(provider)
    root = _read_store(root_path)
    root_tokens = root["providers"][provider]["tokens"]
    root_pool = root["credential_pool"][provider]
    if terminal:
        assert "access_token" not in root_tokens
        assert "refresh_token" not in root_tokens
        assert root_pool == []
    else:
        assert root_tokens["refresh_token"] == "stale-refresh"
        assert root_pool[0]["last_status"] == CP.STATUS_EXHAUSTED
        assert "shared_root_source" not in root_pool[0]
        assert "write_through_ok" not in root_pool[0]


def test_terminal_inherited_refresh_does_not_deadlock_profile_then_root_writer(
    tmp_path,
):
    """A terminal root refresh must not wait on the profile lock it does not own."""
    profile_path = tmp_path / "profiles" / "talos" / "auth.json"
    root_path = tmp_path / "auth.json"
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                "xai-oauth": {
                    "tokens": {
                        "access_token": "stale-access",
                        "refresh_token": "stale-refresh",
                    }
                }
            },
        },
    )
    ctx = multiprocessing.get_context("fork")
    root_lock_held = ctx.Event()
    profile_lock_held = ctx.Event()
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_terminal_refresh_worker,
            args=(
                str(profile_path),
                str(root_path),
                root_lock_held,
                profile_lock_held,
                results,
            ),
        ),
        ctx.Process(
            target=_profile_then_root_lock_worker,
            args=(
                str(profile_path),
                str(root_path),
                root_lock_held,
                profile_lock_held,
                results,
            ),
        ),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)

    assert [process.exitcode for process in processes] == [0, 0]
    outcomes = {
        name: (ok, error)
        for name, ok, error in (results.get(timeout=2) for _ in processes)
    }
    assert outcomes == {"refresh": (True, None), "writer": (True, None)}
    profile = _read_store(profile_path)
    assert not profile.get("providers", {}).get("xai-oauth")
    assert not profile.get("credential_pool", {}).get("xai-oauth")


def test_failed_shared_write_through_is_recorded_without_local_shadow(
    monkeypatch, tmp_path
):
    provider = "openai-codex"
    profile_path = tmp_path / "profiles" / "talos" / "auth.json"
    root_path = tmp_path / "auth.json"
    _write_store(profile_path, {"version": 1, "providers": {}})
    _write_store(
        root_path,
        {
            "version": 1,
            "providers": {
                provider: {
                    "tokens": {
                        "access_token": "old-access",
                        "refresh_token": "old-refresh",
                    }
                }
            },
        },
    )
    monkeypatch.setattr(A, "_auth_file_path", lambda: profile_path)
    monkeypatch.setattr(
        A,
        "_global_auth_file_path",
        lambda provider_id=None: root_path,
    )
    real_save = A._save_auth_store

    def fail_root_save(store, target_path=None):
        if target_path == root_path:
            raise OSError("simulated root write failure")
        return real_save(store, target_path=target_path)

    monkeypatch.setattr(A, "_save_auth_store", fail_root_save)
    monkeypatch.setattr(CP, "_save_auth_store", fail_root_save)
    entry = _entry(
        provider,
        id="shared-codex",
        access_token="new-access",
        refresh_token="new-refresh",
    )
    entry.extra["shared_root_source"] = True
    pool = CredentialPool(provider, [entry])

    assert pool._sync_device_code_entry_to_auth_store(entry) is False
    pool._record_shared_write_through_status(entry)

    status = A._credential_trust_status(provider)
    assert status["write_through_ok"] is False
    profile = _read_store(profile_path)
    assert not profile.get("providers", {}).get(provider)
    assert not profile.get("credential_pool", {}).get(provider)


def test_two_profiles_racing_refresh_converge_under_root_lock(tmp_path):
    """Two OS processes consume one single-use token once and converge."""
    from hermes_constants import get_default_hermes_root

    root_home = get_default_hermes_root()
    root_home.mkdir(parents=True, exist_ok=True)
    profile_names = ("lane-a", "lane-b")
    for profile_name in profile_names:
        profile_home = root_home / "profiles" / profile_name
        profile_home.mkdir(parents=True)
        _write_store(profile_home / "auth.json", {"version": 1, "providers": {}})

    (root_home / "config.yaml").write_text(
        json.dumps(
            {
                "profiles": {
                    "shared_credentials": {
                        "enabled": True,
                        "trusted_profiles": list(profile_names),
                        "providers": ["openai-codex"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    expired_access = _jwt_with_exp(int(time.time()) - 3600)
    _write_store(
        root_home / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": expired_access,
                        "refresh_token": "initial-refresh",
                    }
                }
            },
            "credential_pool": {
                "openai-codex": [
                    {
                        "id": "shared-codex",
                        "provider": "openai-codex",
                        "auth_type": "oauth",
                        "source": "device_code",
                        "access_token": expired_access,
                        "refresh_token": "initial-refresh",
                        "enabled": True,
                        "priority": 0,
                    }
                ]
            },
        },
    )

    ctx = multiprocessing.get_context("fork")
    manager = ctx.Manager()
    start_barrier = ctx.Barrier(2)
    finish_barrier = ctx.Barrier(2)
    consume_lock = ctx.Lock()
    consumed = manager.dict()
    post_count = ctx.Value("i", 0)
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_multiprocess_refresh_worker,
            args=(
                str(root_home),
                profile_name,
                start_barrier,
                finish_barrier,
                consume_lock,
                consumed,
                post_count,
                results,
            ),
        )
        for profile_name in profile_names
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)

    assert [process.exitcode for process in processes] == [0, 0]
    assert [results.get(timeout=2) for _ in processes] == [True, True]
    assert post_count.value == 1
    root = _read_store(root_home / "auth.json")
    assert (
        root["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == "rotated-refresh"
    )


def test_two_profiles_runtime_refresh_converges_under_root_lock(tmp_path):
    """The singleton runtime path must serialize on the root authority too."""
    from hermes_constants import get_default_hermes_root

    root_home = get_default_hermes_root()
    root_home.mkdir(parents=True, exist_ok=True)
    profile_names = ("runtime-a", "runtime-b")
    for profile_name in profile_names:
        profile_home = root_home / "profiles" / profile_name
        profile_home.mkdir(parents=True)
        _write_store(profile_home / "auth.json", {"version": 1, "providers": {}})

    (root_home / "config.yaml").write_text(
        json.dumps(
            {
                "profiles": {
                    "shared_credentials": {
                        "enabled": True,
                        "trusted_profiles": list(profile_names),
                        "providers": ["openai-codex"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    expired_access = _jwt_with_exp(int(time.time()) - 3600)
    _write_store(
        root_home / "auth.json",
        {
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": expired_access,
                        "refresh_token": "initial-runtime-refresh",
                    }
                }
            },
        },
    )

    ctx = multiprocessing.get_context("fork")
    manager = ctx.Manager()
    start_barrier = ctx.Barrier(2)
    finish_barrier = ctx.Barrier(2)
    consume_lock = ctx.Lock()
    consumed = manager.dict()
    post_count = ctx.Value("i", 0)
    results = ctx.Queue()
    processes = [
        ctx.Process(
            target=_multiprocess_runtime_refresh_worker,
            args=(
                str(root_home),
                profile_name,
                start_barrier,
                finish_barrier,
                consume_lock,
                consumed,
                post_count,
                results,
            ),
        )
        for profile_name in profile_names
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)

    assert [process.exitcode for process in processes] == [0, 0]
    assert [results.get(timeout=2) for _ in processes] == [True, True]
    assert post_count.value == 1
    root = _read_store(root_home / "auth.json")
    assert (
        root["providers"]["openai-codex"]["tokens"]["refresh_token"]
        == "rotated-runtime-refresh"
    )

