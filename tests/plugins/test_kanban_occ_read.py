"""Contract tests for the OCC read-only Kanban endpoints (OCC #263).

These drive the real FastAPI app (real routing, real middleware, real Starlette
``Request``) against a disposable SQLite board built from the production schema,
plus deliberately malformed rows. No production board, credential or network
resource is touched: ``HERMES_KANBAN_HOME`` points at a temp dir and the token is
generated per test in memory.

Coverage maps to Athena's binding A–H conditions and the packet's acceptance
criteria: credential isolation, board binding, read-only enforcement, status and
count reconciliation, malformed-record preservation, cursor semantics, order
matrix, secrecy allowlist and response bounds.
"""
from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
import unicodedata

import pytest
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from hermes_cli import kanban_occ_read as occ
from hermes_cli.dashboard_auth import clear_providers, token_auth

TOKEN = "A" * 43  # token68, 43 chars: the minimum accepted length


# ---------------------------------------------------------------------------
# Disposable board
# ---------------------------------------------------------------------------


def _schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL,
            priority INTEGER DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            branch_name TEXT,
            claim_lock TEXT,
            claim_expires INTEGER,
            tenant TEXT,
            result TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER,
            block_kind TEXT
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            run_id INTEGER,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            profile TEXT,
            step_key TEXT,
            status TEXT NOT NULL,
            claim_lock TEXT,
            claim_expires INTEGER,
            worker_pid INTEGER,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            outcome TEXT,
            summary TEXT,
            metadata TEXT,
            error TEXT
        );
        CREATE TABLE task_links (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            PRIMARY KEY (parent_id, child_id)
        );
        CREATE TABLE task_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE task_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            content_type TEXT,
            size INTEGER NOT NULL DEFAULT 0,
            uploaded_by TEXT,
            created_at INTEGER NOT NULL
        );
        """
    )


@pytest.fixture()
def home(tmp_path):
    """A temp kanban home with a synthetic board, clearly labelled as such."""
    kanban_home = tmp_path / "hermes"
    kanban_home.mkdir()
    board = kanban_home / "kanban.db"
    conn = sqlite3.connect(board)
    # WAL is what the production dispatcher uses; it is also the only mode in
    # which the "-wal"/"-shm" sidecars exist, so the read-only guarantee can
    # actually be tested against them (B9).
    conn.execute("PRAGMA journal_mode = WAL")
    _schema(conn)
    # Anchored in the recent past so "expired claim" / "old heartbeat" are real
    # relative to wall-clock ``now`` inside the request transaction.
    now = int(time.time())
    rows = [
        # id, title, status, created, assignee, tenant, priority, extra
        ("t_00000001", "Blocked payment work", "blocked", now - 900, "talos", "or7", 2),
        ("t_00000002", "Running migration", "running", now - 600, "talos", "or7", 1),
        ("t_00000003", "Review security packet", "review", now - 300, "athena", "or7", 5),
        ("t_00000004", "Ready feature", "ready", now - 100, "talos", "or7", 9),
        ("t_00000005", "Todo item", "todo", now - 50, "apollo", "or7", 0),
        ("t_00000006", "Done work", "done", now - 1200, "talos", "or7", 0),
        ("t_00000007", "Archived work", "archived", now - 5000, "talos", "or7", 0),
        ("t_00000008", "Synthetic lifecycle", "todo", now - 10, "talos",
         "synthetic-lifecycle-verification", 0),
    ]
    for tid, title, status, created, assignee, tenant, priority in rows:
        conn.execute(
            "INSERT INTO tasks(id, title, status, created_at, assignee, tenant, priority,"
            " body, result, last_failure_error, worker_pid, workspace_path)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                tid, title, status, created, assignee, tenant, priority,
                f"SECRET BODY {tid}", f"SECRET RESULT {tid}",
                f"SECRET ERROR {tid}", 4242, "/host/secret/path",
            ),
        )
    # A running task with an expired claim and an old heartbeat => claim_stale.
    conn.execute(
        "UPDATE tasks SET last_heartbeat_at = ?, claim_expires = ?, current_run_id = 1,"
        " started_at = ? WHERE id = 't_00000002'",
        (now - 7200, now - 60, now - 600),
    )
    conn.execute(
        "INSERT INTO task_runs(id, task_id, status, started_at, last_heartbeat_at, summary,"
        " metadata, error) VALUES(1, 't_00000002', 'running', ?, ?, 'SECRET RUN SUMMARY',"
        " 'SECRET META', 'SECRET RUN ERROR')",
        (now - 600, now - 7200),
    )
    conn.execute(
        "INSERT INTO task_events(task_id, kind, payload, created_at) VALUES"
        "('t_00000001', 'blocked', 'SECRET EVENT PAYLOAD', ?)",
        (now - 800,),
    )
    conn.execute(
        "INSERT INTO task_comments(task_id, author, body, created_at) VALUES"
        "('t_00000001', 'talos', 'SECRET COMMENT BODY', ?)",
        (now - 700,),
    )
    conn.execute(
        "INSERT INTO task_attachments(task_id, filename, stored_path, size, created_at)"
        " VALUES('t_00000001', 'f.pdf', '/host/attachments/f.pdf', 10, ?)",
        (now - 650,),
    )
    # Value columns are INTEGER-encoded timestamps; tasks.completed_at for done.
    conn.execute("UPDATE tasks SET completed_at = ? WHERE id = 't_00000006'", (now - 1000,))
    conn.commit()
    conn.close()
    return kanban_home


@pytest.fixture()
def config(monkeypatch, home):
    """Enable the surface against the temp board with an in-memory credential."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_OCC_READ_TOKEN", TOKEN)
    # A worker-style env that must NOT redirect the read.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home.parent / "other.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "some-other-board")
    _patch_config(monkeypatch, enabled=True, board="default")
    # Hold a writer connection open so the board stays in WAL mode with live
    # "-wal"/"-shm" sidecars (closing the last connection checkpoints and
    # removes them, which would make the sidecar tests vacuous).
    holder = sqlite3.connect(home / "kanban.db")
    holder.execute("PRAGMA journal_mode = WAL")
    yield home
    holder.close()


def _patch_config(monkeypatch, *, enabled: bool, board):
    from hermes_cli import kanban_occ_api

    monkeypatch.setattr(
        kanban_occ_api,
        "_load_occ_config",
        lambda: (enabled, board),
        raising=True,
    )


@pytest.fixture()
def app(monkeypatch, config):
    from hermes_cli import kanban_occ_api

    clear_providers()
    token_auth.clear_token_routes()
    # Budgets are process-global by design (they protect the real service), so
    # every test starts from a full bucket rather than inheriting the previous
    # test's consumption.
    kanban_occ_api._AuthorizedBudget = kanban_occ_api._Budget(per_minute=120, burst=30)
    kanban_occ_api._FailedBudget = kanban_occ_api._Budget(per_minute=30, burst=30)
    kanban_occ_api._ExhaustedAudit.reset()
    # Pre-comparison audit windows are process-global too (B1 round 3); a test
    # that asserts "one record was written" must not inherit the previous
    # test's window.
    kanban_occ_api._PreComparisonAudit.reset()
    # The credential is snapshotted once per process (B8). Each test starts
    # from "not yet snapshotted" so its own environment decides the value.
    kanban_occ_api._reset_token_snapshot()
    kanban_occ_api.register_routes()

    application = FastAPI()
    application.middleware("http")(token_auth.token_auth_middleware)
    application.include_router(kanban_occ_api.router, prefix="/api/plugins/kanban")

    @application.get("/api/plugins/kanban/board")
    async def _board_alias():
        return {"leaked": True}

    yield application
    clear_providers()
    token_auth.clear_token_routes()


def _auth(token: str = TOKEN, **extra):
    headers = {"Authorization": f"Bearer {token}"}
    headers.update(extra)
    return headers


def _get(client: TestClient, path: str, *, token: str = TOKEN, query: str = "", **extra):
    url = f"/api/plugins/kanban/occ/v1/{path}"
    if query:
        url += "?" + query
    return client.get(url, headers=_auth(token) if token else None, **extra)


# ---------------------------------------------------------------------------
# Authentication and isolation
# ---------------------------------------------------------------------------


def test_summary_requires_the_dedicated_credential(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert client.get("/api/plugins/kanban/occ/v1/summary").status_code == 401
        assert _get(client, "summary", token="B" * 43).status_code == 401
        assert _get(client, "summary").status_code == 200


def test_occ_credential_does_not_authenticate_the_dashboard_board_route(app, config):
    """The scoped credential is not a dashboard credential.

    B9 round 3: this used to assert registry facts only. It now *sends the
    request* carrying a valid OCC credential to the real ``/board`` path and
    asserts the seam left it unauthenticated, which is precisely the flag the
    production session gate keys on. The local app mounts no session gate, so a
    200 here would be the harness, not the product — the assertion is therefore
    on ``token_authenticated``, captured by an inner probe handler.
    """
    from hermes_cli.dashboard_auth import token_auth as seam

    probe = FastAPI()
    probe.middleware("http")(seam.token_auth_middleware)
    observed: dict[str, bool] = {}

    # A dedicated probe path: the shared `app` fixture also mounts a
    # `/api/plugins/kanban/board` route, and re-registering the same path in a
    # second app makes FastAPI treat `request` as a query field. The property
    # under test is path-independent: an unregistered dashboard path must never
    # become token-authenticated.
    board_path = "/api/plugins/kanban/board-probe"

    @probe.get(board_path)
    async def _board(request: Request):
        observed["authenticated"] = bool(
            getattr(request.state, "token_authenticated", False)
        )
        return {"reached": True}

    assert seam.is_token_route(board_path) is False
    assert seam.route_verifier(board_path) is None
    with TestClient(probe, client=("127.0.0.1", 40000)) as client:
        resp = client.get(board_path, headers=_auth())
    assert resp.status_code == 200, resp.text
    assert observed["authenticated"] is False, (
        "the OCC credential must never mark a dashboard route token-authenticated"
    )


def test_cookie_session_is_refused_on_the_machine_route(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary",
            headers={"Cookie": "hermes_session=whatever"},
        )
    assert resp.status_code == 401


def test_session_token_header_is_refused(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary",
            headers={"X-Hermes-Session-Token": "whatever"},
        )
    assert resp.status_code == 401


def test_query_token_is_refused(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(f"/api/plugins/kanban/occ/v1/summary?token={TOKEN}")
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "header,value",
    [
        ("Forwarded", "for=1.2.3.4"),
        ("X-Forwarded-For", "1.2.3.4"),
        ("X-Real-IP", "1.2.3.4"),
        ("X-Forwarded-Proto", "https"),
        # B7: names that were absent from the old fixed list. A caller must not
        # be able to pick a forwarding header the server happens not to know.
        ("X-Forwarded-Prefix", "/evil"),
        ("X-Forwarded-Uri", "/admin"),
        ("X-Forwarded-Ssl", "on"),
        ("X-Forwarded-Scheme", "https"),
        ("X-Forwarded-Client-Ip", "1.2.3.4"),
        ("X-Forwarded-Invented-By-A-Future-Proxy", "x"),
    ],
)
def test_forwarding_headers_are_refused(app, header, value):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary", headers=_auth(**{header: value})
        )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Peer allowlist (Athena A5 + B3)
# ---------------------------------------------------------------------------


def _verify_from(host):
    """Call the real verifier with a synthetic scope carrying ``host`` as peer."""
    from starlette.requests import Request

    from hermes_cli import kanban_occ_api

    scope = {
        "type": "http",
        "method": "GET",
        "path": kanban_occ_api.SUMMARY_PATH,
        "raw_path": kanban_occ_api.SUMMARY_PATH.encode("ascii"),
        "root_path": "",
        "headers": [(b"authorization", f"Bearer {TOKEN}".encode("ascii"))],
        "client": (host, 40000) if host is not None else None,
        "query_string": b"",
    }
    return kanban_occ_api._verify_occ_request(Request(scope))


@pytest.mark.parametrize(
    "peer",
    [
        "127.0.0.1",
        "127.9.9.9",
        "10.1.2.3",
        "172.16.0.5",
        "172.31.255.254",
        "192.168.1.10",
        "::1",
        "fd00::1",
        "fc00::1",
        "::ffff:127.0.0.1",   # IPv4-mapped loopback unwraps first
        "::ffff:10.0.0.7",    # IPv4-mapped RFC1918 unwraps first
    ],
)
def test_allowlisted_peers_are_accepted(app, peer):
    assert _verify_from(peer).ok is True, peer


@pytest.mark.parametrize(
    "peer",
    [
        "8.8.8.8",            # public IPv4
        "1.1.1.1",            # public IPv4
        "169.254.169.254",    # link-local / cloud metadata
        "100.64.0.1",         # CGNAT (RFC 6598)
        "2001::1",            # Teredo range: is_private accepted this
        "2001:db8::1",        # IPv6 documentation
        "192.0.2.1",          # IPv4 documentation
        "198.18.0.1",         # benchmarking range accepted by is_private
        "203.0.113.9",        # IPv4 documentation
        "0.0.0.0",            # unspecified
        "::ffff:8.8.8.8",     # mapped public address
        "not-an-ip",          # unparsable
        "",                   # empty
        None,                 # no transport peer at all
    ],
)
def test_non_allowlisted_peers_are_refused(app, peer):
    decision = _verify_from(peer)
    assert decision.ok is False, peer
    assert decision.status_code == 401


def test_repeated_authorization_header_is_refused(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary",
            headers=[("Authorization", f"Bearer {TOKEN}"), ("Authorization", f"Bearer {TOKEN}")],
        )
    assert resp.status_code == 401


def test_bearer_scheme_is_required(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary",
            headers={"Authorization": f"Basic {TOKEN}"},
        )
    assert resp.status_code == 401


def test_trailing_slash_and_case_variants_are_not_the_registered_path(app):
    """Only the exact registered bytes authenticates (seam-level guarantee)."""
    from hermes_cli.dashboard_auth import token_auth as seam

    for path in (
        "/api/plugins/kanban/occ/v1/summary/",
        "/api/plugins/kanban/occ/v1/SUMMARY",
        "/api/plugins/kanban/occ/v1//summary",
    ):
        assert seam.is_token_route(path) is False


def test_trailing_slash_request_does_not_reach_a_handler_as_scoped(app):
    """A trailing-slash request is redirected/refused, never served as OCC."""
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary/", headers=_auth(), follow_redirects=False
        )
    assert resp.status_code in {307, 401, 404}


def _handler_guard_result(*, principal, raw_path):
    """Run the real handler guard with a synthetic request state."""
    from starlette.requests import Request

    from hermes_cli import kanban_occ_api

    scope = {
        "type": "http",
        "method": "GET",
        "path": kanban_occ_api.SUMMARY_PATH,
        "raw_path": raw_path,
        "root_path": "",
        "headers": [],
        "client": ("127.0.0.1", 40000),
        "query_string": b"",
    }
    request = Request(scope)
    request.state.token_principal = principal
    return kanban_occ_api._handler_guard(request, kanban_occ_api.SUMMARY_PATH)


def test_handler_guard_404s_for_a_session_principal(app):
    """B9: the handler's own identity check must actually be exercised.

    A cookie session that arrives through an unregistered alias path reaches
    the handler with *some* principal. It must get 404 — not the data — because
    it is not the sentinel OCC principal object.
    """
    from hermes_cli import kanban_occ_api
    from hermes_cli.dashboard_auth.base import TokenPrincipal

    sentinel = kanban_occ_api.KANBAN_OCC_PRINCIPAL
    impostor = TokenPrincipal(
        principal=sentinel.principal, provider=sentinel.provider, scopes=sentinel.scopes
    )
    # Equal by value, different by identity — which is the whole point.
    assert impostor == sentinel and impostor is not sentinel
    result = _handler_guard_result(
        principal=impostor, raw_path=b"/api/plugins/kanban/occ/v1/summary"
    )
    assert result is not None
    assert result.status_code == 404
    assert json.loads(result.body) == {"error": "not_found"}


def test_handler_guard_404s_without_a_principal(app):
    result = _handler_guard_result(
        principal=None, raw_path=b"/api/plugins/kanban/occ/v1/summary"
    )
    assert result is not None and result.status_code == 404


def test_handler_guard_404s_on_a_mismatched_raw_path(app):
    """Reached through an alias, even the real principal gets 404."""
    from hermes_cli import kanban_occ_api

    result = _handler_guard_result(
        principal=kanban_occ_api.KANBAN_OCC_PRINCIPAL,
        raw_path=b"/api/plugins/kanban/occ/v1/summary/",
    )
    assert result is not None and result.status_code == 404


def test_handler_guard_accepts_the_real_principal_on_the_exact_path(app):
    from hermes_cli import kanban_occ_api

    assert (
        _handler_guard_result(
            principal=kanban_occ_api.KANBAN_OCC_PRINCIPAL,
            raw_path=b"/api/plugins/kanban/occ/v1/summary",
        )
        is None
    )


def test_disabled_surface_returns_401_even_with_a_valid_token(monkeypatch, app, config):
    _patch_config(monkeypatch, enabled=False, board="default")
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 401


def test_missing_token_env_returns_401(monkeypatch, app):
    """An unset credential closes the surface (no snapshot has been taken yet)."""
    from hermes_cli import kanban_occ_api

    monkeypatch.delenv("HERMES_KANBAN_OCC_READ_TOKEN", raising=False)
    kanban_occ_api._reset_token_snapshot()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 401


def test_short_configured_token_is_treated_as_unconfigured(monkeypatch, app):
    from hermes_cli import kanban_occ_api

    monkeypatch.setenv("HERMES_KANBAN_OCC_READ_TOKEN", "tooshort")
    kanban_occ_api._reset_token_snapshot()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary", token="tooshort").status_code == 401


def test_token_is_snapshotted_once_and_env_rewrite_does_not_rotate_it(monkeypatch, app, config):
    """B8: the credential is read once. A live environ rewrite must not work.

    The runbook says rotation requires a process restart. If the verifier
    re-read ``os.environ`` per request, any code in the process could swap the
    credential without one — so this asserts the *documented* behaviour, in
    both directions: the old value keeps working and the new one does not.
    """
    new_token = "B" * 43
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 200
        monkeypatch.setenv("HERMES_KANBAN_OCC_READ_TOKEN", new_token)
        assert _get(client, "summary", token=new_token).status_code == 401
        assert _get(client, "summary", token=TOKEN).status_code == 200


def test_registration_must_not_snapshot_the_credential(app, monkeypatch):
    """B8 round 3: eager snapshotting at registration deadlocks the import.

    Athena round 3 asked for the credential to be frozen inside
    ``register_routes()``. That cannot work: registration runs from the kanban
    plugin's API module, which ``web_server`` imports from inside
    ``_mount_plugin_api_routes()`` while ``web_server`` is still executing its
    own module body, and validating the token consults
    ``web_server._SESSION_TOKEN``. Doing it eagerly re-enters the in-progress
    import and hangs the process forever (measured with a faulthandler dump).

    This pins the safe contract instead: after registration the snapshot is
    still unset, and the first request is what freezes it.
    """
    from hermes_cli import kanban_occ_api

    kanban_occ_api._reset_token_snapshot()
    kanban_occ_api.register_routes()
    assert kanban_occ_api._configured_token is kanban_occ_api._UNSET, (
        "register_routes() must not snapshot the credential; web_server is still "
        "importing and validation would re-enter it"
    )
    # A request is safe: by then web_server has finished importing.
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 200
    assert kanban_occ_api._configured_token == TOKEN


# ---------------------------------------------------------------------------
# Rate limiting, 429 and audit hygiene (Athena A7 + B1/B2)
# ---------------------------------------------------------------------------


@pytest.fixture()
def audit_records(monkeypatch):
    """Capture audit writes from **both** audit call sites.

    Two modules write auth-failure records on a route-scoped rejection:
    ``kanban_occ_api`` (the verifier) and ``token_auth`` (the seam's alias
    probes). ``token_auth`` imported ``audit_log`` by name at module scope
    (``from ...audit import audit_log``), so patching only the source module
    cannot intercept it. R3-2: without patching ``token_auth.audit_log`` as
    well, the "the seam must not duplicate" assertion could never fail.
    """
    from hermes_cli import kanban_occ_api
    from hermes_cli.dashboard_auth import audit as audit_module
    from hermes_cli.dashboard_auth import token_auth as token_auth_module

    records: list[dict] = []

    def _capture(event, **fields):
        records.append({"event": getattr(event, "value", str(event)), **fields})

    monkeypatch.setattr(audit_module, "audit_log", _capture)
    monkeypatch.setattr(token_auth_module, "audit_log", _capture)
    # _reject_audit imports the symbol lazily, so patching the module attribute
    # above is enough; this assertion keeps that contract honest.
    assert kanban_occ_api._reject_audit.__module__ == "hermes_cli.kanban_occ_api"
    return records


def test_successful_requests_do_not_consume_the_failure_budget(app):
    """B1: a legitimate caller must never be throttled by someone's guessing.

    Before the fix ``_FailedBudget.take()`` ran before the digest compare, so
    ~30 wrong tokens a minute locked the real caller out of the 120/min ceiling
    it was promised. Here the failure bucket is small and the success bucket is
    not: guessing must not touch the caller's budget.
    """
    from hermes_cli import kanban_occ_api

    kanban_occ_api._FailedBudget = kanban_occ_api._Budget(per_minute=1, burst=3)
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for _ in range(10):
            assert _get(client, "summary", token="Z" * 43).status_code in {401, 429}
        # The legitimate caller is still served after the guesser is throttled.
        assert _get(client, "summary").status_code == 200
        # Moving to a fresh "exhausted" budget, a success still does not debit
        # the failure bucket: its token count is unchanged by a passing request.
        assert kanban_occ_api._FailedBudget.retry_after() > 0


def test_passing_requests_leave_the_failure_bucket_untouched(app):
    """The debit is released on success, not merely checked."""
    from hermes_cli import kanban_occ_api

    budget = kanban_occ_api._Budget(per_minute=1, burst=3)
    kanban_occ_api._FailedBudget = budget
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 200
    # A passing request must not have consumed failure capacity: the bucket can
    # still absorb its full burst of wrong attempts.
    for _ in range(3):
        assert budget.take() is None
    assert budget.take() is not None


def test_failure_budget_exhaustion_returns_429_with_retry_after(app):
    """B1: throttling is 429 ``rate_limited``, never a credential-shaped 401."""
    from hermes_cli import kanban_occ_api

    kanban_occ_api._FailedBudget = kanban_occ_api._Budget(per_minute=1, burst=1)
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary", token="Z" * 43).status_code == 401
        resp = _get(client, "summary", token="Z" * 43)
    assert resp.status_code == 429
    assert json.loads(resp.content) == {"error": "rate_limited"}
    assert int(resp.headers["retry-after"]) >= 1


def test_authorized_budget_exhaustion_returns_429_with_retry_after(app):
    """B1: the *success* budget is separate and also answers 429."""
    from hermes_cli import kanban_occ_api

    kanban_occ_api._AuthorizedBudget = kanban_occ_api._Budget(per_minute=1, burst=1)
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 200
        resp = _get(client, "summary")
    assert resp.status_code == 429
    assert json.loads(resp.content) == {"error": "rate_limited"}
    assert resp.headers["retry-after"]


def test_429_body_and_headers_are_safe(app):
    from hermes_cli import kanban_occ_api

    kanban_occ_api._AuthorizedBudget = kanban_occ_api._Budget(per_minute=1, burst=1)
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        _get(client, "summary")
        resp = _get(client, "summary")
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert TOKEN not in resp.text


def test_exhausted_budget_audits_once_per_window(app, audit_records):
    """B1: after exhaustion, auditing is summarized — not one line per attempt."""
    from hermes_cli import kanban_occ_api

    kanban_occ_api._FailedBudget = kanban_occ_api._Budget(per_minute=1, burst=1)
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        _get(client, "summary", token="Z" * 43)          # 401, one audit line
        for _ in range(20):
            assert _get(client, "summary", token="Z" * 43).status_code == 429
    exhausted = [
        r for r in audit_records if r.get("reason") == "failure_budget_exhausted"
    ]
    assert len(exhausted) == 1, exhausted
    # The 20 suppressed attempts are counted, but the window wrote one record.
    assert sum(kanban_occ_api._ExhaustedAudit.suppressed.values()) >= 19


def test_each_rejection_is_audited_exactly_once(app, audit_records):
    """B2: the verifier owns auditing; the seam must not write a second line."""
    with TestClient(app, client=("192.168.5.5", 40000)) as client:
        assert _get(client, "summary", token="Z" * 43).status_code == 401
    mismatch = [r for r in audit_records if r.get("reason") == "credential_mismatch"]
    seam = [r for r in audit_records if r.get("reason") == "route_credential_rejected"]
    assert len(mismatch) == 1, audit_records
    assert seam == [], "the seam must not duplicate a verifier-route rejection"
    assert mismatch[0]["provider"] == "kanban-occ-read"


def test_audit_ip_comes_from_the_peer_not_the_forwarded_header(app, audit_records):
    """B2: the audit IP is the transport peer, never attacker-controlled text."""
    with TestClient(app, client=("192.168.5.5", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary",
            headers=_auth(token="Z" * 43, **{"X-Forwarded-For": "203.0.113.66"}),
        )
    assert resp.status_code == 401
    assert audit_records, "a rejection must still be audited"
    for record in audit_records:
        assert record["ip"] == "192.168.5.5", record
        assert "203.0.113.66" not in json.dumps(record)


def test_credential_mismatch_is_audited_for_every_peer_r4_2(app, audit_records):
    """R4-2: the window must not suppress ``credential_mismatch``.

    Round 3's window was applied to every rejection, so a second guessing peer
    inside the same minute went unrecorded. Only the pre-comparison reasons are
    windowed; a wrong credential is the record the failure budget exists for.
    """
    from hermes_cli import kanban_occ_api

    kanban_occ_api._PreComparisonAudit.reset()
    with TestClient(app, client=("192.168.5.5", 40000)) as client:
        assert _get(client, "summary", token="Z" * 43).status_code == 401
    with TestClient(app, client=("10.1.2.3", 40000)) as client:
        assert _get(client, "summary", token="Y" * 43).status_code == 401
    mismatches = [r for r in audit_records if r.get("reason") == "credential_mismatch"]
    assert len(mismatches) == 2, mismatches
    assert {r["ip"] for r in mismatches} == {"192.168.5.5", "10.1.2.3"}


def test_pre_comparison_rejections_are_audited_once_per_window(app, audit_records):
    """B1 round 3: an unauthenticated loop cannot amplify the audit log.

    Ten rejections happen before the credential comparison (config, peer,
    headers, query, syntax). A caller can trigger them in a loop without ever
    presenting a token, so each reason is audited once per window instead of
    once per request.
    """
    from hermes_cli import kanban_occ_api

    kanban_occ_api._PreComparisonAudit.reset()
    # No Authorization header at all -> authorization_header_count, 20 times.
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for _ in range(20):
            assert client.get("/api/plugins/kanban/occ/v1/summary").status_code == 401
    counted = [
        r for r in audit_records if r.get("reason") == "authorization_header_count"
    ]
    assert len(counted) == 1, counted
    # Suppression is per reason, so other classes still surface.
    assert kanban_occ_api._PreComparisonAudit.should_write("peer_not_private") is True


def test_pre_comparison_window_does_not_throttle_the_legitimate_caller(app):
    """B1 round 3: windowing the audit must not change any response status."""
    kanban_occ_api = __import__("hermes_cli.kanban_occ_api", fromlist=["x"])
    kanban_occ_api._PreComparisonAudit.reset()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for _ in range(5):
            assert client.get("/api/plugins/kanban/occ/v1/summary").status_code == 401
        # A correct credential is still served throughout the window.
        assert _get(client, "summary").status_code == 200


# ---------------------------------------------------------------------------
# Board binding (Athena C1 + H)
# ---------------------------------------------------------------------------


def test_worker_board_env_cannot_redirect_the_configured_board(app, config):
    """HERMES_KANBAN_DB points at a non-existent DB; the read must still work."""
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = _get(client, "summary")
    assert resp.status_code == 200
    assert json.loads(resp.content)["board"] == "default"


def test_missing_board_returns_503_board_unavailable(monkeypatch, app, config):
    (config / "kanban.db").unlink()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = _get(client, "summary")
    assert resp.status_code == 503
    assert json.loads(resp.content) == {"error": "board_unavailable"}


def test_schema_without_required_columns_returns_503(app, config):
    conn = sqlite3.connect(config / "kanban.db")
    conn.executescript("DROP TABLE task_events;")
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = _get(client, "summary")
    assert resp.status_code == 503
    assert json.loads(resp.content) == {"error": "schema_unsupported"}


# ---------------------------------------------------------------------------
# Summary counts (Athena E/F2 + packet §10.7)
# ---------------------------------------------------------------------------


def test_summary_counts_reconcile_and_exclude_archived(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "summary").content)
    counts = body["counts_by_status"]
    # 7 non-archived rows; unknown/invalid zero on a clean board.
    assert sum(counts.values()) == body["total_filtered"] == 7
    assert counts["archived"] if "archived" in counts else True  # archived never a key
    assert "archived" not in counts
    assert counts["blocked"] == 1 and counts["running"] == 1 and counts["done"] == 1
    assert body["invalid_records"] == 0
    assert body["schema_version"] == 1


def test_summary_and_list_filters_reconcile(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        summary = json.loads(_get(client, "summary", query="assignee=talos").content)
        listed = json.loads(
            _get(client, "tasks", query="assignee=talos&limit=100&order=updated_desc").content
        )
    assert summary["total_filtered"] == listed["total"]
    assert sum(summary["counts_by_status"].values()) == listed["total"]


def test_exclude_tenant_removes_synthetic_workspace(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(
            _get(client, "summary", query="exclude_tenant=synthetic-lifecycle-verification").content
        )
    assert body["total_filtered"] == 6
    assert "synthetic-lifecycle-verification" not in body["tenants"]


def test_q_matches_task_id_and_title_case_insensitively(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        by_title = json.loads(_get(client, "summary", query="q=PAYMENT").content)
        by_id = json.loads(_get(client, "summary", query="q=t_00000003").content)
    assert by_title["total_filtered"] == 1
    assert by_id["total_filtered"] == 1


def test_q_wildcards_are_literal(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "summary", query="q=%25").content)
    # No title or id contains a literal '%'; a LIKE-based match would return all.
    assert body["total_filtered"] == 0


def test_q_over_80_scalars_is_rejected(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary", query="q=" + "a" * 81).status_code == 400


def test_q_control_characters_are_rejected(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary", query="q=a%00b").status_code == 400


def test_q_nfc_normalisation(app, config):
    """A decomposed query still matches a precomposed stored title.

    B9: this used to assert ``total_filtered >= 0``, which no implementation
    could fail. It now inserts a genuinely precomposed row and asserts the
    match, so breaking the NFC step makes it red.
    """
    composed = unicodedata.normalize("NFC", "Café Payment")
    decomposed = unicodedata.normalize("NFD", "Café Payment")
    assert composed != decomposed

    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000094',?,?,?)",
        (composed, "todo", 1_799_999_000),
    )
    conn.commit()
    conn.close()

    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        # The needle is normalised to NFC by ``normalize_q``; the haystack is
        # NFC-folded in SQL. Without either step this returns nothing.
        matched = json.loads(_get(client, "summary", query="q=cafe%CC%81").content)
        unmatched = json.loads(_get(client, "summary", query="q=cafe").content)
    assert matched["total_filtered"] == 1
    assert unmatched["total_filtered"] == 0


# ---------------------------------------------------------------------------
# Malformed records survive (Athena E + packet §10.6)
# ---------------------------------------------------------------------------


def _insert_raw(conn, tid_blob, status_value, *, title="Malformed"):
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES(?,?,?,?)",
        (tid_blob, title, status_value, 1_799_999_000),
    )


@pytest.mark.parametrize(
    "status_value",
    ["run\x00ning", "Running", " running", "RUNNING", "future_state", ""],
)
def test_unsupported_status_values_stay_visible(app, config, status_value):
    conn = sqlite3.connect(config / "kanban.db")
    _insert_raw(conn, "t_00000099", status_value)
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "tasks", query="limit=100&order=updated_desc").content)
    row = next((t for t in body["tasks"] if t["id"] == "t_00000099"), None)
    assert row is not None, "malformed record must never be dropped"
    assert row["status_class"] in {"unknown", "invalid"}


def test_non_status_strings_are_unknown_and_blobs_are_invalid(app, config):
    """TEXT-affinity behaviour is respected, not fought.

    ``tasks.status`` has TEXT affinity, so a bound integer arrives as the string
    "7" — a clean but unsupported status, i.e. ``unknown``. A BLOB keeps BLOB
    storage class and is ``invalid``. Neither is dropped or coerced to a known
    status.
    """
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000098','Intish',7,?)",
        (1_799_999_000,),
    )
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000097','Blobby2',?,?)",
        (b"weird", 1_799_999_000),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "tasks", query="limit=100&order=updated_desc").content)
    classes = {t["id"]: t["status_class"] for t in body["tasks"]}
    assert classes.get("t_00000098") == "unknown"
    assert classes.get("t_00000097") == "invalid"


def test_blob_status_is_classified_not_decoded_as_text(app, config):
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000096','Blobby',?,?)",
        (b"running", 1_799_999_000),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "tasks", query="limit=100&order=updated_desc").content)
    row = next(t for t in body["tasks"] if t["id"] == "t_00000096")
    assert row["status_class"] == "invalid"


def test_blob_status_counts_as_invalid_in_the_summary_too(app, config):
    """B4: the summary must classify a BLOB b"running" exactly like the list row.

    Before the fix the summary decoded the bytes and added them to
    ``counts_by_status["running"]`` while the row rendered ``invalid`` — so the
    strip and the rows disagreed for the same task.
    """
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000093','BlobRun',?,?)",
        (b"running", 1_799_999_000),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        summary = json.loads(_get(client, "summary").content)
        listed = json.loads(_get(client, "tasks", query="limit=100&order=updated_desc").content)

    counts = summary["counts_by_status"]
    row = next(t for t in listed["tasks"] if t["id"] == "t_00000093")
    assert row["status_class"] == "invalid"
    assert row["status"] is None, "non-TEXT storage has no text to display"
    # The seeded board has exactly one real running task (t_00000002).
    assert counts["running"] == 1, counts
    assert counts["invalid"] == 1, counts
    assert sum(counts.values()) == summary["total_filtered"] == listed["total"]


def test_summary_and_list_reconcile_per_status_class(app, config):
    """Every class bucket in the summary equals the filtered list total."""
    conn = sqlite3.connect(config / "kanban.db")
    for row in (
        ("t_00000092", "BlobRow", b"todo"),
        ("t_00000091", "IntRow", 7),
        ("t_00000090", "OddRow", "future_state"),
        ("t_0000008f", "NulRow", "run\x00ning"),
    ):
        conn.execute(
            "INSERT INTO tasks(id, title, status, created_at) VALUES(?,?,?,?)",
            (*row, 1_799_999_000),
        )
    conn.commit()
    conn.close()

    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for status_class, bucket in (
            ("invalid", "invalid"),
            ("unknown", "unknown"),
            ("unknown,invalid", None),
        ):
            summary = json.loads(_get(client, "summary", query=f"status_class={status_class}").content)
            listed = json.loads(
                _get(
                    client,
                    "tasks",
                    query=f"status_class={status_class}&limit=100&order=updated_desc",
                ).content
            )
            assert summary["total_filtered"] == listed["total"], status_class
            assert sum(summary["counts_by_status"].values()) == listed["total"], status_class
            if bucket is not None:
                assert summary["counts_by_status"][bucket] == listed["total"], status_class


def test_status_class_invalid_includes_non_text_storage(app, config):
    """B4: non-TEXT statuses must appear under the Unknown/Invalid chip.

    ``tasks.status`` has TEXT affinity, so an INTEGER *bound value* is stored
    as TEXT (that case is ``unknown`` — see
    ``test_non_status_strings_are_unknown_and_blobs_are_invalid``). To get a
    genuinely non-TEXT storage class the column must be written as BLOB.
    """
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000089','BlobOnly',?,?)",
        (b"whatever", 1_799_999_000),
    )
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000088','BlobKnownWord',?,?)",
        (b"done", 1_799_999_000),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        listed = json.loads(
            _get(client, "tasks", query="status_class=invalid&limit=100&order=updated_desc").content
        )
        summary = json.loads(_get(client, "summary", query="status_class=invalid").content)
    ids = {t["id"] for t in listed["tasks"]}
    # b"done" decodes to a known word but is not TEXT storage: still invalid,
    # and still counted under Done by nothing.
    assert {"t_00000089", "t_00000088"} <= ids
    assert summary["total_filtered"] == listed["total"] >= 2
    assert summary["counts_by_status"]["done"] == 0


def test_q_does_not_fail_when_a_row_has_invalid_utf8(app, config):
    """C4/B6: one bad row must never fail the whole page for ``q``.

    CPython's sqlite3 decodes a TEXT argument strictly before calling a Python
    UDF, so folding the raw column raised "user-defined function raised
    exception" and turned summary+list into 503 for every caller.
    """
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000087',?,?,?)",
        (b"Bad \xff\xfe utf8 title", "todo", 1_799_999_000),
    )
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000086','Good searchable row',?,?)",
        ("todo", 1_799_999_000),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        summary = _get(client, "summary", query="q=searchable")
        listed = _get(client, "tasks", query="q=searchable&limit=100&order=updated_desc")
        touched = _get(client, "tasks", query="q=bad&limit=100&order=updated_desc")
    assert summary.status_code == 200, summary.content
    assert listed.status_code == 200, listed.content
    assert json.loads(summary.content)["total_filtered"] == 1
    # The bad row simply does not match a needle; it is not dropped from the
    # unfiltered page (see test_unsupported_status_values_stay_visible).
    assert touched.status_code == 200
    assert json.loads(touched.content)["total"] == 0


def test_q_does_not_match_blob_or_integer_titles_by_repr(app, config):
    """A BLOB title matches on its real text, never the Python ``b'…'`` repr."""
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000085',?,'todo',?)",
        (b"uniqueblobtoken", 1_799_999_000),
    )
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000084',?,'todo',?)",
        (12345, 1_799_999_000),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        blob_match = json.loads(_get(client, "summary", query="q=uniqueblobtoken").content)
        repr_probe = json.loads(_get(client, "summary", query="q=b%27uniqueblobtoken").content)
        int_probe = json.loads(_get(client, "summary", query="q=12345").content)
    # The BLOB's actual bytes are searchable...
    assert blob_match["total_filtered"] == 1
    # ...but the repr wrapper b' is never in the haystack.
    assert repr_probe["total_filtered"] == 0
    # ``title`` is TEXT-affinity, so this INTEGER is stored (and searched) as
    # the string "12345": an honest text match, not a repr of the number.
    assert int_probe["total_filtered"] == 1
    listed = json.loads(_get(client, "tasks", query="q=12345&limit=100").content)
    assert next(t for t in listed["tasks"] if t["id"] == "t_00000084")["title"] == "12345"


def test_invalid_utf8_id_is_counted_and_kept(app, config):
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES(?,?,?,?)",
        (b"t_\xff\xfe\xfd", "Bad id", "todo", 1_799_999_000),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "summary").content)
        listed = json.loads(_get(client, "tasks", query="limit=100&order=updated_desc").content)
    assert body["invalid_records"] >= 1
    assert listed["total"] == body["total_filtered"]


def test_unknown_status_class_filter_returns_only_unknown(app, config):
    conn = sqlite3.connect(config / "kanban.db")
    conn.execute(
        "INSERT INTO tasks(id, title, status, created_at) VALUES('t_00000095','Odd','weird',?)",
        (1_799_999_000,),
    )
    conn.commit()
    conn.close()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(
            _get(client, "tasks", query="status_class=unknown,invalid&limit=100&order=updated_desc").content
        )
    assert body["total"] >= 1
    assert all(t["status_class"] in {"unknown", "invalid"} for t in body["tasks"])


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def test_detail_returns_bounded_event_and_run_summaries(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "task", query="task_id=t_00000002").content)
    assert body["task"]["id"] == "t_00000002"
    assert body["task"]["running"]["claim_stale"] is True
    assert [e["kind"] for e in body["events"]] == []
    assert len(body["runs"]) == 1
    assert set(body["runs"][0]) == {
        "id", "status", "started_at", "ended_at", "last_heartbeat_at", "outcome"
    }
    assert body["counts"]["comments"] == 0


def test_detail_rejects_malformed_ids(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for bad, expected in [
            ("t_short", 400),
            ("T_00000001", 400),
            ("t_0000000G", 400),
            ("t_00000001 ", 400),
            ("", 400),
        ]:
            resp = _get(client, "task", query=f"task_id={bad}")
            assert resp.status_code == expected, (bad, resp.status_code)


def test_detail_unknown_id_is_404(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = _get(client, "task", query="task_id=t_deadbeef")
    assert resp.status_code == 404
    assert json.loads(resp.content) == {"error": "not_found"}


# ---------------------------------------------------------------------------
# Orders (Athena F + packet §10.11)
# ---------------------------------------------------------------------------


def test_order_matrix_sequences_and_tiebreakers(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        blocked = json.loads(
            _get(client, "tasks", query="status=blocked&order=oldest_status").content
        )
        ready = json.loads(
            _get(client, "tasks", query="status=ready&order=priority_age").content
        )
        done = json.loads(
            _get(client, "tasks", query="status=done&order=completed_desc").content
        )
        running = json.loads(
            _get(client, "tasks", query="status=running&order=heartbeat_attention").content
        )
    assert blocked["order"] == "oldest_status"
    assert ready["tasks"][0]["priority"] >= (ready["tasks"][-1]["priority"] or 0)
    assert done["tasks"][0]["completed_at"] is not None
    assert running["tasks"][0]["id"] == "t_00000002"


@pytest.mark.parametrize(
    "query",
    [
        "order=heartbeat_attention",
        "status=done&order=heartbeat_attention",
        "status=blocked&order=priority_age",
        "status=ready&order=completed_desc",
        "order=nonsense",
    ],
)
def test_incompatible_order_combinations_are_rejected(app, query):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = _get(client, "tasks", query=query)
    assert resp.status_code == 400
    assert json.loads(resp.content) == {"error": "invalid_order"}


# ---------------------------------------------------------------------------
# Cursor (Athena F1 + H)
# ---------------------------------------------------------------------------


def test_cursor_paging_is_exact_and_tampering_is_rejected(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        first = json.loads(_get(client, "tasks", query="limit=3&order=updated_desc").content)
        assert first["next_cursor"]
        second = json.loads(
            _get(client, "tasks", query=f"limit=3&order=updated_desc&cursor={first['next_cursor']}").content
        )
        tampered = first["next_cursor"][:-2] + ("AA" if not first["next_cursor"].endswith("AA") else "BB")
        bad = _get(client, "tasks", query=f"limit=3&order=updated_desc&cursor={tampered}")
    ids_first = [t["id"] for t in first["tasks"]]
    ids_second = [t["id"] for t in second["tasks"]]
    assert not set(ids_first) & set(ids_second)
    assert bad.status_code == 400
    assert json.loads(bad.content) == {"error": "cursor_invalid"}


def test_cursor_bound_to_filters(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        first = json.loads(_get(client, "tasks", query="limit=2&order=updated_desc").content)
        mismatched = _get(
            client,
            "tasks",
            query=f"limit=2&order=updated_desc&assignee=talos&cursor={first['next_cursor']}",
        )
    assert mismatched.status_code == 409
    assert json.loads(mismatched.content) == {"error": "cursor_stale"}


def test_cursor_detects_change_without_any_event_row(app, config):
    """A heartbeat update writes no event; the digest must still catch it.

    B9: the old version ran under ``updated_desc`` — where the heartbeat is not
    part of the digest — and accepted 200, so it could never fail. Under
    ``heartbeat_attention`` the heartbeat *is* the sort key, so the digest must
    move and the stale cursor must be refused with 409. The same statement
    under ``updated_desc`` is the control: there the digest does *not* move,
    which is exactly why the original test was vacuous.
    """
    db = config / "kanban.db"
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        # heartbeat_attention is bound to status=running; the seeded board has
        # exactly one running row, so page with limit=1 over a status filter
        # that has several rows instead and rely on the digest, which covers
        # the heartbeat of *every* row in the filtered sequence.
        first = json.loads(
            _get(client, "tasks", query="status=running&order=heartbeat_attention&limit=1").content
        )
        assert [t["id"] for t in first["tasks"]] == ["t_00000002"]

        # Issue a cursor over the same running filter but with a page size that
        # leaves a second page: add a second running row first.
        conn = sqlite3.connect(db)
        conn.execute(
            "INSERT INTO tasks(id, title, status, created_at, last_heartbeat_at)"
            " VALUES('t_00000080','Second runner','running',?,?)",
            (1_799_999_000, 1_799_999_000),
        )
        conn.commit()
        conn.close()

        paged = json.loads(
            _get(client, "tasks", query="status=running&order=heartbeat_attention&limit=1").content
        )
        cursor = paged["next_cursor"]
        assert cursor, paged

        # Touch a heartbeat without writing any event row.
        conn = sqlite3.connect(db)
        events_before = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = last_heartbeat_at + 900"
            " WHERE id = 't_00000002'"
        )
        conn.commit()
        events_after = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        conn.close()
        assert events_before == events_after, "the fixture must not write an event"

        stale = _get(
            client,
            "tasks",
            query=f"status=running&order=heartbeat_attention&limit=1&cursor={cursor}",
        )

    assert stale.status_code == 409, stale.content
    assert json.loads(stale.content) == {"error": "cursor_stale"}


def test_cursor_from_a_previous_process_epoch_is_stale_not_invalid(app, config):
    """B5: a restart must yield 409 ``cursor_stale``, not 400 ``cursor_invalid``.

    The per-process HMAC key is gone after a restart, so an old cursor can
    never verify. Without an explicit epoch the caller would be told its cursor
    was malformed, and the documented recovery ("start again at page 1") would
    never be reached.
    """
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        first = json.loads(_get(client, "tasks", query="limit=2&order=updated_desc").content)
        cursor = first["next_cursor"]
        assert cursor

        # Simulate a restart: new key, new epoch.
        occ._PROCESS_CURSOR_KEY = None
        occ._PROCESS_CURSOR_EPOCH = None

        resp = _get(
            client, "tasks", query=f"limit=2&order=updated_desc&cursor={cursor}"
        )
        # A cursor minted in the *new* epoch still works.
        fresh = json.loads(_get(client, "tasks", query="limit=2&order=updated_desc").content)
        ok = _get(
            client,
            "tasks",
            query=f"limit=2&order=updated_desc&cursor={fresh['next_cursor']}",
        )
    assert resp.status_code == 409, resp.content
    assert json.loads(resp.content) == {"error": "cursor_stale"}
    assert ok.status_code == 200


# ---------------------------------------------------------------------------
# Read-only enforcement (Athena C2 + H)
# ---------------------------------------------------------------------------


def test_connection_refuses_writes(app, config):
    conn = occ.open_readonly_connection(config / "kanban.db")
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("UPDATE tasks SET title = 'pwned'")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("CREATE TABLE nope (x)")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("ATTACH DATABASE ':memory:' AS other")
    finally:
        conn.close()


def test_reads_do_not_modify_the_database(app, config):
    db = config / "kanban.db"
    before = (db.stat().st_size, db.stat().st_mtime_ns)
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for _ in range(3):
            assert _get(client, "summary").status_code == 200
            assert _get(client, "tasks", query="limit=100&order=updated_desc").status_code == 200
            assert _get(client, "task", query="task_id=t_00000001").status_code == 200
    after = (db.stat().st_size, db.stat().st_mtime_ns)
    assert before == after


def test_reads_do_not_touch_the_wal_or_shm_sidecars(app, config):
    """B9: the fixture board is now in WAL mode, so the WAL half is exercised.

    A read-only connection must never checkpoint, truncate or grow the WAL —
    a read that mutates the operator's sidecar files is a write in disguise.
    """
    db = config / "kanban.db"
    wal = db.with_name(db.name + "-wal")
    shm = db.with_name(db.name + "-shm")
    assert wal.exists(), "fixture must be in WAL mode for this test to mean anything"

    def snapshot():
        return {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in (db, wal, shm)
            if path.exists()
        }

    before = snapshot()
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for _ in range(3):
            assert _get(client, "summary").status_code == 200
            assert _get(client, "tasks", query="limit=100&order=updated_desc").status_code == 200
            assert _get(client, "task", query="task_id=t_00000001").status_code == 200
    assert snapshot() == before


def test_readonly_connection_reports_wal_mode(app, config):
    """The read connection sees a WAL board (asserted via its own handle)."""
    db = config / "kanban.db"
    check = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        check.close()
    # ...and the service's own authorizer refuses to run that PRAGMA, which is
    # the belt to ``query_only``'s braces.
    conn = occ.open_readonly_connection(db)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("PRAGMA journal_mode").fetchone()
    finally:
        conn.close()


def test_board_path_cannot_escape_the_kanban_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kanban.db").write_bytes(b"")
    (home / "kanban").mkdir()
    (home / "kanban" / "boards").mkdir()
    os.symlink(outside, home / "kanban" / "boards" / "evil")
    with pytest.raises(occ.BoardUnavailable):
        occ.resolve_board_db_path("evil", kanban_home=home)


# ---------------------------------------------------------------------------
# Secrecy allowlist (Athena + packet §10.9)
# ---------------------------------------------------------------------------


def test_no_forbidden_field_or_secret_ever_appears_in_a_payload(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        bodies = [
            _get(client, "summary").content,
            _get(client, "tasks", query="limit=100&order=updated_desc").content,
            _get(client, "task", query="task_id=t_00000001").content,
        ]
    blob = b"".join(bodies)
    for forbidden in (
        b"SECRET",
        b"4242",          # worker_pid
        b"/host/",        # workspace / attachment path
        b"claim_lock",
        b"worker_pid",
        b"workspace_path",
        b"payload",
        b"branch_name",
        b"session_id",
    ):
        assert forbidden not in blob, forbidden


def test_task_row_carries_exactly_the_allowlisted_keys(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "tasks", query="limit=1&order=updated_desc").content)
    allowed = {
        "id", "title", "status", "status_class", "assignee", "tenant", "priority",
        "created_at", "updated_at", "status_entered_at", "scheduled_at",
        "completed_at", "parent_id", "child_count", "links", "reason_preview",
        "running",
    }
    assert set(body["tasks"][0]) == allowed


def test_documented_null_fields_are_null(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "tasks", query="limit=100&order=updated_desc").content)
    for task in body["tasks"]:
        assert task["status_entered_at"] is None
        assert task["scheduled_at"] is None
        assert task["reason_preview"] is None
        assert task["links"] == []


def test_error_bodies_are_codes_only(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        body = json.loads(_get(client, "tasks", query="q=" + "a" * 90).content)
    assert list(body) == ["error"]


def test_response_headers_are_no_store_and_nosniff(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = _get(client, "summary")
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers["x-content-type-options"] == "nosniff"


# ---------------------------------------------------------------------------
# Query parsing (Athena G1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    ["unknown_param=1", "limit=5&limit=6", "q=a&q=b", "status=todo&status=ready"],
)
def test_unknown_or_repeated_parameters_are_rejected(app, query):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = _get(client, "tasks", query=query)
    assert resp.status_code == 400


@pytest.mark.parametrize("limit", ["0", "101", "abc", "-1"])
def test_limit_bounds(app, limit):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "tasks", query=f"limit={limit}").status_code == 400


def test_status_and_status_class_are_mutually_exclusive(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "tasks", query="status=todo&status_class=known").status_code == 400


def test_archived_status_filter_is_rejected(app):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "tasks", query="status=archived").status_code == 400


# ---------------------------------------------------------------------------
# Performance (packet §10.10: 2,000 tasks, p95 <= 200ms)
# ---------------------------------------------------------------------------


def test_two_thousand_task_board_meets_p95_budget(app, config):
    conn = sqlite3.connect(config / "kanban.db")
    now = 1_800_100_000
    conn.executemany(
        "INSERT INTO tasks(id, title, status, created_at, assignee, tenant, priority)"
        " VALUES(?,?,?,?,?,?,?)",
        [
            (
                f"t_9{i:07x}"[:10],
                f"Bulk task {i}",
                ["todo", "ready", "blocked", "done", "running"][i % 5],
                now - i * 7,
                ["talos", "athena", "apollo"][i % 3],
                "or7",
                i % 11,
            )
            for i in range(2000)
        ],
    )
    conn.commit()
    conn.close()

    timings = []
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        for _ in range(10):
            start = time.perf_counter()
            resp = _get(client, "tasks", query="limit=50&order=updated_desc")
            timings.append((time.perf_counter() - start) * 1000)
            assert resp.status_code == 200
        summary_times = []
        for _ in range(5):
            start = time.perf_counter()
            assert _get(client, "summary").status_code == 200
            summary_times.append((time.perf_counter() - start) * 1000)

    timings.sort()
    summary_times.sort()
    p95_list = timings[int(len(timings) * 0.95) - 1]
    p95_summary = summary_times[int(len(summary_times) * 0.95) - 1]
    assert p95_list <= 200, f"list p95 {p95_list:.1f}ms"
    assert p95_summary <= 200, f"summary p95 {p95_summary:.1f}ms"
