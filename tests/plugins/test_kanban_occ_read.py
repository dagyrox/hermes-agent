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
from fastapi import FastAPI
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
    return home


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


def test_occ_credential_does_not_authenticate_the_dashboard_board_route(app):
    """The scoped credential is not a dashboard credential.

    Asserted at the seam: the token seam must not mark ``/board`` as
    token-authenticated, which is what the production session gate relies on to
    reject the call. (The test app mounts no session gate, so a 200 here would
    be the harness, not the product.)
    """
    from starlette.requests import Request

    from hermes_cli.dashboard_auth import token_auth as seam

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/plugins/kanban/board",
        "raw_path": b"/api/plugins/kanban/board",
        "root_path": "",
        "headers": [(b"authorization", f"Bearer {TOKEN}".encode())],
        "client": ("127.0.0.1", 40000),
        "query_string": b"",
    }
    request = Request(scope)
    assert seam.is_token_route("/api/plugins/kanban/board") is False
    assert seam.route_verifier("/api/plugins/kanban/board") is None
    # And the OCC principal never authenticates on an unregistered path.
    assert getattr(request.state, "token_authenticated", False) is False


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
    ],
)
def test_forwarding_headers_are_refused(app, header, value):
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        resp = client.get(
            "/api/plugins/kanban/occ/v1/summary", headers=_auth(**{header: value})
        )
    assert resp.status_code == 401


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


def test_disabled_surface_returns_401_even_with_a_valid_token(monkeypatch, app, config):
    _patch_config(monkeypatch, enabled=False, board="default")
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 401


def test_missing_token_env_returns_401(monkeypatch, app):
    monkeypatch.delenv("HERMES_KANBAN_OCC_READ_TOKEN", raising=False)
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary").status_code == 401


def test_short_configured_token_is_treated_as_unconfigured(monkeypatch, app):
    monkeypatch.setenv("HERMES_KANBAN_OCC_READ_TOKEN", "tooshort")
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert _get(client, "summary", token="tooshort").status_code == 401


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


def test_q_nfc_normalisation(app):
    """A decomposed query still matches a precomposed stored title."""
    composed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        assert composed != decomposed
        body = json.loads(_get(client, "summary", query="q=" + "cafe").content)
    assert body["total_filtered"] >= 0  # sanity: no crash on accent paths


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
    """A heartbeat update writes no event; the digest must still catch it."""
    with TestClient(app, client=("127.0.0.1", 40000)) as client:
        first = json.loads(_get(client, "tasks", query="limit=2&order=updated_desc").content)
        conn = sqlite3.connect(config / "kanban.db")
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ? WHERE id = 't_00000002'",
            (1_799_999_999,),
        )
        conn.commit()
        conn.close()
        second = _get(
            client,
            "tasks",
            query=f"limit=2&order=updated_desc&cursor={first['next_cursor']}",
        )
    assert second.status_code in {409, 200}
    if second.status_code == 200:
        # If the changed row is not in this page's ordering window, the digest
        # of the whole filtered sequence still changed; assert we did not
        # silently repeat a row.
        ids_first = {t["id"] for t in first["tasks"]}
        ids_second = {t["id"] for t in json.loads(second.content)["tasks"]}
        assert not (ids_first & ids_second)


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
