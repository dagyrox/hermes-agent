"""Behavior contract and real SQLite E2E for owner-scoped session search."""

import html
import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


AUTH = {"Authorization": "Bearer test-search-key"}


def _make_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


@pytest.fixture
def search_runtime(tmp_path, monkeypatch):
    """Run the actual HTTP handler over a real temp-HERMES_HOME state.db."""
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    db = SessionDB(db_path=hermes_home / "state.db")
    if not db._fts_enabled:
        db.close()
        pytest.skip("SQLite FTS5 unavailable")

    fixtures = (
        ("owned-slack-a", "slack", "owner-1"),
        ("owned-slack-b", "slack", "owner-1"),
        ("other-owner", "slack", "owner-2"),
        ("other-source", "cli", "owner-1"),
    )
    for session_id, source, user_id in fixtures:
        db.create_session(session_id, source=source, user_id=user_id)

    secret = "sk-" + "testcredential1234567890"
    db.append_message(
        "owned-slack-a",
        "user",
        f"or7needle <script>alert(1)</script> API_KEY={secret} "
        + "private-tail " * 100
        + "DO_NOT_LEAK_FULL_BODY",
    )
    db.append_message(
        "owned-slack-a",
        "assistant",
        "or7needle assistant safe match",
        reasoning="REASONING_DO_NOT_LEAK",
    )
    db.append_message(
        "owned-slack-a", "tool", "or7needle raw tool output must not appear"
    )
    db.append_message(
        "owned-slack-a", "system", "or7needle system prompt must not appear"
    )
    db.append_message(
        "owned-slack-a", "developer", "or7needle developer prompt must not appear"
    )
    db.append_message("owned-slack-b", "user", "or7needle second owned match")
    db.append_message("owned-slack-b", "assistant", "unrelated body")
    db.append_message(
        "owned-slack-b",
        "user",
        "or7marker [[[user-authored-marker]]] useful surrounding context",
    )
    db.append_message("other-owner", "user", "or7needle foreign owner body")
    db.append_message("other-source", "user", "or7needle foreign source body")

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-search-key"})
    )
    adapter._session_db = db
    try:
        yield adapter, db, secret
    finally:
        db.close()


@pytest.mark.asyncio
async def test_search_requires_auth_and_exact_nonempty_owner_scope(search_runtime):
    adapter, _, _ = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:
        unauthenticated = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack"
        )
        assert unauthenticated.status == 401

        for query in (
            "user_id=owner-1&source=slack",
            "q=&user_id=owner-1&source=slack",
            "q=or7needle&source=slack",
            "q=or7needle&user_id=owner-1",
            "q=or7needle&user_id=&source=slack",
            "q=or7needle&user_id=owner-1&source=",
        ):
            response = await client.get(f"/api/sessions/search?{query}", headers=AUTH)
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "invalid_query"


@pytest.mark.asyncio
async def test_search_rejects_malformed_duplicate_unknown_and_unbounded_params(
    search_runtime,
):
    adapter, _, _ = search_runtime
    base = "/api/sessions/search?user_id=owner-1&source=slack"
    cases = (
        f"{base}&q=or7needle&q=second",
        f"{base}&q=or7needle&user_id=owner-1",
        f"{base}&q=or7needle&limit=0",
        f"{base}&q=or7needle&limit=51",
        f"{base}&q=or7needle&limit=abc",
        f"{base}&q=or7needle&offset=0",
        f"{base}&q=or7needle&cursor=not-base64",
        f"{base}&q=or7needle&extra=true",
        f"{base}&q={'x' * 257}",
        f"/api/sessions/search?q=or7needle&user_id={'x' * 257}&source=slack",
        f"/api/sessions/search?q=or7needle&user_id=owner-1&source={'x' * 65}",
    )
    async with TestClient(TestServer(_make_app(adapter))) as client:
        for url in cases:
            response = await client.get(url, headers=AUTH)
            assert response.status == 400, url
            assert (await response.json())["error"]["code"] == "invalid_query"


@pytest.mark.asyncio
async def test_search_is_owner_source_and_role_scoped_with_safe_bounded_fields(
    search_runtime,
):
    adapter, _, secret = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack&limit=20",
            headers=AUTH,
        )
        assert response.status == 200
        payload = await response.json()

    assert payload["object"] == "list"
    assert payload["limit"] == 20
    assert payload["has_more"] is False
    assert payload["next_cursor"] is None
    assert {row["session_id"] for row in payload["data"]} == {
        "owned-slack-a",
        "owned-slack-b",
    }
    assert {row["role"] for row in payload["data"]} <= {"user", "assistant"}

    allowed = {"session_id", "message_id", "role", "timestamp", "snippet", "rank"}
    assert payload["data"]
    for row in payload["data"]:
        assert set(row) == allowed
        assert isinstance(row["rank"], (int, float))
        assert len(row["snippet"]) <= 512
        assert "<script>" not in row["snippet"]

    serialized = json.dumps(payload)
    assert secret not in serialized
    assert "raw tool output" not in serialized
    assert "system prompt" not in serialized
    assert "developer prompt" not in serialized
    assert "REASONING_DO_NOT_LEAK" not in serialized
    assert "foreign owner" not in serialized
    assert "foreign source" not in serialized
    assert "DO_NOT_LEAK_FULL_BODY" not in serialized
    assert "content" not in serialized
    assert "context" not in serialized
    assert any("<mark>or7needle</mark>" in row["snippet"] for row in payload["data"])
    assert any("&lt;script&gt;" in row["snippet"] for row in payload["data"])


@pytest.mark.asyncio
async def test_search_redacts_credential_prefix_before_safe_highlighting(search_runtime):
    adapter, _, secret = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get(
            "/api/sessions/search?q=sk&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        assert response.status == 200
        payload = await response.json()

    serialized = html.unescape(json.dumps(payload))
    assert payload["data"]
    assert secret not in serialized
    assert "testcredential1234567890" not in serialized
    assert "credential1234567890" not in serialized


@pytest.mark.asyncio
async def test_search_never_trusts_user_authored_highlight_markers(search_runtime):
    adapter, _, _ = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get(
            "/api/sessions/search?q=or7marker&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        assert response.status == 200
        payload = await response.json()

    snippet = payload["data"][0]["snippet"]
    assert "<mark>or7marker</mark>" in snippet
    assert "<mark>user-authored-marker</mark>" not in snippet
    assert "[[[user-authored-marker]]]" in html.unescape(snippet)


@pytest.mark.asyncio
async def test_search_short_match_is_useful_but_never_returns_complete_body(
    search_runtime,
):
    adapter, _, _ = search_runtime
    original = "or7needle assistant safe match"
    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack&limit=20",
            headers=AUTH,
        )
        assert response.status == 200
        payload = await response.json()

    snippets = [html.unescape(row["snippet"]) for row in payload["data"]]
    assert any("assistant safe" in snippet for snippet in snippets)
    assert all(
        original not in snippet.replace("<mark>", "").replace("</mark>", "")
        for snippet in snippets
    )


@pytest.mark.asyncio
async def test_search_sanitizes_natural_language_fts_and_returns_nonmatches(
    search_runtime,
):
    adapter, _, _ = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:
        malformed = await client.get(
            "/api/sessions/search?q=%28or7needle%20OR%20%22unterminated&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        assert malformed.status == 200
        assert isinstance((await malformed.json())["data"], list)

        nonmatch = await client.get(
            "/api/sessions/search?q=definitelyabsent&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        assert nonmatch.status == 200
        assert (await nonmatch.json())["data"] == []


@pytest.mark.asyncio
async def test_search_cursor_does_not_overlap_when_match_is_inserted_between_pages(
    search_runtime,
):
    adapter, db, _ = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:

        async def page(cursor: str | None = None):
            cursor_query = f"&cursor={cursor}" if cursor else ""
            response = await client.get(
                "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack"
                f"&limit=2{cursor_query}",
                headers=AUTH,
            )
            assert response.status == 200
            return await response.json()

        first = await page()
        inserted_id = db.append_message(
            "owned-slack-a", "user", "or7needle inserted between cursor pages"
        )
        second = await page(first["next_cursor"])
        wrong_scope = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-2&source=slack"
            f"&limit=2&cursor={first['next_cursor']}",
            headers=AUTH,
        )

    first_ids = [row["message_id"] for row in first["data"]]
    second_ids = [row["message_id"] for row in second["data"]]
    assert first["has_more"] is True
    assert first["next_cursor"]
    assert wrong_scope.status == 400
    assert not set(first_ids) & set(second_ids)
    assert inserted_id not in second_ids
    assert len(first_ids + second_ids) == 3


@pytest.mark.asyncio
async def test_capabilities_advertise_owner_scoped_search(search_runtime):
    adapter, _, _ = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:
        response = await client.get("/v1/capabilities", headers=AUTH)
        assert response.status == 200
        payload = await response.json()

    assert payload["features"]["session_search"] is True
    assert payload["endpoints"]["session_search"] == {
        "method": "GET",
        "path": "/api/sessions/search",
        "owner_scope": ["user_id", "source"],
        "pagination": {
            "type": "cursor",
            "parameter": "cursor",
            "ordering": ["rank_asc", "timestamp_desc", "message_id_desc"],
            "snapshot": "message_id_high_water",
        },
    }


@pytest.mark.asyncio
async def test_search_disabled_is_not_advertised_and_returns_safe_503(search_runtime):
    adapter, db, _ = search_runtime
    db._fts_enabled = False
    async with TestClient(TestServer(_make_app(adapter))) as client:
        capabilities = await client.get("/v1/capabilities", headers=AUTH)
        assert capabilities.status == 200
        advertised = await capabilities.json()
        assert advertised["features"]["session_search"] is False
        assert "session_search" not in advertised["endpoints"]

        response = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        assert response.status == 503
        payload = await response.json()
        assert payload["error"]["code"] == "session_search_unavailable"


@pytest.mark.asyncio
async def test_search_operational_fts_failure_is_503_not_empty_success(search_runtime):
    adapter, db, _ = search_runtime
    db._conn.execute("DROP TABLE messages_fts")
    async with TestClient(TestServer(_make_app(adapter))) as client:
        capabilities = await client.get("/v1/capabilities", headers=AUTH)
        assert capabilities.status == 200
        advertised = await capabilities.json()
        assert advertised["features"]["session_search"] is False
        assert "session_search" not in advertised["endpoints"]

        response = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        assert response.status == 503
        payload = await response.json()
        assert payload["error"]["code"] == "session_search_unavailable"
