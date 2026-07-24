"""Behavior contract and real SQLite E2E for owner-scoped session search."""

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
        f"{base}&q=or7needle&offset=-1",
        f"{base}&q=or7needle&offset=abc",
        f"{base}&q=or7needle&offset=1000001",
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
    assert payload["offset"] == 0
    assert payload["has_more"] is False
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
async def test_search_pagination_is_stable_and_nonoverlapping(search_runtime):
    adapter, _, _ = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:

        async def page(offset: int):
            response = await client.get(
                f"/api/sessions/search?q=or7needle&user_id=owner-1&source=slack&limit=2&offset={offset}",
                headers=AUTH,
            )
            assert response.status == 200
            return await response.json()

        first = await page(0)
        first_repeat = await page(0)
        second = await page(2)

    first_ids = [row["message_id"] for row in first["data"]]
    second_ids = [row["message_id"] for row in second["data"]]
    assert first_ids == [row["message_id"] for row in first_repeat["data"]]
    assert first["has_more"] is True
    assert not set(first_ids) & set(second_ids)
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
    }
