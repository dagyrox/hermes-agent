"""Behavior contract and real SQLite E2E for owner-scoped session search."""

import base64
import hashlib
import html
import hmac
import json
import time

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

    allowed = {"session_id", "message_id", "role", "timestamp", "snippet"}
    assert payload["data"]
    for row in payload["data"]:
        assert set(row) == allowed
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
    assert any("or7needle" in row["snippet"] for row in payload["data"])
    assert any("&lt;script&gt;" in row["snippet"] for row in payload["data"])


@pytest.mark.asyncio
async def test_search_redacts_complete_credential_without_head_or_tail_fragments(search_runtime):
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
    assert "sk-" not in serialized
    assert "sk-test" not in serialized
    assert "7890" not in serialized


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
    assert "or7marker" in snippet
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
async def test_search_snippet_is_query_independent_and_blocks_reconstruction(search_runtime):
    adapter, db, _ = search_runtime
    message_id = db.append_message(
        "owned-slack-a",
        "user",
        "firstprobe " + "alpha-private " * 30 + "secondprobe " + "omega-private " * 30,
    )

    async with TestClient(TestServer(_make_app(adapter))) as client:
        first = await client.get(
            "/api/sessions/search?q=firstprobe&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        second = await client.get(
            "/api/sessions/search?q=secondprobe&user_id=owner-1&source=slack",
            headers=AUTH,
        )
        first_payload = await first.json()
        second_payload = await second.json()

    assert first.status == second.status == 200
    first_row = next(row for row in first_payload["data"] if row["message_id"] == message_id)
    second_row = next(row for row in second_payload["data"] if row["message_id"] == message_id)
    assert first_row["snippet"] == second_row["snippet"]
    assert "secondprobe" not in html.unescape(first_row["snippet"])


@pytest.mark.parametrize(
    "content, forbidden",
    (
        (
            'credprobe MiXeD_ApI_KeY = "AbCdEfGhIjKlMnOpQrStUvWxYz012345" boundary',
            ("AbCdEf", "012345", "QrStUv"),
        ),
        (
            "credprobe Authorization: Bearer abcDEF0123456789xyzXYZ9876543210 boundary",
            ("abcDEF", "6543210", "xyzXYZ"),
        ),
        (
            "credprobe " + "x" * 150 + " api-token=tok_HEAD_middle_SECRET_tail999 boundary",
            ("tok_HEAD", "tail999", "SECRET"),
        ),
        (
            "credprobe eyJabcdefghijklmno.abcdefghijklmnop.signaturetail boundary",
            ("eyJabc", "signaturetail", "hijklmno"),
        ),
    ),
)
def test_search_snippet_redacts_assignments_before_boundary_truncation(content, forbidden):
    snippet = html.unescape(APIServerAdapter._bounded_search_snippet(content))
    assert "[REDACTED]" in snippet or all(value not in snippet for value in forbidden)
    for value in forbidden:
        assert value not in snippet


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
async def test_search_cursor_is_authenticated_scope_bound_and_expires(
    search_runtime,
    monkeypatch,
):
    adapter, db, _ = search_runtime
    async with TestClient(TestServer(_make_app(adapter))) as client:
        for index in range(4):
            db.append_message(
                "owned-slack-a", "user", f"or7needle cursor fixture {index}"
            )
        first_response = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack&limit=2",
            headers=AUTH,
        )
        first = await first_response.json()
        cursor = first["next_cursor"]
        assert cursor

        replacement = "A" if cursor[-1] != "A" else "B"
        tampered = cursor[:-1] + replacement
        encoded_payload, _signature = cursor.split(".", 1)
        decoded_payload = json.loads(
            base64.urlsafe_b64decode(
                encoded_payload + "=" * (-len(encoded_payload) % 4)
            )
        )
        decoded_payload["v"] = 999
        unsupported_payload = base64.urlsafe_b64encode(
            json.dumps(decoded_payload, separators=(",", ":")).encode()
        ).rstrip(b"=")
        unsupported_signature = base64.urlsafe_b64encode(
            hmac.new(
                adapter._search_cursor_signing_key(),
                unsupported_payload,
                hashlib.sha256,
            ).digest()
        ).rstrip(b"=")
        unsupported = (
            unsupported_payload.decode() + "." + unsupported_signature.decode()
        )
        cases = (
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack"
            f"&limit=2&cursor={tampered}",
            "/api/sessions/search?q=different&user_id=owner-1&source=slack"
            f"&limit=2&cursor={cursor}",
            "/api/sessions/search?q=or7needle&user_id=owner-2&source=slack"
            f"&limit=2&cursor={cursor}",
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=cli"
            f"&limit=2&cursor={cursor}",
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack"
            "&limit=2&cursor=malformed.cursor",
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack"
            f"&limit=2&cursor={unsupported}",
        )
        for url in cases:
            response = await client.get(url, headers=AUTH)
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "invalid_query"

        monkeypatch.setattr(time, "time", lambda: 4_000_000_000)
        expired = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack"
            f"&limit=2&cursor={cursor}",
            headers=AUTH,
        )
        expired_payload = await expired.json()

    assert expired.status == 400
    assert expired_payload["error"]["code"] == "invalid_query"


@pytest.mark.asyncio
async def test_search_stable_multi_page_snapshot_has_no_overlap_or_omission_on_insert_and_rank_update(
    search_runtime,
):
    adapter, db, _ = search_runtime
    initial_ids = []
    for index in range(9):
        initial_ids.append(
            db.append_message(
                "owned-slack-a", "user", f"pageprobe stable fixture {index}"
            )
        )

    async with TestClient(TestServer(_make_app(adapter))) as client:
        async def page(cursor: str | None = None):
            cursor_query = f"&cursor={cursor}" if cursor else ""
            response = await client.get(
                "/api/sessions/search?q=pageprobe&user_id=owner-1&source=slack"
                f"&limit=3{cursor_query}",
                headers=AUTH,
            )
            assert response.status == 200
            return await response.json()

        first = await page()
        inserted_id = db.append_message(
            "owned-slack-a", "user", "pageprobe inserted after snapshot"
        )
        # Change BM25 term frequency for a row already returned. Immutable
        # timestamp/id keyset ordering must make this irrelevant to pagination.
        updated_id = first["data"][0]["message_id"]
        db._conn.execute(
            "UPDATE messages SET content = ? WHERE id = ?",
            ("pageprobe " * 100, updated_id),
        )
        db._conn.commit()

        pages = [first]
        while pages[-1]["has_more"]:
            pages.append(await page(pages[-1]["next_cursor"]))

    returned = [row["message_id"] for payload in pages for row in payload["data"]]
    assert len(returned) == len(set(returned))
    assert set(returned) == set(initial_ids)
    assert inserted_id not in returned

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
            "ordering": ["timestamp_desc", "message_id_desc"],
            "snapshot": "message_id_high_water",
            "cursor": "hmac_authenticated_v2",
        },
    }


@pytest.mark.asyncio
async def test_search_without_cursor_signing_credential_fails_closed(search_runtime):
    adapter, _, _ = search_runtime
    adapter._api_key = ""
    async with TestClient(TestServer(_make_app(adapter))) as client:
        capabilities = await client.get("/v1/capabilities")
        response = await client.get(
            "/api/sessions/search?q=or7needle&user_id=owner-1&source=slack"
        )
        advertised = await capabilities.json()
        failure = await response.json()

    assert advertised["features"]["session_search"] is False
    assert "session_search" not in advertised["endpoints"]
    assert response.status == 503
    assert failure["error"]["code"] == "session_search_unavailable"


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
        serialized = json.dumps(payload).lower()
        assert "messages_fts" not in serialized
        assert "sqlite" not in serialized


@pytest.mark.asyncio
async def test_search_operational_fts_failure_is_503_not_empty_success(
    search_runtime, monkeypatch
):
    adapter, db, _ = search_runtime
    db._conn.execute("DROP TABLE messages_fts")
    # A metadata-only readiness check would still claim success here. The
    # capability probe must execute a harmless scoped MATCH and fail closed.
    monkeypatch.setattr(db, "_fts_table_exists", lambda _name: True)
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
        serialized = json.dumps(payload).lower()
        assert "messages_fts" not in serialized
        assert "sqlite" not in serialized
