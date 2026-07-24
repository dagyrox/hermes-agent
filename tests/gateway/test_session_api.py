"""Focused tests for API server session-control endpoints."""

import json
import sqlite3
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


def _append_completed_turn(
    session_db,
    session_id,
    user="question",
    assistant="answer",
    *,
    finish_reason=None,
):
    session_db.append_message(session_id, "user", user)
    return session_db.append_message(
        session_id, "assistant", assistant, finish_reason=finish_reason
    )


def _tool_call(call_id, name="test_tool"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def _responses_tool_call(call_id, name="test_tool"):
    return {
        "call_id": call_id,
        "type": "function_call",
        "name": name,
        "arguments": {},
    }


def _anthropic_tool_call(call_id, name="test_tool"):
    return {
        "id": call_id,
        "type": "tool_use",
        "name": name,
        "input": {},
    }


def _assert_valid_for_next_user_turn(messages):
    """Strict provider-neutral role/tool shape accepted before a new user turn."""
    expect = "user"
    pending_tool_calls = set()
    for message in messages:
        role = message["role"]
        if expect == "user":
            assert role == "user"
            expect = "assistant"
        elif expect == "assistant":
            assert role == "assistant"
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                pending_tool_calls = {call["id"] for call in tool_calls}
                assert len(pending_tool_calls) == len(tool_calls)
                expect = "tool"
            else:
                expect = "user"
        else:
            assert role == "tool"
            assert message.get("tool_call_id") in pending_tool_calls
            pending_tool_calls.remove(message["tool_call_id"])
            if not pending_tool_calls:
                expect = "assistant"
    assert expect == "user"
    assert not pending_tool_calls


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


@pytest.mark.asyncio
async def test_capabilities_advertises_session_control_surface(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    assert data["object"] == "hermes.api_server.capabilities"
    assert data["platform"] == "hermes-agent"
    assert data["revision"] == "occ.conversation_gateway.h1-h2-h3.v1"
    assert len(data["build_id"]) == 40
    assert int(data["build_id"], 16) >= 0
    features = data["features"]
    assert features["session_resources"] is True
    assert features["session_chat"] is True
    assert features["session_chat_streaming"] is True
    assert features["session_fork"] is True
    assert features["session_fork_anchored_preserve_source"] is True
    assert features["session_fork_anchor_field"] == "anchor_message_id"
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }


@pytest.mark.asyncio
async def test_run_agent_binds_api_session_context_for_tool_env(adapter, monkeypatch):
    """API-server request sessions should reach tools and terminal subprocess env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-session")
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.session_context import get_session_env
            from tools.environments.local import _make_run_env

            observed["task_id"] = task_id
            observed["context_session_id"] = get_session_env("HERMES_SESSION_ID")
            observed["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            observed["child_session_id"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
    )

    assert result["session_id"] == "request-session"
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
    }


@pytest.mark.asyncio
async def test_session_crud_and_message_history(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        create_resp = await cli.post("/api/sessions", json={"title": "Mobile chat", "model": "test-model"})
        assert create_resp.status == 201
        created = await create_resp.json()
        session_id = created["session"]["id"]
        assert created["object"] == "hermes.session"
        assert created["session"]["title"] == "Mobile chat"

        session_db.append_message(session_id, "user", "hello from phone")
        session_db.append_message(session_id, "assistant", "hello from hermes")

        list_resp = await cli.get("/api/sessions?limit=10&offset=0")
        assert list_resp.status == 200
        listed = await list_resp.json()
        assert listed["object"] == "list"
        assert [s["id"] for s in listed["data"]] == [session_id]
        assert listed["data"][0]["message_count"] == 2

        get_resp = await cli.get(f"/api/sessions/{session_id}")
        assert get_resp.status == 200
        got = await get_resp.json()
        assert got["session"]["id"] == session_id
        assert got["session"]["message_count"] == 2

        messages_resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()
        assert messages["object"] == "list"
        assert [m["role"] for m in messages["data"]] == ["user", "assistant"]
        assert messages["data"][0]["content"] == "hello from phone"

        patch_resp = await cli.patch(f"/api/sessions/{session_id}", json={"title": "Renamed"})
        assert patch_resp.status == 200
        patched = await patch_resp.json()
        assert patched["session"]["title"] == "Renamed"

        delete_resp = await cli.delete(f"/api/sessions/{session_id}")
        assert delete_resp.status == 200
        deleted = await delete_resp.json()
        assert deleted == {"object": "hermes.session.deleted", "id": session_id, "deleted": True}
        assert session_db.get_session(session_id) is None


@pytest.mark.asyncio
async def test_session_messages_follow_compression_tip(adapter, session_db):
    source_id = session_db.create_session("source-session", "api_server")
    session_db.append_message(source_id, "user", "before compression")
    session_db.end_session(source_id, "compression")
    session_db.create_session("tip-session", "api_server", parent_session_id=source_id)
    session_db.replace_messages(source_id, [])
    session_db.append_message("tip-session", "user", "after compression")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        messages_resp = await cli.get(f"/api/sessions/{source_id}/messages")
        assert messages_resp.status == 200
        messages = await messages_resp.json()

    assert messages["object"] == "list"
    assert messages["session_id"] == "tip-session"
    assert [m["content"] for m in messages["data"]] == ["after compression"]


@pytest.mark.asyncio
async def test_session_fork_is_authenticated_anchored_and_source_exactly_immutable(
    auth_adapter, session_db
):
    source_id = session_db.create_session(
        "source-session",
        "slack",
        model="test-model",
        model_config={"provider": "openrouter", "route": "safe-route"},
        system_prompt="stable prompt",
        user_id="owner-42",
        cwd="/tmp/project",
    )
    session_db._conn.execute(
        """UPDATE sessions
           SET git_branch = ?, git_repo_root = ?, billing_provider = ?,
               billing_base_url = ?, billing_mode = ?, profile_name = ?,
               pricing_version = ?
           WHERE id = ?""",
        (
            "talos/issue-159",
            "/tmp/project",
            "openrouter",
            "https://provider.invalid/v1",
            "api",
            "talos",
            "2026-07",
            source_id,
        ),
    )
    session_db.record_gateway_session_peer(
        source_id,
        source="slack",
        user_id="owner-42",
        session_key="slack:owner-42:C1:T1",
        chat_id="C1",
        chat_type="channel",
        thread_id="T1",
        display_name="Owner thread",
        origin_json='{"platform":"slack","chat_id":"C1"}',
    )
    session_db.set_session_title(source_id, "Original")
    session_db.append_message(source_id, "user", "first path")
    session_db.append_message(source_id, "assistant", "answer")
    session_db.append_message(source_id, "user", "later turn")
    anchor_id = session_db.get_messages(source_id)[1]["id"]

    source_row_before = session_db.get_session(source_id)
    source_messages_before = session_db.get_messages(source_id, include_inactive=True)
    active_head_before = session_db.find_latest_gateway_session_for_peer(
        source="slack",
        user_id="owner-42",
        session_key="slack:owner-42:C1:T1",
        chat_id="C1",
        chat_type="channel",
        thread_id="T1",
    )

    app = _create_session_app(auth_adapter)
    request_body = {
        "id": "anchored-child",
        "title": "Alternative",
        "anchor_message_id": anchor_id,
        "preserve_source": True,
        "idempotency_key": "fork-request-1",
    }
    async with TestClient(TestServer(app)) as cli:
        unauthenticated = await cli.post(
            f"/api/sessions/{source_id}/fork", json=request_body
        )
        assert unauthenticated.status == 401

        resp = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json=request_body,
            headers={"Authorization": "Bearer sk-test"},
        )
        assert resp.status == 201, await resp.text()
        payload = await resp.json()

        get_resp = await cli.get(
            "/api/sessions/anchored-child",
            headers={"Authorization": "Bearer sk-test"},
        )
        assert get_resp.status == 200
        fetched = await get_resp.json()

    fork = payload["session"]
    assert payload["object"] == "hermes.session"
    assert payload["idempotent_replay"] is False
    assert fork["id"] == "anchored-child"
    assert fork["parent_session_id"] == source_id
    assert fork["branch_point_message_id"] == anchor_id
    assert fork["title"] == "Alternative"
    assert fork["source"] == "slack"
    assert fork["user_id"] == "owner-42"
    assert fork["model"] == "test-model"
    assert fetched["session"]["branch_point_message_id"] == anchor_id
    assert "model_config" not in fork
    assert "system_prompt" not in fork
    assert "idempotency_key" not in json.dumps(payload)

    child_row = session_db.get_session(fork["id"])
    child_config = json.loads(child_row["model_config"])
    assert child_config["provider"] == "openrouter"
    assert child_config["route"] == "safe-route"
    assert child_config["_branched_from"] == source_id
    assert child_config["_branch_point_message_id"] == anchor_id
    assert child_row["system_prompt"] == "stable prompt"
    assert child_row["cwd"] == "/tmp/project"
    assert child_row["git_branch"] == "talos/issue-159"
    assert child_row["git_repo_root"] == "/tmp/project"
    assert child_row["billing_provider"] == "openrouter"
    assert child_row["billing_base_url"] == "https://provider.invalid/v1"
    assert child_row["billing_mode"] == "api"
    assert child_row["profile_name"] == "talos"
    assert child_row["pricing_version"] == "2026-07"
    assert child_row["session_key"] is None
    assert child_row["chat_id"] is None
    assert child_row["thread_id"] is None
    assert child_row["display_name"] is None
    assert child_row["origin_json"] is None
    assert child_row["ended_at"] is None
    assert child_row["end_reason"] is None
    assert [m["content"] for m in session_db.get_messages(fork["id"])] == [
        "first path",
        "answer",
    ]

    assert session_db.get_session(source_id) == source_row_before
    assert (
        session_db.get_messages(source_id, include_inactive=True)
        == source_messages_before
    )
    active_head_after = session_db.find_latest_gateway_session_for_peer(
        source="slack",
        user_id="owner-42",
        session_key="slack:owner-42:C1:T1",
        chat_id="C1",
        chat_type="channel",
        thread_id="T1",
    )
    assert active_head_before["id"] == source_id
    assert active_head_after["id"] == source_id


@pytest.mark.asyncio
async def test_session_fork_slices_only_active_rows_through_exact_anchor(
    auth_adapter, session_db
):
    source_id = session_db.create_session("active-only-source", "api_server")
    inactive_user = session_db.append_message(source_id, "user", "abandoned")
    inactive_assistant = session_db.append_message(
        source_id, "assistant", "abandoned answer", finish_reason="stop"
    )
    session_db._conn.execute(
        "UPDATE messages SET active = 0 WHERE id IN (?, ?)",
        (inactive_user, inactive_assistant),
    )
    session_db.append_message(source_id, "user", "canonical")
    anchor_id = session_db.append_message(
        source_id, "assistant", "canonical answer", finish_reason="stop"
    )
    source_before = session_db.get_messages(source_id, include_inactive=True)

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "active-only-child",
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": "active-only-key",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        response_text = await response.text()

    assert response.status == 201, response_text
    child = session_db.get_messages("active-only-child", include_inactive=True)
    assert [message["content"] for message in child] == [
        "canonical",
        "canonical answer",
    ]
    assert all(message["active"] for message in child)
    assert session_db.get_messages(source_id, include_inactive=True) == source_before


@pytest.mark.asyncio
async def test_session_fork_idempotent_replay_and_conflicting_reuse(
    auth_adapter, session_db
):
    source_id = session_db.create_session("source-session", "api_server")
    anchor_id = _append_completed_turn(session_db, source_id)
    app = _create_session_app(auth_adapter)
    body = {
        "anchor_message_id": anchor_id,
        "preserve_source": True,
        "idempotency_key": "stable-key",
    }
    headers = {"Authorization": "Bearer sk-test"}

    async with TestClient(TestServer(app)) as cli:
        created = await cli.post(
            f"/api/sessions/{source_id}/fork", json=body, headers=headers
        )
        assert created.status == 201
        first = await created.json()

        replay = await cli.post(
            f"/api/sessions/{source_id}/fork", json=body, headers=headers
        )
        assert replay.status == 200
        second = await replay.json()

        conflict = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={**body, "title": "different request"},
            headers=headers,
        )
        assert conflict.status == 409
        conflict_payload = await conflict.json()

    assert second["idempotent_replay"] is True
    assert second["session"]["id"] == first["session"]["id"]
    assert conflict_payload["error"]["code"] == "idempotency_conflict"
    children = session_db._conn.execute(
        "SELECT id FROM sessions WHERE parent_session_id = ?", (source_id,)
    ).fetchall()
    assert [row["id"] for row in children] == [first["session"]["id"]]


@pytest.mark.asyncio
async def test_session_fork_rejects_foreign_anchor_without_creating_child(
    auth_adapter, session_db
):
    source_id = session_db.create_session("source-session", "api_server")
    session_db.append_message(source_id, "user", "source message")
    other_id = session_db.create_session("other-session", "api_server")
    session_db.append_message(other_id, "user", "foreign message")
    foreign_anchor = session_db.get_messages(other_id)[0]["id"]
    source_before = session_db.get_session(source_id)
    messages_before = session_db.get_messages(source_id, include_inactive=True)

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "must-not-exist",
                "from_message_id": foreign_anchor,
                "preserve_source": True,
                "idempotency_key": "invalid-anchor-key",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert resp.status == 400
        payload = await resp.json()

    assert payload["error"]["code"] == "invalid_anchor"
    assert session_db.get_session("must-not-exist") is None
    assert session_db.get_session(source_id) == source_before
    assert session_db.get_messages(source_id, include_inactive=True) == messages_before
    assert session_db._conn.execute(
        "SELECT 1 FROM session_fork_requests WHERE idempotency_key = ?",
        ("invalid-anchor-key",),
    ).fetchone() is None


@pytest.mark.asyncio
async def test_session_fork_failure_rolls_back_child_messages_and_idempotency(
    auth_adapter, session_db
):
    source_id = session_db.create_session("source-session", "api_server")
    anchor_id = _append_completed_turn(session_db, source_id)
    source_before = session_db.get_session(source_id)
    messages_before = session_db.get_messages(source_id, include_inactive=True)
    session_db._conn.executescript(
        """
        CREATE TRIGGER fail_anchored_child_copy
        BEFORE INSERT ON messages
        WHEN NEW.session_id = 'rollback-child'
        BEGIN
            SELECT RAISE(ABORT, 'injected child copy failure');
        END;
        """
    )

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "rollback-child",
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": "rollback-key",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        assert resp.status == 500
        payload = await resp.json()

    assert payload["error"]["code"] == "session_fork_failed"
    assert session_db.get_session("rollback-child") is None
    assert session_db._conn.execute(
        "SELECT 1 FROM messages WHERE session_id = 'rollback-child'"
    ).fetchone() is None
    assert session_db._conn.execute(
        "SELECT 1 FROM session_fork_requests WHERE idempotency_key = 'rollback-key'"
    ).fetchone() is None
    assert session_db.get_session(source_id) == source_before
    assert session_db.get_messages(source_id, include_inactive=True) == messages_before


@pytest.mark.asyncio
async def test_session_fork_requires_preserve_source_and_bounded_fields(
    auth_adapter, session_db
):
    source_id = session_db.create_session("source-session", "api_server")
    anchor_id = _append_completed_turn(session_db, source_id)
    headers = {"Authorization": "Bearer sk-test"}
    app = _create_session_app(auth_adapter)

    async with TestClient(TestServer(app)) as cli:
        missing_preserve = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "anchor_message_id": anchor_id,
                "idempotency_key": "missing-preserve",
            },
            headers=headers,
        )
        assert missing_preserve.status == 400
        assert (await missing_preserve.json())["error"]["code"] == "preserve_source_required"

        long_title = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": "long-title",
                "title": "x" * (SessionDB.MAX_TITLE_LENGTH + 1),
            },
            headers=headers,
        )
        assert long_title.status == 400
        assert (await long_title.json())["error"]["code"] == "invalid_title"

        long_key = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": "k" * 256,
            },
            headers=headers,
        )
        assert long_key.status == 400
        assert (await long_key.json())["error"]["code"] == "invalid_idempotency_key"

    assert session_db._conn.execute(
        "SELECT COUNT(*) AS n FROM sessions WHERE parent_session_id = ?", (source_id,)
    ).fetchone()["n"] == 0


@pytest.mark.asyncio
async def test_session_fork_anchor_contract_accepts_canonical_and_compatible_alias(
    auth_adapter, session_db
):
    source_id = session_db.create_session("source-session", "api_server")
    anchor_id = _append_completed_turn(session_db, source_id)
    headers = {"Authorization": "Bearer sk-test"}
    app = _create_session_app(auth_adapter)

    requests = [
        ({"anchor_message_id": anchor_id}, "canonical-child", "canonical-key"),
        ({"from_message_id": anchor_id}, "legacy-child", "legacy-key"),
        (
            {"anchor_message_id": anchor_id, "from_message_id": anchor_id},
            "equal-fields-child",
            "equal-fields-key",
        ),
    ]
    async with TestClient(TestServer(app)) as cli:
        for anchor_fields, child_id, key in requests:
            response = await cli.post(
                f"/api/sessions/{source_id}/fork",
                json={
                    **anchor_fields,
                    "id": child_id,
                    "preserve_source": True,
                    "idempotency_key": key,
                },
                headers=headers,
            )
            assert response.status == 201, await response.text()
            assert (await response.json())["session"]["branch_point_message_id"] == anchor_id

        mismatched = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "anchor_message_id": anchor_id,
                "from_message_id": anchor_id + 1,
                "id": "mismatch-child",
                "preserve_source": True,
                "idempotency_key": "mismatch-key",
            },
            headers=headers,
        )
        missing = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "missing-child",
                "preserve_source": True,
                "idempotency_key": "missing-key",
            },
            headers=headers,
        )
        mismatched_payload = await mismatched.json()
        missing_payload = await missing.json()

    assert mismatched.status == 400
    assert mismatched_payload["error"]["code"] == "invalid_anchor"
    assert missing.status == 400
    assert missing_payload["error"]["code"] == "invalid_anchor"
    assert session_db.get_session("mismatch-child") is None
    assert session_db.get_session("missing-child") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prefix_kind",
    [
        "user-anchor",
        "assistant-tool-request-anchor",
        "partial-multi-tool-results",
        "complete-multi-tool-results-tool-anchor",
        "assistant-native-tool-use-block",
        "system-anchor",
        "developer-anchor",
        "consecutive-users-before-final",
        "orphan-tool-result",
    ],
)
async def test_session_fork_rejects_prefixes_unsafe_for_a_new_user_turn(
    auth_adapter, session_db, prefix_kind
):
    source_id = session_db.create_session("source-session", "api_server")
    if prefix_kind in {"system-anchor", "developer-anchor"}:
        anchor_id = session_db.append_message(source_id, prefix_kind.split("-")[0], "policy")
    else:
        anchor_id = session_db.append_message(source_id, "user", "first")
        if prefix_kind == "assistant-tool-request-anchor":
            anchor_id = session_db.append_message(
                source_id, "assistant", "calling", tool_calls=[_tool_call("call-1")]
            )
        elif prefix_kind in {
            "partial-multi-tool-results",
            "complete-multi-tool-results-tool-anchor",
        }:
            session_db.append_message(
                source_id,
                "assistant",
                "calling",
                tool_calls=[_tool_call("call-1"), _tool_call("call-2")],
            )
            anchor_id = session_db.append_message(
                source_id, "tool", "one", tool_call_id="call-1", tool_name="test_tool"
            )
            if prefix_kind == "complete-multi-tool-results-tool-anchor":
                anchor_id = session_db.append_message(
                    source_id, "tool", "two", tool_call_id="call-2", tool_name="test_tool"
                )
        elif prefix_kind == "assistant-native-tool-use-block":
            anchor_id = session_db.append_message(
                source_id,
                "assistant",
                [{"type": "tool_use", "id": "native-1", "name": "test_tool"}],
            )
        elif prefix_kind == "consecutive-users-before-final":
            session_db.append_message(source_id, "user", "second")
            anchor_id = session_db.append_message(source_id, "assistant", "answer")
        elif prefix_kind == "orphan-tool-result":
            anchor_id = session_db.append_message(
                source_id, "tool", "orphan", tool_call_id="unknown", tool_name="test_tool"
            )
    source_before = session_db.get_session(source_id)
    messages_before = session_db.get_messages(source_id, include_inactive=True)
    app = _create_session_app(auth_adapter)

    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "unsafe-child",
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": f"unsafe-{prefix_kind}",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        response_payload = await response.json()

    assert response.status == 400
    assert response_payload["error"]["code"] == "unsafe_anchor"
    assert session_db.get_session("unsafe-child") is None
    assert session_db.get_session(source_id) == source_before
    assert session_db.get_messages(source_id, include_inactive=True) == messages_before


@pytest.mark.asyncio
async def test_session_fork_accepts_first_completed_response_and_exact_multi_tool_prefix(
    auth_adapter, session_db
):
    source_id = session_db.create_session("source-session", "api_server")
    first_anchor = _append_completed_turn(
        session_db,
        source_id,
        "first",
        "first answer",
        finish_reason="stop",
    )
    session_db.append_message(source_id, "user", "use tools")
    session_db.append_message(
        source_id,
        "assistant",
        "working",
        tool_calls=[_tool_call("call-1", "alpha"), _tool_call("call-2", "beta")],
        finish_reason="tool_calls",
        api_content="working exactly",
    )
    session_db.append_message(
        source_id, "tool", "alpha result", tool_call_id="call-1", tool_name="alpha"
    )
    session_db.append_message(
        source_id, "tool", "beta result", tool_call_id="call-2", tool_name="beta"
    )
    final_anchor = session_db.append_message(
        source_id, "assistant", "complete", finish_reason="stop"
    )
    source_before = session_db.get_session(source_id)
    source_messages_before = session_db.get_messages(source_id, include_inactive=True)
    app = _create_session_app(auth_adapter)
    headers = {"Authorization": "Bearer sk-test"}

    async with TestClient(TestServer(app)) as cli:
        first = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "first-valid-child",
                "anchor_message_id": first_anchor,
                "preserve_source": True,
                "idempotency_key": "first-valid",
            },
            headers=headers,
        )
        assert first.status == 201, await first.text()

        complete = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "complete-tool-child",
                "anchor_message_id": final_anchor,
                "preserve_source": True,
                "idempotency_key": "complete-tool",
            },
            headers=headers,
        )
        assert complete.status == 201, await complete.text()

    child_messages = session_db.get_messages("complete-tool-child", include_inactive=True)
    source_prefix = [m for m in source_messages_before if m["id"] <= final_anchor]
    immutable_columns = [
        "role",
        "content",
        "tool_call_id",
        "tool_calls",
        "tool_name",
        "finish_reason",
        "api_content",
        "reasoning",
        "reasoning_content",
        "active",
    ]
    assert [
        {key: message.get(key) for key in immutable_columns} for message in child_messages
    ] == [
        {key: message.get(key) for key in immutable_columns} for message in source_prefix
    ]
    assert len(child_messages) == len(source_prefix)
    assert session_db.get_session(source_id) == source_before
    assert session_db.get_messages(source_id, include_inactive=True) == source_messages_before

    provider_history = session_db.get_messages_as_conversation("complete-tool-child")
    _assert_valid_for_next_user_turn(provider_history)
    assert [message["role"] for message in provider_history] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]

    mock_run = AsyncMock(
        return_value=(
            {"final_response": "continued", "session_id": "complete-tool-child"},
            {"total_tokens": 1},
        )
    )
    with (
        patch.object(auth_adapter, "_run_agent", mock_run),
        patch("agent.agent_runtime_helpers.repair_message_sequence") as repair_sequence,
    ):
        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                "/api/sessions/complete-tool-child/chat",
                json={"message": "continue"},
                headers=headers,
            )
            assert response.status == 200, await response.text()

    repair_sequence.assert_not_called()
    assert mock_run.await_args is not None
    first_post_fork_history = mock_run.await_args.kwargs["conversation_history"]
    assert first_post_fork_history == provider_history
    _assert_valid_for_next_user_turn(first_post_fork_history)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish_reason",
    [
        "length",
        "max_tokens",
        "content_filter",
        "interrupted",
        "cancelled",
        "tool_calls",
        "unknown_nonterminal_reason",
        None,
    ],
)
async def test_session_fork_rejects_modern_anchor_without_recognized_completion(
    auth_adapter, session_db, finish_reason
):
    source_id = session_db.create_session(
        f"finish-source-{finish_reason}", "api_server"
    )
    session_db.append_message(source_id, "user", "known completed turn")
    session_db.append_message(
        source_id, "assistant", "complete", finish_reason="stop"
    )
    session_db.append_message(source_id, "user", "candidate turn")
    anchor_id = session_db.append_message(
        source_id,
        "assistant",
        "possibly partial",
        finish_reason=finish_reason,
    )

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": f"finish-child-{anchor_id}",
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": f"finish-key-{anchor_id}",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == "unsafe_anchor"
    assert session_db.get_session(f"finish-child-{anchor_id}") is None


@pytest.mark.asyncio
async def test_session_fork_accepts_recognized_modern_and_all_null_legacy_boundaries(
    auth_adapter, session_db
):
    modern_id = session_db.create_session("modern-finish-source", "api_server")
    session_db.append_message(modern_id, "user", "modern")
    modern_anchor = session_db.append_message(
        modern_id, "assistant", "done", finish_reason="stop"
    )
    legacy_id = session_db.create_session("legacy-finish-source", "api_server")
    session_db.append_message(legacy_id, "user", "legacy")
    legacy_anchor = session_db.append_message(legacy_id, "assistant", "done")

    app = _create_session_app(auth_adapter)
    headers = {"Authorization": "Bearer sk-test"}
    async with TestClient(TestServer(app)) as cli:
        modern = await cli.post(
            f"/api/sessions/{modern_id}/fork",
            json={
                "id": "modern-finish-child",
                "anchor_message_id": modern_anchor,
                "preserve_source": True,
                "idempotency_key": "modern-finish-key",
            },
            headers=headers,
        )
        modern_text = await modern.text()
        legacy = await cli.post(
            f"/api/sessions/{legacy_id}/fork",
            json={
                "id": "legacy-finish-child",
                "anchor_message_id": legacy_anchor,
                "preserve_source": True,
                "idempotency_key": "legacy-finish-key",
            },
            headers=headers,
        )
        legacy_text = await legacy.text()

    assert modern.status == 201, modern_text
    assert legacy.status == 201, legacy_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_calls,tool_result_ids",
    [
        ([{"id": "id-only"}], ["id-only"]),
        ([_tool_call("empty-name", "")], ["empty-name"]),
        (
            [{
                "id": "bad-args",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{not-json"},
            }],
            ["bad-args"],
        ),
        ([_tool_call("duplicate"), _tool_call("duplicate")], ["duplicate"]),
        ([_tool_call("expected")], ["orphan"]),
        ([_tool_call("twice")], ["twice", "twice"]),
    ],
)
async def test_session_fork_rejects_malformed_or_unmatched_tool_call_groups(
    auth_adapter, session_db, tool_calls, tool_result_ids
):
    first_call_id = (
        tool_calls[0].get("id")
        or tool_calls[0].get("call_id")
        or "missing"
    )
    source_id = session_db.create_session(
        f"malformed-tool-source-{first_call_id}-{len(tool_result_ids)}",
        "api_server",
    )
    session_db.append_message(source_id, "user", "use a tool")
    session_db.append_message(
        source_id,
        "assistant",
        "calling",
        tool_calls=tool_calls,
        finish_reason="tool_calls",
    )
    for result_id in tool_result_ids:
        session_db.append_message(
            source_id,
            "tool",
            "result",
            tool_call_id=result_id,
            tool_name="test_tool",
        )
    anchor_id = session_db.append_message(
        source_id, "assistant", "final", finish_reason="stop"
    )

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": f"malformed-tool-child-{anchor_id}",
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": f"malformed-tool-key-{anchor_id}",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        payload = await response.json()

    assert response.status == 400
    assert payload["error"]["code"] == "unsafe_anchor"


@pytest.mark.asyncio
async def test_session_fork_accepts_complete_provider_neutral_tool_call_shapes(
    auth_adapter, session_db
):
    source_id = session_db.create_session("provider-neutral-tools", "api_server")
    shapes = [
        _tool_call("openai-call", "openai_tool"),
        _responses_tool_call("responses-call", "responses_tool"),
        _anthropic_tool_call("anthropic-call", "anthropic_tool"),
    ]
    for index, call in enumerate(shapes):
        session_db.append_message(source_id, "user", f"turn {index}")
        session_db.append_message(
            source_id,
            "assistant",
            "calling",
            tool_calls=[call],
            finish_reason="tool_calls",
        )
        call_id = call.get("call_id") or call["id"]
        session_db.append_message(
            source_id,
            "tool",
            "result",
            tool_call_id=call_id,
            tool_name=(call.get("function") or {}).get("name") or call.get("name"),
        )
        anchor_id = session_db.append_message(
            source_id, "assistant", "done", finish_reason="stop"
        )

    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post(
            f"/api/sessions/{source_id}/fork",
            json={
                "id": "provider-neutral-child",
                "anchor_message_id": anchor_id,
                "preserve_source": True,
                "idempotency_key": "provider-neutral-key",
            },
            headers={"Authorization": "Bearer sk-test"},
        )
        response_text = await response.text()

    assert response.status == 201, response_text
    assert len(session_db.get_messages("provider-neutral-child")) == 12


@pytest.mark.asyncio
async def test_consumed_fork_key_survives_child_delete_and_cannot_create_again(
    auth_adapter, session_db
):
    source_id = session_db.create_session("delete-reuse-source", "api_server")
    session_db.append_message(source_id, "user", "branch")
    anchor_id = session_db.append_message(
        source_id, "assistant", "done", finish_reason="stop"
    )
    body = {
        "id": "delete-reuse-child",
        "anchor_message_id": anchor_id,
        "preserve_source": True,
        "idempotency_key": "delete-reuse-key",
    }
    headers = {"Authorization": "Bearer sk-test"}
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        created = await cli.post(
            f"/api/sessions/{source_id}/fork", json=body, headers=headers
        )
        assert created.status == 201, await created.text()
        assert session_db.delete_session("delete-reuse-child") is True
        reservation = session_db._conn.execute(
            "SELECT child_session_id FROM session_fork_requests WHERE idempotency_key = ?",
            ("delete-reuse-key",),
        ).fetchone()
        assert reservation["child_session_id"] == "delete-reuse-child"

        stale = await cli.post(
            f"/api/sessions/{source_id}/fork", json=body, headers=headers
        )
        stale_payload = await stale.json()

    assert stale.status == 409
    assert stale_payload["error"]["code"] == "idempotency_stale"
    assert session_db.get_session("delete-reuse-child") is None


@pytest.mark.asyncio
async def test_session_chat_loads_history_and_preserves_session_headers(auth_adapter, session_db):
    session_id = session_db.create_session("chat-session", "api_server")
    session_db.set_session_title(session_id, "Chat")
    session_db.append_message(session_id, "user", "earlier")
    session_db.append_message(session_id, "assistant", "prior answer")

    mock_run = AsyncMock(return_value=({"final_response": "fresh answer", "session_id": session_id}, {"total_tokens": 3}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "next", "system_message": "stay focused"},
                headers={"Authorization": "Bearer sk-test", "X-Hermes-Session-Key": "client-42"},
            )
            assert resp.status == 200
            payload = await resp.json()

    assert resp.headers["X-Hermes-Session-Id"] == session_id
    assert resp.headers["X-Hermes-Session-Key"] == "client-42"
    assert payload["object"] == "hermes.session.chat.completion"
    assert payload["session_id"] == session_id
    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "fresh answer"
    mock_run.assert_awaited_once()
    _, kwargs = mock_run.call_args
    assert kwargs["session_id"] == session_id
    assert kwargs["gateway_session_key"] == "client-42"
    assert kwargs["ephemeral_system_prompt"] == "stay focused"
    history = kwargs["conversation_history"]
    assert len(history) == 2
    assert isinstance(history[0].pop("timestamp"), (int, float))
    assert isinstance(history[1].pop("timestamp"), (int, float))
    assert history == [
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "prior answer"},
    ]


@pytest.mark.asyncio
async def test_session_chat_fails_closed_on_history_error_before_model(auth_adapter, session_db):
    session_id = session_db.create_session("chat-history-fail", "api_server")
    mock_run = AsyncMock()
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run), patch.object(
        session_db, "get_messages_as_conversation", side_effect=sqlite3.DatabaseError("secret db path")
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "next"},
                headers={"Authorization": "Bearer sk-test"},
            )
            payload = await resp.json()
    assert resp.status == 503
    assert payload["error"]["code"] == "session_history_unavailable"
    assert "secret" not in json.dumps(payload)
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_chat_stream_fails_closed_on_history_error_before_model(adapter, session_db):
    session_id = session_db.create_session("stream-history-fail", "api_server")
    mock_run = AsyncMock()
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run), patch.object(
        session_db, "get_messages_as_conversation", side_effect=sqlite3.DatabaseError("secret db path")
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream", json={"message": "next"}
            )
            body = await resp.text()

    assert resp.status == 200
    assert "session_history_unavailable" in body
    assert "secret db path" not in body
    assert "event: run.started" not in body
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_chat_conflicts_with_canonical_durable_owner(auth_adapter, session_db):
    session_id = session_db.create_session("chat-lease-conflict", "api_server")
    assert session_db.acquire_session_execution_lease(session_id, "gateway-owner") is True
    mock_run = AsyncMock()
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            unauthorized = await cli.post(
                f"/api/sessions/{session_id}/chat", json={"message": "next"}
            )
            conflict = await cli.post(
                f"/api/sessions/{session_id}/chat", json={"message": "next"},
                headers={"Authorization": "Bearer sk-test"},
            )
            conflict_payload = await conflict.json()
    assert unauthorized.status == 401
    assert conflict.status == 409
    assert conflict_payload["error"]["code"] == "active_session_execution"
    mock_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_chat_accepts_multimodal_message(auth_adapter, session_db):
    session_id = session_db.create_session("image-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]

    mock_run = AsyncMock(return_value=({"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}))
    app = _create_session_app(auth_adapter)
    with patch.object(auth_adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": image_payload},
                headers={"Authorization": "Bearer sk-test"},
            )
            assert resp.status == 200, await resp.text()

    _, kwargs = mock_run.call_args
    assert kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_accepts_multimodal_message(adapter, session_db):
    session_id = session_db.create_session("image-stream-session", "api_server")
    image_payload = [
        {"type": "input_text", "text": "What's in this image?"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
    ]
    expected_user_message = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        kwargs["stream_delta_callback"]("A cat.")
        return {"final_response": "A cat.", "session_id": session_id}, {"total_tokens": 4}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": image_payload},
            )
            assert resp.status == 200, await resp.text()
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: assistant.completed" in body
    assert captured_kwargs["user_message"] == expected_user_message


@pytest.mark.asyncio
async def test_session_chat_stream_emits_lifecycle_events_and_keepalive_safe_shape(adapter, session_db):
    session_id = session_db.create_session("stream-session", "api_server")
    session_db.set_session_title(session_id, "Stream")

    async def fake_run(**kwargs):
        kwargs["stream_delta_callback"]("Hello")
        kwargs["stream_delta_callback"](" world")
        kwargs["tool_progress_callback"]("reasoning.available", tool_name="_thinking", preview="thinking")
        return {"final_response": "Hello world", "session_id": session_id}, {"total_tokens": 2}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/api/sessions/{session_id}/chat/stream", json={"message": "stream please"})
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            body = await resp.text()

    assert "event: run.started" in body
    assert "event: message.started" in body
    assert "event: assistant.delta" in body
    assert "Hello world" in body
    assert "event: tool.progress" in body
    assert "event: assistant.completed" in body
    assert "event: run.completed" in body
    assert "event: done" in body


@pytest.mark.asyncio
async def test_session_chat_stream_run_completed_carries_turn_transcript(adapter, session_db):
    """run.completed must include the full interleaved turn transcript so a
    client that lost intermediate (pre-tool-call) assistant text from the live
    delta stream can reconcile without a separate /messages fetch. Refs #34703.
    """
    import json as _json

    session_id = session_db.create_session("transcript-session", "api_server")

    async def fake_run(**kwargs):
        # Stream the intermediate planning text the way a real turn would.
        kwargs["stream_delta_callback"]("Let me search for that:")
        kwargs["stream_delta_callback"]("Here is the summary.")
        result = {
            "final_response": "Here is the summary.",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "search then summarize"},
                {
                    "role": "assistant",
                    "content": "Let me search for that:",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        }
        return result, {"total_tokens": 6}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "search then summarize"},
            )
            assert resp.status == 200
            body = await resp.text()

    # Pull the run.completed event payload out of the SSE body.
    run_completed_payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_completed_payload = _json.loads(line[len("data: "):])
            break
    assert run_completed_payload is not None, body
    messages = run_completed_payload.get("messages")
    assert isinstance(messages, list) and messages, run_completed_payload

    # The colon-ended intermediate text that preceded the tool call must be present.
    contents = [m.get("content") for m in messages]
    assert "Let me search for that:" in contents
    assert "Here is the summary." in contents
    # No prior-turn user message should leak into the per-turn slice.
    assert all(m.get("role") in ("assistant", "tool") for m in messages)
    # The tool call is preserved alongside the intermediate text.
    assert any(m.get("tool_calls") for m in messages)



@pytest.mark.asyncio
async def test_session_endpoints_require_auth_when_key_configured(auth_adapter):
    app = _create_session_app(auth_adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/sessions")
        assert resp.status == 401
        body = await resp.json()
        assert body["error"]["code"] == "gateway_auth_failed"

        ok = await cli.get("/api/sessions", headers={"Authorization": "Bearer sk-test"})
        assert ok.status == 200
        data = await ok.json()
        assert data["object"] == "list"
        assert data["data"] == []


@pytest.mark.asyncio
async def test_session_header_rejected_without_api_key(adapter, session_db):
    session_id = session_db.create_session("unsafe-session", "api_server")
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            f"/api/sessions/{session_id}/chat",
            json={"message": "hello"},
            headers={"X-Hermes-Session-Key": "client-42"},
        )
        assert resp.status == 403
        data = await resp.json()
        assert "X-Hermes-Session-Key requires API key" in data["error"]["message"]
