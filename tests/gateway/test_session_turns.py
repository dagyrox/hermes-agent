"""Behavior contract for durable API session turns (Or7 #159)."""

import asyncio
import base64
import io
import json
import sqlite3
import struct
import zlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import SendResult
from gateway.platforms.session_turns import (
    ConflictingTurn,
    SessionTurnStore,
    TurnConflict,
    TurnInputError,
    project_safe_tool_event,
    resolve_slack_binding,
)
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionDB:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "api_server")
    return db


def _payload(text: str = "hello", mode: str = "occ_only") -> dict:
    return {"input": {"text": text, "images": []}, "delivery_mode": mode}


def _png(width: int = 1, height: int = 1) -> str:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"\x89PNG\r\n\x1a\n"
    raw += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    raw += chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
    raw += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def test_duplicate_submit_reuses_same_turn_without_second_reservation(session_db):
    store = SessionTurnStore(session_db)
    first, created = store.reserve("s1", "turn-1", _payload())
    second, reused = store.reserve("s1", "turn-1", _payload())

    assert created is True
    assert reused is False
    assert second["turn_id"] == first["turn_id"]
    assert second["fingerprint"] == first["fingerprint"]


def test_conflicting_fingerprint_reuse_is_rejected(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "turn-1", _payload("first"))

    with pytest.raises(ConflictingTurn):
        store.reserve("s1", "turn-1", _payload("different"))


def test_one_active_turn_per_session_is_enforced(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "turn-1", _payload())

    with pytest.raises(TurnConflict):
        store.reserve("s1", "turn-2", _payload("second"))


def test_restart_reconciliation_interrupts_uncertain_work_without_execution(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "queued", _payload())
    store.set_running("queued")

    assert store.reconcile_uncertain() == 1
    turn = store.get("queued")
    assert turn["status"] == "interrupted"
    assert turn["safe_error_code"] == "process_restarted"
    events = store.events_after("queued", 0)
    assert events[-1]["event"] == "turn.interrupted"
    assert events[-1]["data"]["error_code"] == "process_restarted"


def test_restart_reconciliation_preserves_live_unexpired_lease(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "live", _payload())
    store.set_running("live")
    assert session_db.acquire_session_execution_lease("s1", "live-owner", ttl_seconds=30)

    assert store.reconcile_uncertain() == 0
    assert store.get("live")["status"] == "running"
    assert session_db.get_session_execution_lease("s1")["owner_id"] == "live-owner"


def test_restart_reconciliation_interrupts_dead_owner_and_removes_lease(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "dead", _payload())
    store.set_running("dead")
    assert session_db.acquire_session_execution_lease("s1", "dead-owner", ttl_seconds=30)
    session_db._execute_write(lambda conn: conn.execute(
        "UPDATE session_execution_leases SET owner_pid = ? WHERE session_id = ?",
        (999_999_999, "s1"),
    ))

    assert store.reconcile_uncertain() == 1
    assert store.get("dead")["status"] == "interrupted"
    assert session_db.get_session_execution_lease("s1") is None


def test_reconnect_replays_events_after_last_sequence(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "turn-1", _payload())
    one = store.append_event("turn-1", "turn.started", {"status": "running"})
    two = store.append_event("turn-1", "assistant.delta", {"delta": "Hi"})

    assert one["seq"] < two["seq"]
    assert store.events_after("turn-1", one["seq"]) == [two]


def test_durable_event_rows_never_store_assistant_delta_bodies(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "turn-body-free", _payload("user body sentinel"))
    store.append_event("turn-body-free", "assistant.delta", {"delta": "assistant body sentinel"})

    persisted = session_db._conn.execute(
        "SELECT data_json FROM api_session_turn_events WHERE turn_id = ?",
        ("turn-body-free",),
    ).fetchone()[0]
    ledger = repr(session_db._conn.execute(
        "SELECT * FROM api_session_turns WHERE turn_id = ?", ("turn-body-free",)
    ).fetchone())
    assert "assistant body sentinel" not in persisted
    assert "user body sentinel" not in ledger


def test_safe_tool_projection_never_contains_sensitive_fields():
    projected = project_safe_tool_event(
        "tool.started",
        tool_call_id="call-1",
        tool_name="terminal",
        args={"command": "print secret"},
        preview="token=secret",
        output="credential",
        reasoning="hidden",
        stack_trace="trace",
    )

    assert projected == {
        "event": "tool.started",
        "data": {
            "tool_call_id": "call-1",
            "label": "Terminal",
            "status": "running",
        },
    }
    wire = json.dumps(projected)
    for forbidden in ("command", "preview", "output", "reasoning", "trace", "token"):
        assert forbidden not in wire


def test_inline_images_are_bounded_and_remote_images_rejected(
    session_db, monkeypatch: pytest.MonkeyPatch
):
    import gateway.platforms.session_turns as session_turns

    store = SessionTurnStore(session_db)
    inline = _png()
    turn, created = store.reserve(
        "s1",
        "turn-inline",
        {"input": {"text": "look", "images": [inline]}, "delivery_mode": "occ_only"},
    )
    assert created is True
    assert turn["status"] == "queued"
    store.finish("turn-inline", "interrupted")

    with pytest.raises(TurnInputError, match="inline"):
        store.reserve(
            "s1",
            "turn-image",
            {"input": {"text": "look", "images": ["https://example.test/x.png"]}, "delivery_mode": "occ_only"},
        )
    with pytest.raises(TurnInputError, match="Too many"):
        store.reserve(
            "s1",
            "turn-count",
            {"input": {"text": "look", "images": [inline] * 5}, "delivery_mode": "occ_only"},
        )
    monkeypatch.setattr(session_turns, "MAX_INLINE_IMAGE_BYTES", 1)
    with pytest.raises(TurnInputError, match="per-image"):
        store.reserve(
            "s1",
            "turn-size",
            {"input": {"text": "look", "images": [inline]}, "delivery_mode": "occ_only"},
        )


@pytest.mark.parametrize(
    "image,match",
    [
        ("data:image/jpeg;base64," + _png().split(",", 1)[1], "MIME"),
        ("data:image/png;base64,iVBORw0KGgo=", "truncated"),
        (_png(100_000, 100_000), "dimensions"),
    ],
)
def test_inline_images_reject_spoofed_truncated_and_bomb_dimensions(session_db, image, match):
    store = SessionTurnStore(session_db)
    with pytest.raises(TurnInputError, match=match):
        store.reserve(
            "s1",
            f"turn-image-{match.lower()}",
            {"input": {"text": "look", "images": [image]}, "delivery_mode": "occ_only"},
        )


def test_inline_image_decoder_rejects_corrupt_compressed_pixels(session_db):
    from PIL import Image

    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, format="PNG")
    corrupt = bytearray(output.getvalue())
    idat = corrupt.index(b"IDAT")
    corrupt[idat + 8] ^= 0xFF
    image = "data:image/png;base64," + base64.b64encode(corrupt).decode()

    with pytest.raises(TurnInputError):
        SessionTurnStore(session_db).reserve(
            "s1", "turn-corrupt-pixels",
            {"input": {"text": "look", "images": [image]}, "delivery_mode": "occ_only"},
        )


def test_session_execution_lease_is_persistent_owner_checked_and_reconciles_dead_owner(tmp_path):
    first = SessionDB(db_path=tmp_path / "lease.db")
    first.create_session("shared", "api_server")
    second = SessionDB(db_path=tmp_path / "lease.db")

    assert first.acquire_session_execution_lease("shared", "owner-a", ttl_seconds=30)
    assert not second.acquire_session_execution_lease("shared", "owner-b", ttl_seconds=30)
    assert not second.release_session_execution_lease("shared", "owner-b")
    assert first.release_session_execution_lease("shared", "owner-a")
    assert second.acquire_session_execution_lease("shared", "owner-b", ttl_seconds=30)
    second._execute_write(lambda conn: conn.execute(
        "UPDATE session_execution_leases SET owner_pid = ?, expires_at = ? WHERE session_id = ?",
        (999_999_999, 0, "shared"),
    ))
    assert first.acquire_session_execution_lease("shared", "owner-c", ttl_seconds=30)


def test_execution_lease_can_serialize_admitted_session_before_session_row_exists(tmp_path):
    db = SessionDB(db_path=tmp_path / "lease-without-session.db")

    assert db.acquire_session_execution_lease("admitted-not-yet-persisted", "owner-a")
    assert not db.acquire_session_execution_lease("admitted-not-yet-persisted", "owner-b")


def test_browser_supplied_routing_fields_are_rejected(session_db):
    store = SessionTurnStore(session_db)
    with pytest.raises(TurnInputError, match="Unsupported turn field"):
        store.reserve(
            "s1",
            "turn-route",
            {**_payload(mode="slack_only"), "channel_id": "C_ATTACKER"},
        )


def test_destination_outbox_is_unique_and_pending_is_ambiguous_on_restart(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "turn-1", _payload(mode="both"))
    assert store.begin_delivery("turn-1", "slack") is True
    assert store.begin_delivery("turn-1", "slack") is False

    store.reconcile_uncertain()
    delivery = store.get_delivery("turn-1", "slack")
    assert delivery["state"] == "needs_manual_retry"
    assert delivery["safe_error_code"] == "delivery_outcome_unknown"


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        if "/api/sessions/{session_id}/turns" in path or path == "/v1/capabilities":
            app.router.add_route(method, path, handler)
    return app


@pytest.mark.asyncio
async def test_history_read_failure_fails_durable_turn_before_agent_execution(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db
    adapter._run_agent = AsyncMock()
    original_history = session_db.get_messages_as_conversation
    session_db.get_messages_as_conversation = MagicMock(side_effect=sqlite3.DatabaseError("secret"))

    async with TestClient(TestServer(_app(adapter))) as client:
        submitted = await client.post(
            "/api/sessions/s1/turns", headers={"Idempotency-Key": "turn-history-fail"}, json=_payload(),
        )
        assert submitted.status == 202
        await adapter._session_turn_service.wait_for_turn("turn-history-fail")
        status = await client.get("/api/sessions/s1/turns/turn-history-fail")
        turn = (await status.json())["turn"]

    assert turn["status"] == "failed"
    assert turn["safe_error_code"] == "history_read_failed"
    adapter._run_agent.assert_not_awaited()

    # The failed history read must not strand the canonical lease.
    session_db.get_messages_as_conversation = original_history
    adapter._run_agent = AsyncMock(return_value=({"final_response": "ok", "messages": []}, {}))
    async with TestClient(TestServer(_app(adapter))) as client:
        retry = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-history-retry"},
            json=_payload("retry"),
        )
        await adapter._session_turn_service.wait_for_turn("turn-history-retry")
    assert retry.status == 202
    adapter._run_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_legacy_chat_history_read_failure_returns_safe_503_without_agent_execution(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db
    adapter._run_agent = AsyncMock()
    session_db.get_messages_as_conversation = MagicMock(side_effect=sqlite3.DatabaseError("/secret/state.db"))
    app = web.Application()
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)

    async with TestClient(TestServer(app)) as client:
        response = await client.post("/api/sessions/s1/chat", json={"message": "hello"})
        data = await response.json()

    assert response.status == 503
    assert data["error"]["code"] == "session_history_unavailable"
    assert "/secret" not in json.dumps(data)
    adapter._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_durable_and_legacy_session_paths_share_one_persistent_execution_lease(session_db):
    gate = asyncio.Event()
    started = asyncio.Event()
    calls = 0

    async def run_agent(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await gate.wait()
        return {"final_response": "done", "messages": []}, {}

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db
    adapter._run_agent = run_agent
    app = _app(adapter)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)

    async with TestClient(TestServer(app)) as client:
        submitted = await client.post(
            "/api/sessions/s1/turns", headers={"Idempotency-Key": "turn-held"}, json=_payload(),
        )
        assert submitted.status == 202
        await started.wait()
        conflict = await client.post("/api/sessions/s1/chat", json={"message": "second"})
        assert conflict.status == 409
        assert (await conflict.json())["error"]["code"] == "active_session_execution"
        gate.set()
        await adapter._session_turn_service.wait_for_turn("turn-held")
        after = await client.post("/api/sessions/s1/chat", json={"message": "third"})
        assert after.status == 200


@pytest.mark.asyncio
async def test_occ_only_executes_once_attempts_zero_slack_sends_and_does_not_duplicate_transcript(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db

    async def canonical_run(**kwargs):
        # The edge must not pre-append a user message; AIAgent/SessionDB owns
        # the complete turn atomically through the existing run machinery.
        assert session_db.get_messages("s1") == []
        session_db.replace_messages(
            "s1",
            [
                {"role": "user", "content": kwargs["user_message"]},
                {"role": "assistant", "content": "done"},
            ],
        )
        return {"final_response": "done", "messages": session_db.get_messages_as_conversation("s1")}, {}

    adapter._run_agent = AsyncMock(side_effect=canonical_run)
    slack = AsyncMock()
    adapter.gateway_runner = type("Runner", (), {"adapters": {}})()

    async with TestClient(TestServer(_app(adapter))) as client:
        first = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-1"},
            json=_payload(),
        )
        assert first.status == 202
        await adapter._session_turn_service.wait_for_turn("turn-1")
        retry = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-1"},
            json=_payload(),
        )
        assert retry.status == 200

    assert adapter._run_agent.await_count == 1
    assert [message["role"] for message in session_db.get_messages("s1")] == ["user", "assistant"]
    slack.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_slack_mode_sends_exactly_once_via_connected_adapter(session_db):
    origin = {
        "platform": "slack",
        "chat_id": "C123",
        "thread_id": "171.2",
        "scope_id": "T123",
    }
    session_db.record_gateway_session_peer(
        "s1",
        source="slack",
        session_key="agent:main:slack:T123:channel:C123:thread:171.2",
        chat_id="C123",
        thread_id="171.2",
        origin_json=json.dumps(origin),
    )
    slack = AsyncMock()
    slack.send.return_value = SendResult(success=True, message_id="171.3")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db

    def canonical_run(**kwargs):
        session_db.append_message("s1", "user", kwargs["user_message"])
        session_db.append_message("s1", "assistant", "done")
        return {"final_response": "done", "messages": session_db.get_messages("s1")}, {}

    adapter._run_agent = AsyncMock(side_effect=canonical_run)
    adapter._get_platform_callback_adapter = lambda _request, name: slack if name == "slack" else None

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-slack"},
            json=_payload(mode="both"),
        )
        assert response.status == 202
        await adapter._session_turn_service.wait_for_turn("turn-slack")
        events = await client.get("/api/sessions/s1/turns/turn-slack/events")
        safe_wire = await events.text()

    slack.send.assert_awaited_once()
    args, kwargs = slack.send.await_args
    assert args[:2] == ("C123", "done")
    assert kwargs["reply_to"] == "171.2"
    assert kwargs["metadata"]["client_msg_id"]
    assert adapter._run_agent.await_count == 1
    assert "C123" not in safe_wire
    assert "171.2" not in safe_wire
    assert "T123" not in safe_wire
    assert kwargs["metadata"]["client_msg_id"] not in safe_wire
    persisted = SessionTurnStore(session_db)
    delivery = persisted.get_delivery("turn-slack", "slack")
    turn = persisted.get("turn-slack")
    assert turn is not None and turn["assistant_message_id"]
    assert delivery is not None and delivery["provider_message_id"] == "171.3"
    assert delivery["provider_message_id"] != delivery["delivery_id"]


def test_slack_binding_walks_to_nearest_bound_ancestor_and_checks_all_metadata(session_db):
    origin = {"platform": "slack", "chat_id": "C123", "thread_id": "171.2", "scope_id": "T123"}
    session_db.record_gateway_session_peer(
        "s1", source="slack", session_key="agent:main:slack:T123:channel:C123:thread:171.2",
        chat_id="C123", thread_id="171.2", origin_json=json.dumps(origin),
    )
    session_db.end_session("s1", "compression")
    session_db.create_session("child", "slack", parent_session_id="s1")
    session_db.save_gateway_routing_entry(
        "agent:main:slack:T123:channel:C123:thread:171.2",
        json.dumps({"session_id": "s1", "source": {"platform": "slack", "chat_id": "C123", "thread_id": "171.2", "scope_id": "T123"}}),
    )
    binding = resolve_slack_binding(session_db, "child")
    assert (binding.chat_id, binding.thread_id) == ("C123", "171.2")

    session_db.save_gateway_routing_entry(
        "conflict",
        json.dumps({"session_id": "child", "source": {"platform": "slack", "chat_id": "C999", "thread_id": "171.2", "scope_id": "T123"}}),
    )
    with pytest.raises(Exception, match="ambiguous_slack_binding"):
        resolve_slack_binding(session_db, "child")


def test_slack_binding_uses_nearest_bound_row_without_aggregating_valid_ancestor(session_db):
    parent = {"platform": "slack", "chat_id": "C_PARENT", "thread_id": "1", "scope_id": "T1"}
    child = {"platform": "slack", "chat_id": "C_CHILD", "thread_id": "2", "scope_id": "T1"}
    session_db.record_gateway_session_peer(
        "s1", source="slack", session_key="parent", origin_json=json.dumps(parent)
    )
    session_db.create_session("child-nearest", "slack", parent_session_id="s1")
    session_db.record_gateway_session_peer(
        "child-nearest", source="slack", session_key="child", origin_json=json.dumps(child)
    )

    binding = resolve_slack_binding(session_db, "child-nearest")
    assert (binding.chat_id, binding.thread_id) == ("C_CHILD", "2")


@pytest.mark.asyncio
async def test_ambiguous_slack_send_failure_is_not_automatically_retried(session_db):
    origin = {
        "platform": "slack",
        "chat_id": "C123",
        "thread_id": "171.2",
        "scope_id": "T123",
    }
    session_db.record_gateway_session_peer(
        "s1", source="slack", session_key="binding", origin_json=json.dumps(origin)
    )
    slack = AsyncMock()
    slack.send.side_effect = TimeoutError("ambiguous transport outcome")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db

    def canonical_run(**kwargs):
        session_db.append_message("s1", "user", kwargs["user_message"])
        session_db.append_message("s1", "assistant", "done")
        return {
            "final_response": "done",
            "messages": session_db.get_messages_as_conversation("s1"),
        }, {}

    adapter._run_agent = AsyncMock(side_effect=canonical_run)
    adapter._get_platform_callback_adapter = (
        lambda _request, name: slack if name == "slack" else None
    )

    async with TestClient(TestServer(_app(adapter))) as client:
        first = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-ambiguous"},
            json=_payload(mode="slack_only"),
        )
        assert first.status == 202
        await adapter._session_turn_service.wait_for_turn("turn-ambiguous")
        # Durable reuse must not depend on the adapter still being connected.
        adapter._get_platform_callback_adapter = lambda request, name: None
        retry = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-ambiguous"},
            json=_payload(mode="slack_only"),
        )
        assert retry.status == 200
        status = await client.get("/api/sessions/s1/turns/turn-ambiguous")
        delivery = (await status.json())["turn"]["deliveries"][0]

    slack.send.assert_awaited_once()
    assert delivery["state"] == "needs_manual_retry"
    assert delivery["safe_error_code"] == "delivery_outcome_unknown"


@pytest.mark.asyncio
async def test_explicit_deliver_is_idempotent_and_never_retries_unknown_send(session_db):
    origin = {"platform": "slack", "chat_id": "C123", "thread_id": "171.2", "scope_id": "T123"}
    session_db.record_gateway_session_peer(
        "s1", source="slack", session_key="binding", chat_id="C123", thread_id="171.2",
        origin_json=json.dumps(origin),
    )
    slack = AsyncMock()
    slack.send.side_effect = TimeoutError("ambiguous")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db

    def canonical_run(**kwargs):
        session_db.append_message("s1", "user", kwargs["user_message"])
        session_db.append_message("s1", "assistant", "done")
        return {
            "final_response": "done",
            "messages": session_db.get_messages_as_conversation("s1"),
        }, {}

    adapter._run_agent = AsyncMock(side_effect=canonical_run)
    adapter._get_platform_callback_adapter = lambda _request, name: slack if name == "slack" else None

    async with TestClient(TestServer(_app(adapter))) as client:
        submitted = await client.post(
            "/api/sessions/s1/turns", headers={"Idempotency-Key": "turn-deliver"}, json=_payload(mode="both"),
        )
        await adapter._session_turn_service.wait_for_turn("turn-deliver")
        first = await client.post("/api/sessions/s1/turns/turn-deliver/deliver", json={"destination": "slack"})
        second = await client.post("/api/sessions/s1/turns/turn-deliver/deliver", json={"destination": "slack"})
        status = await client.get("/api/sessions/s1/turns/turn-deliver")
        delivery = (await status.json())["turn"]["deliveries"][0]

    assert submitted.status == 202
    assert first.status == second.status == 409
    assert slack.send.await_count == 1
    assert delivery["delivery_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin,code",
    [
        ({"platform": "api_server"}, "no_slack_binding"),
        (
            {"platform": "slack", "chat_id": "C1", "channel_id": "C2", "thread_id": "1"},
            "ambiguous_slack_binding",
        ),
    ],
)
async def test_slack_binding_missing_or_ambiguous_fails_closed_before_run(session_db, origin, code):
    session_db.record_gateway_session_peer(
        "s1", source=origin["platform"], session_key="binding", origin_json=json.dumps(origin)
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db
    adapter._run_agent = AsyncMock()

    async with TestClient(TestServer(_app(adapter))) as client:
        response = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-slack"},
            json=_payload(mode="slack_only"),
        )
        assert response.status == 409
        assert (await response.json())["error"]["code"] == code

    adapter._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_reports_stopping_until_worker_actually_exits(session_db):
    gate = asyncio.Event()
    agent = type("Agent", (), {"interrupt": lambda self, reason: None})()

    async def run_agent(**kwargs):
        kwargs["agent_ref"][0] = agent
        await gate.wait()
        return {"final_response": "late", "messages": []}, {}

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db
    adapter._run_agent = run_agent

    async with TestClient(TestServer(_app(adapter))) as client:
        submit = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-stop"},
            json=_payload(),
        )
        assert submit.status == 202
        await asyncio.sleep(0)
        stop = await client.post("/api/sessions/s1/turns/turn-stop/stop")
        assert stop.status == 202
        assert (await stop.json())["turn"]["status"] == "stopping"
        status = await client.get("/api/sessions/s1/turns/turn-stop")
        assert (await status.json())["turn"]["status"] == "stopping"
        gate.set()
        await adapter._session_turn_service.wait_for_turn("turn-stop")
        final = await client.get("/api/sessions/s1/turns/turn-stop")
        assert (await final.json())["turn"]["status"] == "interrupted"


def test_no_transcript_duplication_store_has_no_content_columns(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "turn-1", _payload("body must not be here"))
    columns = {
        row[1]
        for row in session_db._conn.execute("PRAGMA table_info(api_session_turns)").fetchall()
    }
    assert not ({"input", "text", "prompt", "assistant_content", "output", "reasoning"} & columns)


def test_capabilities_advertise_durable_turn_contract():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))

    async def check():
        async with TestClient(TestServer(_app(adapter))) as client:
            response = await client.get("/v1/capabilities")
            data = await response.json()
            assert data["features"]["conversation_turn_contract"] == "h1"
            assert data["features"]["durable_turn_idempotency"] is True
            assert data["features"]["durable_turn_status"] is True
            assert data["features"]["durable_turn_resumable_events"] is True
            assert data["features"]["durable_turn_stop"] is True
            assert data["endpoints"]["conversation_turn_create"]["path"].endswith("/turns")
            turns = data["session_turns"]
            assert turns["durable"] is True
            assert turns["routing"] == "session_origin_only"
            assert turns["events"]["resumable"] is True
            assert turns["events"]["last_event_id"] is True
            assert turns["safe_progress"] is True
            assert turns["inline_images"]["remote_urls"] is False
            assert turns["inline_images"]["max_dimension"] == 16_384
            assert turns["inline_images"]["max_pixels"] == 40_000_000
            assert turns["cancellation"] == "cooperative_truthful"
            assert data["endpoints"]["session_turn_deliver"] == {
                "method": "POST",
                "path": "/api/sessions/{session_id}/turns/{turn_id}/deliver",
            }
            assert turns["delivery"] == {
                "manual_operation": "POST /api/sessions/{session_id}/turns/{turn_id}/deliver",
                "delivery_id": True,
                "automatic_retry": False,
                "ambiguous_retry": False,
                "slack_atomic_max_chars": 3_000,
            }

    asyncio.run(check())


def test_slack_client_msg_id_validation_is_uuid_only():
    from plugins.platforms.slack.adapter import SlackAdapter

    valid = "3f60c8d8-4dd8-5dd2-9bf4-f66db1c600f8"
    assert SlackAdapter._metadata_client_msg_id({"client_msg_id": valid}) == valid
    assert SlackAdapter._metadata_client_msg_id({"client_msg_id": "browser-controlled"}) == ""


@pytest.mark.asyncio
async def test_slack_send_forwards_valid_client_msg_id_once():
    from plugins.platforms.slack.adapter import SlackAdapter

    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test"))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.chat_postMessage = AsyncMock(return_value={"ts": "1.0"})
    adapter._running = True
    client_msg_id = "3f60c8d8-4dd8-5dd2-9bf4-f66db1c600f8"

    result = await adapter.send(
        "C123",
        "done",
        metadata={"client_msg_id": client_msg_id},
    )

    assert result.success is True
    adapter._app.client.chat_postMessage.assert_awaited_once_with(
        channel="C123",
        text="done",
        mrkdwn=True,
        client_msg_id=client_msg_id,
    )


@pytest.mark.asyncio
async def test_sse_last_event_id_replays_only_later_durable_frames(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db

    async def streamed_run(**kwargs):
        kwargs["stream_delta_callback"]("one")
        kwargs["stream_delta_callback"]("two")
        session_db.append_message("s1", "user", kwargs["user_message"])
        session_db.append_message("s1", "assistant", "onetwo")
        return {
            "final_response": "onetwo",
            "messages": session_db.get_messages_as_conversation("s1"),
        }, {}

    adapter._run_agent = streamed_run
    async with TestClient(TestServer(_app(adapter))) as client:
        submitted = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-events"},
            json=_payload(),
        )
        assert submitted.status == 202
        await adapter._session_turn_service.wait_for_turn("turn-events")
        replay = await client.get(
            "/api/sessions/s1/turns/turn-events/events",
            headers={"Last-Event-ID": "1"},
        )
        wire = await replay.text()

    assert replay.status == 200
    assert "id: 1\n" not in wire
    assert "event: assistant.delta" in wire
    assert 'data: {"delta": "one"}' in wire
    assert "event: turn.completed" in wire


@pytest.mark.asyncio
async def test_sse_impossible_high_water_and_durable_gap_require_canonical_resync(session_db):
    store = SessionTurnStore(session_db)
    store.reserve("s1", "turn-gap", _payload())
    store.append_event("turn-gap", "turn.started", {"status": "running"})
    store.append_event("turn-gap", "assistant.delta", {"delta": "safe"})
    store.finish("turn-gap", "completed")
    session_db._execute_write(lambda conn: conn.execute(
        "DELETE FROM api_session_turn_events WHERE turn_id = ? AND seq = 1", ("turn-gap",)
    ))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db

    async with TestClient(TestServer(_app(adapter))) as client:
        high = await client.get(
            "/api/sessions/s1/turns/turn-gap/events", headers={"Last-Event-ID": "99"}
        )
        gap = await client.get(
            "/api/sessions/s1/turns/turn-gap/events", headers={"Last-Event-ID": "0"}
        )
        high_data = await high.json()
        gap_data = await gap.json()

    assert high.status == gap.status == 409
    assert high_data["error"]["code"] == "event_cursor_ahead"
    assert gap_data["error"]["code"] == "event_gap"
    assert high_data["error"]["resync_required"] is True


@pytest.mark.asyncio
async def test_sse_after_volatile_delta_loss_requires_canonical_resync(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db

    async def streamed_run(**kwargs):
        kwargs["stream_delta_callback"]("volatile body")
        session_db.append_message("s1", "user", kwargs["user_message"])
        session_db.append_message("s1", "assistant", "volatile body")
        return {"final_response": "volatile body", "messages": session_db.get_messages("s1")}, {}

    adapter._run_agent = streamed_run
    async with TestClient(TestServer(_app(adapter))) as client:
        submitted = await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-volatile-loss"},
            json=_payload(),
        )
        assert submitted.status == 202
        await adapter._session_turn_service.wait_for_turn("turn-volatile-loss")

    adapter._session_turn_service = type(adapter._session_turn_service)(adapter)
    # Avoid startup reconciliation changing the already-completed turn.
    async with TestClient(TestServer(_app(adapter))) as client:
        replay = await client.get(
            "/api/sessions/s1/turns/turn-volatile-loss/events",
            headers={"Last-Event-ID": "0"},
        )
        data = await replay.json()

    assert replay.status == 409
    assert data["error"]["code"] == "volatile_events_lost"
    assert data["error"]["resync_required"] is True


@pytest.mark.asyncio
async def test_slack_only_turn_never_exposes_assistant_deltas_to_occ(session_db):
    origin = {"platform": "slack", "chat_id": "C123", "thread_id": "1", "scope_id": "T1"}
    session_db.record_gateway_session_peer(
        "s1", source="slack", session_key="bound", origin_json=json.dumps(origin)
    )
    slack = AsyncMock()
    slack.send.return_value = SendResult(success=True, message_id="provider-receipt")
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._session_db = session_db
    adapter._get_platform_callback_adapter = lambda _request, name: slack if name == "slack" else None

    async def streamed_run(**kwargs):
        kwargs["stream_delta_callback"]("must not reach OCC")
        session_db.append_message("s1", "user", kwargs["user_message"])
        session_db.append_message("s1", "assistant", "must not reach OCC")
        return {"final_response": "must not reach OCC", "messages": session_db.get_messages("s1")}, {}

    adapter._run_agent = streamed_run
    async with TestClient(TestServer(_app(adapter))) as client:
        await client.post(
            "/api/sessions/s1/turns",
            headers={"Idempotency-Key": "turn-slack-only"},
            json=_payload(mode="slack_only"),
        )
        await adapter._session_turn_service.wait_for_turn("turn-slack-only")
        events = await client.get("/api/sessions/s1/turns/turn-slack-only/events")
        wire = await events.text()

    assert "assistant.delta" not in wire
    assert "must not reach OCC" not in wire
