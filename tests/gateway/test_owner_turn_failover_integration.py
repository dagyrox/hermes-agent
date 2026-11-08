from __future__ import annotations

import json
import threading

import pytest

from gateway.backend_failover import AttemptObservation, FrozenTurn, GatewayFailoverLedger, ReplayDenied
from gateway.owner_tool_bridge import OwnerToolBridge, OwnerTurnExecution
from gateway.owner_turn_delivery import OwnerFinalDelivery


def _turn() -> FrozenTurn:
    return FrozenTurn.create(
        session_key="agent:talos:slack:channel:C1:thread:1",
        session_id="session-1",
        event_id="C1:event-1",
        generation=3,
        prompt={
            "system": "frozen-system",
            "history": [{"role": "assistant", "content": "prior"}],
            "current": [{"type": "text", "text": "hello"}],
        },
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "todo",
                    "description": "owner todo",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        media=[{"type": "image_url", "sha256": "a" * 64}],
        legs=[
            {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
            {
                "kind": "claude_code",
                "model": "claude-sonnet-5-5",
                "capability_mode": "full",
                "reasoning_effort": "high",
            },
        ],
    )


class _Owner:
    def __init__(self) -> None:
        self.calls = 0

    def _execute_tool_calls(self, assistant, messages, effective_task_id, api_call_count=0):
        self.calls += 1
        call = assistant.tool_calls[0]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": "owner-result",
            }
        )


def test_default_slack_safe_fallback_preserves_owner_snapshot(tmp_path):
    ledger = GatewayFailoverLedger(tmp_path / "failover.db")
    turn = _turn()
    record = ledger.plan(turn, owner="gateway:3")
    first = ledger.start_leg(record.coordinator_id, 0, turn=turn, owner="gateway:3", generation=3)
    ledger.finish_leg(
        record.coordinator_id,
        0,
        AttemptObservation.availability_failure("rate_limit"),
        owner="gateway:3",
        generation=3,
    )
    assert ledger.next_leg(record.coordinator_id, owner="gateway:3", generation=3) == 1
    second = ledger.start_leg(record.coordinator_id, 1, turn=turn, owner="gateway:3", generation=3)

    assert second.prompt_json == first.prompt_json
    assert second.tools_json == first.tools_json
    assert second.media_json == first.media_json
    assert json.loads(second.leg_json)["reasoning_effort"] == "high"


def test_owner_approval_roundtrip_through_both_guards(tmp_path):
    ledger = GatewayFailoverLedger(tmp_path / "failover.db")
    turn = _turn()
    record = ledger.plan(turn, owner="gateway:3")
    ledger.start_leg(record.coordinator_id, 0, turn=turn, owner="gateway:3", generation=3)
    ledger.finish_leg(
        record.coordinator_id,
        0,
        AttemptObservation.availability_failure("rate_limit"),
        owner="gateway:3",
        generation=3,
    )
    dispatch = ledger.start_leg(record.coordinator_id, 1, turn=turn, owner="gateway:3", generation=3)
    owner = _Owner()
    execution = OwnerTurnExecution(
        agent=owner,
        messages=[],
        dispatch=dispatch,
        effective_task_id="session-1",
        turn_id="turn-1",
        coordinator_id=record.coordinator_id,
        attempt_id=f"{record.coordinator_id}:1",
        generation=3,
        snapshot_digest=dispatch.prompt_digest,
        identity=("talos", "slack", "C1", "U1", "session-1"),
        expected_identity=("talos", "slack", "C1", "U1", "session-1"),
        record_intent=lambda call_id, name, args: ledger.record_tool_intent(
            record.coordinator_id, 1, call_id, name, args, owner="gateway:3", generation=3
        ),
        record_result=lambda call_id, content, is_error: ledger.record_tool_result(
            record.coordinator_id, 1, call_id, content, is_error, owner="gateway:3", generation=3
        ),
    )
    bridge = OwnerToolBridge.open(dispatch, execution, runtime_root=tmp_path)
    frame = {
        "op": "call",
        "capability": bridge.capability,
        "coordinator_id": record.coordinator_id,
        "attempt_id": f"{record.coordinator_id}:1",
        "generation": 3,
        "snapshot_digest": dispatch.prompt_digest,
        "call_id": "call-1",
        "name": "todo",
        "arguments": {},
    }
    responses: list[dict] = []
    child = threading.Thread(target=lambda: responses.append(bridge.submit(frame)))
    child.start()
    assert bridge.drain_one(timeout=1) is True
    child.join(timeout=1)
    assert owner.calls == 1
    assert responses[0]["disposition"] == "result"
    assert [row["state"] for row in ledger.debug_tool_calls(record.coordinator_id)] == ["result"]
    bridge.close()


@pytest.mark.parametrize(
    "observation",
    [
        AttemptObservation.failure("tool_intent", tool_activity=True),
        AttemptObservation.failure("partial_output", partial_output=True),
        AttemptObservation.refusal("refusal"),
        AttemptObservation.failure("pending_approval", pending_approval=True),
        AttemptObservation.failure("timeout", timed_out=True),
        AttemptObservation.failure("cancel", cancelled=True),
        AttemptObservation.failure("ipc_loss"),
        AttemptObservation.failure("format_error"),
    ],
)
def test_replay_denied_after_observable_boundary(tmp_path, observation):
    ledger = GatewayFailoverLedger(tmp_path / "failover.db")
    turn = _turn()
    record = ledger.plan(turn, owner="gateway:3")
    ledger.start_leg(record.coordinator_id, 0, turn=turn, owner="gateway:3", generation=3)
    outcome = ledger.finish_leg(
        record.coordinator_id, 0, observation, owner="gateway:3", generation=3
    )
    assert outcome.replayable is False
    with pytest.raises(ReplayDenied):
        ledger.next_leg(record.coordinator_id, owner="gateway:3", generation=3)


def test_owner_lifecycle_revokes_child_and_late_callbacks(tmp_path):
    ledger = GatewayFailoverLedger(tmp_path / "failover.db")
    turn = _turn()
    record = ledger.plan(turn, owner="gateway:3")
    dispatch = ledger.start_leg(record.coordinator_id, 0, turn=turn, owner="gateway:3", generation=3)
    owner = _Owner()
    execution = OwnerTurnExecution(
        agent=owner,
        messages=[],
        dispatch=dispatch,
        effective_task_id="session-1",
        turn_id="turn-1",
        coordinator_id=record.coordinator_id,
        attempt_id=f"{record.coordinator_id}:0",
        generation=3,
        snapshot_digest=dispatch.prompt_digest,
        identity=("talos", "slack", "C1", "U1", "session-1"),
        expected_identity=("talos", "slack", "C1", "U1", "session-1"),
    )
    bridge = OwnerToolBridge.open(dispatch, execution, runtime_root=tmp_path)
    bridge.revoke("stop")
    assert execution.cancellation.is_set()
    assert bridge.terminated.wait(timeout=1)
    assert owner.calls == 0
    bridge.close()


def test_success_is_rejected_while_owner_tool_result_is_unresolved(tmp_path):
    ledger = GatewayFailoverLedger(tmp_path / "failover.db")
    turn = _turn()
    record = ledger.plan(turn, owner="gateway:3")
    ledger.start_leg(
        record.coordinator_id, 0, turn=turn, owner="gateway:3", generation=3
    )
    ledger.record_tool_intent(
        record.coordinator_id,
        0,
        "call-1",
        "todo",
        {},
        owner="gateway:3",
        generation=3,
    )

    with pytest.raises(ReplayDenied, match="unresolved owner tool call"):
        ledger.finish_leg(
            record.coordinator_id,
            0,
            AttemptObservation.success("final"),
            owner="gateway:3",
            generation=3,
        )


@pytest.mark.parametrize("crash_point", ["before_send", "after_claim", "after_send_lost_ack", "after_receipt"])
def test_final_delivery_restart_receipt_and_transcript(tmp_path, crash_point):
    ledger = GatewayFailoverLedger(tmp_path / "failover.db")
    turn = _turn()
    record = ledger.plan(turn, owner="gateway:3")
    ledger.start_leg(record.coordinator_id, 0, turn=turn, owner="gateway:3", generation=3)
    ledger.finish_leg(
        record.coordinator_id,
        0,
        AttemptObservation.success("final"),
        owner="gateway:3",
        generation=3,
    )
    delivery = OwnerFinalDelivery(
        ledger=ledger,
        coordinator_id=record.coordinator_id,
        owner="gateway:3",
        generation=3,
        payload="final",
        parts=("final",),
    )
    if crash_point == "before_send":
        assert delivery.claim() == ("final",)
        return
    assert delivery.claim() == ("final",)
    if crash_point in {"after_claim", "after_send_lost_ack"}:
        restarted = OwnerFinalDelivery(
            ledger=GatewayFailoverLedger(tmp_path / "failover.db"),
            coordinator_id=record.coordinator_id,
            owner="gateway:3",
            generation=3,
            payload="final",
            parts=("final",),
        )
        assert restarted.reconcile_after_restart() == "ambiguous"
        assert restarted.claim() == ()
        return
    delivery.record_receipt(0, "slack-ts-1")
    delivery.complete()
    assert delivery.state() == "delivered"
    assert delivery.claim() == ()


def test_final_delivery_records_each_transport_chunk_receipt(tmp_path):
    ledger = GatewayFailoverLedger(tmp_path / "failover.db")
    turn = _turn()
    record = ledger.plan(turn, owner="gateway:3")
    ledger.start_leg(
        record.coordinator_id, 0, turn=turn, owner="gateway:3", generation=3
    )
    ledger.finish_leg(
        record.coordinator_id,
        0,
        AttemptObservation.success("firstsecond"),
        owner="gateway:3",
        generation=3,
    )
    delivery = OwnerFinalDelivery(
        ledger=ledger,
        coordinator_id=record.coordinator_id,
        owner="gateway:3",
        generation=3,
        payload="firstsecond",
        parts=("firstsecond",),
    )

    assert delivery.claim(("first", "second")) == ("first", "second")
    delivery.record_receipt(0, "slack-ts-1")
    with pytest.raises(ReplayDenied, match="missing part receipts"):
        delivery.complete()
    delivery.record_receipt(1, "slack-ts-2")
    delivery.complete()
    assert delivery.state() == "delivered"
