"""Behavioral contract for the gateway-owned failover coordinator ledger.

This file exercises the durable state machine only. The default Slack turn and
Claude subscription executor are intentionally not wired until this contract is
complete and reviewed.
"""

from __future__ import annotations

import pytest

from gateway.backend_failover import (
    AttemptObservation,
    FrozenTurn,
    GatewayFailoverLedger,
    OwnershipLost,
    ReplayDenied,
)


def _turn(*, generation: int = 7) -> FrozenTurn:
    return FrozenTurn.create(
        session_key="agent:talos:slack:channel:C1:thread:171.001",
        session_id="session-1",
        event_id="event-42",
        generation=generation,
        prompt={"system": "fixed", "history": [{"role": "user", "content": "hello"}]},
        tools=[{"function": {"name": "terminal"}}],
        media=[{"path": "/tmp/image.png", "sha256": "abc"}],
        legs=[
            {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol"},
            {"kind": "claude_code", "model": "claude-sonnet-5-5"},
            {"kind": "api", "provider": "deepseek", "model": "deepseek-flash"},
        ],
    )


@pytest.fixture
def ledger(tmp_path):
    return GatewayFailoverLedger(tmp_path / "failover.db")


def test_plan_is_idempotent_only_for_the_identical_frozen_turn(ledger):
    turn = _turn()
    planned = ledger.plan(turn, owner="gateway:100")
    replay = ledger.plan(turn, owner="gateway:100")

    assert replay.coordinator_id == planned.coordinator_id
    assert replay.prompt_digest == turn.prompt_digest
    assert replay.tool_digest == turn.tool_digest
    assert replay.media_digest == turn.media_digest

    changed = FrozenTurn.create(
        session_key=turn.session_key,
        session_id=turn.session_id,
        event_id=turn.event_id,
        generation=turn.generation,
        prompt={"system": "changed"},
        tools=turn.tools,
        media=turn.media,
        legs=turn.legs,
    )
    with pytest.raises(ReplayDenied, match="snapshot mismatch"):
        ledger.plan(changed, owner="gateway:100")


def test_frozen_turn_detaches_nested_mutable_input_before_digest_and_persist(ledger):
    prompt = {"history": [{"role": "user", "content": ["original"]}]}
    tools = [{"function": {"name": "terminal"}}]
    media = [{"path": "/tmp/image.png", "metadata": {"size": 3}}]
    turn = FrozenTurn.create(
        session_key="agent:talos:slack:channel:C1",
        session_id="session-mutable",
        event_id="event-mutable",
        generation=1,
        prompt=prompt,
        tools=tools,
        media=media,
        legs=[{"kind": "api", "provider": "deepseek", "model": "deepseek-flash"}],
    )
    prompt["history"][0]["content"].append("mutated")
    tools[0]["function"]["name"] = "write_file"
    media[0]["metadata"]["size"] = 999

    planned = ledger.plan(turn, owner="gateway:100")
    snapshot = ledger.load_snapshot(planned.coordinator_id)

    assert snapshot["prompt"] == {
        "history": [{"role": "user", "content": ["original"]}]
    }
    assert snapshot["tools"] == [{"function": {"name": "terminal"}}]
    assert snapshot["media"] == [
        {"path": "/tmp/image.png", "metadata": {"size": 3}}
    ]

    turn.prompt["history"][0]["content"].append("late mutation")
    with pytest.raises(ReplayDenied, match="mutated after creation"):
        ledger.plan(turn, owner="gateway:other")


def test_start_leg_rejects_post_plan_nested_snapshot_mutation(ledger):
    turn = _turn()
    planned = ledger.plan(turn, owner="gateway:100")
    turn.prompt["history"][0]["content"] = "rewritten after plan"
    turn.tools[0]["function"]["name"] = "write_file"
    turn.media[0]["sha256"] = "different"

    with pytest.raises(ReplayDenied, match="snapshot diverged before dispatch"):
        ledger.start_leg(
            planned.coordinator_id,
            0,
            turn=turn,
            owner="gateway:100",
            generation=7,
        )

    assert ledger.debug_attempts(planned.coordinator_id) == []


def test_start_leg_returns_the_exact_persisted_dispatch_bytes(ledger):
    turn = _turn()
    original_prompt = '{"history":[{"content":"hello","role":"user"}],"system":"fixed"}'
    original_tools = '[{"function":{"name":"terminal"}}]'
    original_media = '[{"path":"/tmp/image.png","sha256":"abc"}]'
    original_leg = '{"kind":"api","model":"gpt-6.1-sol","provider":"openai-codex"}'
    planned = ledger.plan(turn, owner="gateway:100")

    dispatch = ledger.start_leg(
        planned.coordinator_id,
        0,
        turn=turn,
        owner="gateway:100",
        generation=7,
    )

    assert dispatch.prompt_json == original_prompt
    assert dispatch.tools_json == original_tools
    assert dispatch.media_json == original_media
    assert dispatch.leg_json == original_leg
    assert dispatch.prompt_digest == turn.prompt_digest
    assert dispatch.tool_digest == turn.tool_digest
    assert dispatch.media_digest == turn.media_digest
    assert dispatch.chain_digest == turn.chain_digest


def test_only_pre_tool_pre_output_availability_failure_can_advance(ledger):
    turn = _turn()
    planned = ledger.plan(turn, owner="gateway:100")
    ledger.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )
    outcome = ledger.finish_leg(
        planned.coordinator_id,
        0,
        AttemptObservation.availability_failure("rate_limit"),
        owner="gateway:100",
        generation=7,
    )

    assert outcome.replayable is True
    assert ledger.next_leg(planned.coordinator_id, owner="gateway:100", generation=7) == 1
    assert ledger.debug_attempts(planned.coordinator_id) == [
        {
            "leg_index": 0,
            "state": "finished",
            "outcome": "failure",
            "reason": "rate_limit",
            "replayable": 1,
            "partial_output": 0,
            "tool_activity": 0,
            "pending_approval": 0,
            "timed_out": 0,
            "cancelled": 0,
        }
    ]


@pytest.mark.parametrize(
    "observation",
    [
        AttemptObservation.refusal("content_policy_blocked"),
        AttemptObservation.failure("server_error", partial_output=True),
        AttemptObservation.failure("server_error", tool_activity=True),
        AttemptObservation.failure("server_error", pending_approval=True),
        AttemptObservation.failure("timeout", timed_out=True),
        AttemptObservation.failure("cancelled", cancelled=True),
        AttemptObservation.failure("server_error"),
    ],
    ids=[
        "refusal",
        "partial-output",
        "tool-activity",
        "pending-approval",
        "timeout",
        "cancellation",
        "unclassified-failure",
    ],
)
def test_every_unsafe_failure_denies_replay(ledger, observation):
    turn = _turn()
    planned = ledger.plan(turn, owner="gateway:100")
    ledger.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )
    outcome = ledger.finish_leg(
        planned.coordinator_id,
        0,
        observation,
        owner="gateway:100",
        generation=7,
    )

    assert outcome.replayable is False
    with pytest.raises(ReplayDenied):
        ledger.next_leg(planned.coordinator_id, owner="gateway:100", generation=7)


def test_restart_with_started_attempt_requires_manual_reconciliation(tmp_path):
    path = tmp_path / "failover.db"
    first = GatewayFailoverLedger(path)
    turn = _turn()
    planned = first.plan(turn, owner="gateway:100")
    first.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )

    restarted = GatewayFailoverLedger(path)
    recovered = restarted.recover(
        planned.coordinator_id, owner="gateway:100", generation=7
    )

    assert recovered.state == "reconciliation_required"
    assert recovered.reason == "unfinished_attempt"
    with pytest.raises(ReplayDenied, match="unfinished attempt"):
        restarted.claim(
            planned.coordinator_id,
            owner="gateway:200",
            previous_owner="gateway:100",
            generation=7,
        )


def test_failed_restart_claim_persists_reconciliation_before_raising(tmp_path):
    path = tmp_path / "failover.db"
    first = GatewayFailoverLedger(path)
    turn = _turn()
    planned = first.plan(turn, owner="gateway:100")
    first.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )

    restarted = GatewayFailoverLedger(path)
    with pytest.raises(ReplayDenied, match="unfinished attempt"):
        restarted.claim(
            planned.coordinator_id,
            owner="gateway:200",
            previous_owner="gateway:100",
            generation=7,
        )

    assert restarted.recover(
        planned.coordinator_id, owner="gateway:100", generation=7
    ).state == "reconciliation_required"


def test_generation_and_owner_fence_every_transition(ledger):
    turn = _turn()
    planned = ledger.plan(turn, owner="gateway:100")

    with pytest.raises(OwnershipLost, match="generation"):
        ledger.start_leg(
            planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=8
        )
    with pytest.raises(OwnershipLost, match="owner"):
        ledger.start_leg(
            planned.coordinator_id, 0, turn=turn, owner="gateway:200", generation=7
        )
    with pytest.raises(OwnershipLost, match="owner"):
        ledger.recover(planned.coordinator_id, owner="gateway:200", generation=7)
    with pytest.raises(OwnershipLost, match="generation"):
        ledger.recover(planned.coordinator_id, owner="gateway:100", generation=8)
    with pytest.raises(OwnershipLost, match="owner"):
        ledger.claim(
            planned.coordinator_id,
            owner="gateway:300",
            previous_owner="gateway:wrong",
            generation=7,
        )

    ledger.cancel(planned.coordinator_id, owner="gateway:100", generation=7)
    with pytest.raises(ReplayDenied, match="cancelled"):
        ledger.start_leg(
            planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
        )


def test_cancelled_turn_rejects_late_finish(ledger):
    turn = _turn()
    planned = ledger.plan(turn, owner="gateway:100")
    ledger.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )
    ledger.cancel(planned.coordinator_id, owner="gateway:100", generation=7)

    with pytest.raises(ReplayDenied, match="running state"):
        ledger.finish_leg(
            planned.coordinator_id,
            0,
            AttemptObservation.success("late"),
            owner="gateway:100",
            generation=7,
        )


def test_one_event_cannot_be_replanned_under_a_different_generation(ledger):
    ledger.plan(_turn(generation=7), owner="gateway:100")
    with pytest.raises(ReplayDenied, match="generation"):
        ledger.plan(_turn(generation=8), owner="gateway:200")


def test_attempt_observation_rejects_malformed_success():
    with pytest.raises(ValueError, match="successful attempts require"):
        AttemptObservation(outcome="success", reason="success")


@pytest.mark.parametrize("reason", ["timeout", "cancelled", "refusal", "arbitrary"])
def test_availability_factory_cannot_self_certify_blocked_failures(reason):
    with pytest.raises(ValueError, match="not a replayable availability failure"):
        AttemptObservation.availability_failure(reason)


@pytest.mark.parametrize(
    ("reason", "evidence"),
    [
        ("refusal", {}),
        ("partial_output", {"partial_output": True}),
        ("tool_activity", {"tool_activity": True}),
        ("pending_approval", {"pending_approval": True}),
        ("timeout", {"timed_out": True}),
        ("cancellation", {"cancelled": True}),
        ("format_error", {}),
    ],
)
def test_public_constructor_cannot_self_certify_nonavailability_reason(
    reason, evidence
):
    with pytest.raises(ValueError, match="not a replayable availability failure"):
        AttemptObservation(
            outcome="failure",
            reason=reason,
            is_availability_failure=True,
            **evidence,
        )


@pytest.mark.parametrize(
    "reason",
    [
        "auth",
        "auth_permanent",
        "billing",
        "model_not_found",
        "overloaded",
        "rate_limit",
        "server_error",
        "upstream_rate_limit",
    ],
)
def test_legitimate_availability_reasons_remain_replayable(reason):
    assert AttemptObservation.availability_failure(reason).replayable is True


def test_final_delivery_can_be_claimed_exactly_once_across_restart(tmp_path):
    path = tmp_path / "failover.db"
    first = GatewayFailoverLedger(path)
    turn = _turn()
    planned = first.plan(turn, owner="gateway:100")
    first.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )
    first.finish_leg(
        planned.coordinator_id,
        0,
        AttemptObservation.success("final answer"),
        owner="gateway:100",
        generation=7,
    )

    assert first.claim_delivery(
        planned.coordinator_id, owner="gateway:100", generation=7
    ) == "final answer"
    assert first.claim_delivery(
        planned.coordinator_id, owner="gateway:100", generation=7
    ) is None

    restarted = GatewayFailoverLedger(path)
    recovered = restarted.recover(
        planned.coordinator_id, owner="gateway:100", generation=7
    )
    assert recovered.delivery_state == "ambiguous"
    assert recovered.reason == "unfinished_delivery"
    assert restarted.claim_delivery(
        planned.coordinator_id, owner="gateway:100", generation=7
    ) is None
    with pytest.raises(ReplayDenied, match="not claimed"):
        restarted.mark_delivered(
            planned.coordinator_id, owner="gateway:100", generation=7
        )


def test_restart_before_delivery_accepts_completed_output(tmp_path):
    path = tmp_path / "failover.db"
    first = GatewayFailoverLedger(path)
    turn = _turn()
    planned = first.plan(turn, owner="gateway:100")
    first.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )
    first.finish_leg(
        planned.coordinator_id,
        0,
        AttemptObservation.success("durable answer"),
        owner="gateway:100",
        generation=7,
    )

    restarted = GatewayFailoverLedger(path)
    restarted.claim(
        planned.coordinator_id,
        owner="gateway:200",
        previous_owner="gateway:100",
        generation=7,
    )
    recovered = restarted.recover(
        planned.coordinator_id, owner="gateway:200", generation=7
    )
    assert recovered.state == "completed"
    assert recovered.delivery_state == "pending"
    assert restarted.claim_delivery(
        planned.coordinator_id, owner="gateway:200", generation=7
    ) == "durable answer"


def test_confirmed_delivery_is_terminal_and_not_claimable_again(ledger):
    turn = _turn()
    planned = ledger.plan(turn, owner="gateway:100")
    ledger.start_leg(
        planned.coordinator_id, 0, turn=turn, owner="gateway:100", generation=7
    )
    ledger.finish_leg(
        planned.coordinator_id,
        0,
        AttemptObservation.success("delivered answer"),
        owner="gateway:100",
        generation=7,
    )
    assert ledger.claim_delivery(
        planned.coordinator_id, owner="gateway:100", generation=7
    ) == "delivered answer"

    ledger.mark_delivered(
        planned.coordinator_id, owner="gateway:100", generation=7
    )

    assert ledger.recover(
        planned.coordinator_id, owner="gateway:100", generation=7
    ).delivery_state == "delivered"
    assert ledger.claim_delivery(
        planned.coordinator_id, owner="gateway:100", generation=7
    ) is None
