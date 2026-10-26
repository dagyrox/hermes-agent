"""Durable completion handoffs: real disposable SQLite, no source snapshots."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tests.hermes_cli import test_kanban_sentinel as processes
from tests.hermes_cli.test_kanban_sentinel import handoff_fixture, native_fixture  # noqa: F401


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    with kb.connect_closing() as conn:
        yield conn


def source(conn, **kwargs):
    task = kb.create_task(conn, title="handoff fixture", assignee="talos", **kwargs)
    assert kb.claim_task(conn, task, claimer="fixture")
    run = kb.get_task(conn, task).current_run_id
    return task, run


def claim_payload(**overrides):
    return {
        "v": 1, "repo": "fixture/project", "pr": 24,
        "head_sha": "a" * 40, "tree_sha": "b" * 40,
        "work_packet": "/fixture/packet.md", "next_action": "none",
        "review_task_id": None, "notes": "fixture",
        **overrides,
    }


def test_completion_and_transition_are_one_transaction(board):
    task, run = source(board)
    # A failed intent INSERT must roll back the terminal state and run too.
    board.execute("CREATE TRIGGER reject_intent BEFORE INSERT ON task_transitions "
                  "BEGIN SELECT RAISE(ABORT, 'fixture reject'); END")
    with pytest.raises(Exception, match="fixture reject"):
        kb.complete_task(board, task, summary="fixture", expected_run_id=run)
    assert kb.get_task(board, task).status == "running"
    assert not [e for e in kb.list_events(board, task) if e.kind == "completed"]
    board.execute("DROP TRIGGER reject_intent")
    assert kb.complete_task(board, task, summary="fixture", expected_run_id=run)
    rows = board.execute("SELECT * FROM task_transitions WHERE source_task_id=?", (task,)).fetchall()
    assert len(rows) == 1
    assert rows[0]["source_run_id"] == run
    assert rows[0]["state"] == "pending"
    assert not kb.complete_task(board, task, summary="duplicate", expected_run_id=run)
    assert board.execute("SELECT COUNT(*) FROM task_transitions").fetchone()[0] == 1


def test_pending_handoff_preserves_managed_scratch_until_successor_completes(board):
    task, run = source(board)
    source_task = kb.get_task(board, task)
    assert source_task is not None
    workspace = kb.resolve_workspace(source_task)
    kb.set_workspace_path(board, task, workspace)
    artifact = workspace / "handoff-proof.txt"
    artifact.write_text("bounded fixture artifact", encoding="utf-8")
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    assert artifact.exists()
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    kb.consume_transition(board, key)
    successor = board.execute("SELECT successor_task_id FROM task_transitions WHERE transition_key=?", (key,)).fetchone()[0]
    successor_task = kb.get_task(board, successor)
    assert successor_task is not None and successor_task.workspace_path == str(workspace)
    assert artifact.exists()
    kb.complete_task(board, successor, summary="fixture read source artifact")
    assert not workspace.exists()


def test_plain_completion_consumes_once_without_successor(board):
    task, run = source(board)
    assert kb.complete_task(board, task, summary="no next action")
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "consumed" and row["outcome"] == "no_successor"
    assert not kb.consume_transition(board, key)
    assert row["attempt_count"] == 1
    assert board.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


@pytest.mark.parametrize("bad", [
    {"approved": True}, {"head_sha": "bad"}, {"v": True}, {"pr": True},
    {"notes": "x" * 2001}, {"work_packet": "relative.md"},
    {"work_packet": "/fixture/../packet.md"}, {"next_action": "merge"},
])
def test_untrusted_claim_fails_closed_without_losing_completion(board, bad):
    task, run = source(board)
    assert kb.complete_task(board, task, metadata={"transition": claim_payload(**bad)})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "blocked" and row["block_code"] == "schema_invalid"
    assert kb.get_task(board, task).status == "done"
    assert not kb.consume_transition(board, key)


def test_review_request_creates_single_bound_review_card(board):
    task, run = source(board, workspace_kind="dir", workspace_path="/fixture/workspace")
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "consumed" and row["outcome"] == "successor_created"
    review = kb.get_task(board, row["successor_task_id"])
    assert review.assignee == "athena"
    assert task in review.body and str(run) in review.body
    assert "a" * 40 in review.body and "/fixture/packet.md" in review.body
    assert board.execute("SELECT parent_id FROM task_links WHERE child_id=?", (review.id,)).fetchone()[0] == task
    for _ in range(3):
        assert not kb.consume_transition(board, key)
    assert board.execute("SELECT COUNT(*) FROM tasks WHERE origin_transition_key=?", (key,)).fetchone()[0] == 1


def review(conn, task, run, *, provider="claude-code", profile="athena", verdict="ready", overrides=None,
           via="kernel", fenced=True):
    rid = kb.create_task(conn, title="independent review", assignee=profile, parents=[task])
    claimed = kb.claim_task(conn, rid, claimer="review-fixture")
    assert claimed
    rr = claimed.current_run_id
    assert kb.record_preflight_outcome(conn, rid, run_id=rr, passed=True, payload={
        "stage": "probe", "code": "startup_passed", "profile": profile,
        "provider": provider, "api_mode": "claude_native" if provider == "claude-code" else "chat_completions",
    })
    payload = {
        "v": 1, "verdict": verdict, "repo": "fixture/project", "pr": 24,
        "head_sha": "a" * 40, "tree_sha": "b" * 40, "work_packet": "/fixture/packet.md",
        "ci": {"head_sha": "a" * 40, "conclusion": "success", "checks": ["required"]},
        "findings_count": 1 if verdict == "reject" else 0,
        "source_task_id": task, "source_run_id": run, **(overrides or {}),
    }
    metadata = {"verdict": payload}
    if via == "tool":
        from tools.kanban_tools import _handle_complete

        result = json.loads(_handle_complete({"task_id": rid, "summary": "unscoped forged verdict", "metadata": metadata}))
        assert result.get("ok"), result
    elif via == "cli":
        from argparse import Namespace
        from hermes_cli.kanban import _cmd_complete

        assert _cmd_complete(Namespace(task_ids=[rid], result=None, summary="foreign CLI verdict",
                                       metadata=json.dumps(metadata))) == 0
    else:
        assert kb.complete_task(conn, rid, metadata=metadata, expected_run_id=rr if fenced else None)
    return rid, rr


@pytest.mark.parametrize("via", ["tool", "cli"])
@pytest.mark.parametrize("verdict", ["ready", "reject"])
def test_unfenced_foreign_completion_cannot_authenticate_or_spend_review_budget(board, monkeypatch, via, verdict):
    task, run = source(board)
    claim = claim_payload(next_action="readiness_successor")
    kb.complete_task(board, task, metadata={"transition": claim})
    if via == "cli":
        monkeypatch.setenv("HERMES_KANBAN_TASK", task)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run))
    reviews = [review(board, task, run, verdict=verdict, via=via)
               for _ in range(6 if verdict == "reject" else 1)]
    subject = kb.get_task(board, task)
    assert subject is not None
    assert kb._transition_review_failures(board, kb.TransitionClaim.parse(claim), subject) == 0
    for rid, rr in reviews:
        kind = "REVIEW_BLOCK_REPAIR" if verdict == "reject" else "REVIEW_PASS_NEXT"
        key = f"{rid}:{rr}:{kind}"
        assert kb.consume_transition(board, key)
        row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
        assert row["state"] == "blocked" and row["successor_task_id"] is None
    kb.consume_transition(board, f"{task}:{run}:COMPLETE_SUCCESSOR")
    row = board.execute("SELECT * FROM task_transitions WHERE source_task_id=?", (task,)).fetchone()
    assert row["block_code"] == "review_unauthenticated" and row["successor_task_id"] is None


@pytest.mark.parametrize("original", ["ready", "reject"])
def test_administrative_metadata_edit_cannot_reseal_authenticated_review(board, original):
    task, run = source(board)
    claim = claim_payload(next_action="readiness_successor")
    kb.complete_task(board, task, metadata={"transition": claim})
    rid, rr = review(board, task, run, verdict=original)
    metadata = json.loads(board.execute("SELECT metadata FROM task_runs WHERE id=?", (rr,)).fetchone()[0])
    metadata["verdict"].update(verdict="reject" if original == "ready" else "ready",
                               findings_count=1 if original == "ready" else 0)
    assert kb.edit_completed_task_result(board, rid, result="edited claim", metadata=metadata)
    subject = kb.get_task(board, task)
    assert subject is not None
    assert kb._transition_review_failures(board, kb.TransitionClaim.parse(claim), subject) == 0
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["block_code"] == "review_unauthenticated" and row["successor_task_id"] is None


def test_authenticated_review_is_readiness_only_and_two_triggers_share_effect(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    rid, rr = review(board, task, run)
    keys = [f"{task}:{run}:COMPLETE_SUCCESSOR", f"{rid}:{rr}:REVIEW_PASS_NEXT"]
    for key in keys:
        assert kb.consume_transition(board, key)
    rows = board.execute("SELECT * FROM task_transitions WHERE transition_key IN (?,?)", keys).fetchall()
    assert all(r["state"] == "consumed" for r in rows)
    assert len({r["successor_task_id"] for r in rows}) == 1
    successor = kb.get_task(board, rows[0]["successor_task_id"])
    assert successor.assignee == "talos"
    assert '"authority": "readiness_only"' in successor.body
    assert '"requires_live_exact_head_recheck": true' in successor.body
    assert '"review_run_id": ' + str(rr) in successor.body


@pytest.mark.parametrize("options,code", [
    ({"provider": "custom:fixture"}, "review_unauthenticated"),
    ({"profile": "talos"}, "review_unauthenticated"),
    ({"overrides": {"head_sha": "c" * 40}}, "binding_mismatch"),
    ({"overrides": {"work_packet": "/different.md"}}, "binding_mismatch"),
    ({"overrides": {"source_run_id": 99999}}, "review_stale"),
    ({"overrides": {"ci": {"head_sha": "a" * 40, "conclusion": "pending", "checks": []}}}, "ci_not_success"),
    ({"verdict": "needs_input"}, "verdict_needs_input"),
    ({"overrides": {"approved": True}}, "schema_invalid"),
])
def test_review_claims_cannot_grant_authority(board, options, code):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    rid, rr = review(board, task, run, **options)
    key = f"{rid}:{rr}:REVIEW_PASS_NEXT"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "blocked" and row["block_code"] == code
    assert not row["successor_task_id"]


def test_review_failures_one_to_five_repair_six_escalates_across_heads(board):
    for failure in range(1, 7):
        task, run = source(board)
        head = f"{failure:040x}"
        kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="repair_requested", head_sha=head)})
        rid, rr = review(board, task, run, verdict="reject", overrides={
            "head_sha": head, "ci": {"head_sha": head, "conclusion": "success", "checks": ["required"]},
        })
        key = f"{rid}:{rr}:REVIEW_BLOCK_REPAIR"
        kb.consume_transition(board, key)
        row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
        if failure <= 5:
            assert row["state"] == "consumed", dict(row)
            repair = kb.get_task(board, row["successor_task_id"])
            assert repair.assignee == "talos" and "repair_requested" in repair.body
            kb.complete_task(board, repair.id, summary="fixture repair completed")
        else:
            assert row["state"] == "blocked" and row["block_code"] == "review_ceiling"
            assert not row["successor_task_id"]


def settings(**handoff):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(exist_ok=True, parents=True)
    home.joinpath("config.yaml").write_text(json.dumps({"kanban": {"handoff": handoff}}))


def test_sweep_modes_and_dispatch_dry_run_preserve_no_effect(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    assert kb.sweep_transitions(board) == 0  # default-off rollout
    settings(typed_guard="shadow")
    assert kb.sweep_transitions(board) == 0
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "pending" and row["attempt_count"] == 0
    assert "review_requested" in row["decision_json"]
    settings(typed_guard="enforce")
    kb.dispatch_once(board, dry_run=True, max_spawn=1)
    assert board.execute("SELECT state FROM task_transitions WHERE transition_key=?", (key,)).fetchone()[0] == "pending"
    assert kb.sweep_transitions(board) == 1
    assert board.execute("SELECT state FROM task_transitions WHERE transition_key=?", (key,)).fetchone()[0] == "consumed"


@pytest.mark.parametrize("gate", ["approval_required", "auth_or_credential", "sensitive_action", "ambiguous_write", "ambiguous_input"])
def test_hard_gate_unblock_is_audit_not_permission(board, gate):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested"), "gate": gate})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    kb.consume_transition(board, key)
    for _ in range(3):
        assert not kb.consume_transition(board, key)
    assert kb.unblock_task(board, task, actor="operator", reason="audit only; not a grant")
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "pending"
    kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "blocked" and row["block_code"] == gate
    assert row["block_recurrences"] == 2
    assert kb.get_task(board, task).status == "done"  # never respawn the completed builder
    audit = [e for e in kb.list_events(board, task) if e.kind == "transition_unblocked"]
    assert audit[0].payload["actor"] == "operator"
    assert audit[0].payload["prior_gate_code"] == gate
    assert any(e.kind == "transition_triage" for e in kb.list_events(board, task))


def test_new_authenticated_run_resolves_review_gate_only_after_audited_unblock(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    kb.consume_transition(board, key)
    review(board, task, run)
    assert not kb.consume_transition(board, key)
    assert kb.unblock_task(board, task, actor="operator", reason="new review run available")
    assert kb.consume_transition(board, key)
    assert board.execute("SELECT state FROM task_transitions WHERE transition_key=?", (key,)).fetchone()[0] == "consumed"


def test_effect_exception_rolls_back_then_exhausts_without_leaking_secret(board):
    task, run = source(board, max_retries=2)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    board.execute("CREATE TRIGGER reject_successor BEFORE INSERT ON tasks "
                  "WHEN NEW.origin_transition_key IS NOT NULL BEGIN SELECT RAISE(ABORT, 'SECRET_CANARY'); END")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="consumer_error") as caught:
            kb.consume_transition(board, key)
        assert "SECRET_CANARY" not in str(caught.value)
        assert board.execute("SELECT COUNT(*) FROM tasks WHERE origin_transition_key=?", (key,)).fetchone()[0] == 0
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "blocked" and row["block_code"] == "retry_exhausted"
    assert row["last_error"] == "consumer_error"
    assert row["attempt_count"] == 2
    assert not kb.consume_transition(board, key)


def test_named_crash_points_are_inert_without_test_setting(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TEST_CRASHPOINT", raising=False)
    def forbidden(_code):
        pytest.fail("production crash point was not inert")
    monkeypatch.setattr(os, "_exit", forbidden)
    for name in ("after_complete_commit", "after_claim", "after_decision", "in_effect_before_commit", "after_effect_commit"):
        kb._transition_crashpoint(name)


def test_live_owner_and_expired_owner_cannot_duplicate_effect(board, monkeypatch):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    now = kb.time.time()
    monkeypatch.setattr(kb.time, "time", lambda: now)
    assert kb.claim_transition(board, key, owner="A", ttl_seconds=1)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    decision = kb._evaluate_transition(board, row)
    assert not kb.claim_transition(board, key, owner="B", ttl_seconds=1)
    monkeypatch.setattr(kb.time, "time", lambda: now + 2)
    assert kb.claim_transition(board, key, owner="B", ttl_seconds=10)
    assert kb._apply_transition_decision(board, key, "B", decision)
    assert not kb._apply_transition_decision(board, key, "A", decision)
    assert board.execute("SELECT COUNT(*) FROM tasks WHERE origin_transition_key=?", (key,)).fetchone()[0] == 1


def test_sixth_review_counts_across_real_head_changes(board):
    for failure in range(1, 7):
        task, run = source(board)
        head = format(failure, "040x")
        kb.complete_task(board, task, metadata={"transition": claim_payload(head_sha=head, next_action="repair_requested")})
        rid, rr = review(board, task, run, verdict="reject", overrides={
            "head_sha": head, "ci": {"head_sha": head, "conclusion": "success", "checks": ["required"]},
        })
        kb.consume_transition(board, f"{rid}:{rr}:REVIEW_BLOCK_REPAIR")
        row = board.execute("SELECT * FROM task_transitions WHERE source_run_id=?", (rr,)).fetchone()
        if failure <= 5:
            assert row["state"] == "consumed"
            kb.complete_task(board, row["successor_task_id"], summary="repair")
        else:
            assert row["block_code"] == "review_ceiling"


def test_none_never_becomes_action_from_unsolicited_review(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload()})
    rid, rr = review(board, task, run)
    kb.consume_transition(board, f"{rid}:{rr}:REVIEW_PASS_NEXT")
    row = board.execute("SELECT * FROM task_transitions WHERE source_run_id=?", (rr,)).fetchone()
    assert row["state"] == "consumed" and row["outcome"] == "no_successor"


def test_active_workspace_writer_blocks_other_action(board):
    workspace = str(Path(os.environ["HERMES_HOME"]) / "workspace")
    task, run = source(board, workspace_kind="dir", workspace_path=workspace)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    kb.create_task(board, title="still writing", assignee="talos", workspace_kind="dir", workspace_path=workspace)
    kb.consume_transition(board, f"{task}:{run}:COMPLETE_SUCCESSOR")
    row = board.execute("SELECT * FROM task_transitions WHERE source_run_id=?", (run,)).fetchone()
    assert row["state"] == "blocked" and row["block_code"] == "active_writer_hold"
    assert not row["successor_task_id"]


@pytest.mark.parametrize("location", ["body", "comment"])
def test_manual_writer_in_another_worktree_blocks_same_pr_handoff(board, location):
    task, run = source(board, workspace_kind="dir", workspace_path="/fixture/source")
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    url = "https://github.com/fixture/project/pull/24"
    writer = kb.create_task(board, title="manual existing PR repair", assignee="talos",
                            body=url if location == "body" else "repair",
                            workspace_kind="dir", workspace_path="/fixture/another-worktree")
    assert kb.claim_task(board, writer, claimer="manual")
    if location == "comment":
        kb.add_comment(board, writer, "talos", url)
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "blocked" and row["block_code"] == "active_writer_hold"
    assert row["successor_task_id"] is None


@pytest.mark.parametrize("location", ["workspace", "body", "comment"])
def test_active_writer_hold_recovers_after_exit_without_replaying_producer(board, location):
    workspace = "/fixture/source"
    task, run = source(board, workspace_kind="dir", workspace_path=workspace, max_retries=3)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    url = "https://github.com/fixture/project/pull/24"
    writer = kb.create_task(board, title="temporary writer", assignee="athena",
                            body=url if location == "body" else "fixture",
                            workspace_kind="dir", workspace_path=workspace if location == "workspace" else "/fixture/other")
    if location == "comment":
        kb.add_comment(board, writer, "athena", url)
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    for _ in range(2):
        assert kb.consume_transition(board, key)
        row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
        assert row["state"] == "blocked" and row["successor_task_id"] is None
        assert kb.unblock_task(board, task, actor="operator", reason="re-evaluate current writer")
    assert kb.complete_task(board, writer, summary="writer finished")
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "consumed" and row["successor_task_id"] is not None
    assert row["attempt_count"] == 3
    assert board.execute("SELECT COUNT(*) FROM tasks WHERE origin_transition_key=?", (key,)).fetchone()[0] == 1
    assert not kb.consume_transition(board, key)
    blocked = [e for e in kb.list_events(board, task) if e.kind == "transition_blocked"]
    assert all(e.payload and e.payload["code"] == "active_writer_hold" and e.payload["owner"] == "athena" for e in blocked)
    source_task = kb.get_task(board, task)
    assert source_task is not None and source_task.status == "done"


def test_crashed_claims_cannot_exceed_existing_attempt_budget(board, monkeypatch):
    task, run = source(board, max_retries=1)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="review_requested")})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    now = kb.time.time()
    monkeypatch.setattr(kb.time, "time", lambda: now)
    assert kb.claim_transition(board, key, owner="died", ttl_seconds=1)
    monkeypatch.setattr(kb.time, "time", lambda: now + 2)
    kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["state"] == "blocked" and row["block_code"] == "retry_exhausted"
    assert not row["successor_task_id"]


def test_old_review_transition_cannot_override_newer_reject(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    rid, old = review(board, task, run)
    key = f"{rid}:{old}:REVIEW_PASS_NEXT"
    board.execute("UPDATE tasks SET status='ready',claim_lock=NULL WHERE id=?", (rid,))
    newer = kb.claim_task(board, rid, claimer="new-review")
    kb.record_preflight_outcome(board, rid, run_id=newer.current_run_id, passed=True, payload={
        "profile": "athena", "provider": "claude-code", "api_mode": "claude_native", "stage": "probe", "code": "startup_passed",
    })
    payload = json.loads(board.execute("SELECT metadata FROM task_runs WHERE id=?", (old,)).fetchone()[0])
    payload["verdict"]["verdict"] = "reject"
    payload["verdict"]["findings_count"] = 1
    kb.complete_task(board, rid, metadata=payload)
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["block_code"] == "review_stale" and row["successor_task_id"] is None


@pytest.mark.parametrize("gate", ["approval_required", "auth_or_credential", "sensitive_action", "ambiguous_write", "ambiguous_input"])
def test_review_side_hard_gate_blocks_source_and_review_consumers(board, gate):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    rid, rr = review(board, task, run)
    payload = json.loads(board.execute("SELECT metadata FROM task_runs WHERE id=?", (rr,)).fetchone()[0])
    payload["gate"] = gate
    board.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps(payload), rr))
    for key in (f"{task}:{run}:COMPLETE_SUCCESSOR", f"{rid}:{rr}:REVIEW_PASS_NEXT"):
        assert kb.consume_transition(board, key)
        row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
        assert row["block_code"] == gate and row["successor_task_id"] is None


def test_preflight_inserted_after_completed_event_is_not_authority(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    rid, rr = review(board, task, run, provider="custom:fixture")
    board.execute("DELETE FROM task_events WHERE task_id=? AND kind='preflight_passed'", (rid,))
    kb.record_preflight_outcome(board, rid, run_id=rr, passed=True, payload={
        "profile": "athena", "provider": "claude-code", "api_mode": "claude_native", "stage": "probe", "code": "startup_passed",
    })
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["block_code"] == "review_unauthenticated"


def test_authenticated_reject_with_failed_ci_still_requests_bounded_repair(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="repair_requested")})
    rid, rr = review(board, task, run, verdict="reject", overrides={
        "ci": {"head_sha": "a" * 40, "conclusion": "failure", "checks": ["required"]},
    })
    key = f"{rid}:{rr}:REVIEW_BLOCK_REPAIR"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert json.loads(row["decision_json"])["code"] == "repair_requested"


def test_prior_gate_does_not_refill_attempt_budget(board):
    task, run = source(board, max_retries=1)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    kb.consume_transition(board, key)
    review(board, task, run)
    kb.unblock_task(board, task, actor="operator", reason="new authenticated review")
    kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["block_code"] == "retry_exhausted" and row["successor_task_id"] is None


def test_optional_tree_and_readonly_diagnostics(board):
    task, run = source(board)
    payload = claim_payload(next_action="review_requested")
    payload.pop("tree_sha")
    kb.complete_task(board, task, metadata={"transition": payload})
    key = f"{task}:{run}:COMPLETE_SUCCESSOR"
    kb.consume_transition(board, key)
    diagnostics = kb.list_transitions(board, task)
    assert diagnostics[0]["state"] == "consumed"
    assert "lease_owner" not in diagnostics[0]
    assert kb.list_transitions(board, task) == diagnostics


def test_ready_with_unresolved_findings_is_not_readiness(board):
    task, run = source(board)
    kb.complete_task(board, task, metadata={"transition": claim_payload(next_action="readiness_successor")})
    rid, rr = review(board, task, run, overrides={"findings_count": 1})
    key = f"{rid}:{rr}:REVIEW_PASS_NEXT"
    kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["block_code"] == "verdict_reject" and row["successor_task_id"] is None


@pytest.mark.parametrize("explicit", [False, True])
def test_old_review_card_cannot_bypass_newer_review_card(board, explicit):
    task, run = source(board)
    payload = claim_payload(next_action="readiness_successor")
    kb.complete_task(board, task, metadata={"transition": payload})
    old, rr = review(board, task, run)
    newer, _ = review(board, task, run, verdict="reject")
    if explicit:
        payload["review_task_id"] = newer
        board.execute("UPDATE task_runs SET metadata=? WHERE id=?", (json.dumps({"transition": payload}), run))
    key = f"{old}:{rr}:REVIEW_PASS_NEXT"
    assert kb.consume_transition(board, key)
    row = board.execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    assert row["block_code"] == "review_stale" and row["successor_task_id"] is None


# Separate the repeated race from other process scenarios to respect the
# canonical runner's 300s per-file cap without changing CI configuration.
@pytest.mark.parametrize("iteration", range(20))
def test_r6_hook_sweep_race_single_effect_and_useful_worker(handoff_fixture, tmp_path, iteration):
    f = handoff_fixture
    row = processes.start_source(f)
    key = row["transition_key"]
    processes.configure(f, lease=10)
    release = tmp_path / "race-release"
    left_ready, right_ready = tmp_path / "left-ready", tmp_path / "right-ready"
    left = processes.barrier_process(f, "pass # BARRIER k._nudge_transition(c, " + repr(key) + ")", left_ready, release)
    right = processes.barrier_process(f, "pass # BARRIER k.sweep_transitions(c)", right_ready, release)
    try:
        processes.wait_for(lambda: left_ready.exists() and right_ready.exists(), "two consumer barriers")
        release.touch()
        for process in (left, right):
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
        claims = [e for e in kb.list_events(f["conn"], f["task_id"]) if e.kind == "transition_claimed"]
        assert len(claims) == 1
        assert processes.transition(f, key)["attempt_count"] == 1
        processes.tick(f)
        processes.useful_successor(f, key)
        processes.report(f, "R6-" + str(iteration), key)
    finally:
        for process in (left, right):
            if process.poll() is None:
                process.kill()
                process.wait()
