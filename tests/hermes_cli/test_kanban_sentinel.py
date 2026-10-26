"""Feature2 process recovery, not a Feature6 Sentinel implementation.

A scripted loopback provider drives the unmodified real worker entrypoint and
real tools. No test substitutes spawn_fn, cli_main or the terminal command.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

from tests.hermes_cli.test_kanban_worker_preflight_e2e import native_fixture  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX process kill/restart acceptance")


def wait_for(predicate, description, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    pytest.fail("timed out: " + description)


@pytest.fixture
def handoff_fixture(native_fixture, tmp_path):
    f = native_fixture
    assert Path(os.environ["HERMES_HOME"]).resolve().is_relative_to(tmp_path)
    assert Path(os.environ["HERMES_KANBAN_DB"]).resolve().is_relative_to(tmp_path)
    f["state"]["status"] = 200
    f["state"]["handoff"] = True
    for name in ("athena", "talos"):
        shutil.copytree(f["profile"], f["home"] / "profiles" / name)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=f["workspace"], text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=f["workspace"], text=True).strip()
    f["claim"] = {
        "v": 1, "repo": "fixture/project", "pr": 24, "head_sha": head, "tree_sha": tree,
        "work_packet": str(tmp_path / "packet.md"), "next_action": "review_requested",
        "review_task_id": None, "notes": "disposable process fixture; no approval grant",
    }
    f["state"]["metadata_by_task"] = {f["task_id"]: {"transition": f["claim"]}}
    yield f
    for row in f["conn"].execute("SELECT worker_pid FROM tasks WHERE worker_pid IS NOT NULL"):
        try:
            os.kill(row[0], signal.SIGTERM)
        except ProcessLookupError:
            pass


def configure(f, mode="enforce", lease=2):
    f["home"].joinpath("config.yaml").write_text(yaml.safe_dump({
        "kanban": {"handoff": {"typed_guard": mode, "lease_seconds": lease}},
    }))


def tick(f, crashpoint=None):
    env = dict(os.environ)
    env.pop("HERMES_KANBAN_TEST_CRASHPOINT", None)
    if crashpoint:
        env["HERMES_KANBAN_TEST_CRASHPOINT"] = crashpoint
    result = subprocess.run(
        [sys.executable, "-c", "from hermes_cli import kanban_db as k; "
         "c=k.connect(); r=k.dispatch_once(c, max_spawn=1); print(r); c.close()"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60,
    )
    if not crashpoint:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def transition(f, key=None):
    if key is None:
        row = f["conn"].execute("SELECT * FROM task_transitions WHERE source_task_id=? ORDER BY source_run_id LIMIT 1", (f["task_id"],)).fetchone()
    else:
        row = f["conn"].execute("SELECT * FROM task_transitions WHERE transition_key=?", (key,)).fetchone()
    return dict(row) if row else None


def useful_successor(f, key):
    row = transition(f, key)
    assert row["state"] == "consumed" and row["outcome"] == "successor_created", row
    successor_id = row["successor_task_id"]
    kb, conn = f["kb"], f["conn"]
    def terminal():
        status = kb.get_task(conn, successor_id).status
        if status == "ready":
            # The real source may still be exiting after committing done.
            # Only real dispatcher ticks may admit the next worker.
            tick(f)
        return status in {"done", "blocked", "triage"}
    wait_for(terminal, "useful successor")
    task = kb.get_task(conn, successor_id)
    assert task.status == "done", kb.worker_log_path(successor_id).read_text()[-4000:]
    assert f["task_id"] in task.body
    assert f["task_id"] in kb.parent_ids(conn, successor_id)
    events = kb.list_events(conn, successor_id)
    kinds = [e.kind for e in events]
    assert {"claimed", "spawned", "preflight_passed", "heartbeat", "completed"} <= set(kinds)
    assert kinds.index("preflight_passed") < kinds.index("completed")
    spawned = [e for e in events if e.kind == "spawned"]
    assert len(spawned) == 1 and spawned[0].payload["pid"] > 0
    proof = f["workspace"] / ("merge-readiness-" + successor_id + ".json")
    artifact = json.loads(proof.read_text())
    assert artifact["task_id"] == successor_id
    assert artifact["workspace"] == str(f["workspace"])
    origin = conn.execute("SELECT origin_transition_key FROM tasks WHERE id=?", (successor_id,)).fetchone()[0]
    # Source-completion and review-completion can race to the same semantic
    # action. The first committed origin owns it; the other must adopt it.
    assert origin and conn.execute("SELECT COUNT(*) FROM tasks WHERE origin_transition_key=?", (origin,)).fetchone()[0] == 1
    runs = kb.list_runs(conn, successor_id)
    assert len(runs) == 1 and runs[0].outcome == "completed"
    assert runs[0].metadata["artifact_path"] == str(proof)
    notes = subprocess.check_output(["git", "notes", "--ref=fixture", "show"], cwd=f["workspace"], text=True)
    assert notes.splitlines().count(successor_id) == 1
    return successor_id


def test_automatic_completion_handoff_executes_useful_real_worker(handoff_fixture):
    f = handoff_fixture
    f["launch"]()
    assert f["await_terminal"]().status == "done"
    row = transition(f)
    assert row["state"] == "pending"
    configure(f)
    tick(f)
    useful_successor(f, row["transition_key"])


def verdict_payload(f, run):
    return {
        **{k: f["claim"][k] for k in ("v", "repo", "pr", "head_sha", "tree_sha", "work_packet")},
        "verdict": "ready", "ci": {"head_sha": f["claim"]["head_sha"], "conclusion": "success", "checks": ["fixture-required"]},
        "findings_count": 0, "source_task_id": f["task_id"], "source_run_id": run,
    }


def seed_authenticated_review(f, *, verdict="ready"):
    """Synthetic trusted-event rows via kernel APIs, NOT live Claude evidence."""
    kb, conn = f["kb"], f["conn"]
    run = kb.list_runs(conn, f["task_id"])[-1].id
    rid = kb.create_task(conn, title="kernel-seeded authenticated fixture", assignee="athena", parents=[f["task_id"]])
    claimed = kb.claim_task(conn, rid, claimer="fixture-seed")
    rr = claimed.current_run_id
    kb.record_preflight_outcome(conn, rid, run_id=rr, passed=True, payload={
        "stage": "probe", "code": "startup_passed", "profile": "athena", "provider": "claude-code", "api_mode": "claude_native",
    })
    payload = verdict_payload(f, run)
    payload.update(verdict=verdict, findings_count=1 if verdict == "reject" else 0)
    kb.complete_task(conn, rid, metadata={"verdict": payload}, expected_run_id=rr)
    return rr


def start_source(f, monkeypatch=None, crashpoint=None):
    if crashpoint:
        monkeypatch.setenv("HERMES_KANBAN_TEST_CRASHPOINT", crashpoint)
    result = f["launch"]()
    assert result.spawned
    assert f["await_terminal"]().status == "done"
    if crashpoint:
        monkeypatch.delenv("HERMES_KANBAN_TEST_CRASHPOINT")
    return transition(f)


def report(f, name, key):
    row = transition(f, key)
    events = f["kb"].list_events(f["conn"], row["successor_task_id"]) if row["successor_task_id"] else []
    print("FEATURE2_EVIDENCE " + json.dumps({
        "scenario": name, "transition": row,
        "successor_events": [{"kind": e.kind, "run_id": e.run_id, "payload": e.payload} for e in events],
    }, sort_keys=True))


def test_r1_after_completion_commit_recovers_exact_head_readiness(handoff_fixture, monkeypatch):
    f = handoff_fixture
    f["claim"]["next_action"] = "readiness_successor"
    row = start_source(f, monkeypatch, "after_complete_commit")
    assert row["state"] == "pending" and row["successor_task_id"] is None
    rr = seed_authenticated_review(f)
    configure(f)
    tick(f)
    successor = useful_successor(f, row["transition_key"])
    artifact = json.loads(f["workspace"].joinpath("merge-readiness-" + successor + ".json").read_text())
    assert artifact["head_sha"] == f["claim"]["head_sha"]
    assert artifact["review_ref"] == rr and artifact["ci_ref"]["conclusion"] == "success"
    assert artifact["ci_ref"]["head_sha"] == artifact["head_sha"]
    report(f, "R1", row["transition_key"])


@pytest.mark.parametrize("crashpoint", ["after_claim", "after_decision", "in_effect_before_commit", "after_effect_commit"])
def test_r2_to_r5_process_death_recovery(handoff_fixture, crashpoint):
    f = handoff_fixture
    row = start_source(f)
    key = row["transition_key"]
    configure(f, lease=5)
    killed = tick(f, crashpoint)
    assert killed.returncode == 137, killed.stdout + killed.stderr
    row = transition(f, key)
    if crashpoint == "after_effect_commit":
        assert row["state"] == "consumed" and row["successor_task_id"]
    else:
        assert row["state"] == "consuming" and row["successor_task_id"] is None
        assert f["conn"].execute("SELECT COUNT(*) FROM tasks WHERE origin_transition_key=?", (key,)).fetchone()[0] == 0
        persisted = row["decision_json"]
        for _ in range(2):
            tick(f)
            assert transition(f, key)["state"] == "consuming"
        wait_for(lambda: int(time.time()) > row["lease_expires_at"], "lease expiry", timeout=10)
        tick(f)
        assert transition(f, key)["attempt_count"] == 2
        if persisted:
            assert transition(f, key)["decision_json"] == persisted
    if crashpoint == "after_effect_commit":
        tick(f)
    successor = useful_successor(f, key)
    for _ in range(3):
        tick(f)
    f["kb"]._nudge_transition(f["conn"], key)
    assert transition(f, key)["successor_task_id"] == successor
    assert len(f["kb"].list_runs(f["conn"], successor)) == 1
    report(f, crashpoint, key)


def barrier_process(f, body, ready, release):
    code = (
        "import time; from pathlib import Path; from hermes_cli import kanban_db as k; "
        "c=k.connect(); " + body.split("# BARRIER")[0].strip()
        + "; Path(" + repr(str(ready)) + ").touch(); "
        "\nwhile not Path(" + repr(str(release)) + ").exists(): time.sleep(.01)\n"
        + body.split("# BARRIER")[1].strip() + "; c.close()"
    )
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=dict(os.environ),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_r7_stale_process_cannot_commit_after_new_owner(handoff_fixture, tmp_path):
    f = handoff_fixture
    row = start_source(f)
    key = row["transition_key"]
    configure(f, lease=2)
    ready, release = tmp_path / "stale-ready", tmp_path / "stale-release"
    body = (
        "assert k.claim_transition(c," + repr(key) + ",owner='A',ttl_seconds=1); "
        "row=c.execute('SELECT * FROM task_transitions WHERE transition_key=?',(" + repr(key) + ",)).fetchone(); "
        "decision=k._evaluate_transition(c,row) # BARRIER "
        "assert not k._apply_transition_decision(c," + repr(key) + ",'A',decision)"
    )
    stale = barrier_process(f, body, ready, release)
    try:
        wait_for(ready.exists, "old consumer claimed")
        expires = transition(f, key)["lease_expires_at"]
        wait_for(lambda: int(time.time()) > expires, "stale lease expiry", timeout=5)
        tick(f)
        release.touch()
        stdout, stderr = stale.communicate(timeout=30)
        assert stale.returncode == 0, stdout + stderr
        useful_successor(f, key)
        assert transition(f, key)["attempt_count"] == 2
        report(f, "R7", key)
    finally:
        if stale.poll() is None:
            stale.kill()
            stale.wait()


@pytest.mark.parametrize("gate", ["approval_required", "auth_or_credential", "sensitive_action", "ambiguous_write", "ambiguous_input"])
def test_r8_hard_gates_survive_fresh_processes_and_audited_unblock(handoff_fixture, gate):
    f = handoff_fixture
    f["state"]["metadata_by_task"][f["task_id"]]["gate"] = gate
    row = start_source(f)
    key = row["transition_key"]
    configure(f)
    for _ in range(3):
        tick(f)
    assert transition(f, key)["block_code"] == gate
    assert transition(f, key)["attempt_count"] == 1
    assert f["kb"].unblock_task(f["conn"], f["task_id"], actor="fixture-operator", reason="not an approval")
    tick(f)
    assert transition(f, key)["state"] == "blocked" and transition(f, key)["block_code"] == gate
    assert transition(f, key)["successor_task_id"] is None
    report(f, "R8-" + gate, key)


def test_r8_sixth_review_gate_survives_restart_and_unblock(handoff_fixture):
    f = handoff_fixture
    f["claim"]["next_action"] = "repair_requested"
    row = start_source(f)
    assert row is not None
    key = row["transition_key"]
    for _ in range(6):
        seed_authenticated_review(f, verdict="reject")
    configure(f)
    for _ in range(3):
        tick(f)
    held = transition(f, key)
    assert held is not None and held["state"] == "blocked"
    assert held["block_code"] == "review_ceiling"
    assert f["kb"].unblock_task(f["conn"], f["task_id"], actor="fixture-operator", reason="not permission")
    tick(f)
    held = transition(f, key)
    assert held is not None and held["block_code"] == "review_ceiling"
    assert held["successor_task_id"] is None
    report(f, "R8-review_ceiling", key)


def test_r9_effect_fault_exhaustion_is_atomic_and_sanitized(handoff_fixture):
    f = handoff_fixture
    row = start_source(f)
    key = row["transition_key"]
    f["conn"].execute("CREATE TRIGGER effect_fault BEFORE INSERT ON tasks WHEN NEW.origin_transition_key IS NOT NULL "
                      "BEGIN SELECT RAISE(ABORT, 'PRIVATE_CANARY'); END")
    configure(f)
    for attempt in range(2):
        # A real dispatcher error must be nonzero and carry only sanitized text.
        result = tick(f, "unused-no-crash")
        assert result.returncode != 0
        assert "PRIVATE_CANARY" not in result.stdout + result.stderr
        assert transition(f, key)["attempt_count"] == attempt + 1
        assert f["conn"].execute("SELECT COUNT(*) FROM tasks WHERE origin_transition_key=?", (key,)).fetchone()[0] == 0
    tick(f)
    assert transition(f, key)["state"] == "blocked" and transition(f, key)["block_code"] == "retry_exhausted"
    report(f, "R9", key)


def test_unregistered_fixture_provider_fails_preflight_without_completion(handoff_fixture):
    f = handoff_fixture
    f["config"]["custom_providers"] = []
    f["config"]["model"].pop("base_url", None)
    f["launch"]()
    assert f["await_terminal"]().status == "blocked"
    events = f["kb"].list_events(f["conn"], f["task_id"])
    assert any(e.kind == "preflight_failed" for e in events)
    assert not any(e.kind == "completed" for e in events)
    assert transition(f) is None


def test_real_fixture_athena_verdict_is_not_authenticated_claude(handoff_fixture):
    f = handoff_fixture
    row = start_source(f)
    run = f["kb"].list_runs(f["conn"], f["task_id"])[-1].id
    f["state"]["metadata_by_profile"] = {"athena": {"verdict": verdict_payload(f, run)}}
    configure(f)
    tick(f)
    successor = useful_successor(f, row["transition_key"])
    tick(f)
    verdict_row = f["conn"].execute("SELECT * FROM task_transitions WHERE source_task_id=?", (successor,)).fetchone()
    assert verdict_row["state"] == "blocked" and verdict_row["block_code"] == "review_unauthenticated"
    assert verdict_row["successor_task_id"] is None


@pytest.mark.parametrize("after", [False, True])
def test_sigkill_before_and_after_real_complete_transaction(handoff_fixture, after):
    f = handoff_fixture
    barrier = threading.Event()
    f["state"]["after_complete" if after else "before_complete"] = barrier
    f["state"]["hold_after_complete" if after else "hold_complete"] = 120
    f["launch"]()
    assert barrier.wait(30)
    events = f["kb"].list_events(f["conn"], f["task_id"])
    pid = next(e.payload["pid"] for e in events if e.kind == "spawned")
    os.kill(pid, signal.SIGKILL)
    wait_for(lambda: not f["kb"]._pid_alive(pid), "SIGKILL worker exit")
    row = transition(f)
    if after:
        assert row["state"] == "pending"
        f["state"].pop("after_complete")
        configure(f)
        tick(f)
        useful_successor(f, row["transition_key"])
    else:
        assert row is None
        assert not any(e.kind == "completed" for e in events)
