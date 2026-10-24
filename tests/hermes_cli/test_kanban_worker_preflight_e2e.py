"""Local HTTP protocol fixtures drive the real dispatcher and worker entrypoint."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
KEY = "fixture-credential-9d4d621c"
CANARIES = ("CANARY_TASK_71a2", "CANARY_MEM_71a2", "CANARY_SKILL_71a2")


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb

    for key in list(os.environ):
        if key.startswith((
            "HERMES_",
            "ANTHROPIC_",
            "OPENAI_",
            "DEEPSEEK_",
            "AUXILIARY_",
        )):
            monkeypatch.delenv(key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "fixture"
    profile.mkdir(parents=True)
    workspace = tmp_path / "workspace"
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-qb", "fixture", str(workspace)],
        check=True,
    )
    guards = tmp_path / "guards"
    guards.mkdir()
    context_reads = tmp_path / "context-reads.txt"
    guards.joinpath("sitecustomize.py").write_text("""import socket
import os
import sys
_original = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
        raise OSError("fixture rejects external network")
    return _original(sock, address)
socket.socket.connect = local_only
def record_context(event, args):
    sensitive = event == "import" and args[0] == "cli"
    if event == "open" and isinstance(args[0], str):
        sensitive = args[0].endswith("/memories/MEMORY.md") or args[0].endswith("/canary/SKILL.md")
    if sensitive:
        fd = os.open(os.environ["PREFLIGHT_FIXTURE_READS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.write(fd, b"context-loaded\\n")
        os.close(fd)
sys.addaudithook(record_context)
""")
    for key, value in {
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
        "HERMES_KANBAN_DB": str(home / "kanban.db"),
        "HERMES_BIN": "missing-hermes-fixture-use-module",
        "PYTHONPATH": os.pathsep.join((str(guards), str(ROOT))),
        "HERMES_NO_TMUX": "1",
        "HERMES_DISABLE_UPDATE_CHECK": "1",
        "PREFLIGHT_FIXTURE_READS": str(context_reads),
    }.items():
        monkeypatch.setenv(key, value)
    requests = []
    state = {"status": 401, "delay": 0, "useful": False}
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            if "model" not in request:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"model_info": {}}')
                return
            if state["delay"]:
                release.wait(state["delay"])
            status = state["status"]
            if status != 200:
                body = {"error": {"message": KEY + " SECRET_RESPONSE"}}
            else:
                tool = None
                # The fixed diagnostic probe has no tools. Subsequent requests
                # execute real tools; the fixture is not an agent substitute.
                if request.get("tools"):
                    calls = [m for m in request["messages"] if m.get("role") == "tool"]
                    if not calls:
                        tool = ("kanban_show", {})
                    elif len(calls) == 1:
                        tool = (
                            "terminal",
                            {
                                "command": "git rev-parse --show-toplevel > preflight_proof.txt",
                                "timeout": 10,
                            },
                        )
                    elif len(calls) == 2:
                        tool = (
                            "kanban_complete",
                            {"summary": "fixture command verified"},
                        )
                message = {"role": "assistant", "content": "OK"}
                if tool:
                    message.update(
                        content=None,
                        tool_calls=[
                            {
                                "id": "fixture_call_" + str(len(requests)),
                                "type": "function",
                                "function": {
                                    "name": tool[0],
                                    "arguments": json.dumps(tool[1]),
                                },
                            }
                        ],
                    )
                body = {
                    "id": "fixture-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": request["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            streaming = status == 200 and request.get("stream")
            if streaming:
                chunk = {
                    **body,
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": message, "finish_reason": None}],
                }
                if tool:
                    chunk["choices"][0]["delta"]["tool_calls"][0]["index"] = 0
                final = {
                    **body,
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls" if tool else "stop",
                        }
                    ],
                }
                raw = (
                    "data: "
                    + json.dumps(chunk)
                    + "\n\ndata: "
                    + json.dumps(final)
                    + "\n\ndata: [DONE]\n\n"
                ).encode()
            else:
                raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header(
                "Content-Type", "text/event-stream" if streaming else "application/json"
            )
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = {
        "model": {
            "provider": "custom:fixture",
            "default": "fixture-model",
            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
        },
        "custom_providers": [
            {
                "name": "fixture",
                "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                "key_env": "FIXTURE_API_KEY",
            }
        ],
        "kanban": {"worker_preflight": {"enabled": True, "timeout_seconds": 15}},
        "platform_toolsets": {"cli": ["terminal", "kanban"]},
        "display": {"streaming": False},
        "agent": {"max_turns": 5},
        "curator": {"enabled": False},
        "memory": {"provider": "none"},
    }
    home.joinpath("config.yaml").write_text("{}\n")
    profile.joinpath("config.yaml").write_text(yaml.safe_dump(config))
    profile.joinpath(".env").write_text("FIXTURE_API_KEY=" + KEY + "\n")
    profile.joinpath("memories").mkdir()
    profile.joinpath("memories", "MEMORY.md").write_text(CANARIES[1])
    skill = profile / "skills" / "canary"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text(
        "---\nname: canary\ndescription: Fixture canary.\n---\n" + CANARIES[2]
    )
    conn = kb.connect()
    task_id = kb.create_task(
        conn,
        title="native preflight fixture",
        body=CANARIES[0],
        assignee="fixture",
        workspace_kind="worktree",
        workspace_path=str(workspace),
        branch_name="fixture",
        skills=["canary"],
    )

    def launch():
        profile.joinpath("config.yaml").write_text(yaml.safe_dump(config))
        return kb.dispatch_once(conn, max_spawn=1)

    def await_terminal():
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            task = kb.get_task(conn, task_id)
            if task.status in {"done", "blocked", "triage"}:
                return task
            time.sleep(0.05)
        log_path = kb.worker_log_path(task_id)
        pytest.fail(
            "native worker did not reach a terminal outcome: "
            + (log_path.read_text()[-3000:] if log_path.exists() else str(log_path))
        )

    yield {
        "kb": kb,
        "conn": conn,
        "task_id": task_id,
        "profile": profile,
        "home": home,
        "workspace": workspace,
        "requests": requests,
        "state": state,
        "config": config,
        "launch": launch,
        "await_terminal": await_terminal,
        "context_reads": context_reads,
    }
    task = kb.get_task(conn, task_id)
    if task and task.worker_pid:
        try:
            os.kill(task.worker_pid, 15)
        except ProcessLookupError:
            pass
    release.set()
    server.shutdown()
    server.server_close()
    conn.close()


def test_rejected_credential_stops_before_context(native_fixture):
    f = native_fixture
    f["launch"]()
    task = f["await_terminal"]()
    assert task.status == "blocked"
    events = f["kb"].list_events(f["conn"], f["task_id"])
    failures = [e for e in events if e.kind == "preflight_failed"]
    assert failures and failures[0].payload["code"] == "credential_rejected"
    assert len(f["requests"]) == 1
    assert f["requests"][0]["messages"] == [
        {"role": "user", "content": "Reply with OK."}
    ]
    assert f["requests"][0]["max_tokens"] <= 16
    assert not f["requests"][0].get("tools")
    log = f["kb"].worker_log_path(f["task_id"]).read_text()
    evidence = json.dumps(f["requests"]) + json.dumps([e.payload for e in events]) + log
    assert KEY not in evidence
    assert "SECRET_RESPONSE" not in evidence
    assert not any(canary in evidence for canary in CANARIES)
    assert not f["context_reads"].exists()
    assert not f["kb"].dispatch_once(f["conn"], max_spawn=1).spawned


def test_healthy_worker_executes_useful_terminal_command(native_fixture):
    f = native_fixture
    f["state"]["status"] = 200
    f["launch"]()
    task = f["await_terminal"]()
    assert task.status == "done"
    proof = f["workspace"] / "preflight_proof.txt"
    assert proof.read_text().strip() == str(f["workspace"])
    events = f["kb"].list_events(f["conn"], f["task_id"])
    kinds = [e.kind for e in events]
    assert kinds.index("preflight_passed") < kinds.index("completed")
    assert f["requests"][0]["messages"] == [
        {"role": "user", "content": "Reply with OK."}
    ]
    assert CANARIES[0] in json.dumps(f["requests"][1:])
    assert f["context_reads"].exists()


@pytest.mark.parametrize(
    "scenario,code,request_count",
    [
        ("missing", "credential_missing", 0),
        ("deepseek_missing", "credential_missing", 0),
        ("anthropic_rejected", "credential_rejected", 1),
        ("tool", "tool_unavailable", 0),
        ("transient", "probe_transient", 1),
        ("timeout", "probe_timeout", 1),
        ("unsupported", "transport_unsupported", 0),
        ("operation", "operation_unavailable", 0),
    ],
)
def test_failure_matrix_without_fallback_or_context(
    native_fixture, scenario, code, request_count
):
    f = native_fixture
    if scenario in {"missing", "deepseek_missing"}:
        f["profile"].joinpath(".env").write_text("")
        f["home"].joinpath(".env").write_text(
            "FIXTURE_API_KEY=ROOT_CREDENTIAL_CANARY\nDEEPSEEK_API_KEY=ROOT_CREDENTIAL_CANARY\n"
        )
        if scenario == "deepseek_missing":
            f["config"]["model"].update(provider="deepseek", default="deepseek-flash")
    elif scenario == "anthropic_rejected":
        f["config"]["custom_providers"][0]["api_mode"] = "anthropic_messages"
    elif scenario == "tool":
        f["config"]["kanban"]["worker_preflight"]["required_tools"] = [
            "nonexistent_tool"
        ]
    elif scenario == "transient":
        f["state"]["status"] = 503
    elif scenario == "timeout":
        f["config"]["kanban"]["worker_preflight"]["timeout_seconds"] = 5
        f["state"]["delay"] = 20
    elif scenario == "unsupported":
        f["config"]["custom_providers"][0]["api_mode"] = "codex_responses"
    elif scenario == "operation":
        f["config"]["platform_toolsets"]["cli"] = ["terminal"]
        f["config"]["agent"]["disabled_toolsets"] = ["kanban"]
    f["config"]["fallback_providers"] = [
        {"provider": "custom:never", "model": "never-model"}
    ]
    f["launch"]()
    assert f["await_terminal"]().status == "blocked"
    events = f["kb"].list_events(f["conn"], f["task_id"])
    failures = [e for e in events if e.kind == "preflight_failed"]
    assert failures and failures[0].payload["code"] == code
    assert len(f["requests"]) == request_count
    evidence = (
        json.dumps(f["requests"])
        + json.dumps([e.payload for e in events])
        + f["kb"].worker_log_path(f["task_id"]).read_text()
    )
    assert not any(
        canary in evidence
        for canary in (*CANARIES, KEY, "ROOT_CREDENTIAL_CANARY", "SECRET_RESPONSE")
    )
    assert not f["context_reads"].exists()
    assert not f["kb"].dispatch_once(f["conn"], max_spawn=1).spawned


def test_real_launch_model_and_provider_override(native_fixture):
    f = native_fixture
    f["config"]["model"].update(provider="deepseek", default="profile-must-not-run")
    f["kb"].set_model_override(
        f["conn"], f["task_id"], "override-model", "custom:fixture"
    )
    f["launch"]()
    assert f["await_terminal"]().status == "blocked"
    assert len(f["requests"]) == 1
    assert f["requests"][0]["model"] == "override-model"


def test_workspace_disappears_before_popen_without_retry(native_fixture):
    f = native_fixture
    kb = f["kb"]

    def remove_before_spawn(task, workspace, *, board=None):
        Path(workspace).rename(Path(workspace).with_name("moved-workspace"))
        return kb._default_spawn(task, workspace, board=board)

    result = kb.dispatch_once(f["conn"], max_spawn=1, spawn_fn=remove_before_spawn)
    assert not result.spawned
    task = kb.get_task(f["conn"], f["task_id"])
    assert task.status == "blocked"
    assert task.consecutive_failures == 0
    events = kb.list_events(f["conn"], f["task_id"])
    assert "spawn_failed" not in [e.kind for e in events]
    assert [e.payload["code"] for e in events if e.kind == "preflight_failed"] == [
        "workspace_invalid"
    ]
    assert not f["requests"]


def test_child_rejects_wrong_cwd_before_context(native_fixture):
    f = native_fixture
    claim = f["kb"].claim_task(f["conn"], f["task_id"])
    env = dict(
        os.environ,
        HERMES_HOME=str(f["profile"]),
        HERMES_PROFILE="fixture",
        HERMES_KANBAN_TASK=f["task_id"],
        HERMES_KANBAN_RUN_ID=str(claim.current_run_id),
        HERMES_KANBAN_CLAIM_LOCK=claim.claim_lock,
        HERMES_KANBAN_WORKSPACE=str(f["workspace"]),
        TERMINAL_CWD=str(f["workspace"]),
        HERMES_KANBAN_PREFLIGHT="1",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "-p",
            "fixture",
            "--cli",
            "--toolsets",
            "terminal,kanban",
            "chat",
            "-q",
            "work kanban task " + f["task_id"],
        ],
        env=env,
        cwd=f["home"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 2
    assert f["await_terminal"]().status == "blocked"
    events = f["kb"].list_events(f["conn"], f["task_id"])
    assert [e.payload["code"] for e in events if e.kind == "preflight_failed"] == [
        "workspace_invalid"
    ]
    assert not f["requests"]


def test_supervised_route_preserves_unrepresented_media_denial(native_fixture, monkeypatch):
    from hermes_cli import backend_failover
    from agent import image_routing

    f = native_fixture
    monkeypatch.setattr(image_routing, "extract_image_refs", lambda *a: ([], ["https://example.invalid/image.png"]))
    monkeypatch.setattr(
        backend_failover,
        "load_profile_chain",
        lambda *args: (
            backend_failover.BackendLeg("api", "fixture-model", "deepseek"),
        ),
    )
    monkeypatch.setattr(f["kb"], "_native_image_attachment_evidence", lambda *a, **k: ([], 0))
    result = f["launch"]()
    assert not result.spawned
    events = f["kb"].list_events(f["conn"], f["task_id"])
    assert "spawn_failed" in [e.kind for e in events]
    assert not any(e.kind.startswith("preflight_") for e in events)
    assert not f["requests"]


def test_legacy_launch_discards_inherited_admission(native_fixture, monkeypatch):
    from types import SimpleNamespace

    f = native_fixture
    f["config"]["kanban"]["worker_preflight"]["enabled"] = False
    f["profile"].joinpath("config.yaml").write_text(yaml.safe_dump(f["config"]))
    monkeypatch.setenv("HERMES_KANBAN_PREFLIGHT", "1")
    claim = f["kb"].claim_task(f["conn"], f["task_id"])
    captured = {}

    def capture(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=424242)

    monkeypatch.setattr(subprocess, "Popen", capture)
    assert f["kb"]._default_spawn(claim, str(f["workspace"])) == 424242
    assert "HERMES_KANBAN_PREFLIGHT" not in captured["env"]
    assert captured["cwd"] == str(f["workspace"])
    assert not any(
        e.kind.startswith("preflight_")
        for e in f["kb"].list_events(f["conn"], f["task_id"])
    )
