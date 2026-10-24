"""Synthetic local native protocols; never uses live vendor credentials/services."""

from __future__ import annotations

import base64
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
CANARIES = (
    "NATIVE_TASK_CANARY",
    "NATIVE_MEMORY_CANARY",
    "NATIVE_SKILL_CANARY",
    "NATIVE_CONTEXT_CANARY",
    "CANARY_SOUL_NATIVE",
)
TOKEN = "sk-ant-oat-fixture-only-secret"


@pytest.fixture
def native_protocol(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli.backend_failover import _CANONICAL_ROUTES

    for key in list(os.environ):
        if key.startswith(("HERMES_", "ANTHROPIC_", "OPENAI_", "AUXILIARY_")):
            monkeypatch.delenv(key)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    roles = {}
    for name in ("talos", "athena"):
        roles[name] = [
            dict(kind=kind, model=model, **({"provider": provider} if provider else {}))
            for kind, provider, model in _CANONICAL_ROUTES[name]
        ]
    root_config = {
        "profiles": {
            "shared_credentials": {
                "enabled": True,
                "trusted_profiles": ["talos", "athena"],
                "providers": ["openai-codex"],
            }
        },
        "agent_backend_failover": {"enabled": True, "roles": roles},
    }
    home.joinpath("config.yaml").write_text(yaml.safe_dump(root_config))
    jwt = (
        "e30."
        + base64
        .urlsafe_b64encode(json.dumps({"exp": 4102444800}).encode())
        .decode()
        .rstrip("=")
        + ".fixture"
    )
    home.joinpath("auth.json").write_text(
        json.dumps({
            "providers": {
                "openai-codex": {
                    "tokens": {
                        "access_token": jwt,
                        "refresh_token": "fixture-refresh-secret",
                    }
                }
            }
        })
    )
    home.joinpath("claude").mkdir()
    home.joinpath("claude/subscription.token").write_text(TOKEN)
    home.joinpath("claude/subscription.token").chmod(0o600)
    for name in roles:
        p = home / "profiles" / name
        p.mkdir(parents=True)
        p.joinpath("SOUL.md").write_text(CANARIES[4])
        from agent.models_dev import PROVIDER_TO_MODELS_DEV

        p.joinpath("models_dev_cache.json").write_text(
            json.dumps({
                PROVIDER_TO_MODELS_DEV["openai-codex"]: {
                    "models": {
                        "gpt-6.1-sol": {
                            "tool_call": True,
                            "limit": {"context": 200000, "output": 8192},
                        }
                    }
                }
            })
        )
        p.joinpath("MEMORY.md").write_text(CANARIES[1])
        skill = p / "skills" / "canary"
        skill.mkdir(parents=True)
        skill.joinpath("SKILL.md").write_text(
            "---\nname: canary\ndescription: Fixture canary.\n---\n" + CANARIES[2]
        )
    repo, workspace = tmp_path / "repo", tmp_path / "workspace"
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
    workspace.joinpath("CLAUDE.md").write_text(CANARIES[3])
    workspace.joinpath("AGENTS.md").write_text(CANARIES[3])
    audit = tmp_path / "context-audit"
    guards = tmp_path / "guards"
    guards.mkdir()
    guards.joinpath("sitecustomize.py").write_text("""import os, socket, sys
from pathlib import Path
from hermes_cli import provider_access
provider_access.native_home = lambda: Path(os.environ["HOME"])
from hermes_cli import auth
auth.DEFAULT_CODEX_BASE_URL = Path(os.environ["HOME"]).joinpath("codex-url").read_text()

if (Path(os.environ["HOME"]) / "synthetic-athena-schemas").exists() and os.environ.get("HERMES_PROFILE") == "athena":
    import json
    import model_tools
    from hermes_cli.backend_failover import BackendLeg
    from hermes_cli.backend_failover_worker import _claude_bridged_tool_names
    original_definitions = model_tools.get_tool_definitions
    supported = _claude_bridged_tool_names(BackendLeg("claude_code", "claude-opus-5-5", capability_mode="read_only"))
    def synthetic_schemas(*args, **kwargs):
        definitions = original_definitions(*args, **kwargs)
        filtered = [d for d in definitions if d["function"]["name"] in supported]
        removed = sorted(d["function"]["name"] for d in definitions if d not in filtered)
        assert all(d in definitions and d["function"]["name"] in supported for d in filtered)
        Path(os.environ["HOME"]).joinpath("removed-schemas.json").write_text(json.dumps(removed))
        return filtered
    model_tools.get_tool_definitions = synthetic_schemas
_original = socket.socket.connect
def local_only(sock, address):
    if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
        raise OSError("external network forbidden")
    return _original(sock, address)
socket.socket.connect = local_only
def audit(event, args):
    sensitive = event == "import" and args[0] == "cli"
    if event == "open" and isinstance(args[0], str):
        sensitive = args[0].endswith(("/MEMORY.md", "/SKILL.md", "/AGENTS.md", "/CLAUDE.md"))
    if sensitive:
        fd = os.open(os.environ["HOME"] + "/context-audit", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.write(fd, b"context\\n")
        os.close(fd)
sys.addaudithook(audit)
""")
    requests, paths = [], []
    state = {"status": 401, "delay": False}
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            paths.append(self.path)
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            paths.append(self.path)
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "model" not in request:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"model_info": {}}')
                return
            requests.append(request)
            if state["delay"]:
                release.wait(20)
            if state["status"] != 200:
                raw = json.dumps({
                    "error": {
                        "message": "SECRET_RESPONSE",
                        "type": "authentication_error",
                    }
                }).encode()
                self.send_response(state["status"])
                self.send_header("Content-Type", "application/json")
            else:
                calls = [
                    item
                    for item in request.get("input", [])
                    if item.get("type") == "function_call_output"
                ]
                tool = None
                if request.get("tools"):
                    if not calls:
                        tool = (
                            "terminal",
                            {
                                "command": "git rev-parse --show-toplevel > preflight_proof.txt",
                                "timeout": 10,
                            },
                        )
                    elif len(calls) == 1:
                        tool = (
                            "kanban_complete",
                            {"summary": "native fixture verified"},
                        )
                item = (
                    {
                        "id": "fc_fixture",
                        "type": "function_call",
                        "call_id": "call_" + str(len(requests)),
                        "name": tool[0],
                        "arguments": json.dumps(tool[1]),
                        "status": "completed",
                    }
                    if tool
                    else {
                        "id": "msg_fixture",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "OK", "annotations": []}
                        ],
                    }
                )
                response = {
                    "id": "resp_fixture",
                    "object": "response",
                    "created_at": 1,
                    "model": request["model"],
                    "status": "completed",
                    "output": [item],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                }
                events = [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": item,
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": item,
                    },
                    {"type": "response.completed", "response": response},
                ]
                raw = "".join(
                    "data: " + json.dumps(e) + "\n\n" for e in events
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    tmp_path.joinpath("codex-url").write_text(
        f"http://127.0.0.1:{server.server_port}/backend-api/codex"
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    fake = fakebin / "claude"
    fake.write_text(f"""#!{sys.executable}
import json, os, pathlib, subprocess, sys, time
home = pathlib.Path(os.environ['HOME'])
if '--version' in sys.argv:
    print('2.1.999'); raise SystemExit(0)
if sys.argv[1:4] == ['auth', 'status', '--json']:
    print('{{"loggedIn":true}}'); raise SystemExit(0)
probe = '--append-system-prompt-file' not in sys.argv
with (home / 'claude-calls.jsonl').open('a') as f:
    f.write(json.dumps(dict(argv=sys.argv[1:], cwd=os.getcwd(), env_names=sorted(os.environ), probe=probe, pid=os.getpid())) + '\\n')
scenario = (home / 'claude-scenario').read_text() if (home / 'claude-scenario').exists() else 'healthy'
if probe:
    assert '--safe-mode' in sys.argv and '--bare' not in sys.argv
    if scenario == 'flood':
        print('SECRET_RESPONSE' * 6000); raise SystemExit(0)
    if scenario == 'timeout': time.sleep(60)
    if scenario == 'reject':
        print(json.dumps(dict(type='result', subtype='error', is_error=True, error='authentication_error', api_error_status=401, message='SECRET_RESPONSE'))); raise SystemExit(1)
    print(json.dumps(dict(type='result', subtype='success', is_error=False, result='OK'))); raise SystemExit(0)
assert '--restricted' in sys.argv and '--strict-mcp-config' in sys.argv
assert sys.argv[sys.argv.index('--tools') + 1] == 'Read,Glob,Grep'
subprocess.run(['git', 'rev-parse', '--show-toplevel'], stdout=open('preflight_proof.txt', 'w'), check=True)
server = json.loads(pathlib.Path(sys.argv[sys.argv.index('--mcp-config') + 1]).read_text())['mcpServers']['hermes-tools']
# Real MCP stdio transport and fenced production lifecycle, not a fake completion.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
async def complete():
    params = StdioServerParameters(command=server['command'], args=server['args'], env=server['env'])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            shown = await session.call_tool('kanban_show', {{}})
            assert not shown.isError
            result = await session.call_tool('kanban_complete', {{'summary': 'native fixture verified'}})
            assert not result.isError
asyncio.run(complete())
print(json.dumps(dict(type='result', subtype='success', is_error=False)))
""")
    fake.chmod(0o700)
    for key, value in {
        "HOME": str(tmp_path),
        "HERMES_HOME": str(home),
        "HERMES_KANBAN_DB": str(home / "kanban.db"),
        "HERMES_BIN": "missing-use-module",
        "PYTHONPATH": os.pathsep.join((str(guards), str(ROOT))),
        "PATH": str(fakebin) + os.pathsep + os.environ["PATH"],
        "HERMES_NO_TMUX": "1",
        "HERMES_DISABLE_UPDATE_CHECK": "1",
        "HERMES_CODEX_BASE_URL": f"http://127.0.0.1:{server.server_port}/backend-api/codex",
    }.items():
        monkeypatch.setenv(key, value)
    conn = kb.connect()
    tasks = []

    def launch(
        profile="talos", *, direct=False, required_tools=None, toolsets=None, timeout=45
    ):
        # The auth seatbelt compares HOME/.hermes to "real" home under pytest.
        # Only these subprocesses use the wholly disposable HOME and auth store.
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        if profile == "athena" and toolsets is None:
            tmp_path.joinpath("synthetic-athena-schemas").touch()
        config = {
            "model": {"provider": "openai-codex", "default": "gpt-6.1-sol"},
            "kanban": {
                "worker_preflight": {
                    "enabled": True,
                    "native_timeout_seconds": timeout,
                    "required_tools": required_tools
                    if required_tools is not None
                    else (["kanban_show"] if profile == "athena" else ["terminal"]),
                }
            },
            "platform_toolsets": {
                "cli": toolsets
                if toolsets is not None
                else (["kanban"] if profile == "athena" else ["terminal", "kanban"])
            },
            "curator": {"enabled": False},
            "memory": {"provider": "none"},
            "agent": {"max_turns": 5},
            "display": {"streaming": False},
        }
        home.joinpath("profiles", profile, "config.yaml").write_text(
            yaml.safe_dump(config)
        )
        task_id = kb.create_task(
            conn,
            title="native protocol fixture",
            body=CANARIES[0],
            assignee=profile,
            workspace_kind="worktree",
            workspace_path=str(workspace),
            branch_name="fixture",
            skills=["canary"],
            max_runtime_seconds=60,
        )
        tasks.append(task_id)
        if direct:
            kb.set_model_override(conn, task_id, "gpt-6.1-sol", "openai-codex")
        kb.dispatch_once(conn, max_spawn=1)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            task = kb.get_task(conn, task_id)
            if task.status in {"done", "blocked", "triage"}:
                return task, kb.list_events(conn, task_id)
            time.sleep(0.05)
        pytest.fail(kb.worker_log_path(task_id).read_text()[-5000:])

    yield dict(
        home=home,
        workspace=workspace,
        state=state,
        requests=requests,
        paths=paths,
        launch=launch,
        kb=kb,
        conn=conn,
        audit=audit,
        jwt=jwt,
        fake=fake,
    )
    for task_id in tasks:
        task = kb.get_task(conn, task_id)
        if task.worker_pid:
            try:
                os.kill(task.worker_pid, 15)
            except ProcessLookupError:
                pass
    release.set()
    server.shutdown()
    server.server_close()
    conn.close()


def assert_failure(f, task, events, code):
    assert task.status == "blocked"
    failures = [e for e in events if e.kind == "preflight_failed"]
    assert failures and failures[0].payload["code"] == code
    evidence = (
        json.dumps(f["requests"])
        + json.dumps([e.payload for e in events])
        + f["kb"].worker_log_path(task.id).read_text()
    )
    assert not any(
        c in evidence for c in (*CANARIES, TOKEN, f["jwt"], "SECRET_RESPONSE")
    )
    assert not f["audit"].exists()
    assert not list(f["home"].glob("backend-failover-ledgers/*"))
    assert not list(f["home"].glob("profiles/*/cache/scratch/backend-failover-*"))
    assert not f["kb"].dispatch_once(f["conn"], max_spawn=1).spawned


def test_codex_primary_rejection_before_context(native_protocol):
    f = native_protocol
    task, events = f["launch"]()
    assert_failure(f, task, events, "credential_rejected")
    assert len(f["requests"]) == 1
    assert not (f["home"].parent / "claude-calls.jsonl").exists()


def test_healthy_codex_supervisor_executes_real_terminal(native_protocol):
    f = native_protocol
    f["state"]["status"] = 200
    task, events = f["launch"]()
    assert task.status == "done", f["kb"].worker_log_path(task.id).read_text()[-5000:]
    assert (f["workspace"] / "preflight_proof.txt").read_text().strip() == str(
        f["workspace"]
    )
    probe = f["requests"][0]
    assert probe["instructions"] == "Reply with OK."
    assert probe["store"] is False and probe["stream"] is True
    assert not probe.get("tools") and "max_output_tokens" not in probe
    assert not any(c in json.dumps(probe) for c in CANARIES)
    passed = next(e for e in events if e.kind == "preflight_passed")
    assert passed.payload["route"] == "supervisor"
    attempt = next(e for e in events if e.kind == "commented")
    assert passed.id < attempt.id < next(e.id for e in events if e.kind == "completed")
    assert not any("oauth" in path for path in f["paths"])


def test_claude_primary_rejection_before_context(native_protocol):
    f = native_protocol
    f["home"].parent.joinpath("claude-scenario").write_text("reject")
    task, events = f["launch"]("athena")
    assert_failure(f, task, events, "credential_rejected")
    assert not f["requests"]


def test_synthetic_schema_healthy_claude_uses_real_mcp_lifecycle(native_protocol):
    f = native_protocol
    task, events = f["launch"]("athena")
    assert task.status == "done", f["kb"].worker_log_path(task.id).read_text()[-5000:]
    assert (f["workspace"] / "preflight_proof.txt").read_text().strip() == str(
        f["workspace"]
    )
    calls = [
        json.loads(line)
        for line in (f["home"].parent / "claude-calls.jsonl").read_text().splitlines()
    ]
    assert [call["probe"] for call in calls] == [True, False]
    probe = calls[0]
    assert probe["cwd"] != str(f["workspace"])
    assert probe["argv"][-1] == "Reply with OK."
    assert probe["argv"][probe["argv"].index("--tools") + 1] == ""
    assert "--append-system-prompt-file" not in probe["argv"]
    assert all("HERMES_KANBAN_PREFLIGHT" not in call["env_names"] for call in calls)
    assert not any(c in json.dumps(probe) for c in (*CANARIES, TOKEN, f["jwt"]))
    removed = json.loads(f["home"].parent.joinpath("removed-schemas.json").read_text())
    assert "kanban_attach" in removed and "kanban_create" in removed
    passed = next(e for e in events if e.kind == "preflight_passed")
    assert passed.payload["api_mode"] == "claude_native"
    assert (
        passed.id
        < next(e.id for e in events if e.kind == "commented")
        < next(e.id for e in events if e.kind == "completed")
    )
    assert not f["requests"]


@pytest.mark.parametrize(
    "profile,scenario,code",
    [
        ("talos", "missing", "credential_missing"),
        ("talos", "timeout", "probe_timeout"),
        ("talos", "catalog", "capability_mismatch"),
        ("athena", "missing", "credential_missing"),
        ("athena", "timeout", "probe_timeout"),
        ("athena", "flood", "probe_failed"),
        ("athena", "cli", "transport_unsupported"),
        ("athena", "tools", "tool_unavailable"),
    ],
)
def test_native_failure_matrix(native_protocol, monkeypatch, profile, scenario, code):
    f = native_protocol
    if scenario == "missing":
        f["home"].joinpath(
            "auth.json" if profile == "talos" else "claude/subscription.token"
        ).unlink()
    elif scenario == "catalog":
        f["home"].joinpath("profiles/talos/models_dev_cache.json").write_text(
            '{"fixture": {}}'
        )
    elif scenario == "timeout" and profile == "talos":
        f["state"]["delay"] = True
    elif scenario == "cli":
        f["fake"].unlink()
        monkeypatch.setenv("PATH", str(f["fake"].parent) + ":/usr/bin:/bin")
    else:
        f["home"].parent.joinpath("claude-scenario").write_text(scenario)
    started = time.monotonic()
    task, events = f["launch"](
        profile,
        timeout=4 if scenario == "timeout" else 45,
        toolsets=["hermes-cli"] if scenario == "tools" else None,
    )
    assert_failure(f, task, events, code)
    if scenario == "tools":
        missing = next(
            e.payload["missing"] for e in events if e.kind == "preflight_failed"
        )
        assert "terminal" in missing and "kanban_attach" in missing
    if scenario == "timeout":
        assert time.monotonic() - started < 12
    if profile == "athena" or scenario in {"missing", "catalog"}:
        assert not f["requests"]
    if profile == "talos":
        assert not f["home"].parent.joinpath("claude-calls.jsonl").exists()


def test_direct_codex_override_proceeds_after_probe(native_protocol):
    f = native_protocol
    f["state"]["status"] = 200
    task, events = f["launch"](direct=True)
    assert task.status == "done", f["kb"].worker_log_path(task.id).read_text()[-4000:]
    passed = next(e for e in events if e.kind == "preflight_passed")
    assert passed.payload["route"] == "direct"
    assert (f["workspace"] / "preflight_proof.txt").read_text().strip() == str(
        f["workspace"]
    )
