from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from gateway.backend_failover import FrozenLegDispatch
from gateway.owner_tool_bridge import (
    OwnerToolBridge,
    OwnerTurnExecution,
    ToolBridgeError,
    tool_allowed_for_capability,
)


def test_role_capabilities_apply_a_hard_ceiling_before_normal_approval():
    assert tool_allowed_for_capability("read_file", "read_only") is True
    assert tool_allowed_for_capability("terminal", "read_only") is False
    assert tool_allowed_for_capability("kanban_create", "orchestrator") is True
    assert tool_allowed_for_capability("terminal", "orchestrator") is False
    assert tool_allowed_for_capability("terminal", "full") is True
    assert tool_allowed_for_capability("read_file", "unknown") is False


def _dispatch() -> FrozenLegDispatch:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "todo",
                "description": "Update the owner todo list",
                "parameters": {
                    "type": "object",
                    "properties": {"merge": {"type": "boolean"}},
                    "additionalProperties": False,
                },
            },
        }
    ]
    return FrozenLegDispatch(
        prompt_json=json.dumps({"messages": []}),
        tools_json=json.dumps(tools),
        media_json="[]",
        leg_json=json.dumps({"kind": "claude_code"}),
        prompt_digest="prompt-digest",
        tool_digest="tool-digest",
        media_digest="media-digest",
        chain_digest="chain-digest",
    )


class _OwnerAgent:
    def __init__(self) -> None:
        self.calls = []
        self._interrupt_requested = False

    def _execute_tool_calls(self, assistant, messages, effective_task_id, api_call_count=0):
        call = assistant.tool_calls[0]
        self.calls.append(
            (
                call.id,
                call.function.name,
                json.loads(call.function.arguments),
                effective_task_id,
                threading.get_ident(),
            )
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": json.dumps({"ok": True, "owner": effective_task_id}),
            }
        )


def _execution(agent: _OwnerAgent) -> OwnerTurnExecution:
    return OwnerTurnExecution(
        agent=agent,
        messages=[{"role": "user", "content": "hello"}],
        dispatch=_dispatch(),
        effective_task_id="gateway-task",
        turn_id="turn-1",
        coordinator_id="coordinator-1",
        attempt_id="attempt-1",
        generation=7,
        snapshot_digest="prompt-digest",
        identity=("talos", "slack", "C1", "U1", "session-1"),
    )


def _call(bridge: OwnerToolBridge, *, call_id: str = "call-1", merge: bool = True):
    return {
        "op": "call",
        "capability": bridge.capability,
        "coordinator_id": "coordinator-1",
        "attempt_id": "attempt-1",
        "generation": 7,
        "snapshot_digest": "prompt-digest",
        "call_id": call_id,
        "name": "todo",
        "arguments": {"merge": merge},
    }


def test_owner_thread_executes_frozen_tool_and_duplicate_replays_cached_result(tmp_path):
    agent = _OwnerAgent()
    bridge = OwnerToolBridge.open(_dispatch(), _execution(agent), runtime_root=tmp_path)
    owner_thread = threading.get_ident()
    responses = []

    def submit() -> None:
        responses.append(bridge.submit(_call(bridge)))

    child = threading.Thread(target=submit)
    child.start()
    assert bridge.drain_one(timeout=1.0) is True
    child.join(timeout=1.0)

    assert responses == [
        {
            "call_id": "call-1",
            "disposition": "result",
            "content": json.dumps({"ok": True, "owner": "gateway-task"}),
            "is_error": False,
        }
    ]
    assert agent.calls == [
        ("call-1", "todo", {"merge": True}, "gateway-task", owner_thread)
    ]
    assert bridge.submit(_call(bridge)) == responses[0]
    assert len(agent.calls) == 1
    bridge.close()


def test_changed_duplicate_unknown_tool_and_owner_identity_override_fail_closed(tmp_path):
    agent = _OwnerAgent()
    bridge = OwnerToolBridge.open(_dispatch(), _execution(agent), runtime_root=tmp_path)

    request = _call(bridge)
    child = threading.Thread(target=lambda: bridge.submit(request))
    child.start()
    assert bridge.drain_one(timeout=1.0) is True
    child.join(timeout=1.0)

    with pytest.raises(ToolBridgeError, match="different arguments"):
        bridge.submit(_call(bridge, merge=False))
    bad_name = {**_call(bridge, call_id="call-2"), "name": "terminal"}
    with pytest.raises(ToolBridgeError, match="not in the frozen tool snapshot"):
        bridge.submit(bad_name)
    override = {**_call(bridge, call_id="call-3"), "user_id": "other"}
    with pytest.raises(ToolBridgeError, match="unexpected frame fields"):
        bridge.submit(override)
    bridge.close()


def test_revoke_rejects_queued_and_future_calls_without_execution(tmp_path):
    agent = _OwnerAgent()
    bridge = OwnerToolBridge.open(_dispatch(), _execution(agent), runtime_root=tmp_path)
    result = []
    child = threading.Thread(target=lambda: result.append(bridge.submit(_call(bridge))))
    child.start()
    bridge.revoke("stale_generation")
    child.join(timeout=1.0)

    assert result == [
        {
            "call_id": "call-1",
            "disposition": "revoked",
            "content": "stale_generation",
            "is_error": True,
        }
    ]
    assert agent.calls == []
    with pytest.raises(ToolBridgeError, match="revoked"):
        bridge.submit(_call(bridge, call_id="call-2"))
    bridge.close()


def test_live_owner_identity_is_rechecked_before_tool_execution(tmp_path):
    agent = _OwnerAgent()
    execution = _execution(agent)
    execution.expected_identity = execution.identity
    live_identity = [execution.identity]
    execution.identity_provider = lambda: live_identity[0]
    bridge = OwnerToolBridge.open(_dispatch(), execution, runtime_root=tmp_path)
    result = []

    child = threading.Thread(
        target=lambda: result.append(bridge.submit(_call(bridge)))
    )
    child.start()
    live_identity[0] = ("other-owner",)
    assert bridge.drain_one(timeout=1.0) is True
    child.join(timeout=1.0)

    assert result[0]["disposition"] == "revoked"
    assert result[0]["content"] == "owner_identity_changed"
    assert agent.calls == []
    bridge.close()


def test_runtime_endpoint_is_private_and_capability_is_not_in_argv_material(tmp_path):
    bridge = OwnerToolBridge.open(_dispatch(), _execution(_OwnerAgent()), runtime_root=tmp_path)
    assert bridge.runtime_dir.stat().st_mode & 0o777 == 0o700
    assert bridge.endpoint.stat().st_mode & 0o777 == 0o600
    config = bridge.client_config()
    assert config["endpoint"] == str(bridge.endpoint)
    assert config["capability"] == bridge.capability
    assert bridge.capability not in " ".join(config["command"])
    bridge.close()


def test_request_private_socket_accepts_a_real_child_process(tmp_path):
    agent = _OwnerAgent()
    bridge = OwnerToolBridge.open(_dispatch(), _execution(agent), runtime_root=tmp_path)
    script = """
import json, sys
from agent.transports.gateway_owner_mcp import SocketOwnerClient
config = json.load(open(sys.argv[1], encoding='utf-8'))
response = SocketOwnerClient(config).submit({
    'call_id': 'child-call', 'name': 'todo', 'arguments': {'merge': True}
})
print(json.dumps(response, sort_keys=True))
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(bridge.config_path)],
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert bridge.drain_one(timeout=2.0) is True
    stdout, stderr = child.communicate(timeout=2.0)

    assert child.returncode == 0, stderr
    assert json.loads(stdout) == {
        "call_id": "child-call",
        "disposition": "result",
        "content": json.dumps({"ok": True, "owner": "gateway-task"}),
        "is_error": False,
    }
    assert len(agent.calls) == 1
    bridge.close()


def test_real_mcp_stdio_child_lists_frozen_schema_and_calls_owner(tmp_path):
    agent = _OwnerAgent()
    bridge = OwnerToolBridge.open(_dispatch(), _execution(agent), runtime_root=tmp_path)
    observed = {}
    failure = []

    async def child_main():
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "agent.transports.gateway_owner_mcp",
                str(bridge.config_path),
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                observed["listed"] = await session.list_tools()
                observed["called"] = await session.call_tool(
                    "mcp__gateway-owner__todo", {"merge": True}
                )

    def child_worker():
        try:
            asyncio.run(child_main())
        except BaseException as exc:
            failure.append(exc)

    child = threading.Thread(target=child_worker, daemon=True)
    child.start()
    while child.is_alive():
        bridge.drain_one(timeout=0.05)
    child.join(timeout=1.0)

    assert not failure
    assert [tool.name for tool in observed["listed"].tools] == [
        "mcp__gateway-owner__todo"
    ]
    assert observed["listed"].tools[0].inputSchema == {
        "type": "object",
        "properties": {"merge": {"type": "boolean"}},
        "additionalProperties": False,
    }
    assert observed["called"].isError is False
    assert len(agent.calls) == 1
    bridge.close()
