from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from agent.transports.gateway_claude_sdk import (
    ClaudeSubscriptionTurn,
    run_claude_subscription_turn,
)
from gateway.owner_tool_bridge import FrozenLegDispatch


class _TextBlock:
    def __init__(self, text: str):
        self.text = text


class _AssistantMessage:
    def __init__(self, content):
        self.content = content


class _ResultMessage:
    def __init__(self, result: str, *, is_error: bool = False):
        self.result = result
        self.is_error = is_error


class _Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Bridge:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.drains = 0
        self.revoked = []

    def drain_one(self, timeout: float) -> bool:
        self.drains += 1
        return False

    def revoke(self, reason: str) -> None:
        self.revoked.append(reason)

    def client_config(self):
        return {"command": [sys.executable, "-m", "owner-mcp", str(self.config_path)]}


def test_sdk_leg_uses_frozen_owner_mcp_and_request_local_roleful_resume(
    tmp_path, monkeypatch
):
    captured = {}
    fake = ModuleType("claude_agent_sdk")
    fake.AssistantMessage = _AssistantMessage
    fake.ResultMessage = _ResultMessage
    fake.TextBlock = _TextBlock
    setattr(
        fake,
        "PermissionResultDeny",
        lambda message: SimpleNamespace(behavior="deny", message=message),
    )
    fake.ClaudeAgentOptions = _Options
    setattr(fake, "project_key_for_directory", lambda cwd: f"project:{cwd}")

    async def query(*, prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        seeded = await options.session_store.load(
            {"project_key": f"project:{tmp_path}", "session_id": options.resume}
        )
        captured["seeded"] = seeded
        yield _AssistantMessage([_TextBlock("draft")])
        yield _ResultMessage("final")

    fake.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    monkeypatch.setattr(
        "agent.transports.gateway_claude_sdk.ensure", lambda *_a, **_kw: None
    )

    config_path = tmp_path / "owner.json"
    config_path.write_text(
        json.dumps({"command": [sys.executable, "-m", "owner-mcp", "cfg"]}),
        encoding="utf-8",
    )
    bridge = _Bridge(config_path)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "todo",
                "description": "owner todo",
                "parameters": {"type": "object"},
            },
        }
    ]
    dispatch = FrozenLegDispatch(
        prompt_json=json.dumps(
            {"system": "frozen system", "history": [{"role": "user", "content": "prior"}]}
        ),
        tools_json=json.dumps(tools),
        media_json="[]",
        leg_json=json.dumps({"kind": "claude_code", "model": "claude-sonnet-5-5"}),
        prompt_digest="p",
        tool_digest="t",
        media_digest="m",
        chain_digest="c",
    )

    result = run_claude_subscription_turn(
        ClaudeSubscriptionTurn(
            model="claude-sonnet-5-5",
            prompt="current",
            history=[{"role": "user", "content": "prior"}],
            system_prompt="frozen system",
            cwd=str(tmp_path),
            reasoning_effort="high",
            capability_mode="full",
        ),
        dispatch,
        bridge,
    )

    assert result == "final"
    assert captured["prompt"] == "current"
    options = captured["options"]
    assert options.tools == []
    assert options.strict_mcp_config is True
    assert options.setting_sources == []
    assert options.allowed_tools == ["mcp__gateway-owner__todo"]
    assert options.effort == "high"
    denied = asyncio.run(options.can_use_tool("Bash", {"command": "true"}, None))
    assert denied.behavior == "deny"
    assert options.system_prompt == "frozen system"
    assert options.resume
    assert captured["seeded"][0]["message"] == {
        "role": "user",
        "content": "prior",
    }
    assert bridge.drains >= 1
    assert bridge.revoked == []
