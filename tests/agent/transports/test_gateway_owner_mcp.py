from __future__ import annotations

import json

import pytest

from agent.transports.gateway_claude_turn import (
    InMemorySessionStore,
    ProjectionError,
    project_hermes_history,
)
from agent.transports.gateway_owner_mcp import FrozenOwnerMcpFacade, decode_mcp_tool_name


def _tools_json() -> str:
    return json.dumps(
        [
            {
                "type": "function",
                "function": {
                    "name": "todo",
                    "description": "Owner todo",
                    "parameters": {
                        "type": "object",
                        "properties": {"merge": {"type": "boolean"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Owner terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
        ]
    )


def test_mcp_facade_lists_exact_frozen_schemas_in_order_and_reverses_names():
    calls = []
    facade = FrozenOwnerMcpFacade(
        tools_json=_tools_json(),
        submit=lambda request: calls.append(request) or {
            "call_id": request["call_id"],
            "disposition": "result",
            "content": "ok",
            "is_error": False,
        },
    )

    listed = facade.list_tools()
    assert [tool["name"] for tool in listed] == [
        "mcp__gateway-owner__todo",
        "mcp__gateway-owner__terminal",
    ]
    assert listed[0]["inputSchema"] == json.loads(_tools_json())[0]["function"][
        "parameters"
    ]
    assert decode_mcp_tool_name("mcp__gateway-owner__terminal") == "terminal"

    result = facade.call_tool(
        "mcp__gateway-owner__terminal", {"command": "pwd"}, call_id="call-9"
    )
    assert result["content"] == "ok"
    assert calls == [
        {"call_id": "call-9", "name": "terminal", "arguments": {"command": "pwd"}}
    ]


@pytest.mark.parametrize("name", ["terminal", "mcp__other__terminal", "mcp__gateway-owner__"])
def test_mcp_facade_rejects_non_owner_encoded_names(name):
    with pytest.raises(ValueError):
        decode_mcp_tool_name(name)


def test_roleful_projector_preserves_tool_use_result_and_current_user_is_not_seeded():
    history = [
        {"role": "user", "content": "previous user"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "tool-1",
                    "type": "function",
                    "function": {"name": "todo", "arguments": '{"merge":false}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "tool-1",
            "name": "todo",
            "content": "saved",
        },
    ]

    entries = project_hermes_history(history, session_id="claude-session")

    assert [entry["type"] for entry in entries] == ["user", "assistant", "user"]
    assert entries[1]["message"]["content"] == [
        {"type": "text", "text": "checking"},
        {
            "type": "tool_use",
            "id": "tool-1",
            "name": "todo",
            "input": {"merge": False},
        },
    ]
    assert entries[2]["message"]["content"] == [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "saved"}
    ]
    assert all("current user" not in json.dumps(entry) for entry in entries)
    assert entries[0]["sessionId"] == "claude-session"
    assert entries[1]["parentUuid"] == entries[0]["uuid"]
    assert entries[2]["parentUuid"] == entries[1]["uuid"]

    store = InMemorySessionStore()
    store.append("claude-session", entries)
    loaded = store.load("claude-session")
    assert loaded == entries
    loaded[0]["message"]["content"] = "mutated"
    assert store.load("claude-session") == entries


def test_projector_rejects_orphan_tool_result_and_malformed_arguments():
    with pytest.raises(ProjectionError, match="orphan tool result"):
        project_hermes_history(
            [{"role": "tool", "tool_call_id": "missing", "content": "x"}],
            session_id="s",
        )
    with pytest.raises(ProjectionError, match="arguments"):
        project_hermes_history(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "bad",
                            "type": "function",
                            "function": {"name": "todo", "arguments": "{"},
                        }
                    ],
                }
            ],
            session_id="s",
        )
