"""Synchronous gateway-owner adapter for one Claude subscription turn.

The Claude Agent SDK is asynchronous and runs a Claude Code child process.  A
GatewayRunner turn is synchronous inside its executor thread, so this adapter
runs the SDK coroutine on a helper thread while the original gateway thread
pumps :class:`gateway.owner_tool_bridge.OwnerToolBridge`.  Tool handlers never
run on the SDK, socket-reader, or helper threads.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any, Callable, Mapping, Sequence
import uuid

from agent.transports.gateway_claude_turn import (
    InMemorySessionStore,
    project_bounded_history,
)
from agent.transports.gateway_owner_mcp import encode_mcp_tool_name
from gateway.owner_tool_bridge import FrozenLegDispatch, OwnerToolBridge
from tools.lazy_deps import FeatureUnavailable, ensure


class ClaudeSubscriptionTurnError(RuntimeError):
    """The isolated Claude subscription leg could not produce a final answer."""


@dataclass(frozen=True)
class ClaudeSubscriptionTurn:
    model: str
    prompt: str
    history: Sequence[Mapping[str, Any]]
    system_prompt: str | None
    cwd: str
    reasoning_effort: str | None = None
    capability_mode: str = "full"
    max_history_messages: int = 120
    max_history_bytes: int = 512_000
    max_turns: int = 60


def _mcp_config(bridge: OwnerToolBridge) -> dict[str, Any]:
    config = bridge.client_config()
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) and part for part in command
    ):
        raise ClaudeSubscriptionTurnError("owner bridge emitted an invalid MCP command")
    return {
        "type": "stdio",
        "command": command[0],
        "args": command[1:],
    }


def _allowed_tools(dispatch: FrozenLegDispatch) -> list[str]:
    raw = json.loads(dispatch.tools_json)
    if not isinstance(raw, list):
        raise ClaudeSubscriptionTurnError("frozen tool schema is not a list")
    names: list[str] = []
    for item in raw:
        function = item.get("function") if isinstance(item, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if not isinstance(name, str) or not name:
            raise ClaudeSubscriptionTurnError("frozen tool schema contains an invalid name")
        names.append(encode_mcp_tool_name(name))
    return names


async def _query_sdk(
    turn: ClaudeSubscriptionTurn,
    dispatch: FrozenLegDispatch,
    bridge: OwnerToolBridge,
) -> str:
    try:
        ensure("provider.claude_subscription", prompt=False)
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            project_key_for_directory,
            ResultMessage,
            TextBlock,
            PermissionResultDeny,
            query,
        )
    except (ImportError, FeatureUnavailable) as exc:  # pragma: no cover - install boundary
        raise ClaudeSubscriptionTurnError(
            "Claude subscription failover requires hermes-agent[claude-subscription]"
        ) from exc

    session_id = str(uuid.uuid4())
    key = {
        "project_key": project_key_for_directory(turn.cwd),
        "session_id": session_id,
    }
    history = project_bounded_history(
        turn.history,
        session_id=session_id,
        max_messages=turn.max_history_messages,
        max_bytes=turn.max_history_bytes,
        cwd=turn.cwd,
    )
    store = InMemorySessionStore()
    if history:
        await store.append(key, history)

    allowed_tools = _allowed_tools(dispatch)

    async def deny_unlisted_tool(name: str, _input: dict[str, Any], _context: Any):
        if name in allowed_tools:
            # Listed MCP calls are auto-approved by the SDK and still pass the
            # authenticated owner bridge, which applies Hermes policy/approval.
            return PermissionResultDeny("owner MCP permission routing error")
        return PermissionResultDeny(f"native Claude tool {name!r} is disabled")

    options = ClaudeAgentOptions(
        tools=[],
        allowed_tools=allowed_tools,
        can_use_tool=deny_unlisted_tool,
        system_prompt=turn.system_prompt,
        mcp_servers={"gateway-owner": _mcp_config(bridge)},
        strict_mcp_config=True,
        permission_mode="dontAsk",
        resume=session_id if history else None,
        session_id=None if history else session_id,
        max_turns=turn.max_turns,
        model=turn.model,
        effort=turn.reasoning_effort,
        cwd=Path(turn.cwd),
        setting_sources=[],
        verbatim_prompts=True,
        session_store=store,
    )

    text: list[str] = []
    result_text: str | None = None
    async for message in query(prompt=turn.prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text.append(block.text)
        elif isinstance(message, ResultMessage):
            result = getattr(message, "result", None)
            if isinstance(result, str) and result:
                result_text = result
            if getattr(message, "is_error", False):
                raise ClaudeSubscriptionTurnError(
                    result_text or "Claude subscription leg returned an error"
                )
    final = result_text or "".join(text)
    if not final.strip():
        raise ClaudeSubscriptionTurnError("Claude subscription leg returned no final text")
    return final


def run_claude_subscription_turn(
    turn: ClaudeSubscriptionTurn,
    dispatch: FrozenLegDispatch,
    bridge: OwnerToolBridge,
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> str:
    """Run one SDK leg while executing MCP calls on the calling owner thread."""
    result: list[str] = []
    failure: list[BaseException] = []
    loop_ready = threading.Event()
    state: dict[str, Any] = {}

    def worker() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(_query_sdk(turn, dispatch, bridge))
        state.update(loop=loop, task=task)
        loop_ready.set()
        try:
            result.append(loop.run_until_complete(task))
        except asyncio.CancelledError:
            failure.append(ClaudeSubscriptionTurnError("Claude subscription turn cancelled"))
        except BaseException as exc:  # propagate on the owner thread
            failure.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=worker, name="claude-subscription-turn", daemon=True)
    thread.start()
    loop_ready.wait(timeout=5.0)
    while thread.is_alive():
        if is_cancelled is not None and is_cancelled():
            bridge.revoke("gateway_run_cancelled")
            loop = state.get("loop")
            task = state.get("task")
            if loop is not None and task is not None:
                loop.call_soon_threadsafe(task.cancel)
            thread.join(timeout=5.0)
            if thread.is_alive():
                raise ClaudeSubscriptionTurnError(
                    "Claude subscription child did not terminate after cancellation"
                )
            raise ClaudeSubscriptionTurnError("Claude subscription turn cancelled")
        bridge.drain_one(timeout=0.05)
    thread.join()
    while bridge.drain_one(timeout=0.0):
        pass
    if failure:
        exc = failure[0]
        if isinstance(exc, ClaudeSubscriptionTurnError):
            raise exc
        raise ClaudeSubscriptionTurnError(str(exc)) from exc
    if not result:
        raise ClaudeSubscriptionTurnError("Claude subscription worker exited without a result")
    return result[0]
