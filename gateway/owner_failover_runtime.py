"""Fail-closed helpers for the gateway-owned backend failover bridge."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


class GatewayOwnerFailoverError(RuntimeError):
    """The gateway cannot prove a failover invariant for this turn."""


def freeze_owner_tool_snapshot(
    primary_tools: Any,
    *,
    enabled_toolsets: Sequence[str] | None,
    disabled_toolsets: Sequence[str] | None,
    capability_mode: str,
) -> list[dict[str, Any]]:
    """Freeze the actual primary schemas, or resolve them for older agents."""

    from gateway.owner_tool_bridge import tool_allowed_for_capability
    from model_tools import get_tool_definitions

    if isinstance(primary_tools, list):
        source_tools = deepcopy(primary_tools)
    else:
        source_tools = get_tool_definitions(
            enabled_toolsets=(
                list(enabled_toolsets) if enabled_toolsets is not None else None
            ),
            disabled_toolsets=(
                list(disabled_toolsets) if disabled_toolsets is not None else None
            ),
            quiet_mode=True,
        )
    if not isinstance(source_tools, list):
        raise GatewayOwnerFailoverError(
            "gateway owner bridge requires a primary tool snapshot"
        )
    return [
        deepcopy(tool)
        for tool in source_tools
        if isinstance(tool, dict)
        and tool_allowed_for_capability(
            str((tool.get("function") or {}).get("name") or ""),
            capability_mode,
        )
    ]


def freeze_owner_wire_prompt(
    api_messages: Any,
    *,
    transcript_messages: Any,
    history_offset: int,
) -> dict[str, Any]:
    """Freeze final wire bytes plus the exact internal replay boundary."""

    if not isinstance(api_messages, list) or not api_messages:
        raise GatewayOwnerFailoverError(
            "gateway owner bridge requires final API messages"
        )
    wire_messages = deepcopy(api_messages)
    if not all(isinstance(message, Mapping) for message in wire_messages):
        raise GatewayOwnerFailoverError("final API messages must be objects")
    if wire_messages[0].get("role") != "system" or not isinstance(
        wire_messages[0].get("content"), str
    ):
        raise GatewayOwnerFailoverError(
            "gateway owner bridge requires one leading system message"
        )
    if any(message.get("role") == "system" for message in wire_messages[1:]):
        raise GatewayOwnerFailoverError(
            "gateway owner bridge cannot project multiple system messages"
        )
    current_idx = next(
        (
            idx
            for idx in range(len(wire_messages) - 1, 0, -1)
            if wire_messages[idx].get("role") == "user"
        ),
        -1,
    )
    if current_idx < 1:
        raise GatewayOwnerFailoverError(
            "gateway owner bridge requires a current API user message"
        )
    current = wire_messages[current_idx].get("content")
    if current is None:
        raise GatewayOwnerFailoverError(
            "gateway owner bridge current API user message has no content"
        )
    if (
        not isinstance(transcript_messages, list)
        or history_offset < 0
        or history_offset >= len(transcript_messages)
        or not isinstance(transcript_messages[history_offset], Mapping)
        or transcript_messages[history_offset].get("role") != "user"
    ):
        raise GatewayOwnerFailoverError(
            "gateway owner bridge requires a current transcript user boundary"
        )
    return {
        "system": wire_messages[0]["content"],
        "history": wire_messages[1:current_idx],
        "current": current,
        "wire_messages": wire_messages,
        "transcript_prefix": deepcopy(transcript_messages[:history_offset]),
        "transcript_current": deepcopy(transcript_messages[history_offset]),
    }


def observe_current_attempt(
    result: Mapping[str, Any],
    *,
    history_offset: int,
    transcript_prefix: Sequence[Mapping[str, Any]],
    transcript_current: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return observed ``(tool_activity, partial_output)``; unknown is adverse."""

    messages = result.get("messages")
    if (
        not isinstance(messages, list)
        or not isinstance(history_offset, int)
        or history_offset < 0
        or history_offset >= len(messages)
        or messages[:history_offset] != list(transcript_prefix)
        or messages[history_offset] != transcript_current
    ):
        return True, True
    current_messages = messages[history_offset:]
    if sum(
        1
        for message in current_messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    ) != 1:
        return True, True

    tool_activity = False
    partial_output = bool(result.get("partial") or result.get("response_previewed"))
    for message in current_messages:
        if not isinstance(message, Mapping):
            return True, True
        role = message.get("role")
        tool_calls = message.get("tool_calls")
        if role == "tool" or (isinstance(tool_calls, list) and bool(tool_calls)):
            tool_activity = True
        content = message.get("content")
        if role != "assistant":
            continue
        if isinstance(content, str):
            partial_output = partial_output or bool(content.strip())
        elif isinstance(content, list):
            partial_output = partial_output or any(
                not isinstance(part, Mapping)
                or part.get("type") != "text"
                or bool(str(part.get("text") or "").strip())
                for part in content
            )
        elif content is not None:
            partial_output = True
    return tool_activity, partial_output


def prepare_owner_replay_messages(
    result: Mapping[str, Any],
    *,
    history_offset: int,
    transcript_prefix: Sequence[Mapping[str, Any]],
    transcript_current: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build replay input only from the frozen, revalidated transcript boundary."""

    tool_activity, partial_output = observe_current_attempt(
        result,
        history_offset=history_offset,
        transcript_prefix=transcript_prefix,
        transcript_current=transcript_current,
    )
    if tool_activity or partial_output:
        raise GatewayOwnerFailoverError(
            "gateway replay projection cannot prove an untouched transcript boundary"
        )
    return deepcopy(list(transcript_prefix)) + [deepcopy(dict(transcript_current))]


def project_owner_final_messages(
    result: Mapping[str, Any],
    *,
    history_offset: int,
    transcript_prefix: Sequence[Mapping[str, Any]],
    transcript_current: Mapping[str, Any],
    final_response: str,
    coordinator_id: str,
    attempt_id: str,
    leg_index: int,
) -> list[dict[str, Any]]:
    """Replace failed-attempt traffic with one identity-stamped final message."""

    if not final_response:
        raise GatewayOwnerFailoverError(
            "gateway owner bridge cannot project an empty final response"
        )
    prefix = prepare_owner_replay_messages(
        result,
        history_offset=history_offset,
        transcript_prefix=transcript_prefix,
        transcript_current=transcript_current,
    )
    metadata = {
        "backend_failover_coordinator_id": coordinator_id,
        "backend_failover_attempt_id": attempt_id,
        "backend_failover_leg_index": int(leg_index),
    }
    prefix.append(
        {
            "role": "assistant",
            "content": final_response,
            **metadata,
            "display_metadata": metadata,
        }
    )
    return prefix
