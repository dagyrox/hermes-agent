"""Role-preserving Hermes history projection for one Claude subscription turn.

This module contains no provider calls.  The request-local store is deliberately
non-authoritative; Hermes' SessionDB and failover ledger remain the commit log.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable, Mapping
import uuid


class ProjectionError(ValueError):
    """Hermes history cannot be represented without changing its roles."""


class InMemorySessionStore:
    """Minimal request-local SessionStore boundary used by the Agent SDK adapter."""

    def __init__(self) -> None:
        self._entries: dict[str, list[dict[str, Any]]] = {}

    def append(self, key: str, entries: Iterable[Mapping[str, Any]]) -> None:
        if not key:
            raise ValueError("session store key is required")
        bucket = self._entries.setdefault(key, [])
        bucket.extend(deepcopy([dict(entry) for entry in entries]))

    def load(self, key: str) -> list[dict[str, Any]]:
        return deepcopy(self._entries.get(key, []))


def _text_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None or content == "":
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        blocks: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ProjectionError("message content contains a non-object block")
            kind = part.get("type")
            if kind in {"text", "image"}:
                blocks.append(deepcopy(part))
            elif kind in {"image_url", "input_image"}:
                blocks.append(deepcopy(part))
            else:
                raise ProjectionError(f"unsupported message content block {kind!r}")
        return blocks
    raise ProjectionError("message content must be text or content blocks")


def project_hermes_history(
    history: Iterable[Mapping[str, Any]], *, session_id: str
) -> list[dict[str, Any]]:
    """Project previous Hermes roles into linked Claude transcript entries.

    The caller must pass previous history only.  The current real user content
    remains the Agent SDK query prompt and is therefore sent exactly once.
    """
    if not session_id:
        raise ValueError("Claude request-local session id is required")
    entries: list[dict[str, Any]] = []
    parent_uuid: str | None = None
    known_tool_uses: set[str] = set()
    consumed_tool_uses: set[str] = set()

    for raw in history:
        if not isinstance(raw, Mapping):
            raise ProjectionError("history entry must be an object")
        role = raw.get("role")
        if role == "system":
            raise ProjectionError("system messages belong in the frozen system prompt")
        content: Any
        projected_role: str
        if role == "assistant":
            projected_role = "assistant"
            blocks = _text_blocks(raw.get("content"))
            for call in raw.get("tool_calls") or []:
                if not isinstance(call, Mapping):
                    raise ProjectionError("tool call must be an object")
                call_id = call.get("id")
                function = call.get("function")
                if not isinstance(call_id, str) or not call_id or call_id in known_tool_uses:
                    raise ProjectionError("tool call ids must be unique non-empty strings")
                if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
                    raise ProjectionError("tool call function is malformed")
                arguments = function.get("arguments", "{}")
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError as exc:
                    raise ProjectionError("tool call arguments are not valid JSON") from exc
                if not isinstance(parsed, dict):
                    raise ProjectionError("tool call arguments must decode to an object")
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": function["name"],
                        "input": deepcopy(parsed),
                    }
                )
                known_tool_uses.add(call_id)
            content = blocks
        elif role == "tool":
            call_id = raw.get("tool_call_id")
            if (
                not isinstance(call_id, str)
                or call_id not in known_tool_uses
                or call_id in consumed_tool_uses
            ):
                raise ProjectionError("orphan tool result in Hermes history")
            projected_role = "user"
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": deepcopy(raw.get("content", "")),
                }
            ]
            consumed_tool_uses.add(call_id)
        elif role == "user":
            projected_role = "user"
            content = deepcopy(raw.get("content", ""))
        else:
            raise ProjectionError(f"unsupported Hermes history role {role!r}")

        entry_uuid = str(uuid.uuid4())
        entries.append(
            {
                "type": projected_role,
                "uuid": entry_uuid,
                "parentUuid": parent_uuid,
                "sessionId": session_id,
                "message": {"role": projected_role, "content": content},
            }
        )
        parent_uuid = entry_uuid

    return entries
