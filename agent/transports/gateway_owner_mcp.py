"""Schema-preserving MCP facade for a request-scoped gateway owner bridge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import struct
import sys
from typing import Any, Callable, Mapping
import uuid

_PREFIX = "mcp__gateway-owner__"
_MAX_FRAME_BYTES = 4 * 1024 * 1024


def encode_mcp_tool_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("canonical tool name is required")
    return _PREFIX + name


def decode_mcp_tool_name(name: str) -> str:
    if not isinstance(name, str) or not name.startswith(_PREFIX):
        raise ValueError("tool is not in the gateway-owner MCP namespace")
    canonical = name[len(_PREFIX) :]
    if not canonical:
        raise ValueError("gateway-owner MCP tool name is empty")
    return canonical


class FrozenOwnerMcpFacade:
    """Expose exactly one frozen OpenAI-style tool snapshot through MCP."""

    def __init__(
        self,
        *,
        tools_json: str,
        submit: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        definitions = json.loads(tools_json)
        if not isinstance(definitions, list):
            raise ValueError("frozen tools must be a list")
        self._submit = submit
        self._tools: list[dict[str, Any]] = []
        self._canonical_names: set[str] = set()
        for definition in definitions:
            function = definition.get("function") if isinstance(definition, dict) else None
            if not isinstance(function, dict):
                raise ValueError("frozen tool entry has no function schema")
            name = function.get("name")
            if not isinstance(name, str) or not name or name in self._canonical_names:
                raise ValueError("frozen tool names must be unique non-empty strings")
            self._canonical_names.add(name)
            self._tools.append(
                {
                    "name": encode_mcp_tool_name(name),
                    "description": function.get("description") or f"Hermes {name} tool",
                    "inputSchema": function.get("parameters")
                    or {"type": "object", "properties": {}},
                }
            )

    def list_tools(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._tools))

    def call_tool(
        self, name: str, arguments: Mapping[str, Any], *, call_id: str | None = None
    ) -> dict[str, Any]:
        canonical = decode_mcp_tool_name(name)
        if canonical not in self._canonical_names:
            raise ValueError("tool is not in the frozen tool snapshot")
        if not isinstance(arguments, Mapping):
            raise ValueError("tool arguments must be an object")
        response = self._submit(
            {
                "call_id": call_id or str(uuid.uuid4()),
                "name": canonical,
                "arguments": dict(arguments),
            }
        )
        return dict(response)


class SocketOwnerClient:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.endpoint = str(config["endpoint"])
        self._identity = {
            "op": "call",
            "capability": config["capability"],
            "coordinator_id": config["coordinator_id"],
            "attempt_id": config["attempt_id"],
            "generation": config["generation"],
            "snapshot_digest": config["snapshot_digest"],
        }

    def submit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        frame = {**self._identity, **dict(request)}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(self.endpoint)
            _send_frame(client, frame)
            return _recv_frame(client)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = conn.recv(size)
        if not chunk:
            raise RuntimeError("owner bridge closed mid-frame")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _recv_frame(conn: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(conn, 4))[0]
    if size <= 0 or size > _MAX_FRAME_BYTES:
        raise RuntimeError("owner bridge frame length is invalid")
    value = json.loads(_recv_exact(conn, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("owner bridge response must be an object")
    return value


def _send_frame(conn: socket.socket, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > _MAX_FRAME_BYTES:
        raise RuntimeError("owner bridge request is too large")
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def _build_server(facade: FrozenOwnerMcpFacade):
    from mcp import types
    from mcp.server.lowlevel import Server

    server = Server("gateway-owner", version="1")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"],
            )
            for tool in facade.list_tools()
        ]

    @server.call_tool(validate_input=True)
    async def call_tool(name: str, arguments: dict[str, Any]):
        response = await asyncio.to_thread(facade.call_tool, name, arguments)
        content = response.get("content", "")
        if isinstance(content, str):
            blocks = [types.TextContent(type="text", text=content)]
        else:
            blocks = [
                types.TextContent(
                    type="text",
                    text=json.dumps(content, ensure_ascii=False, separators=(",", ":")),
                )
            ]
        return types.CallToolResult(
            content=blocks,
            isError=bool(response.get("is_error"))
            or response.get("disposition") != "result",
        )

    return server


async def _run(config_path: Path) -> None:
    from mcp.server.stdio import stdio_server

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != 1:
        raise RuntimeError("unsupported gateway-owner bridge protocol")
    client = SocketOwnerClient(config)
    facade = FrozenOwnerMcpFacade(tools_json=config["tools_json"], submit=client.submit)
    server = _build_server(facade)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
            raise_exceptions=False,
        )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m agent.transports.gateway_owner_mcp <private-config>", file=sys.stderr)
        return 2
    config_path = Path(args[0])
    if not config_path.is_file():
        print("gateway-owner MCP config is unavailable", file=sys.stderr)
        return 2
    asyncio.run(_run(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
