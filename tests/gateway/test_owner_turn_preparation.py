"""Real agent/transport/owner-callback composition, without provider inference."""
from copy import deepcopy

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import gateway.run as gateway_run
from tests.gateway.test_owner_turn_failover_runner import (
    _configure_runner_test, _make_runner, _source,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", ["codex_responses", "chat_completions"])
async def test_real_primary_preparation_freezes_transport_wire_shape(
    monkeypatch, tmp_path, api_mode
):
    import run_agent
    import hermes_cli.middleware as middleware
    from agent.transports.codex import ResponsesApiTransport
    from gateway.backend_failover import GatewayFailoverLedger

    _configure_runner_test(monkeypatch, tmp_path, fake_agent=False)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex", "api_mode": api_mode,
            "base_url": "https://example.test/v1", "api_key": "test-key",
        },
    )
    monkeypatch.setattr(run_agent, "OpenAI", MagicMock())
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda **kwargs: [_TOOL])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda **kwargs: {})
    real_agent = run_agent.AIAgent
    real_init = real_agent.__init__
    captured = {"wire": [], "frozen": []}

    def initialize_agent(agent, **kwargs):
        real_init(agent, **kwargs, skip_context_files=True, skip_memory=True)
        captured["agent"] = agent
        agent._cached_system_prompt = "cached instructions"
        agent.compression_enabled = False
        agent._disable_streaming = True
        agent.save_trajectories = False

        def provider(request):
            captured["wire"].append(deepcopy(request))
            if api_mode == "codex_responses":
                return SimpleNamespace(
                    status="completed",
                    output=[SimpleNamespace(
                        type="message", status="completed",
                        content=[SimpleNamespace(type="output_text", text="primary final")],
                    )],
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="primary final", tool_calls=None),
                    finish_reason="stop",
                )], usage=None,
            )

        # Substitute only the network boundary, not preparation or the callback.
        agent._interruptible_api_call = provider

    monkeypatch.setattr(real_agent, "__init__", initialize_agent)
    apply_middleware = middleware.apply_llm_request_middleware

    def mutate_final_request(*args, **kwargs):
        result = apply_middleware(*args, **kwargs)
        payload = result.payload
        if api_mode == "codex_responses":
            payload["instructions"] = "  final middleware instructions\n"
            payload["input"][-1]["content"] = "final middleware current"
        else:
            payload["messages"][0]["content"] = "  final middleware instructions\n"
            payload["messages"][-1]["content"] = "final middleware current"
        captured["prepared"] = deepcopy(payload)
        return result

    monkeypatch.setattr(middleware, "apply_llm_request_middleware", mutate_final_request)
    plan = GatewayFailoverLedger.plan

    def capture_plan(self, turn, **kwargs):
        captured["frozen"].append(deepcopy(turn.prompt))
        return plan(self, turn, **kwargs)

    monkeypatch.setattr(GatewayFailoverLedger, "plan", capture_plan)
    result = await _make_runner()._run_agent(
        message="raw prompt", context_prompt="",
        history=[{"role": "user", "content": "prior question"},
                 {"role": "assistant", "content": "prior answer"}],
        source=_source(), session_id="session-1",
        session_key="agent:talos:telegram:dm:12345", run_generation=7,
        event_message_id=f"event-real-{api_mode}",
    )

    assert result["final_response"] == "primary final", result
    assert isinstance(captured["agent"], real_agent)
    if api_mode == "codex_responses":
        assert isinstance(captured["agent"]._get_transport(), ResponsesApiTransport)
    assert len(captured["frozen"]) == len(captured["wire"]) == 1
    frozen = captured["frozen"][0]
    wire = captured["wire"][0]
    expected = captured["prepared"]
    if api_mode == "codex_responses":
        expected = ResponsesApiTransport().preflight_kwargs(expected, allow_stream=False)
    assert wire == expected
    assert frozen["system"] == (
        wire["instructions"] if api_mode == "codex_responses"
        else wire["messages"][0]["content"]
    )
    assert frozen["current"] == "final middleware current"
    assert [message["role"] for message in frozen["history"]] == ["user", "assistant"]
    key = "input" if api_mode == "codex_responses" else "messages"
    assert frozen["wire_messages"] == wire[key]
    assert frozen["wire_mode"] == ("responses" if key == "input" else "messages")
    keys = ("instructions", "input") if key == "input" else ("messages",)
    assert frozen["wire_request"] == {k: wire[k] for k in keys}
    if key == "input":
        assert not any(message.get("role") == "system" for message in wire[key])


_TOOL = {
    "type": "function",
    "function": {
        "name": "todo", "description": "owner todo",
        "parameters": {"type": "object", "properties": {}},
    },
}
