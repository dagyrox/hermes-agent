from __future__ import annotations

import json
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource


_TOOL = {
    "type": "function",
    "function": {
        "name": "todo",
        "description": "owner todo",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _failover_config() -> dict:
    return {
        "agent_backend_failover": {
            "enabled": True,
            "gateway_owner_bridge": True,
            "max_attempts_per_leg": 1,
            "roles": {
                "talos": [
                    {
                        "kind": "api",
                        "provider": "openai-codex",
                        "model": "gpt-6.1-sol",
                        "reasoning_effort": "high",
                    },
                    {
                        "kind": "claude_code",
                        "model": "claude-sonnet-5-5",
                        "reasoning_effort": "high",
                        "capability_mode": "full",
                    },
                    {
                        "kind": "api",
                        "provider": "deepseek",
                        "model": "deepseek-flash",
                    },
                ]
            },
        }
    }


class _FailingPrimaryAgent:
    result: dict = {}
    last_instance = None

    def __init__(self, *args, **kwargs):
        type(self).last_instance = self
        self.session_id = kwargs["session_id"]
        self._cached_system_prompt = None
        self._interrupt_requested = False
        self.stream_delta_callback = None
        self._stream_callback = None
        self.flushed_messages = None

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        callback = getattr(
            self, "_gateway_owner_turn_prepared_callback", None
        )
        if callable(callback):
            callback(
                api_messages=[
                    {
                        "role": "system",
                        "content": "post-preparation-system",
                    },
                    {"role": "user", "content": "prepared prompt"},
                ],
                transcript_messages=[
                    {"role": "user", "content": user_message}
                ],
                tools=self.tools,
                history_offset=0,
            )
        return dict(type(self).result)

    def _flush_messages_to_session_db(self, messages, conversation_history=None):
        self.flushed_messages = list(messages)


def _install_fake_agent(monkeypatch) -> None:
    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = _FailingPrimaryAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)


def _make_runner() -> gateway_run.GatewayRunner:
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._service_tier = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner._pending_model_notes = {}
    runner._session_db = None
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._session_run_generation = {
        "agent:talos:telegram:dm:12345": 7,
    }
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(streaming=None, multiplex_profiles=False)
    runner.session_store = SimpleNamespace(
        get_or_create_session=lambda source: SimpleNamespace(session_id="session-1"),
        load_transcript=lambda session_id: [],
    )
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    runner._enrich_message_with_vision = AsyncMock(return_value="ENRICHED")
    runner._active_profile_name = lambda: "talos"
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        user_id="user-1",
    )


def _configure_runner_test(monkeypatch, tmp_path, *, fake_agent=True) -> None:
    if fake_agent:
        _install_fake_agent(monkeypatch)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_env_path", tmp_path / ".env")
    monkeypatch.setattr(gateway_run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", _failover_config)
    monkeypatch.setattr(gateway_run, "_load_gateway_runtime_config", _failover_config)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_gateway_model",
        lambda config=None: "gpt-6.1-sol",
    )
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://example.test/v1",
            "api_key": "test-key",
        },
    )
    import hermes_cli.tools_config as tools_config
    import model_tools

    monkeypatch.setattr(
        tools_config,
        "_get_platform_tools",
        lambda user_config, platform_key: {"core"},
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kwargs: [_TOOL],
    )


@pytest.mark.asyncio
async def test_enabled_gateway_runner_projects_one_owner_final_from_frozen_turn(
    monkeypatch, tmp_path
):
    _configure_runner_test(monkeypatch, tmp_path)
    _FailingPrimaryAgent.result = {
        "final_response": "",
        "messages": [{"role": "user", "content": "raw prompt"}],
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "error": "rate limited",
        "failure_reason": "rate_limit",
        "should_fallback": True,
        "current_attempt_progress": {
            "observed": True,
            "tool_activity": False,
            "assistant_output": False,
            "approval_activity": False,
            "uncertain": False,
        },
    }
    captured = {}

    def _claude(turn, dispatch, bridge, **kwargs):
        captured["turn"] = turn
        captured["dispatch"] = dispatch
        return "claude final"

    import agent.transports.gateway_claude_sdk as claude_sdk

    monkeypatch.setattr(claude_sdk, "run_claude_subscription_turn", _claude)
    runner = _make_runner()

    result = await runner._run_agent(
        message="raw prompt",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="session-1",
        session_key="agent:talos:telegram:dm:12345",
        run_generation=7,
        event_message_id="event-1",
    )

    assert result["final_response"] == "claude final"
    assert result.get("failed", False) is False
    assistant_messages = [
        message for message in result["messages"] if message.get("role") == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["content"] == "claude final"
    assert assistant_messages[0]["backend_failover_attempt_id"].endswith(":1")
    assert assistant_messages[0]["display_metadata"][
        "backend_failover_attempt_id"
    ].endswith(":1")
    assert result["owner_final_delivery"]["payload"] == "claude final"
    assert captured["turn"].prompt == "prepared prompt"
    frozen = json.loads(captured["dispatch"].prompt_json)
    assert frozen["system"] == "post-preparation-system"
    assert frozen["current"] == "prepared prompt"
    assert frozen["wire_messages"][-1]["content"] == "prepared prompt"
    assert json.loads(captured["dispatch"].tools_json) == [_TOOL]
    assert json.loads(captured["dispatch"].leg_json)["kind"] == "claude_code"


@pytest.mark.asyncio
async def test_enabled_gateway_runner_fails_closed_after_current_attempt_tool_activity(
    monkeypatch, tmp_path
):
    _configure_runner_test(monkeypatch, tmp_path)
    _FailingPrimaryAgent.result = {
        "final_response": "",
        "messages": [
            {"role": "user", "content": "raw prompt"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "todo", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "todo",
                "content": "owner result",
            },
        ],
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "error": "rate limited after tool",
        "failure_reason": "rate_limit",
        "should_fallback": True,
        "current_attempt_progress": {
            "observed": True,
            "tool_activity": False,
            "assistant_output": False,
            "approval_activity": False,
            "uncertain": False,
        },
    }
    claude = MagicMock(return_value="must not run")
    import agent.transports.gateway_claude_sdk as claude_sdk

    monkeypatch.setattr(claude_sdk, "run_claude_subscription_turn", claude)
    runner = _make_runner()

    result = await runner._run_agent(
        message="raw prompt",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="session-1",
        session_key="agent:talos:telegram:dm:12345",
        run_generation=7,
        event_message_id="event-2",
    )

    assert result["failed"] is True
    assert "failed" in result["final_response"].lower()
    claude.assert_not_called()


@pytest.mark.asyncio
async def test_enabled_gateway_runner_rechecks_live_owner_identity_before_bridge(
    monkeypatch, tmp_path
):
    _configure_runner_test(monkeypatch, tmp_path)
    _FailingPrimaryAgent.result = {
        "final_response": "",
        "messages": [{"role": "user", "content": "raw prompt"}],
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "error": "rate limited",
        "failure_reason": "rate_limit",
        "should_fallback": True,
    }

    import agent.transports.gateway_claude_sdk as claude_sdk

    claude = MagicMock(return_value="must not run")
    _FailingPrimaryAgent.result["current_attempt_progress"] = {
        "observed": True,
        "tool_activity": False,
        "assistant_output": False,
        "approval_activity": False,
        "uncertain": False,
    }
    monkeypatch.setattr(claude_sdk, "run_claude_subscription_turn", claude)
    runner = _make_runner()
    profiles = iter(("talos", "talos", "athena"))
    runner._active_profile_name = lambda: next(profiles)

    result = await runner._run_agent(
        message="raw prompt",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="session-1",
        session_key="agent:talos:telegram:dm:12345",
        run_generation=7,
        event_message_id="event-1",
    )

    assert "failed" in result["final_response"].lower()
    claude.assert_not_called()


@pytest.mark.asyncio
async def test_enabled_gateway_runner_terminalizes_started_fallback_on_error(
    monkeypatch, tmp_path
):
    _configure_runner_test(monkeypatch, tmp_path)
    _FailingPrimaryAgent.result = {
        "final_response": "",
        "messages": [{"role": "user", "content": "raw prompt"}],
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "error": "rate limited",
        "failure_reason": "rate_limit",
        "should_fallback": True,
    }

    import agent.transports.gateway_claude_sdk as claude_sdk
    from gateway.backend_failover import GatewayFailoverLedger

    # A known zero-event primary failure is required to reach the fallback.
    # Missing progress must stay non-replayable in the production classifier.
    _FailingPrimaryAgent.result["current_attempt_progress"] = {
        "observed": True,
        "tool_activity": False,
        "assistant_output": False,
        "approval_activity": False,
        "uncertain": False,
    }
    monkeypatch.setattr(
        claude_sdk,
        "run_claude_subscription_turn",
        MagicMock(side_effect=RuntimeError("subscription unavailable")),
    )
    observations = []
    original_finish_leg = GatewayFailoverLedger.finish_leg

    def _finish_leg(self, coordinator_id, leg_index, observation, **kwargs):
        observations.append((leg_index, observation))
        return original_finish_leg(
            self, coordinator_id, leg_index, observation, **kwargs
        )

    monkeypatch.setattr(GatewayFailoverLedger, "finish_leg", _finish_leg)
    runner = _make_runner()

    result = await runner._run_agent(
        message="raw prompt",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="session-1",
        session_key="agent:talos:telegram:dm:12345",
        run_generation=7,
        event_message_id="event-terminalize",
    )

    assert "rate limited" in result["final_response"]
    assert observations[-1][0] == 1
    assert observations[-1][1].reason == "bridge_error"
    assert observations[-1][1].partial_output is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["missing_progress", "assistant_output", "approval_activity", "uncertain"],
)
async def test_enabled_gateway_runner_denies_replay_without_zero_event_proof(
    monkeypatch, tmp_path, boundary
):
    _configure_runner_test(monkeypatch, tmp_path)
    _FailingPrimaryAgent.result = {
        "final_response": "",
        "messages": [{"role": "user", "content": "raw prompt"}],
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "error": "rate limited",
        "failure_reason": "rate_limit",
        "should_fallback": True,
    }
    if boundary != "missing_progress":
        progress = {
            "observed": True,
            "tool_activity": False,
            "assistant_output": False,
            "approval_activity": False,
            "uncertain": False,
        }
        progress[boundary] = True
        _FailingPrimaryAgent.result["current_attempt_progress"] = progress
    import agent.transports.gateway_claude_sdk as claude_sdk

    claude = MagicMock(return_value="must not run")
    monkeypatch.setattr(claude_sdk, "run_claude_subscription_turn", claude)
    result = await _make_runner()._run_agent(
        message="raw prompt",
        context_prompt="",
        history=[],
        source=_source(),
        session_id="session-1",
        session_key="agent:talos:telegram:dm:12345",
        run_generation=7,
        event_message_id=f"event-denied-{boundary}",
    )

    assert result["failed"] is True
    claude.assert_not_called()
