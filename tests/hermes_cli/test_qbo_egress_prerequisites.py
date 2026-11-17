from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginManager


def test_opted_in_request_middleware_failure_propagates():
    manager = PluginManager()

    def guard(**_kwargs):
        raise RuntimeError("withhold protected request")

    setattr(guard, "fails_closed", True)
    manager._middleware["llm_request"] = [guard]

    with pytest.raises(RuntimeError, match="withhold protected request"):
        manager.invoke_middleware("llm_request", request={"messages": []})


def test_default_request_middleware_failure_remains_fail_open():
    manager = PluginManager()

    def observer(**_kwargs):
        raise RuntimeError("observer failed")

    manager._middleware["llm_request"] = [observer]

    assert manager.invoke_middleware("llm_request", request={"messages": []}) == []


@pytest.mark.parametrize(
    "task",
    ["title_generation", "compression", "vision", "web_extract"],
)
def test_auxiliary_transport_runs_registered_middleware_immediately_before_send(
    monkeypatch, task,
):
    from agent import auxiliary_client
    from hermes_cli import plugins as plugins_mod

    manager = PluginManager()
    manager._discovered = True
    seen = []

    def guard(**kwargs):
        seen.append(kwargs)

    setattr(guard, "fails_closed", True)
    manager._middleware["llm_request"] = [guard]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

    sent = []
    client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: sent.append(kwargs) or "ok")
        ),
    )

    result = auxiliary_client._create_aux_completion(
        client,
        {"model": "fixture-model", "messages": [{"role": "user", "content": "fixture"}]},
        task=task,
        provider="openrouter",
    )

    assert result == "ok"
    assert sent == [{
        "model": "fixture-model",
        "messages": [{"role": "user", "content": "fixture"}],
    }]
    assert seen[0]["auxiliary_task"] == task
    assert seen[0]["provider"] == "openrouter"
    assert seen[0]["base_url"] == "https://openrouter.ai/api/v1"
    assert seen[0]["is_aggregator"] is True
    assert seen[0]["is_routing_aggregator"] is True


def test_fail_closed_validator_runs_after_all_request_transformers(monkeypatch):
    from agent import auxiliary_client
    from hermes_cli import plugins as plugins_mod

    manager = PluginManager()
    manager._discovered = True
    observed = []

    def validator(**kwargs):
        observed.append(dict(kwargs["request"]))

    setattr(validator, "fails_closed", True)

    def transformer(**kwargs):
        return {"request": {**kwargs["request"], "transformed": True}}

    # Register the validator first to prove security order does not depend on
    # plugin discovery order.
    manager._middleware["llm_request"] = [validator, transformer]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

    sent = []
    client = SimpleNamespace(
        base_url="https://api.openai.com/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: sent.append(kwargs) or "ok")
        ),
    )
    auxiliary_client._create_aux_completion(
        client, {"model": "fixture", "messages": []},
        task="compression", provider="openai-api",
    )

    assert observed == [{"model": "fixture", "messages": [], "transformed": True}]
    assert sent == observed


def test_fail_closed_validator_cannot_mutate_or_replace_request():
    manager = PluginManager()

    def mutating_validator(**kwargs):
        kwargs["request"]["messages"].append({"role": "user", "content": "changed"})

    setattr(mutating_validator, "fails_closed", True)
    manager._middleware["llm_request"] = [mutating_validator]

    original = {"messages": []}
    with pytest.raises(RuntimeError, match="mutated its validation payload"):
        manager.invoke_middleware("llm_request", request=original)
    assert original == {"messages": []}

    def replacing_validator(**kwargs):
        return {"request": {**kwargs["request"], "model": "different"}}

    setattr(replacing_validator, "fails_closed", True)
    manager._middleware["llm_request"] = [replacing_validator]
    with pytest.raises(RuntimeError, match="validator, not a transformer"):
        manager.invoke_middleware("llm_request", request=original)


@pytest.mark.parametrize(
    ("route_label", "base_url"),
    [
        ("auto", "https://openrouter.ai/api/v1"),
        ("fallback_chain[0](openrouter)", "https://openrouter.ai/api/v1"),
    ],
)
def test_auxiliary_policy_gets_concrete_transport_provider(
    monkeypatch, route_label, base_url,
):
    from agent import auxiliary_client
    from hermes_cli import plugins as plugins_mod

    manager = PluginManager()
    manager._discovered = True
    seen = []

    def validator(**kwargs):
        seen.append(kwargs)

    setattr(validator, "fails_closed", True)
    manager._middleware["llm_request"] = [validator]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    client = SimpleNamespace(
        base_url=base_url,
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_kwargs: "ok")
        ),
    )

    auxiliary_client._create_aux_completion(
        client, {"model": "fixture", "messages": []},
        task="vision", provider=route_label,
    )

    assert seen[0]["provider"] == "openrouter"
    assert seen[0]["route_label"] == route_label
    assert seen[0]["is_routing_aggregator"] is True


def test_auxiliary_transport_withholds_when_fail_closed_guard_raises(monkeypatch):
    from agent import auxiliary_client
    from hermes_cli import plugins as plugins_mod

    manager = PluginManager()
    manager._discovered = True

    def guard(**_kwargs):
        raise RuntimeError("destination denied")

    setattr(guard, "fails_closed", True)
    manager._middleware["llm_request"] = [guard]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

    calls = []
    client = SimpleNamespace(
        base_url="https://disallowed.invalid/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: calls.append(kwargs))
        ),
    )

    with pytest.raises(RuntimeError, match="destination denied"):
        auxiliary_client._create_aux_completion(
            client,
            {"model": "fixture-model", "messages": []},
            task="compression",
            provider="custom:disallowed",
        )

    assert calls == []


def test_mid_ladder_fallback_is_revalidated_for_resolved_destination(monkeypatch):
    from agent import auxiliary_client
    from hermes_cli import plugins as plugins_mod

    manager = PluginManager()
    manager._discovered = True
    observed = []

    def guard(**kwargs):
        observed.append(kwargs)
        if kwargs["provider"] == "openrouter":
            raise RuntimeError("fallback destination denied")

    setattr(guard, "fails_closed", True)
    manager._middleware["llm_request"] = [guard]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

    sent = []
    fallback_client = SimpleNamespace(
        base_url="https://openrouter.ai/api/v1",
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: sent.append(kwargs))
        ),
    )

    with pytest.raises(RuntimeError, match="fallback destination denied"):
        auxiliary_client._call_fallback_candidate_sync(
            fallback_client,
            "fixture-model",
            "fallback_chain[0](openrouter)",
            task="compression",
            messages=[{"role": "user", "content": "synthetic fixture"}],
            temperature=None,
            max_tokens=None,
            tools=None,
            effective_timeout=30.0,
            effective_extra_body={},
            reasoning_config=None,
        )

    assert sent == []
    assert observed[0]["provider"] == "openrouter"
    assert observed[0]["route_label"] == "fallback_chain[0](openrouter)"
    assert observed[0]["egress_path"] == "auxiliary"


@pytest.mark.asyncio
async def test_async_auxiliary_transport_uses_same_guarded_seam(monkeypatch):
    from agent import auxiliary_client
    from hermes_cli import plugins as plugins_mod

    manager = PluginManager()
    manager._discovered = True
    seen = []

    def guard(**kwargs):
        seen.append(kwargs)
        return {"request": kwargs["request"]}

    setattr(guard, "fails_closed", True)
    manager._middleware["llm_request"] = [guard]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)

    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        return "ok"

    client = SimpleNamespace(
        base_url="https://api.anthropic.com/v1",
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )
    result = await auxiliary_client._create_aux_completion_async(
        client,
        {"model": "fixture", "messages": [], "stream": True},
        task="compression",
        provider="anthropic",
    )

    assert result == "ok"
    assert sent[0]["stream"] is True
    assert seen[0]["stream"] is True


def test_interactive_owner_context_rejects_worker_and_returns_gateway_owner(monkeypatch):
    from tools import approval

    values = {
        "HERMES_SESSION_PLATFORM": "slack",
        "HERMES_SESSION_USER_ID": "U-owner",
        "HERMES_SESSION_CHAT_ID": "C-private",
        "HERMES_SESSION_THREAD_ID": "1700.1",
    }
    monkeypatch.setattr(
        "gateway.session_context.get_session_env",
        lambda key, default="": values.get(key, default),
    )
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)

    assert approval.get_interactive_owner_context() == {
        "platform": "slack",
        "owner_id": "U-owner",
        "chat_id": "C-private",
        "thread_id": "1700.1",
    }

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_fixture")
    assert approval.get_interactive_owner_context() is None

    monkeypatch.delenv("HERMES_KANBAN_TASK")
    from agent.delegation_context import delegated_child_context

    with delegated_child_context():
        assert approval.get_interactive_owner_context() is None


def test_gateway_resolution_records_authenticated_platform_user():
    from tools import approval

    entry = approval._ApprovalEntry({"command": "fixture"})
    with approval._lock:
        approval._gateway_queues["fixture-session"] = [entry]
    try:
        assert approval.resolve_gateway_approval(
            "fixture-session", "once", decided_by="U-clicker",
        ) == 1
        assert entry.result == "once"
        assert entry.decided_by == "U-clicker"
    finally:
        with approval._lock:
            approval._gateway_queues.pop("fixture-session", None)