"""Transport-specific wire snapshots never rewrite provider input."""
from copy import deepcopy

import pytest

from gateway.owner_failover_runtime import (
    GatewayOwnerFailoverError,
    freeze_owner_wire_prompt,
)


def test_responses_freezes_separate_instructions_without_injecting_system():
    messages = [{"role": "user", "content": "current"}]
    original = deepcopy(messages)
    frozen = freeze_owner_wire_prompt(
        messages, system_instructions="  exact instructions\n",
        transcript_messages=messages, history_offset=0,
    )
    assert frozen["system"] == "  exact instructions\n"
    assert frozen["history"] == []
    assert frozen["current"] == "current"
    assert frozen["wire_mode"] == "responses"
    assert frozen["wire_request"] == {
        "instructions": "  exact instructions\n", "input": original,
    }
    assert messages == original
    messages[0]["content"] = "later mutation"
    assert frozen["wire_messages"] == original
    assert frozen["transcript_current"] == original[0]


@pytest.mark.parametrize("instructions", ["", " \n", 123, [], {}])
def test_responses_rejects_invalid_separate_instructions(instructions):
    messages = [{"role": "user", "content": "current"}]
    with pytest.raises(GatewayOwnerFailoverError, match="non-empty wire instructions"):
        freeze_owner_wire_prompt(
            messages, system_instructions=instructions,
            transcript_messages=messages, history_offset=0,
        )


def test_responses_rejects_mixed_system_sources():
    messages = [
        {"role": "system", "content": "conflicting system"},
        {"role": "user", "content": "current"},
    ]
    with pytest.raises(GatewayOwnerFailoverError, match="cannot mix"):
        freeze_owner_wire_prompt(
            messages, system_instructions="instructions",
            transcript_messages=messages[1:], history_offset=0,
        )


@pytest.mark.parametrize("system_messages", [[], [
    {"role": "system", "content": "one"},
    {"role": "system", "content": "two"},
]])
def test_chat_preserves_leading_single_system_requirement(system_messages):
    messages = [{"role": "user", "content": "current"}]
    with pytest.raises(GatewayOwnerFailoverError, match="system message"):
        freeze_owner_wire_prompt(
            system_messages + messages, transcript_messages=messages, history_offset=0,
        )
