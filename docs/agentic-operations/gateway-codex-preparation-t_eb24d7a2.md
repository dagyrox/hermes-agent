# Talos PR18 Codex preparation repair — incomplete overall checkpoint

Project: Agent Routing & Automatic Failover / Cross-Provider Chains
Primary repository: dagyrox/hermes-agent (PR #18)
Consumer/tracking repository: dagyrox/or7-platform#260
Acting lane/model: Talos / openai-codex / gpt-6.1-sol
Task/run: t_eb24d7a2 / 592; sole-writer successor to t_263dc0a1 / 582
Branch: talos/issue260-t_263dc0a1-run582-recovered-gateway
Worktree: /root/.hermes/kanban/workspaces/t_28664add
Work packet: /root/.hermes/work-packets/fallbacks-recovery-20261021.md
Delta specification: /root/.hermes/run-state/athena-pr18-1c0be87c-delta-verdict.md
Implementation contract: /root/.hermes/run-state/issue260-owner-bridge-implementation.md
Predecessor: 1c0be87c47ea776ac6656fee1d2bf6649ed24f14
Predecessor exact-head CI run 39656777679 concluded success (read back this run).

## Bounded change

The one-shot post-middleware preparation callback now carries transport-normalized
Responses `instructions` separately from `input`. The freeze helper preserves the
complete input list without prepending a synthetic system message, retains the
leading-system contract on chat completions, and records `wire_mode` plus exact
`wire_request` keys (`instructions`/`input` or `messages`) in the digested prompt.
Separate instructions must be non-empty strings; mixed system sources fail closed.

The real regression also exposed that dispatch repeats Responses preflight AFTER
middleware. That existing preflight strips surrounding instruction whitespace.
The enabled callback now performs that same normalization before freezing, so
snapshot instructions equal the actual dispatched instructions. It does not change
the transport normalization policy or the eventual provider payload. The feature-off
path is unchanged. The helper itself preserves its supplied instruction bytes.

Removed the unused `project_owner_final_messages` helper rather than adding a
second projection route. Existing final projection keeps actual Claude tool rows
from `OwnerTurnExecution.messages`, appends one identity-stamped final, and flushes
through the original owner. No identity, approval, replay, lifecycle, or delivery
fences were relaxed.

## Real RED / GREEN evidence

New `tests/gateway/test_owner_turn_preparation.py` enters real
`GatewayRunner._run_agent`, initializes the real `AIAgent`, invokes real conversation
preparation/build/preflight and the actual runner-installed callback, then inspects
the real ledger plan. Only external provider response and deterministic test setup
are substituted. No hand-written `api_messages` are fed to the callback. Request
middleware changes both instructions and current content before freezing; roleful
user/assistant history is retained. Provider-bound kwargs equal the prepared,
transport-normalized request and the frozen selected wire keys exactly.

Before production edits, canonical runner: 2 failed. Codex logged
`GatewayOwnerFailoverError: gateway owner bridge requires one leading system message`
and the runner rejected the missing prepared boundary. Chat reached the boundary
and primary final successfully, then failed the new `wire_mode` assertion.
Initial collection/factory-spy setup errors were corrected before this valid RED.
After repair: both real-path cases and 9 direct boundary cases passed. The direct
cases cover no-history Responses input, exact whitespace preservation by the helper,
source-mutation isolation, invalid instructions, mixed sources and chat guards.

## Executed final verification

From repository root:

```sh
HERMES_PYTHON=/usr/local/lib/hermes-agent/venv/bin/python scripts/run_tests.sh tests/gateway/test_owner_turn_failover_runner.py tests/gateway/test_owner_turn_failover_integration.py tests/gateway/test_backend_failover_coordinator.py tests/gateway/test_owner_tool_bridge.py tests/agent/transports/test_gateway_owner_mcp.py tests/hermes_cli/test_backend_failover.py tests/hermes_cli/test_backend_failover_worker.py tests/hermes_cli/test_worker_only_failover_e2e.py tests/gateway/test_owner_turn_preparation.py tests/gateway/test_owner_wire_prompt.py tests/agent/transports/test_gateway_claude_sdk.py tests/gateway/test_delivery_ledger.py tests/gateway/test_session_state_cleanup.py tests/run_agent/test_conversation_fallback_state.py --file-retries 0 -j 4 --tb=short
```

Result: 14 files, 234 passed, 0 failed, no retries (38.0s).

| File (under tests/) | Passed |
| --- | ---: |
| agent/transports/test_gateway_claude_sdk.py | 1 |
| agent/transports/test_gateway_owner_mcp.py | 7 |
| gateway/test_backend_failover_coordinator.py | 40 |
| gateway/test_delivery_ledger.py | 32 |
| gateway/test_owner_turn_failover_integration.py | 17 |
| gateway/test_owner_wire_prompt.py | 9 |
| gateway/test_owner_tool_bridge.py | 8 |
| gateway/test_owner_turn_failover_runner.py | 8 |
| gateway/test_owner_turn_preparation.py | 2 |
| gateway/test_session_state_cleanup.py | 13 |
| hermes_cli/test_backend_failover.py | 25 |
| hermes_cli/test_backend_failover_worker.py | 67 |
| run_agent/test_conversation_fallback_state.py | 2 |
| hermes_cli/test_worker_only_failover_e2e.py | 3 |

Additional commands: `/usr/local/lib/hermes-agent/venv/bin/python -m ruff check .`
passed; `.../python scripts/check-windows-footguns.py --all` passed (819 scanned);
`uv lock --check` passed (234 packages); `git diff --check` passed.

## Remaining original scope and handoff

This repairs the preparation seam, NOT the full shipping composition. The real
primary test returns a primary final; it does not exercise the primary-failure ->
Claude-provider -> original tools/approval/delivery composition. The third DeepSeek
gateway leg remains unwired. All five shipping acceptance scenario families in the
implementation packet remain open and require real composition tests (the existing
similarly named integration tests are still helper-level).

New exact-head hosted CI and independent Athena DELTA remain required. Exact
published SHA/tree/ref parity, CI URL and reviewer card are recorded in PR18 and the
Kanban handoff after publication, not self-referentially embedded in this commit.
Review must verify NEW-1 closure and dead-helper removal without reopening accepted
invariants. The reviewer should immediately route the next bounded repair or original-
scope acceptance packet to the proper lane; no Jason review request. Any repair
successor must be the sole writer, with this predecessor held. No duplicate writers.

No merge, install, restart, activation, timer, config, credential changes or provider
inference were performed. Root/conversation fallback remains disabled; no readiness
or Issue260 completion is claimed. Historical recovery document describes the old
head and is superseded only for this specific preparation blocker by this delta.
