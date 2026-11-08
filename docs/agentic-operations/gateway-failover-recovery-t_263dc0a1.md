# Talos recovered gateway failover checkpoint — NOT release-ready

Project: Agent Routing & Automatic Failover / Cross-Provider Chains
Primary repository: dagyrox/hermes-agent
Consumer/tracking repository: dagyrox/or7-platform#260
Acting lane/model: Talos / openai-codex / gpt-6.1-sol
Task/run: t_263dc0a1 / 582
Recovery packet: /root/.hermes/work-packets/fallbacks-recovery-20261021.md
Implementation packet: /root/.hermes/run-state/issue260-owner-bridge-implementation.md
Recovered predecessor: t_28664add / run577, HEAD 1b2ee2098c
Historical review baseline: PR14, 3becde75530f53cbffb0766d6959bd44bb2aec07
Base: 34c1e9eed711389ce19f92dd205a0172de207fe2 (already an ancestor)
Candidate branch: talos/issue260-t_263dc0a1-run582-recovered-gateway

## Preservation and publication

The recovered branch and PR14 diverge (15 local / 12 remote commits at recovery).
A merge-tree probe reports seven conflicted paths. Publication uses a new branch
without rewriting either ancestry. PR14 remains untouched except a supersession
comment. No reset, force push, merge, installation, activation, restart, timer,
credential or runtime-config change is authorized or performed.

The inherited nine modified files and two untracked files were preserved before
edits under `.git/talos-recovery-t_263dc0a1/` (binary patch and untracked copies).
Inherited production changes are checkpointed, not represented as newly authored
or fully accepted work by this run.

## Regression repair and executed evidence

The canonical runner reproduced 3 passed / 1 failed: the fallback-error fixture
omitted `current_attempt_progress`, so the accepted fail-closed classifier never
started fallback leg 1. The fixture now supplies affirmative zero-event evidence.
The identity-drift fixture had the same masking omission and was corrected too.
Four additional runner cases assert no fallback for missing progress, observed
assistant output, approval activity, and uncertain progress. Production replay
safeguards were not weakened.

Commands (from repository root):

- `HERMES_PYTHON=/usr/local/lib/hermes-agent/venv/bin/python scripts/run_tests.sh tests/gateway/test_owner_turn_failover_runner.py tests/gateway/test_owner_turn_failover_integration.py tests/gateway/test_backend_failover_coordinator.py tests/gateway/test_owner_tool_bridge.py tests/agent/transports/test_gateway_owner_mcp.py tests/hermes_cli/test_backend_failover.py tests/hermes_cli/test_backend_failover_worker.py tests/hermes_cli/test_worker_only_failover_e2e.py --file-retries 0 -j 4`
  Result: 8 files, 175 passed, 0 failed.
- `HERMES_PYTHON=/usr/local/lib/hermes-agent/venv/bin/python scripts/run_tests.sh tests/agent/transports/test_gateway_claude_sdk.py tests/gateway/test_delivery_ledger.py tests/gateway/test_session_state_cleanup.py tests/run_agent/test_conversation_fallback_state.py --file-retries 0 -j 4`
  Result: 4 files, 48 passed, 0 failed.
- `python -m ruff check` on the ten changed/new Python files: passed;
  existing invalid-noqa warning at run_agent.py:107 remains.
- `uv lock --check`: passed (234 packages resolved).
- `git diff --check`: passed. Canonical runner also precompiled tracked Python.

These are isolated fixture tests, not live provider or platform acceptance.
The runner tests import and enter real GatewayRunner._run_agent, but replace
AIAgent and the Claude provider source. They do NOT prove the real primary
preparation seam, original owner executor/approval roundtrip or receipt delivery
through the full shipping composition. Existing five named integration scenarios
are helper-level tests despite their names; no full five-scenario acceptance is
claimed. Hosted CI and exact commit/tree parity are recorded separately in the
PR and Kanban handoff after publication.

## Numbered residual blockers and deterministic continuation

1. **Real Codex wire preparation is still broken (Talos implementation lane).**
   A direct no-network call to `ResponsesApiTransport().preflight_kwargs` with
   model, instructions, input and store returns separate `instructions` and
   `input` fields; input begins with the user, not a system message.
   `agent/conversation_loop.py` forwards that input list to the recovered
   preparation callback, while `freeze_owner_wire_prompt` requires a leading
   system message. Calling that helper on the real preflight output reproduces
   `GatewayOwnerFailoverError: gateway owner bridge requires one leading system message`.
   Thus the configured Codex-primary enabled path fails closed before dispatch.
   Do not activate or call the runner fixtures proof this path works. Next repair
   must preserve separate wire instructions, roleful canonical history, final
   middleware-mutated request/tool bytes and exact frozen equality; do not prepend
   raw pre-preparation text or flatten history merely to make the check pass.
   Add real AIAgent/transport preparation coverage before claiming closure.
2. **Independent DELTA review and exact-head hosted CI remain gates (Athena / CI).**
   Review recovered prompt/schema/progress/identity/final projection and delivery
   claim-token changes against PR14 findings; do not repeat already accepted
   worker architecture. Carry blocker 1 explicitly rather than issue a positive
   release verdict from fixture counts. Review other inherited safety gaps,
   including crash-safe final projection and owner approval lifetime, as needed.
3. **Gateway canonical third leg is not wired (existing scope limitation).**
   Recovered gateway code slices the configured chain to primary -> Claude.
   DeepSeek is not executed on this gateway path. Full chain completion, live
   acceptance and conversation readiness are NOT claimed. Worker three-leg
   behavior is a separate already-integrated surface.

Keep predecessor cards t_28664add and t_13828b39 HELD. Continue in this preserved
checkout only after the owning coordinator routes the independent review and
bounded repair; no second writer. Root/conversation fallback stays disabled.
