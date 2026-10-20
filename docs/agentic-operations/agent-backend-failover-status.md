# Worker-only supervised backend failover — staged candidate status

Project: Agent Wiring / Agent Routing & Automatic Failover
Primary repository: `dagyrox/hermes-agent`
Consumer repository: `dagyrox/or7-platform`, Issues #259 and #260
Acting lane: Talos
Implementation card/run: `t_8a13918e` / run 549
Review packet: `/root/.hermes/run-state/worker-only-staged-release.md`
Dependency: PR #13 head `2949cbbe24c7d07c0367712a6ad453a1a1c2168f` (open and unaccepted)
Extracted worker source boundary: `99521b4ac82f`

## Scope and status

This branch is a worker-only extraction built on the open PR #13 dependency. It retains the dispatcher/Kanban supervised-worker path and its narrow shared CLI/conversation metadata, model capability evidence, MCP lifecycle surface, configuration, and tests.

It deliberately excludes `gateway/backend_failover.py`, the gateway coordinator tests, owner conversation bridges, owner-turn delivery, gateway adapters, and the later PR #14 gateway-ledger commits `0a4f95fc719` and `afb9a7154b4`. Ordinary gateway behavior is unchanged; no worker-only coordinator or bridge is imported there.

The feature is disabled by default. This candidate is not activation, merge, deployment, or Issue #260 completion authority. PR #13 has not been accepted or merged, its required actual Claude subscription rereview remains outstanding, and the worker candidate requires independent exact-head review.

## Preserved worker surfaces

- `agent/conversation_loop.py`: narrow classified failed-result metadata.
- `agent/models_dev.py`: observed capability evidence.
- `agent/transports/hermes_tools_mcp_server.py`: task-fenced worker lifecycle tools.
- `cli.py`: supervised-child media and classified-exit mapping.
- `hermes_cli/backend_failover.py`, `backend_failover_worker.py`, and `backend_smoke.py`: route validation, supervision, status, and bounded probes.
- `hermes_cli/config.py`, `fallback_config.py`, `kanban.py`, `kanban_db.py`, and the dispatcher annotation: disabled configuration, nested-child suppression, status, and spawn selection.
- Focused tests under `tests/hermes_cli/` and `tests/agent/transports/`.

## Binding safety boundary

A later backend may start only when the current attempt has affirmative pre-execution evidence and a known availability-class failure. Refusal, assistant output, tool activity (including read-only tools), approval wait or denial, timeout/cancellation, malformed or unknown stream evidence, lost ownership, and a durable started-but-unfinished ledger all deny replay and require terminal handling or reconciliation.

Task identity, active run, assignee profile, claim ownership, and workspace stay fixed across legs. Model/provider overrides and `goal_mode` bypass supervised routing. Unsupported tool, media, context, credential, or authority capability fails closed before child execution. Human approval remains external to every backend; no fallback may self-approve.

## Credential dependency

The branch consumes PR #13 through an explicit stacked base. It does not copy credentials, widen trusted profiles/providers, alter login/config stores, or modify the credential owner's branch. Trusted-profile membership and provider allowlists remain exact; guest/client profiles and Anthropic shared inheritance remain denied; local profile overrides remain authoritative.

The two stale zero-argument xAI test mocks identified on PR #13 remain the credential owner's correction. Their candidate-specific failures must not be reported as frozen-main baseline failures.

## Evidence limits

Synthetic scripted transports in the isolated acceptance suite are test fixtures, not provider responses and not proof of a live three-provider chain. A real-backend canary requires separate current authorization, finite scope, existing credentials/subscriptions, exact model/source readback, and observed board terminal acknowledgement. No provider spend, install, restart, activation, merge, or deployment is authorized by this document.

## Release gates

1. PR #13 correction and mandated actual Claude subscription exact-head acceptance, plus authorized adjudication of hosted failures.
2. Worker-only isolated E2E/negative/credential acceptance and independent exact-head Athena review.
3. Separate human approval for merge, installation, activation, restart, exact canary roles/providers, finite budget, and rollback ownership.
4. One newly created eligible, no-override, non-`goal_mode` canary with exact supported profile identity and observed terminal acknowledgement.

Until all gates are satisfied, activation remains NO-GO.
