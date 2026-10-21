# Worker-only supervised backend failover — staged candidate status

Project: Agent Wiring / Agent Routing & Automatic Failover
Primary repository: `dagyrox/hermes-agent`
Consumer repository: `dagyrox/or7-platform`, Issues #259 and #260
Acting lane: Talos
Implementation card/runs: `t_8a13918e` / runs 549 and 558
Current release packet: `/root/.hermes/work-packets/fallbacks-finish-release.md`
Original extraction packet: `/root/.hermes/run-state/worker-only-staged-release.md`
Integrated dependency: `origin/main` at `cb85f67a575b139433bba61d957f243bba927efc`
Extracted worker source boundary: `99521b4ac82f`

## Scope and status

This branch is a worker-only extraction integrated with the accepted shared-credential and CI-baseline changes on current `main`. It retains the dispatcher/Kanban supervised-worker path and its narrow shared CLI/conversation metadata, model capability evidence, MCP lifecycle surface, configuration, and tests.

It deliberately excludes `gateway/backend_failover.py`, the gateway coordinator tests, owner conversation bridges, owner-turn delivery, gateway adapters, and the later PR #14 gateway-ledger commits `0a4f95fc719` and `afb9a7154b4`. Ordinary gateway behavior is unchanged; no worker-only coordinator or bridge is imported there.

The feature remains disabled by default while this candidate receives independent exact-head review and hosted CI. Worker-only release, installation, a finite authorized canary, and limited trusted-role activation are separate post-review steps under the current release packet. Conversation fallback remains off and outside this branch.

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

The branch consumes the merged shared-credential implementation from current `main`. It does not copy credentials, widen trusted profiles/providers, alter login/config stores, or restore stale tokens. Trusted-profile membership and provider allowlists remain exact; guest/client profiles and Anthropic shared inheritance remain denied; local profile overrides remain authoritative.

## Evidence limits

Synthetic scripted transports in the isolated acceptance suite are test fixtures, not provider responses and not proof of a live three-provider chain. A real-backend canary requires separate current authorization, finite scope, existing credentials/subscriptions, exact model/source readback, and observed board terminal acknowledgement. No provider spend, install, restart, activation, merge, or deployment is authorized by this document.

## Release gates

1. Worker-only isolated subprocess/replay/identity/credential/approval acceptance and independent exact-head Athena review.
2. Hosted CI green at the exact reviewed PR head.
3. Worker-only merge and supported installation, preserving current credential authority and rollback state.
4. One newly created eligible, no-override, non-`goal_mode` trusted internal canary with exact supported profile identity, finite authorized provider use, real artifact readback, and observed terminal acknowledgement.
5. Limited worker-role activation and verification; conversation fallback remains off until its independent review and acceptance.

Until all gates are satisfied, activation remains NO-GO.
