# Agent Routing & Automatic Failover — draft checkpoint status

Project: Agent Routing & Automatic Failover
Primary repository: `dagyrox/hermes-agent`
Consumer repository: `dagyrox/or7-platform`, Issue #260
Acting lane: Talos
Original implementation card/run: `t_3cb377f6` / run 495 (blocked)
Checkpoint publication card/run: `t_b679d087` / run 496
Work packet: `docs/agentic-operations/agent-backend-failover.md`
Branch: `talos/agent-backend-failover`

## Checkpoint classification

This is an incomplete, reviewable draft checkpoint. It preserves the current implementation and tests for independent review; it is not approval or completion of Issue #260. No runtime activation, service restart, deployment, merge, or full-chain activation has been performed or verified.

The canonical configuration remains disabled by default. The five frozen chains each contain exactly three distinct-provider legs and end in DeepSeek Flash, but a configured route is not evidence that its credentials, capability parity, worker invocation, or end-to-end continuation are correct.

## Preserved implementation inventory

Modified tracked files:

- `agent/conversation_loop.py`
- `agent/transports/hermes_tools_mcp_server.py`
- `cli.py`
- `hermes_cli/config.py`
- `hermes_cli/fallback_config.py`
- `hermes_cli/kanban.py`
- `hermes_cli/kanban_db.py`
- `tests/agent/transports/test_hermes_tools_mcp_server.py`

New checkpoint files:

- `docs/agentic-operations/agent-backend-failover.md`
- `docs/agentic-operations/agent-backend-failover-status.md`
- `docs/agentic-operations/backend-failover.canonical.yaml`
- `hermes_cli/backend_failover.py`
- `hermes_cli/backend_failover_worker.py`
- `hermes_cli/backend_smoke.py`
- `tests/hermes_cli/test_backend_failover.py`
- `tests/hermes_cli/test_backend_failover_worker.py`

## Evidence reported by the preceding Talos run

The following results are self-reported by Talos run 495 and remain pending an independent rerun on the published commit:

- Focused failover/MCP suite: **75 passed, 0 failed**.
- Bounded regression matrix: **369 passed, 0 failed** across 17 failover/auth/cooldown/gateway/Kanban files; this matrix preceded the final two small repairs, while the focused suite was rerun afterward.
- Read-only provider smokes: Claude Code Opus 5.5 **PASS**, Claude Code Sonnet 5.5 **PASS**, and Codex Sol 6.1 **PASS**.
- Codex Luna 6 was unavailable with an observed HTTP 401 authentication rejection.
- DeepSeek Flash was unavailable because the assignee profile had no usable `DEEPSEEK_API_KEY`.
- The smoke subprocesses reportedly excluded parent Kanban lifecycle identity after the isolation repair.

These observations do not constitute a verified three-leg failover run. No complete primary → fallback1 → fallback2 chain, exact-head hosted CI result, or independent acceptance review exists at this checkpoint.

## Unresolved merge blockers

Independent review identified four unresolved blocker classes. This publication card intentionally does not repair them:

1. **Assignee-specific toolsets and API child hooks.** Worker argv must resolve required toolsets from the assignee profile rather than the dispatcher home, and API children must preserve `--accept-hooks` plus pinned `--toolsets`.
2. **Fail-closed assignee profile identity.** Missing or mismatched profile identity must fail closed before constructing Claude SOUL/MCP state; dispatcher-home or generic `~/.hermes` fallback must not silently substitute another identity. Tests need an explicit injectable resolver rather than production fallback behavior.
3. **Profile-scoped credential status/execution parity.** API availability/status resolution currently risks using supervisor/dispatcher credentials while execution belongs to the assignee profile. Status and execution must use the same profile-scoped credential source, or the child must authoritatively classify pre-tool auth failure.
4. **Evidence-derived capabilities.** API and cross-runtime route capabilities, especially image input, must be derived from actual backend/profile evidence rather than a blanket declaration before parity checks can be trusted.

The work packet describes event-driven continuation through persisted Kanban transitions. The current checkpoint does not demonstrate that full coordinator wake path. Slack is only a notification surface; no Slack subscription or message proves Kanban continuation, acknowledgement, or next-leg execution. This Slack-versus-Kanban integration remains uncovered end to end.

## Bounded repair run 498

Talos repair card `t_d298814c` separated dispatcher-owned route configuration from assignee-owned tool configuration, preserved `--accept-hooks` and the exact `--toolsets` pin on supervised API children, and made missing or mismatched task/run/profile/SOUL identity fail closed before scratch MCP or prompt artifacts are created. API availability and `backend-status` now resolve against an isolated assignee-profile secret scope rather than inherited supervisor credentials. The Windows-footgun checker annotations cover both new guarded process-group calls and the inherited guarded dispatcher ownership call.

These repairs address B1, B2, B3's profile-only/supervisor-only credential cases, and B6's static checker. B3's shared-root credential inheritance modes remain coordinated with open PR #13 and are not claimed here. B4 now gives the Claude `full` route restricted workspace shell execution, measures the durable worker handoff plus loaded system prompt, and rejects the unimplemented `delegate_task` surface rather than advertising it; API capability evidence and live image/orchestration acceptance remain unresolved. B5 remains unresolved because the shipping Slack/default gateway path still uses legacy in-process fallback rather than the cross-runtime supervisor, and no persisted coordinator acknowledgement acceptance run exists. This remains an incomplete checkpoint, not a merge or activation candidate.

## Bounded repair run 500

Athena's run-499 architecture decision resolves the internal B3/B4/B5 design
gate and supersedes the packet where it excluded ordinary Slack/default turns
or allowed mid-turn replay. The approved shipping seam is now recorded at the
top of the packet.

This run binds supervisor startup to the existing board in SQLite read-only
mode before constructing any ledger, MCP config, or system prompt. Validation
now requires the live task, current active run, normalized assignee/profile,
claim lock, and persisted workspace to match dispatcher identity; a missing DB
is not created. It also replaces blanket API eligibility with fail-closed
model capability evidence for tools, native image input, and combined
input/output context budget. Athena's four executable probes now pass, and the
canonical worker suite contains equivalent behavioral regressions.

This is still an incomplete checkpoint. B3's shared-root matrix remains
dependent on reviewed PR #13 integration. B4 still lacks actual Claude
restricted-sandbox edit/test and denial evidence plus real schema/transport
intersection for every configured tool. B5's gateway-owned Slack/default turn
coordinator, durable transition/acknowledgement record, owner-executor tool
bridge, and exactly-once delivery acceptance are not implemented in this
checkpoint. No activation, credentials, provider call, merge, deploy, or
restart was performed.

## Bounded repair run 504

Talos continued from clean exact head `a7321d9bef8d81843221115c3afdf1ff384b50aa`
under Hyperion coordinator card `t_45e68439` / run 503. This slice removes the
configuration-time blanket API/Claude capability self-certification. Runtime
eligibility now resolves the profile-visible tool schemas through the actual
tool registry, includes their serialized bytes in the context budget, compares
exact tool names with the installed Claude adapter/MCP surface, rejects unknown
schema evidence, and distinguishes native image payloads from the
`vision_analyze` tool. The Kanban Claude path now explicitly rejects native
image payloads because it does not transport those parts.

A finite Claude Code subscription acceptance was executed with installed CLI
`2.1.304`, model `claude-sonnet-5-5`, `--restricted`, permission mode `auto`,
and only `Read,Glob,Grep,Edit,Write,Bash`. In an isolated `/tmp` workspace the
actual Claude runtime edited `module.py`, ran `python -m pytest -q
test_module.py` with `1 passed`, then attempted to overwrite a sibling sentinel.
The auto-mode classifier denied that outside-workspace Bash operation and an
independent readback confirmed the sentinel remained `unchanged`. The run
reported no web tool use and USD 0.0336137 list-basis usage. This is evidence
for restricted native edit/test execution and one denial boundary only; it is
not proof of gateway owner-tool/approval parity.

Fresh repository verification for this slice includes 73 focused tests passing
across the failover config/worker files and Athena's run-499 executable
acceptance remaining 4/4 green. B3 remains dependent on open, unstable PR #13
at `2949cbbe24c7d07c0367712a6ad453a1a1c2168f`; no credential store or PR13
worktree was modified. B5 remains unimplemented: the real Slack/default
`_run_agent` boundary still lacks the gateway-owned durable turn coordinator,
owner-executor MCP bridge, cancellation/reconciliation, and exactly-once final
delivery contract. This checkpoint must not be activated or represented as
Issue #260 complete.

## Bounded B4 repair run 511

Talos repaired the three evidence gaps identified by Athena's round-2 review at
`b6e71f92f52ba14aa20bc80a68dfa25b9b971706` under card `t_1b0b7dab`, coordinated
by `t_45e68439`. Valid empty assignee toolset resolution is now distinct from a
resolver failure: failure aborts supervised spawn, while an empty explicit set
is pinned with a no-explicit-tools sentinel and still resolves, compares, and
budgets the implicit task-scoped Kanban schemas that the API child receives.

Supervisor startup now reads image attachment metadata from the durable Kanban
task before route selection, verifies the immutable file path and size, records
the native-media requirement and byte count in argv/ledger state, and transports
exactly one image through the API child's native `--image` input. Claude legs
remain ineligible for native image payloads, and API legs require both observed
vision support and an exact supported transport shape; multiple native images
fail closed rather than being partially forwarded. The frozen request budget
now includes the exact serialized tool schemas, request framing, and base64
native-media expansion.

`models.dev` capability results now retain whether context and output limits
were actually observed. Legacy callers still receive conservative numeric
defaults, but backend eligibility rejects missing, non-numeric, boolean,
zero, or negative context/output evidence instead of allowing those defaults
to certify a route. Behavioral coverage includes valid-empty and failed
toolset resolution, real durable PNG attachment spawn-to-parity and API argv
transport, native-media budget accounting, and missing/invalid limit cases.

This remains a reviewable, disabled checkpoint rather than an activation or
Issue #260 completion. B3 remains dependency-blocked on PR #13. B5 remains out
of this bounded repair: the shipping Slack/default gateway coordinator,
durable transition acknowledgement, owner-executor bridge, replay/cancellation
contract, and exactly-once delivery acceptance are still unimplemented. No
credential/store mutation, activation, merge, deploy, restart, or PR #13 write
was performed.

## Claude CLI provenance caveat

A prior run reported that the installed Claude CLI changed despite the original restriction against updating the active binary. Athena's pinned-head review observed version/package `2.1.304`, executable mtime `2026-10-19T22:54:36Z`, and package mtime `2026-10-19T22:54:28Z`; the older `2.1.81` observation was stale. The provenance and operational impact require independent review. Do not repeat an install/update as part of checkpoint review.

## Required next owner and gates

Athena should review the immutable draft PR head against this status and the work packet, then route bounded repairs back to Talos. Before activation or merge, the final candidate still requires:

- repairs for all four blocker classes;
- independent reruns of focused and regression verification on the repaired exact head;
- profile-isolated live evidence for each intended backend, including resolution of or an explicit operator decision for Luna authentication and DeepSeek credential availability;
- a controlled, side-effect-safe full-chain failover exercise;
- exact-head hosted CI and independent architecture/security review;
- explicit human approval before runtime activation, deployment, or restart.

Issue #260 remains incomplete. The original implementation card remains blocked pending independent review and bounded repair.
