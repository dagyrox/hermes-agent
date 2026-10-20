# Agent Backend Failover — Architecture Work Packet

**Project:** Hermes Agent — agent routing & automatic backend failover
**Primary repository:** `dagyrox/hermes-agent`
**Consumer repository:** `dagyrox/or7-platform`
**Tracking issue:** `dagyrox/or7-platform#260`
**Kanban card / run:** `t_3cb377f6` / run 490
**Acting lane:** `[Athena]` architecture (no implementation edits in this packet)
**Coordinating issues:** `#259` credential unification (owns credential storage/rotation — **not modified here**), `#254` Claude worker/runtime integration, `#257` voice (independent)
**Status:** architecture bounded, implementation-ready; corrected cross-provider route freeze acknowledged by Talos

> **Superseding architecture decision (Athena card `t_8d96d438`, run 499):**
> ordinary Slack/default turns are in scope when this feature is enabled. The
> approved minimum is a gateway-owned turn coordinator at the real `_run_agent`
> boundary, with an immutable canonical turn snapshot and only pre-tool,
> pre-output availability failover. Refusals, partial output, tool activity,
> pending approval, timeout, or unfinished durable intent stop for
> reconciliation and are never replayed. Claude subscription attempts use a
> gateway-owned, turn-scoped tool bridge to the existing owner executor; they
> are not Kanban conversation proxies. Capability eligibility is the
> intersection of model/transport evidence, resolved schemas, installed
> adapters, and approval guarantees. Unknown evidence fails closed. This
> supersedes §§3–4 and §15 where they say ordinary conversations stay
> unchanged/all-API or permit mid-turn rewind. The frozen five three-provider
> chains and disabled-by-default posture remain unchanged.

> **Authoritative correction (task comment, 2026-10-19):** every route leg
> must use a distinct provider. This supersedes the same-provider fallback
> rows and examples in the original Athena packet. The corrected route table
> and configuration below are binding, and validation rejects duplicate
> providers even when their models differ.

---

## 1. Scope and non-negotiables

Frozen role chains (not open for debate, reproduced verbatim as the implementation target):

| Role | Primary | Fallback 1 | Fallback 2 (final) |
|---|---|---|---|
| Hermes/default | `openai-codex` `gpt-6.1-sol` | Claude Code subscription `Sonnet5.5` | `deepseek` `deepseek-flash` |
| Hyperion | `openai-codex` `gpt-6-luna` | Claude Code subscription `Sonnet5.5` | `deepseek` `deepseek-flash` |
| Athena | Claude Code subscription `Opus5.5` | `openai-codex` `gpt-6.1-sol` | `deepseek` `deepseek-flash` |
| Talos | `openai-codex` `gpt-6.1-sol` | Claude Code subscription `Sonnet5.5` | `deepseek` `deepseek-flash` |
| Apollo | Claude Code subscription `Sonnet5.5` | `openai-codex` `gpt-6.1-sol` | `deepseek` `deepseek-flash` |

Hard constraints carried into every section below:

- Exactly **primary + fallback1 + fallback2** per role. Every leg uses a distinct provider. No third fallback, no fourth tail.
- `deepseek-flash` (official version `DeepSeek-V4.1-Flash-1017`) is always the **second and final** fallback.
- **No direct OpenAI API.** Codex legs run only through the `openai-codex` provider path.
- **No raw Anthropic API.** The Claude legs are the real Claude Code subscription runtime or nothing.
- Sentinel is model-free. Forge stays unactivated; no Forge profile is created by inference. Guests/GF/retired Orion are excluded from new routing and shared secrets.
- Credential storage, rotation, and unification remain owned by **#259**. This packet consumes credentials only through already-existing resolution seams and introduces **no new credential store, no copying, and no blind revocation**.

---

## 2. Current implementation baseline (verified in this worktree)

Failover already exists as an **in-process, same-runtime provider swap**:

- `hermes_cli/fallback_config.py::get_fallback_chain(config)` — merges `fallback_providers` (ordered, source of truth) with legacy `fallback_model`, dedups on `(provider, model, base_url)`, returns fresh dicts. `resolve_entry_api_key(entry)` resolves `api_key` → `key_env`/`api_key_env`.
- `agent/agent_init.py` (~L1350–1367) — materializes `agent._fallback_chain`, `agent._fallback_model`, and prints the chain at startup.
- `agent/chat_completion_helpers.py::try_activate_fallback(agent, reason)` (L1537+) — advances `agent._fallback_index`, skips entries in `agent._unavailable_fallback_keys`, skips entries matching the current `(provider, model)` or the current `base_url`+model, resolves a client through `agent.auxiliary_client.resolve_provider_client(...)`, normalizes the model via `hermes_cli/model_normalize.py::normalize_model_for_provider`, derives `api_mode` (`codex_responses` / `anthropic_messages` / `bedrock_converse` / `chat_completions`), then **mutates `agent.model/provider/base_url/api_mode` in place**, sets `agent._fallback_activated = True`, clears `agent._transport_cache`, and rebinds the credential pool when the provider changes. Rate-limit/billing/upstream-rate-limit arm a 60 s cooldown; non-rate-limit exhaustion arms `_FALLBACK_EXHAUSTED_COOLDOWN_S` (5 s) to prevent the cross-turn replay storm from #24996.
- `run_agent.py` — `_try_activate_fallback` (L5509) forwarder, `_has_pending_fallback` (L5514), `_emit_pending_fallback_notice` (L1073), primary-runtime restore gating.
- `agent/error_classifier.py::classify_api_error(...)` — returns `ClassifiedError` with `reason: FailoverReason` and the action hints `retryable`, `should_compress`, `should_rotate_credential`, `should_fallback`. This is the **existing, authoritative eligibility signal** and is reused as-is.
- `gateway/run.py` — `get_fallback_chain(cfg)` at L2358/L5757, `self._fallback_model` at L5793, `_apply_fallback_chain_to_agent(agent, chain)` at L5797 (live reload, compares `old_chain`), applied at L21412. `cli.py:4164` and `cron/scheduler.py` read the same chain.
- `tools/kanban_tools.py` — the multi-agent board / worker queue surface; `plugins/kanban/` in-tree holds only the dashboard (`dashboard/plugin_api.py`) and `systemd/hermes-kanban-dispatcher.service`. Recent history (`b8087e0c693`) added a fail-closed embedded dispatcher quiesce, and `83d466e443f` isolated Kanban workers from gateway lifecycle — both are load-bearing for the activation story in §12.

**The real gap:** `try_activate_fallback` can only swap an **API-shaped client inside the same Python process**. Every frozen chain row with a Claude Code subscription leg (Athena primary, Talos fallback1, Apollo primary) is unreachable by that mechanism, because the subscription is usable only through the Claude Code CLI runtime — and raw Anthropic API billing is prohibited. PR #12 history (merged provider-fallback work) extended the API-side chain; it did not and could not cross runtimes.

---

## 3. Decision: thin in-process adapter **vs** supervised task runner

### Decision

**One dispatcher-owned supervised task runner for the complete three-leg chain.**

All legs are selected before a fresh Kanban worker starts. API legs launch the
ordinary Hermes worker pinned to one provider/model; Claude subscription legs
launch the real Claude Code CLI. A leg may advance only after a classified
pre-tool availability failure. Once tool activity is observed, or a child
times out, replay is denied and the task is blocked for reconciliation.

### Justification

- A thin in-process adapter for the Claude leg would require speaking an API protocol from inside the Hermes process. The only such protocol for Opus/Sonnet is the raw Anthropic API, which is explicitly forbidden (and would bill the wrong account). The Claude Code **subscription** is redeemable only by the Claude Code runtime itself, which owns its own OAuth/session handling. Therefore an in-process adapter for this leg is not merely unattractive — it is unimplementable within the stated constraints.
- Keeping every leg under one supervisor makes route order and attempt count
  deterministic, prevents nested in-process fallback from skipping the Claude
  leg, and preserves the dispatcher-owned task/run/workspace identity. Existing
  in-process fallback remains unchanged for ordinary non-Kanban conversations.
- A known CLI-compatibility hazard is carried as a precondition, not a surprise: Claude CLI `2.1.226` is incompatible with the latest model aliases. The runner therefore performs an explicit **version+alias compatibility probe** (§9 `ClaudeRunnerProbe`) and declares the leg ineligible rather than silently substituting an older model.

### Explicit non-claim (binding)

**Private in-process conversation and tool state cannot move across runtimes.** The packet does not claim otherwise. Specifically, the following are **non-transferable** and are never asserted to survive a cross-runtime handoff:

- the live in-memory `messages` array with provider-native shapes (Anthropic thinking blocks + signatures, Codex `encrypted_content` replay blobs, reasoning deltas);
- prompt-cache prefixes (and therefore cache economics — a cross-runtime handoff **costs a cold prefix**, by design);
- open tool handles: live PTY/terminal sessions (`tools/environments/*`), browser contexts, MCP client sessions, open file descriptors, in-flight background delegations;
- partially-applied tool side effects and any tool call whose completion was not observed.

What crosses the boundary is **only** the persisted handoff artifact in §4. Everything else is re-established from durable state by the receiving runtime.

---

## 4. Safe persisted handoff boundary

### 4.1 Handoff points (the only legal ones)

| Boundary | Cross-runtime handoff allowed? |
|---|---|
| Between turns in a session (no tool call in flight) | **Yes** |
| Between Kanban tasks / at task claim time | **Yes** |
| After a tool call whose result was durably recorded | **Yes** |
| Mid-turn, tool call in flight | **No** |
| Mid-stream (partial assistant message) | **No** |
| Mid-approval (approval requested, not yet answered) | **No** |

Mid-turn failure on a Claude leg is handled by: abandon the partial assistant message, mark the leg degraded, **rewind to the last durable boundary**, then hand off. Rewind never re-executes recorded side effects (§6).

### 4.2 Handoff artifact

Written to `$HERMES_HOME/failover/handoff/<task_id>.json`, mode `0600`, via `hermes_constants.get_hermes_home()` (profile-aware; never `Path.home()/".hermes"`).

```json
{
  "schema_version": 1,
  "task_id": "t_3cb377f6",
  "run_id": 490,
  "role": "athena",
  "profile": "athena",
  "identity": {
    "lane": "athena",
    "persona_ref": "profile:athena",
    "authority_ref": "profile:athena",
    "memory_scope": "profile:athena"
  },
  "from_backend": {"kind": "claude_code", "model": "Opus5.5", "leg": "primary"},
  "to_backend":   {"kind": "api", "provider": "openai-codex", "model": "gpt-6.1-sol", "leg": "fallback1"},
  "reason": "auth",
  "classified": {"reason": "auth", "status_code": 401, "retryable": false, "should_fallback": true},
  "boundary": {"kind": "turn_boundary", "last_durable_message_index": 42},
  "transcript_ref": "session:<session_id>#42",
  "approvals_ref": "approvals:<task_id>",
  "ambiguous_side_effects": [],
  "degraded_sources": [{"leg": "primary", "backend": "claude_code", "reason": "auth", "at": "2026-10-19T00:00:00Z"}],
  "attempt_ledger": [{"leg": "primary", "outcome": "auth", "at": "..."}],
  "created_at": "2026-10-19T00:00:00Z"
}
```

Rules:

- **References, not secrets.** `transcript_ref` / `approvals_ref` / `persona_ref` are pointers into the SQLite session store and profile config. No credential material, token, env value, or `.env` content is ever written into the artifact. Redaction reuses the existing Kanban redaction contract (`tests/tools/test_kanban_redaction.py`).
- **Role/task identity is preserved verbatim.** `task_id`, `run_id`, `role`, and `identity.*` are copied, never regenerated. The receiving runtime is the *same* role on a different backend — not a new agent, not a new identity. Profile persona, toolset, memory scope, and authority level are resolved from the unchanged profile, so a Claude leg and a Codex leg of the same role present identically to the operator and to the approval system.
- **No new conversation.** The receiving runtime resumes the existing session id; the degraded-source notice is appended as a visible transcript event, not as a synthetic user message (role-alternation invariant preserved).

---

## 5. Failover eligibility and classification

Eligibility is computed by one new predicate, layered **on top of** the existing classifier (no re-classification, no new string matching):

`agent/backend_failover.py::is_failover_eligible(classified: ClassifiedError, *, leg: str, boundary: str) -> FailoverDecision`

**Eligible (hand off to the next leg):** `auth`, `auth_permanent`, `billing`, `rate_limit`, `upstream_rate_limit`, `overloaded`, `server_error`, `timeout`, `model_not_found`, `format_error` (non-retryable 4xx), and the stale-call circuit-breaker `timeout` with `should_fallback=True`.

**Not eligible — recover in place, never hand off:**

| Reason | Why not |
|---|---|
| `context_overflow`, `payload_too_large` | compress; a different backend has the same or smaller window |
| `image_too_large` | shrink and retry |
| `thinking_signature`, `invalid_encrypted_content`, `multimodal_tool_content_unsupported`, `llama_cpp_grammar_pattern`, `oauth_long_context_beta_forbidden` | deterministic request-shape repairs on the current backend |
| `provider_policy_blocked` | account-level setting; the next leg on the same account fails identically |
| `ssl_cert_verification` | deterministic for the host; fail fast with guidance |
| **`content_policy_blocked`** | **refusal — see §5.1** |

The authoritative gate is: `classified.should_fallback is True` **and** the reason is not in the refusal set **and** `boundary` is a legal handoff point (§4.1) **and** the target leg passes capability parity (§7) and, for Claude legs, `ClaudeRunnerProbe` (§9).

### 5.1 Refusal is never a failover trigger (binding)

`FailoverReason.content_policy_blocked` currently carries `should_fallback=True` in `agent/error_classifier.py` (L659–664). For the **cross-runtime/role-chain** path this packet **overrides that hint to False**:

- A safety refusal is a decision about the *request*, not a capacity or auth fault. Retrying it on another backend is refusal-shopping and is prohibited by the acceptance criteria.
- Implementation: `is_failover_eligible` returns `FailoverDecision(eligible=False, surface="refusal")` for `content_policy_blocked` regardless of `should_fallback`. The classifier itself is **not** edited (other consumers rely on the existing hint); the override lives in the new predicate and is asserted by test `test_refusal_never_fails_over`.
- Operator surface: the refusal is reported as a refusal, with the refusing backend named, and the turn ends. No leg is marked degraded, no attempt is consumed.

---

## 6. Ambiguous side effects — reconciliation, and replay denial

A leg may fail *after* dispatching an external side effect but *before* the result is observed (timeout, stream kill, process death). These are **ambiguous**, and the default is **deny replay**.

- **Classification:** any tool call that is (a) externally observable (network write, message send, VCS push, PR/merge, payment, file mutation outside the workspace, `terminal` command with side effects) and (b) has no recorded terminal result, is appended to `handoff.ambiguous_side_effects[]` with `{tool, args_digest, dispatched_at, idempotency_key?}`. `args_digest` is a hash, not the arguments, so nothing sensitive leaks into the artifact.
- **Replay denial:** the receiving runtime **must not** re-dispatch any entry in `ambiguous_side_effects[]`. It receives them as an explicit, visible reconciliation list.
- **Reconciliation order:** (1) if the tool exposes a read-only verification path, verify and record the actual outcome; (2) if the tool is idempotent *and* an `idempotency_key` was recorded, a single keyed re-dispatch is permitted; (3) otherwise → **no replay**, and the item is surfaced to the operator for confirmation.
- **Hard floor:** if any unreconcilable ambiguous side effect is sensitive (approval-gated class per §8, financial, destructive, outbound/customer-facing), the task does not continue on the new leg. It terminates in `NEEDS_INPUT` (§10) naming the exact uncertain operation. Fail-closed, consistent with the dispatcher quiesce precedent in `b8087e0c693`.
- No ambiguous entry is ever silently dropped, and the list travels with the task if a second handoff occurs.

---

## 7. Capability parity checks

Before a leg is entered, `agent/backend_failover.py::check_capability_parity(task_requirements, target_leg) -> ParityResult` compares what the task actually needs against what the target leg provides. Sources: `agent/model_metadata.py`, `toolsets.py`, the profile's enabled toolsets, and the target backend descriptor.

| Dimension | Requirement derived from | Behavior on mismatch |
|---|---|---|
| **Tools** | toolsets the task has used or is configured to use (incl. MCP, terminal, delegation) | if a *required* tool is unsupported on the target → leg ineligible, advance |
| **Images / multimodal** | presence of image parts in the durable transcript, or a vision toolset in use | leg ineligible if the target cannot accept the already-present image content |
| **Context window** | `approx_tokens` at the handoff boundary vs target `context_length` | compress first on the *source*; if still over target → leg ineligible |
| **Reasoning** | role's reasoning requirement (`reasoning_config` / effort) | degraded reasoning is **allowed but recorded** as a visible capability-downgrade notice; it does not block |

Parity results are recorded in the attempt ledger so an exhaustion message can state precisely *why* each leg was skipped (unconfigured, probe-failed, parity-failed, previously marked unavailable). Existing `_unavailable_fallback_keys` suppression is reused for session-scoped skip memory.

---

## 8. Approval preservation

- **Persisted sensitive-action approvals remain binding across a handoff.** Approvals are referenced by `approvals_ref` and are keyed to `(task_id, action_class)` — never to the backend or runtime. A handoff neither grants nor revokes anything.
- **No auto-approval on failover.** Degradation to a weaker leg does not widen authority. If the original leg required expert/human approval for a class of action, the fallback leg requires exactly the same approval. DeepSeek-final is explicitly **not** permitted to self-approve a risky release; support/analysis/diagnostic work proceeds, release-class actions remain gated.
- **Mid-approval is not a handoff boundary** (§4.1). A pending approval request is either answered on the current leg or the task parks in `NEEDS_INPUT`; an unanswered approval is never re-asked on another backend in a way that could double-authorize.
- **Authority is a property of the role/profile, not the backend.** The receiving runtime loads the same profile authority; Hermes approval rules remain authoritative (per `/root/CLAUDE.md`), and agents still do not approve themselves.
- **Credential posture.** Auth failures mark the source **degraded and visible** and continue eligible work on an alternative leg. The implementation must not copy credentials between legs, must not revoke credentials, and must not claim broken auth has been repaired. Credential mechanics stay with **#259**.

---

## 9. Claude Code supervised runner

New module `agent/backends/claude_code_runner.py`:

- `ClaudeRunnerProbe.check(model: str) -> ProbeResult` — resolves the Claude CLI, reads its version, and verifies the requested alias (`Opus5.5` / `Sonnet5.5`) is served by that version. **No silent old-model substitution**: an incompatible CLI yields `ProbeResult(ok=False, reason="cli_alias_incompatible")`, the leg is declared ineligible, and the skip is visible in the ledger. Probe results are cached per `(cli_version, model)` for the process lifetime.
- `ClaudeCodeRunner.run_task(handoff: Handoff) -> RunnerOutcome` — spawns a supervised child Claude Code process scoped to the role's profile (persona, tools, memory scope, authority), bounded by an explicit wall-clock and iteration budget, with the handoff's durable transcript reference as its input and no inherited secrets beyond what the runtime already owns.
- Supervision: child stdout/stderr are captured into the existing profile-aware logging (`hermes_logging.py`), the child is terminated on interrupt/quiesce, and process death is classified into the same `FailoverReason` space (`timeout`, `auth`, `overloaded`) so §5 applies uniformly.
- Outcome is written back to the durable session as a completed turn **before** any further leg transition, so the next boundary is well-defined.
- The runner performs **no raw Anthropic API calls**. If the subscription runtime is unavailable, the leg is simply ineligible.

---

## 10. Final exhaustion → NEEDS_INPUT

DeepSeek-final attempts are **visible and bounded**: a fixed, small attempt budget per task (config `max_attempts`, default 2), surfaced in the transcript as explicit degraded-source notices — not silent retries. DeepSeek smoke calls for acceptance are finite and authorized; no purchases, top-ups, new subscriptions, or unlimited spend.

When primary, fallback1, and fallback2 are all exhausted or ineligible, the task terminates in **`NEEDS_INPUT`** — never a silent drop, never an invented success, never a fourth leg:

```
NEEDS_INPUT — backend exhaustion
Project: Hermes Agent · Repository: dagyrox/hermes-agent
Role: athena · Task: t_3cb377f6 · Run: 490
Attempted:
  primary   claude_code Opus5.5        → auth (degraded, 401, credential NOT modified)
  fallback1 openai-codex gpt-6.1-sol   → rate_limit (429, retry-after 180s)
  fallback2 deepseek deepseek-flash    → parity_failed (images unsupported, 2/2 attempts)
Unreconciled ambiguous side effects: none
Binding approvals still pending: none
Required operator action: restore Claude Code subscription auth, or re-scope the task without image input.
```

The status block follows the Athena handoff format (names the human-readable project and canonical repository near the top).

---

## 11. Event-driven continuation

Continuation is **event-driven, not a new per-job recurring LLM cron**:

- Completion of a leg, child-process exit, `retry-after` expiry, approval answer, and credential-health change each emit an event that advances the task state machine. The Kanban dispatcher consumes these; worker isolation from the gateway lifecycle (`83d466e443f`) and the fail-closed quiesce (`b8087e0c693`) are preserved unchanged.
- **Timers are permitted only** for (a) a specific deadline / `retry-after` value returned by a provider, and (b) a mechanical missed-event watchdog that re-checks state without invoking an LLM. No recurring per-job LLM cron is added.
- **Evidence requirement:** continuation must produce real acknowledgment/next-step evidence — a persisted state transition and ledger entry — not a Slack notification alone. Slack remains a notify-after surface.
- **No mid-run token/config race.** Chain reload reuses `gateway/run.py::_apply_fallback_chain_to_agent` semantics (compare-and-apply against `old_chain`), and an active worker finishes its current turn on its current leg; a reloaded chain takes effect at the next boundary. Active workers are preserved.

---

## 12. Exact config schema

Role chains live in `config.yaml` (behavioral config — **not** `.env`; `.env` stays secrets-only, per AGENTS.md). No new `HERMES_*` env var for behavior. Added to `DEFAULT_CONFIG` in `hermes_cli/config.py` under a new `agent_backend_failover` section. Adding a new section to `DEFAULT_CONFIG` is deep-merged and does **not** require a `_config_version` bump.

```yaml
agent_backend_failover:
  enabled: false                 # staged activation; off by default
  max_attempts_per_leg: 1        # one bounded attempt at each of three legs
  roles:
    default:
      - {kind: api, provider: openai-codex, model: "gpt-6.1-sol"}
      - {kind: claude_code, model: "claude-sonnet-5-5", capability_mode: full}
      - {kind: api, provider: deepseek, model: "deepseek-flash"}
    hyperion:
      - {kind: api, provider: openai-codex, model: "gpt-6-luna"}
      - {kind: claude_code, model: "claude-sonnet-5-5", capability_mode: orchestrator}
      - {kind: api, provider: deepseek, model: "deepseek-flash"}
    athena:
      - {kind: claude_code, model: "claude-opus-5-5", capability_mode: read_only}
      - {kind: api, provider: openai-codex, model: "gpt-6.1-sol"}
      - {kind: api, provider: deepseek, model: "deepseek-flash"}
    talos:
      - {kind: api, provider: openai-codex, model: "gpt-6.1-sol"}
      - {kind: claude_code, model: "claude-sonnet-5-5", capability_mode: full}
      - {kind: api, provider: deepseek, model: "deepseek-flash"}
    apollo:
      - {kind: claude_code, model: "claude-sonnet-5-5", capability_mode: read_only}
      - {kind: api, provider: openai-codex, model: "gpt-6.1-sol"}
      - {kind: api, provider: deepseek, model: "deepseek-flash"}
```

Schema validation (`validate_backend_failover_config`) is **fail-closed** and rejects at load time:

- a role with anything other than exactly `primary`, `fallback1`, `fallback2`;
- repeated providers anywhere in a role chain (including two `openai-codex` models);
- `fallback2` that is not `deepseek` / `deepseek-flash`;
- any entry with `kind: api` and `provider: openai` or `provider: anthropic` (no direct OpenAI API, no raw Anthropic API);
- a role entry for `sentinel` (model-free) or `forge` (unactivated), or for guests/GF/retired Orion;
- `allow_refusal_failover: true` or `deny_replay_on_ambiguous: false`;
- inline secrets in any role entry (credentials resolve through the existing provider path / #259 only).

API legs are projected onto the existing chain format by `agent/backend_failover.py::role_chain_to_fallback_entries(role_cfg)`, so `get_fallback_chain`-shaped consumers (`cli.py:4164`, `gateway/run.py:2358/5757/5793`, `cron/scheduler.py`) keep working with no change to their call shape.

---

## 13. Integration seams (named files and functions)

**New files**

| File | Contents |
|---|---|
| `agent/backend_failover.py` | `FailoverDecision`, `ParityResult`, `Handoff`, `is_failover_eligible()`, `check_capability_parity()`, `write_handoff()`, `read_handoff()`, `role_chain_to_fallback_entries()`, `record_attempt()`, `needs_input_report()` |
| `agent/backends/claude_code_runner.py` | `ClaudeRunnerProbe`, `ClaudeCodeRunner`, `RunnerOutcome` |
| `docs/agentic-operations/agent-backend-failover.md` | this packet |

**Modified files (narrow, additive)**

| File | Seam |
|---|---|
| `hermes_cli/config.py` | add `agent_backend_failover` to `DEFAULT_CONFIG`; add `validate_backend_failover_config()` |
| `hermes_cli/fallback_config.py` | `get_fallback_chain()` consults the role chain (via `role_chain_to_fallback_entries`) when `agent_backend_failover.enabled` and a role is resolved; otherwise unchanged |
| `agent/chat_completion_helpers.py` | in `try_activate_fallback()`: consult `is_failover_eligible()` + `check_capability_parity()` before activating; when the next leg is `kind: claude_code`, do **not** swap in-process — return a `pending_runtime_handoff` signal. Cooldown/dedup/`_unavailable_fallback_keys` logic untouched. |
| `run_agent.py` | `_try_activate_fallback` forwarder honors `pending_runtime_handoff`; `_emit_pending_fallback_notice` extended to render degraded-source and capability-downgrade notices as transcript events (no synthetic user message, no system-prompt mutation — prompt-cache and alternation invariants preserved) |
| `agent/error_classifier.py` | **no change** (refusal override lives in `is_failover_eligible`) |
| `tools/kanban_tools.py` | at task claim/boundary: honor `pending_runtime_handoff` by writing the handoff artifact and dispatching the Claude leg through `ClaudeCodeRunner`; on exhaustion emit `NEEDS_INPUT`; preserve existing quiesce and worker-isolation behavior |
| `gateway/run.py` | `_apply_fallback_chain_to_agent()` also carries role-chain reload; boundary-only application (no mid-run race) |

Cache/alternation discipline: the system prompt stays byte-stable for the life of a conversation; a cross-runtime handoff starts a cold prefix on the new backend and this cost is accepted and documented rather than worked around by mutating past context.

---

## 14. Acceptance tests

New tests (stdlib + pytest + `unittest.mock`, no live network except the explicitly authorized finite DeepSeek smoke check, which is marked and skipped by default):

`tests/agent/test_backend_failover_chain.py`
- `test_every_role_has_exactly_three_legs` — invariant over config, not a snapshot of model names
- `test_fallback2_is_always_deepseek_flash`
- `test_rejects_direct_openai_and_raw_anthropic_legs`
- `test_rejects_sentinel_and_forge_role_entries`

`tests/agent/test_backend_failover_eligibility.py`
- `test_auth_billing_ratelimit_timeout_are_eligible`
- `test_context_overflow_and_policy_block_are_not_eligible`
- `test_refusal_never_fails_over` (asserts `content_policy_blocked` → `eligible=False`, `surface="refusal"`, no leg consumed)
- `test_mid_tool_call_boundary_is_not_a_handoff_point`

`tests/agent/test_backend_failover_handoff.py`
- `test_handoff_preserves_role_task_identity_and_authority`
- `test_handoff_artifact_contains_no_secrets` (scans for env values, token shapes, `.env` contents)
- `test_handoff_artifact_mode_is_0600_under_temp_hermes_home`
- `test_no_claim_of_transferred_live_tool_state` (handoff carries refs only; no PTY/browser/MCP handles)

`tests/agent/test_backend_failover_side_effects.py`
- `test_ambiguous_side_effect_is_not_replayed`
- `test_idempotent_tool_with_key_may_single_retry`
- `test_unreconcilable_sensitive_effect_forces_needs_input`

`tests/agent/test_backend_failover_parity.py`
- `test_image_task_skips_non_multimodal_leg`
- `test_context_overflow_compresses_before_leg_switch`
- `test_reasoning_downgrade_is_recorded_not_blocking`
- `test_missing_required_tool_skips_leg`

`tests/agent/test_backend_failover_approvals.py`
- `test_persisted_approval_still_binding_after_handoff`
- `test_fallback_does_not_auto_approve_risky_release`
- `test_pending_approval_blocks_handoff`

`tests/agent/test_claude_code_runner_probe.py`
- `test_incompatible_cli_version_marks_leg_ineligible`
- `test_no_silent_old_model_substitution`
- `test_runner_never_calls_raw_anthropic_api`

`tests/agent/test_backend_failover_exhaustion.py`
- `test_exhaustion_emits_needs_input_with_per_leg_reasons`
- `test_deepseek_attempts_are_bounded_and_visible`
- `test_no_fourth_leg_is_ever_attempted`

`tests/agent/test_backend_failover_continuation.py`
- `test_continuation_is_event_driven_no_new_llm_cron`
- `test_timer_only_for_retry_after_or_watchdog`
- `test_chain_reload_does_not_race_active_worker`

E2E (real imports, temp `HERMES_HOME`, per AGENTS.md "E2E validation, not just green unit mocks"):
`tests/e2e/test_backend_failover_e2e.py` — real config load → role-chain resolution → simulated 401 on primary → eligibility → handoff artifact → fallback1 activation → visible degraded notice → session continuity, all against a temp profile.

### Verification commands

```bash
cd /root/projects/worktrees/agent-backend-failover
source .venv/bin/activate 2>/dev/null || source venv/bin/activate

# New suites
scripts/run_tests.sh tests/agent/test_backend_failover_chain.py -q
scripts/run_tests.sh tests/agent/test_backend_failover_eligibility.py -q
scripts/run_tests.sh tests/agent/test_backend_failover_handoff.py -q
scripts/run_tests.sh tests/agent/test_backend_failover_side_effects.py -q
scripts/run_tests.sh tests/agent/test_backend_failover_parity.py -q
scripts/run_tests.sh tests/agent/test_backend_failover_approvals.py -q
scripts/run_tests.sh tests/agent/test_claude_code_runner_probe.py -q
scripts/run_tests.sh tests/agent/test_backend_failover_exhaustion.py -q
scripts/run_tests.sh tests/agent/test_backend_failover_continuation.py -q
scripts/run_tests.sh tests/e2e/test_backend_failover_e2e.py -q

# Existing suites that must stay green (no regression in the in-process path)
scripts/run_tests.sh tests/run_agent/test_provider_fallback.py tests/run_agent/test_auth_provider_failover.py -q
scripts/run_tests.sh tests/run_agent/test_24996_fallback_exhaustion_cooldown.py \
                     tests/run_agent/test_32646_fallback_429_after_timeout.py \
                     tests/run_agent/test_primary_runtime_restore.py -q
scripts/run_tests.sh tests/gateway/test_fallback_chain_reload.py tests/gateway/test_auth_fallback.py -q
scripts/run_tests.sh tests/hermes_cli/test_fallback_cmd.py tests/hermes_cli/test_config_validation.py -q
scripts/run_tests.sh tests/tools/test_kanban_tools.py tests/tools/test_kanban_redaction.py \
                     tests/tools/test_delegate_kanban_isolation.py -q

# Config + probe sanity (no model calls)
python -c "from hermes_cli.config import load_config; from agent.backend_failover import role_chain_to_fallback_entries as r; c=load_config(); print(r(c['agent_backend_failover']['roles']['athena']))"
python -c "from agent.backends.claude_code_runner import ClaudeRunnerProbe as P; print(P.check('Opus5.5'))"
```

---

## 15. Activation and rollback

Staged, reversible, no production action without Jason's approval.

**Stage 0 — merged, dormant.** `enabled: false`. Code present, zero behavior change. Verified by the existing-suite block above.

**Stage 1 — one role, non-production profile.** Enable for `talos` only (its primary is an API leg, so the Claude leg is exercised as *fallback1* — the lower-risk shape). Verify: forced 401 on primary → visible degraded notice → Claude leg probe → handoff artifact → task completes with identity preserved.

**Stage 2 — Claude-primary roles.** Enable `athena`, then `apollo`. These exercise Claude-as-primary and the cold-prefix cost on handoff.

**Stage 3 — `hyperion` and `default`.** All-API chains; lowest risk, enabled last only because they carry the most traffic.

**Rollback** (secure, immediate, at any stage): set `agent_backend_failover.enabled: false` in `config.yaml` → next boundary reverts every role to the pre-existing `get_fallback_chain` behavior. No credential change, no data migration, no artifact cleanup required (handoff artifacts are inert task-scoped files under `$HERMES_HOME/failover/handoff/`). Per-role rollback is available by removing that role's block. Git-level rollback is a single revert of the narrow PR.

**Approval gate:** production install/deploy/service restart (including any `hermes-kanban-dispatcher.service` restart) requires Jason's approval and will be requested as an explicit activation request naming the stage, the restart target, and the rollback command. Active workers are preserved across activation; chain changes apply only at boundaries.

**Coordination:** #259 owns credential architecture — this packet reads credentials only through existing provider resolution and must land without touching credential storage/rotation. Sequencing with #259 is a coordination note, not a blocker: Stage 0 and Stage 1 are safe under either ordering. No other profile's memories, skills, plugins, or cron are modified.

---

## 16. Residual risks (stated, not hidden)

1. **Cold prompt-cache cost on every cross-runtime handoff.** Unavoidable given §3's non-claim; bounded by boundary-only handoff and by the fact that handoffs are failure-triggered, not routine.
2. **Coarser recovery granularity on Claude legs.** A Claude leg failing deep into a turn rewinds to the last durable boundary; work since that boundary that produced no durable record is redone (never replayed side effects).
3. **Claude CLI alias compatibility is an external dependency.** Until a compatible CLI version is verified, Claude legs may be probe-ineligible and those roles effectively run their API legs. This is visible in the ledger, not silent, and is explicitly *not* worked around by old-model substitution or raw API billing.
4. **DeepSeek-final capability floor.** Image- or heavy-reasoning tasks may legitimately exhaust to `NEEDS_INPUT` rather than complete degraded. That is the intended, safe outcome.

---

## 17. Status block

```text
[Athena] Planning/doc check-in:
- Project: Hermes Agent — agent backend failover
- Repository: dagyrox/hermes-agent (consumer: dagyrox/or7-platform)
- Goal: bounded, implementation-ready architecture for real automatic cross-runtime
  agent backend failover across the frozen primary+fallback1+fallback2 role chains.
- Artifacts/files touched: docs/agentic-operations/agent-backend-failover.md (new, this
  packet). No implementation files edited.
- Evidence/verification: baseline read in-worktree — hermes_cli/fallback_config.py,
  agent/chat_completion_helpers.py::try_activate_fallback (L1537+),
  agent/error_classifier.py (FailoverReason/should_fallback),
  agent/agent_init.py (~L1350), gateway/run.py (L2358/5757/5793/5797/21412),
  cli.py:4164, tools/kanban_tools.py, plugins/kanban/systemd/, AGENTS.md config/plugin
  policy, and commits b8087e0c693 / 83d466e443f. Verification commands for the
  implementation lane are in §14.
- Open decisions: sequencing against #259 (coordination only; Stages 0–1 safe either way);
  confirmation of a Claude CLI version that serves Opus5.5/Sonnet5.5 aliases.
- Approval needed: production install/deploy/dispatcher restart at Stage 1+ (Jason).
  Merge of the dormant Stage 0 PR runs under standing code authority.
- Recommended next step: Talos implements §13 behind `enabled: false`, lands §14 suites
  green, then Athena reviews and prepares the Stage 1 activation request.
```

---

## 18. Verdict

**READY_TO_IMPLEMENT** — the thin-adapter / supervised-runner split is decided and justified, every required contract is specified with named files, functions, config keys, tests, and verification commands, and the one external dependency (Claude CLI alias compatibility) is handled as a fail-visible probe rather than a blocker.
