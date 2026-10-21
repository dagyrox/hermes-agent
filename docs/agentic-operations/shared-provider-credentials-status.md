# Shared Provider Credentials — Talos Status

Project: Agent Network Credential Unification  
Primary repository: `dagyrox/hermes-agent`  
Consumer/tracking repository: `dagyrox/or7-platform`  
Issue: `dagyrox/or7-platform#259`  
Hermes Kanban: `t_1af34c61`, repair run `552`  
Acting lane: Talos implementation  
Architecture packet: `docs/agentic-operations/shared-provider-credentials.md`

## Status

`ROUND_5_ATHENA_BLOCKERS_REPAIRED_ON_CURRENT_MAIN__FINAL_REVIEW_CI_PENDING`

Round-5 code candidate `7a1e7739a7d061fba3dddcfbcd9cc80d799f3540` (tree `e8214cd0bf1f1d72c81f75d034f9022794bbc37d`) repairs Athena DeepSeek review card `t_1b31cf48` / run `554` findings and includes current `origin/main` at `95638faeaf6b446cd21c6e57fa9383fc9dfaecf8` through merge commit `82eacf1bea7`.

The blocking missing-refresh-token short circuit now takes only the owning root-store lock, persists exhausted state at root, and never calls profile persistence for an inherited row. Terminal xAI/Codex quarantine now removes only the failed grant rather than deleting independent sibling grants from the shared root pool.

The prerequisite main-hygiene repair was independently Athena-approved at exact head `6a93c343b91f1f46a451fa31b1fa5979a3c080db` and merged through PR #15 as `95638faeaf6b446cd21c6e57fa9383fc9dfaecf8`. PR #13's prior CI run `39524394371` failed on the known stale test-contract baseline; the merged dependency repairs those test fixtures. A new hosted run and independent exact-final-head Athena DeepSeek review remain required.

No live migration, installation, activation, restart, deployment, credential mutation, or PR #13 merge has occurred.

## Repository state

- Workspace: `/root/projects/hermes-agent-credential-unification`
- Branch: `talos/shared-provider-credentials`
- Current integrated base: `origin/main` at `95638faeaf6b446cd21c6e57fa9383fc9dfaecf8`
- Pull request: https://github.com/dagyrox/hermes-agent/pull/13
- Round-5 code candidate: `7a1e7739a7d061fba3dddcfbcd9cc80d799f3540`
- Round-5 code tree: `e8214cd0bf1f1d72c81f75d034f9022794bbc37d`
- Main integration commit: `82eacf1bea7`
- Athena round-4 review: `t_1b31cf48` / run `554`, `BLOCK` at `606b0d31d29`
- Prerequisite PR #15 merge: `95638faeaf6b446cd21c6e57fa9383fc9dfaecf8`
- Pull request CI pending after publication
- Provenance: Issue `dagyrox/or7-platform#259`, card `t_1af34c61`, run `552`, packet `docs/agentic-operations/shared-provider-credentials.md`

## Round-5 review repairs

1. The no-refresh-token forced-refresh path now recognizes inherited rows before `_mark_exhausted`, acquires the resolved root lock, disables profile persistence, and writes exhausted metadata back to the owning root pool.
2. Terminal xAI/Codex quarantine scopes pool removal to the failed entry id, preserving independent grants in the same root pool and in-memory rotation set.
3. Regression coverage proves both providers take no profile lock or profile write for inherited rows lacking refresh tokens and preserve independent sibling grants after terminal failure.

## Round-4 failure-path repair

1. Failure handling now retains the root source path selected before the serialized refresh transaction instead of re-resolving policy and risking a profile fallback while the root lock is held.
2. Terminal Codex/xAI quarantine clears provider tokens and matching root pool rows in the owning root-store transaction without taking the profile lock.
3. Transient inherited failures persist cooldown metadata into the matching root row; singleton-derived entries without a root pool row materialize one in the root store rather than losing state on reload.
4. Newly materialized root rows omit runtime-only `shared_root_source` and `write_through_ok` markers.
5. Regression coverage includes direct lock-order instrumentation, eligibility-change fail-closed behavior, root persistence without profile shadows, and a deterministic two-process root/profile ABBA contention schedule.

## Round-3 repairs in candidate head

1. Codex and xAI direct runtime refresh now acquire the profile lock and owning source-store lock, re-read inside the lock, hold it across refresh POST, and persist back to the same source.
2. Two trusted profiles sharing one root Codex grant now converge through one single-use rotation; no profile-local token shadow is written.
3. Inherited credential-pool success and already-rotated adoption paths no longer invoke profile persistence while holding the root lock.
4. Legacy zero-argument `_global_auth_file_path` helpers are no longer accepted for shared-root resolution; signature mismatch fails closed.
5. Root lock timeout and root-save failure preserve root bytes, create no profile provider shadow, and record metadata-only `write_through_ok: false` diagnostics.
6. Added real two-process direct-runtime convergence, explicit no root→profile inversion, atomic pre-replace crash cleanup, robust no-secret-output, DeepSeek direct-route separation, and non-auth artifact hash invariants.
7. Updated stale Codex/xAI tests to mock the pure refresh seam used by the source-store transaction path rather than bypassing the owning-store persistence contract.

## Verification

Round-5 candidate verification:

- Focused failure-path suite: **17 passed, 0 failed**; pre-fix RED evidence was **3 failed, 14 passed**.
- Canonical credential and credential-pool matrix: **183 passed, 0 failed** across 10 discovered files.
- Full Hermes auth test surface (`tests/hermes_cli/test_auth*.py`): **371 passed, 0 failed**.
- Merged prerequisite baseline matrix: **142 passed, 0 failed** across four previously stale files.
- Changed-file Ruff, Python byte compilation, and `git diff --check`: passed.
- Changed-file Windows footgun scan: passed, 3 files scanned.
- Athena DeepSeek round-4 review: `BLOCK` at `606b0d31d29`; blocking finding and sibling-grant residual risk are repaired in `7a1e7739a7d`.

Prior round-4 evidence retained for provenance:

- Focused failure-path suite: **15 passed, 0 failed**.
- Canonical credential and credential-pool matrix: **275 passed, 0 failed** across 11 files.
- Full Hermes auth test surface (`tests/hermes_cli/test_auth*.py`): **371 passed, 0 failed**.
- Independent pre-commit review round 3: `PASS`, no security or logic findings.

Prior round-3 evidence retained for provenance:

- Canonical shared-credential matrix: **211 passed, 0 failed**.
- Full Hermes auth test surface (`tests/hermes_cli/test_auth*.py`): **371 passed, 0 failed**.
- Additional credential-pool/recovery suites: **193 passed, 0 failed** before two stale Codex test seams were corrected; the corrected Codex file then passed **32/32**, and xAI auth provider passed **73/73**.
- Changed-file Ruff: passed.
- `git diff --check`: passed.
- `python scripts/check-windows-footguns.py --all`: passed, 809 files scanned.
- Exact-head parity:
  - local: `290450a0b92bc1ff6ed34b3a7a05e44738128fcd`
  - remote branch: `290450a0b92bc1ff6ed34b3a7a05e44738128fcd`
  - pull ref: `290450a0b92bc1ff6ed34b3a7a05e44738128fcd`
  - live PR API/CI head: `290450a0b92bc1ff6ed34b3a7a05e44738128fcd`
- Hosted CI passed affected-area detection, OSV, attribution/common-ancestor, supply-chain scan, e2e, Desktop E2E, blocking Ruff, Windows footguns, Ruff+ty diff, and lock verification. It remained red on the four slices containing the nine frozen-base failures named above.

## Provider source mapping (metadata only)

No credential values, identities, fingerprints, account identifiers, or credential-file contents were read or recorded.

| Provider route | Canonical source | Eligible profiles | Exclusions |
|---|---|---|---|
| OpenAI Codex subscription OAuth (`openai-codex`) | Root `auth.json`, root provider state plus root pool rows | Exact names in root-owned policy only | Every unlisted profile, including `guestwhatsapp`; no external Codex CLI writes |
| Claude Code subscription | External Claude Code CLI store | External CLI lane only | Native Hermes Anthropic inheritance remains hard-denied |
| Direct DeepSeek (`deepseek`) | Root credential-pool row with DeepSeek base URL; static API-key semantics | Exact names in root-owned policy only | Every unlisted profile; OpenRouter-hosted DeepSeek is distinct |
| xAI OAuth and other providers | Existing profile-local behavior unless separately allowlisted | Unchanged by planned activation | Not enabled by this scope |

## Runtime state

- Live runtime changed: **no**
- `/usr/local/lib/hermes-agent` changed: **no**
- Root/profile config or credential state changed: **no**
- Service restarted/deployed: **no**
- Paid DeepSeek inference called: **no**
- Hyperion broadly unblocked: **no**

## Deterministic resume

1. Commit and push the round-5 status artifact, then verify local branch, remote branch, pull ref, and live PR head parity.
2. Obtain an independent exact-final-head Athena DeepSeek review of the published round-5 head; the prior `BLOCK` is fully addressed but does not approve the new bytes.
3. Require a durable `READY_TO_MERGE` covering AC1–AC18/H1–H6, the missing-refresh-token path, sibling-grant preservation, and the integrated PR #15 dependency.
4. Require hosted CI green at that exact head; the old red run predates the PR #15 integration and is not final evidence.
5. After review and CI pass, merge PR #13 only under Jason's explicit current authorization, snapshot the installed tree/config, perform a scoped install, apply the packet's minimal root-owned inheritance policy without credential exposure, restart only the affected gateway/service, and run finite direct proof.
6. Roll back code/config immediately if health, inheritance source, exclusion, or secret-safety checks fail; do not continue to broader rollout.
