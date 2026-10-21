# Shared Provider Credentials — Talos Status

Project: Agent Network Credential Unification  
Primary repository: `dagyrox/hermes-agent`  
Consumer/tracking repository: `dagyrox/or7-platform`  
Issue: `dagyrox/or7-platform#259`  
Hermes Kanban: `t_1af34c61`, repair run `552`  
Acting lane: Talos implementation  
Architecture packet: `docs/agentic-operations/shared-provider-credentials.md`

## Status

`ROUND_4_FAILURE_PATH_REPAIRED_AND_LOCAL_VERIFIED__PUBLISH_REVIEW_CI_PENDING`

Round-4 code candidate `0114e6e7e6d597ae90b7589201d779ce641696aa` (tree `e158d68147b96b06e61b8b7f5e8bf4480a429af2`) repairs inherited Codex/xAI OAuth failure handling under shared-root contention. The candidate is locally committed and verified but has not yet been pushed; remote/PR parity, hosted CI, and exact-head Athena review remain pending.

The round-4 repair removes the root→profile ABBA edge from transient and terminal failure paths, keeps root-owned cooldown/quarantine state in the root store selected at transaction entry, creates no profile credential shadow, and strips runtime-only inheritance markers when a singleton-derived cooldown row must be materialized in the root pool.

A separate pre-commit review found and drove fixes for two additional failure-path defects: lost singleton cooldown state and unstable source-path recomputation. The final pre-commit review returned `PASS` with no security or logic findings. This does not replace the required exact-head Athena DeepSeek review.

No live migration, installation, activation, restart, deployment, credential mutation, merge, paid inference, or Hyperion unblocking occurred.

## Repository state

- Workspace: `/root/projects/hermes-agent-credential-unification`
- Branch: `talos/shared-provider-credentials`
- Base observed: `origin/main` at `b8087e0c69302e9fc4b947e2e96fb49d42a0f5f2`
- Pull request: https://github.com/dagyrox/hermes-agent/pull/13
- Round-4 code candidate: `0114e6e7e6d597ae90b7589201d779ce641696aa`
- Round-4 code tree: `e158d68147b96b06e61b8b7f5e8bf4480a429af2`
- xAI fixture repair commit: `87bfbc9390077d3991e3981070c60b8f17a1d5c2`
- Round-3 repair commit: `290450a0b92bc1ff6ed34b3a7a05e44738128fcd`
- Athena round-3 review artifact: `docs/agentic-operations/athena-review-pr13-87bfbc.md`
- Hosted CI: https://github.com/dagyrox/hermes-agent/actions/runs/39510318075
- Provenance: Issue `dagyrox/or7-platform#259`, card `t_1af34c61`, run `552`, packet `docs/agentic-operations/shared-provider-credentials-work-packet.md`

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

Round-4 candidate verification:

- Focused failure-path suite: **15 passed, 0 failed**.
- Canonical credential and credential-pool matrix: **275 passed, 0 failed** across 11 files.
- Full Hermes auth test surface (`tests/hermes_cli/test_auth*.py`): **371 passed, 0 failed**.
- Changed-file Ruff, Python byte compilation, and `git diff --check`: passed.
- Changed-file Windows footgun scan: passed, 2 files scanned.
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

1. Push the round-4 candidate and verify local branch, remote branch, pull ref, and live PR head parity.
2. Obtain an independent exact-head Athena DeepSeek review of the published round-4 head; prior review artifacts do not approve this candidate.
3. Require a durable `READY_TO_MERGE` or `BLOCK` result covering AC1–AC18/H1–H6 and all round-4 failure-path repairs.
4. Resolve or receive explicit authorized maintainer adjudication for any required hosted CI failures; red checks must not be relabeled green.
5. Only after those gates pass may the separately authorized install/activation phase begin with snapshot, protected rollback, minimal authorized policy, runtime source readback, and finite smokes.
6. Do not merge, migrate, activate, install, restart, deploy, mutate/revoke credentials, call paid DeepSeek inference, or unblock Hyperion without the packet's separate approvals and quiescence checks.
