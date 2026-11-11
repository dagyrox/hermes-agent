# OCC read-only Kanban release and rotation runbook

Project: Or7 Command Center (OCC). Repositories: `dagyrox/hermes-agent` (provider), `dagyrox/or7-platform` (consumer).
Talos · OCC Issue263 · t_1174b82d/run927 · design predecessor t_b2553ec2/run833.
Authoritative packet: consumer `docs/work-packets/occ-kanban-visual.md`; refinement `docs/work-packets/occ-kanban-provider-implementation.md`.

## Implemented contract (this branch, not installed)

Three static GET endpoints, all under the existing Kanban plugin namespace:

- `GET /api/plugins/kanban/occ/v1/summary`
- `GET /api/plugins/kanban/occ/v1/tasks`
- `GET /api/plugins/kanban/occ/v1/task?task_id=t_xxxxxxxx`

Wire contract `schema_version: 1`. Errors are `{"error":"<code>"}` only:
`unauthenticated` (401), `invalid_request`/`invalid_order`/`invalid_task_id`/
`cursor_invalid` (400), `cursor_stale` (409), `not_found` (404),
`board_unavailable`/`schema_unsupported`/`response_too_large`/`auth_unavailable`
(503). Responses carry `Cache-Control: no-store` and
`X-Content-Type-Options: nosniff` and are capped at 1 MiB.

Fields the design asks for that the current schema cannot support are
**always null**: `status_entered_at`, `scheduled_at`, `reason_preview`, and
`links` is always `[]`. OCC must not render "in status since" or a repository
derived from anything other than typed references (of which v1 has none).

## Release boundary

This is preparation, not authorization to execute release. Keep provider `kanban.occ_read.enabled` false and OCC `HERMES_KANBAN_READ` false until exact-head Athena reviews, Apollo built-screen audit, green exact-head CI, provider installation verification and Jason/operator production credential/activation approvals exist. Do not install from an unreviewed dirty tree. Merging a PR does not prove installed capability. Existing installed runtime is not claimed to implement this API.

## Configuration contract (no live values)

Provider behavior belongs in Hermes `config.yaml`, using the supported `hermes config set` path:
- `kanban.occ_read.enabled`: false by default; approved operator controls activation.
- `kanban.occ_read.board`: one explicit canonical board slug, never a request-selected database.
- Dedicated credential: `HERMES_KANBAN_OCC_READ_TOKEN`, in the approved provider secret store. Never in config.yaml, command arguments, logs or screenshots.

Consumer counterpart in approved `/etc/or7-command.env`, variable names only:
- `HERMES_KANBAN_READ`: false until approved activation.
- `HERMES_KANBAN_BASE_URL`: verified private/loopback provider origin.
- `HERMES_KANBAN_BOARD`: same explicit board identity as provider.
- `HERMES_KANBAN_OCC_READ_TOKEN`: matching dedicated read-only credential; no NEXT_PUBLIC variant.

No broad dashboard session token, OAuth refresh token, raw SQLite path or DB access belongs in OCC.

## Rotation — authorized operator only, no token output

The provider reads `HERMES_KANBAN_OCC_READ_TOKEN` **once at process start**, so
rotation always requires a provider process restart. There is no hot reload and
no rotation endpoint; that is deliberate, because a live rotation path would be
a second credential-acceptance surface.

A rotation is a four-part transaction. If any step is ambiguous, stop with the
surface off — an unrotated credential is a smaller risk than a half-rotated one.

1. Obtain separate production credential/activation authority, identify the approved provider/consumer SHA tuple and the approved secret manager paths. Do not paste credentials in GitHub, Kanban or chat.
2. Disable consumer reads and verify neutral feature-off behavior. Disable provider read surface if necessary. Preserve unrelated configuration and service state.
3. Generate at least 256 bits with the secret manager's native generator directly into a new secret version. The value must be ASCII `token68` of 43–256 characters or the provider treats the surface as unconfigured. Do not print, shell-expand into argv, or place in shell history. No commands in this runbook generate a production token.
4. Use the approved secret-manager installation mechanism to update provider and consumer dedicated secret entries with restrictive existing ownership/modes. Verify matching secret versions via protected metadata or boolean equality only, never values or token fragments. Update only the dedicated entries.
5. Apply the separately approved process reload/release procedure **for both sides**. This implementation task does not authorize a restart. Order matters: restart the provider first (it must accept the new token before the consumer sends it), then the consumer.
6. With a non-logging private probe, read the three exact GET endpoints using the new credential; check schema_version, bound board and bounded allowlisted response. Then assert the negative cases, each expecting 401: the **old** credential, a `Cookie` header instead of the bearer, a request carrying `X-Forwarded-For`, and the same new credential against `/api/gateway/drain`. Record only status codes, SHA tuple, timestamps, sizes and secret version IDs.
7. Re-enable approved provider/consumer flags only after probe success. Verify owner access, counts, freshness, readonly network and feature-off rollback path. On ambiguity, remain off.
8. Revoke the old secret version in the approved store according to operator retention policy. Do not delete unrelated files or credentials.

### Expected negative results (record these, they are the security proof)

| Probe | Expected |
|---|---|
| old token on `/occ/v1/summary` | 401 `unauthenticated` |
| valid token + `Cookie:` header | 401 `unauthenticated` |
| valid token + `X-Forwarded-For` | 401 `unauthenticated` |
| valid OCC token on `/api/gateway/drain` | 401 `unauthenticated` |
| valid token on `/occ/v1/summary/` (trailing slash) | not served as an OCC route |
| any method other than GET | 405 with `Allow: GET`, handler not reached |

## Rollback / failure

Disable OCC feature flag first, then provider read flag through approved configuration controls. Keep source/worktree/evidence intact; do not modify or repair the Kanban database. Restore prior approved secret versions only through the protected secret manager and approved process reload. A failed read must show explicit unavailability, never fixture data, zero counts or inferred execution health.

## Evidence required before release

Record reviewed/merged/installed provider SHA separately, consumer PR/head, machine-readable contract witness, exact-head CI, local integration fixture label, production supported-read probe (only once authorized), Apollo/Athena verdicts, and operator approval IDs. Until those exist this is a local build candidate, not a live feature.
