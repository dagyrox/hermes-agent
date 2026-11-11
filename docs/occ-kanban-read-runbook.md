# OCC read-only Kanban release and rotation runbook

Project: Or7 Command Center (OCC). Repositories: `dagyrox/hermes-agent` (provider), `dagyrox/or7-platform` (consumer).
Talos · OCC Issue263 · t_1174b82d/run835 · design predecessor t_b2553ec2/run833.
Authoritative packet: consumer `docs/work-packets/occ-kanban-visual.md`; refinement `docs/work-packets/occ-kanban-provider-implementation.md`.

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

1. Obtain separate production credential/activation authority, identify the approved provider/consumer SHA tuple and the approved secret manager paths. Do not paste credentials in GitHub, Kanban or chat.
2. Disable consumer reads and verify neutral feature-off behavior. Disable provider read surface if necessary. Preserve unrelated configuration and service state.
3. Generate at least256 bits with the secret manager's native generator directly into a new secret version. Do not print, shell-expand into argv, or place in shell history. No commands in this runbook generate a production token.
4. Use the approved secret-manager installation mechanism to update provider and consumer dedicated secret entries with restrictive existing ownership/modes. Verify matching secret versions via protected metadata or boolean equality only, never values or token fragments. Update only the dedicated entries.
5. Apply the separately approved process reload/release procedure. This implementation task does not authorize a restart.
6. With a non-logging private probe, read the three exact GET endpoints using the new credential; check schema_version, bound board and bounded allowlisted response. Assert old credential gets401, cookies fail and a protected route outside the three rejects the new credential. Record only status codes, SHA tuple, timestamps, sizes and secret version IDs.
7. Re-enable approved provider/consumer flags only after probe success. Verify owner access, counts, freshness, readonly network and feature-off rollback path. On ambiguity, remain off.
8. Revoke the old secret version in the approved store according to operator retention policy. Do not delete unrelated files or credentials.

## Rollback / failure

Disable OCC feature flag first, then provider read flag through approved configuration controls. Keep source/worktree/evidence intact; do not modify or repair the Kanban database. Restore prior approved secret versions only through the protected secret manager and approved process reload. A failed read must show explicit unavailability, never fixture data, zero counts or inferred execution health.

## Evidence required before release

Record reviewed/merged/installed provider SHA separately, consumer PR/head, machine-readable contract witness, exact-head CI, local integration fixture label, production supported-read probe (only once authorized), Apollo/Athena verdicts, and operator approval IDs. Until those exist this is a local build candidate, not a live feature.
