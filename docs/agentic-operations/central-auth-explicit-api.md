# Central vendor authentication and explicit API switching

Project: Agent Authentication / Shared Credentials.
Repository: dagyrox/hermes-agent; consumer: dagyrox/or7-platform #259/#260.
Lane: Talos; card t_ba92d6df, run613.
Packet: `/root/.hermes/work-packets/central-auth-explicit-api-switch.md`.
Architecture authority: Athena decision t_0c464834, copied into the implementation card.

## Contract

For managed internal deployments (`profiles.shared_credentials.enabled: true`),
the root configuration owns vendor access for the default/root instance and
explicitly trusted internal profiles. Guests, customer profiles, retired Orion,
Sentinel, Forge, and other untrusted profiles receive no new credential access.
`agent/credential_trust.py`'s raw Anthropic inheritance prohibition is unchanged.

An absent vendor or mode means `subscription`. `api` is an explicit operator
choice, global per named vendor for trusted lanes, not a per-lane override.
Profile `provider_access` settings are ignored with a warning. Unknown vendors,
settings, modes, or incomplete API selections fail closed. No top-up, purchase,
account creation, revocation, or automatic subscription-to-paid escalation occurs.

Canonical sources:

| Vendor / access | Source | Lifecycle owner |
| --- | --- | --- |
| Claude subscription | Root `claude/subscription.token` if securely configured; otherwise the OS account's native `.claude/.credentials.json` | Native Claude CLI only |
| OpenAI subscription | Root `auth.json`, existing Codex OAuth resolution and locks | Existing root Hermes OAuth machinery |
| Anthropic API | Root `.env`, `ANTHROPIC_API_KEY`, true API-key shape only | Operator |
| OpenAI API | Root `.env`, `OPENAI_API_KEY`, true API-key shape only | Operator |

On this host the root is `/root/.hermes`, native HOME is `/root`.
No profile `.env`, inherited ambient token, lane HOME, Hermes OAuth pool, or
Anthropic-compatible URL can shadow the native Claude source. A canonical
setup-token file must be a regular owner-only 0600 file owned by the launching
account. Hermes only reads it into the native subprocess environment, never
refreshes/writes it. Do not seed a revoked token. If that source is absent,
Athena's D6 alternative uses one native HOME store; the native CLI owns refresh.
The managed raw Anthropic adapter cannot read or write the native store, even
from an untrusted profile. Guests may retain their own profile-scoped credentials.

## Supported control

Use the existing config command against the ROOT home, not an active lane home.
Configure provider and model first while the mode remains subscription:

```sh
HERMES_HOME=/root/.hermes hermes config set provider_access.anthropic.api_provider anthropic
HERMES_HOME=/root/.hermes hermes config set provider_access.anthropic.api_model '<approved-Anthropic-API-model>'
HERMES_HOME=/root/.hermes hermes config set provider_access.openai.api_provider openai-direct
HERMES_HOME=/root/.hermes hermes config set provider_access.openai.api_model '<approved-OpenAI-API-model>'
```

Model placeholders above are NOT runnable model IDs. Choose an approved real API
model before switching; a subscription model alias is not assumed API-compatible.
Provide the separate key through the approved secret mechanism, never chat,
work packets, task comments, per-lane copies, or a command argument.

Explicit operator instructions map to:

```sh
# "Switch Claude to API" / "switch OpenAI to API" (NEW runs only):
HERMES_HOME=/root/.hermes hermes config set provider_access.anthropic.mode api
HERMES_HOME=/root/.hermes hermes config set provider_access.openai.mode api

# "Back to subscription" for the named vendor (NEW runs only):
HERMES_HOME=/root/.hermes hermes config set provider_access.anthropic.mode subscription
HERMES_HOME=/root/.hermes hermes config set provider_access.openai.mode subscription

# Secret-free readback, not a live model call:
HERMES_HOME=/root/.hermes hermes kanban backend-status --json
python -m hermes_cli.native_claude --profile default --status
```

These commands document capability; this delivery did NOT activate production API
mode or perform a paid canary. Missing/invalid keys fail before a request. Funding
and server usability of both API routes remain UNVERIFIED. No automatic mode
change, spending recovery, or silent switch back to subscription is attempted.

Supervised dispatch validates the original canonical chain, substitutes only its
first leg when that primary vendor is explicitly API-selected, and serializes the
selected route into the worker command. A frozen API snapshot is revalidated
against the canonical route, not the current mode file. Fallback legs retain their
subscription routes and the final approved DeepSeek Flash route. API providers
cannot be inserted after a subscription leg. API preflight/capability failures
are terminal rather than silently substituting another mode. Nested in-process
fallback is disabled in supervised children. Managed legacy fallback config
cannot contain direct vendor API routes or remap a Codex fallback to paid API.
CLI credential refresh retains its initial vendor-mode snapshot rather than
adopting a later root edit. Existing override/goal-mode dispatcher bypasses remain
unchanged, but managed direct-runtime credential boundaries still apply.

## Native interactive launcher and activation

`hermes_cli/native_claude.py` and the supervised worker use the SAME
`claude_environment` builder. The module selects native Claude in subscription
mode, or Hermes' true Anthropic API runtime with its explicitly configured model
in API mode. Native-only CLI arguments are not replayed across API runtimes.
`--login` is an explicit native operator recovery action, never automatic.

The reviewed deployment launcher is `scripts/launchers/athena`. After merge and
independent release approval, install it at `/root/.local/bin/athena` with mode
0755. On this host `/root/.local/bin/athena` is a symlink to
`/usr/local/bin/athena`, which OCC invokes: install the canonical executable,
preserve the symlink. The existing launcher is not changed by committing this file. It retains
attach/status/login/smoke/escalation/explicit-fallback/logs/kill control surfaces.
Existing tmux sessions are attached unchanged; mode switches apply to newly
created sessions only. `athena smoke` follows the explicitly selected mode and
can incur API cost if the operator has selected API; do not run it as a paid
acceptance test without separate authorization.

Root trust currently admits Hyperion and Talos. Activation must explicitly add
Athena and Apollo as approved internal lanes and configure their canonical
supervised roles before claiming centralized native launch coverage there.
The existing `athena-hermes` explicit fallback must ALSO be admitted, with
`openai-codex/gpt-6.1-sol` replacing its erroneous raw-Anthropic primary while
preserving its own persona and other config. Independent Athena review F1
requires that closure before installing the launcher; do not repoint it to
Talos/default and thereby change persona isolation.
Do not widen the provider inheritance list to Anthropic. Keep persona, skills,
plugins, memory, tool configuration, and unrelated DeepSeek policy unchanged.
Known revoked profile-token remnants are NOT used or copied by the new path;
credential removal/rotation is not performed by the implementation run.

No gateway restart from a worker execution cgroup is permitted. An authorized
release lane must verify independent exact-head review and green hosted CI,
install the approved commit, stage subscription defaults, safely drain workers,
and arrange any required gateway reload through the external service manager.
Report actual installed/activated evidence separately; a configured chain is
not proof that a running dispatcher has loaded it. Do not interrupt live workers.

## Evidence from run613 (not an installation claim)

- Installed source remained clean at `fb739936576acc4b54f0244978a0201f56c96483`.
- Candidate OpenAI subscription: real `gpt-6.1-sol` turn through root `auth.json`,
  exit 0, exact `BACKEND_SMOKE_OK`, zero tools. Candidate CLI module was executed
  from this worktree, not inferred from the installed launcher's acknowledgment.
- Candidate native Claude subscription: real `claude-sonnet-5-5` turn via
  `/root/.claude/.credentials.json`, forced HOME `/root`, exit 0, exact
  `BACKEND_SMOKE_OK`, zero tool events. No setup-token file was created.
- Both probe children scrubbed inherited Kanban/session/delegation ownership,
  disabled model-call tools and lifecycle mutations, and made no paid API call.
- An initial native probe's overly broad `tool_use` substring check matched a
  usage-counter field. The existing structural smoke validator was used for one
  corrected verification; this was a verification-parser failure, not auth failure.
- Historical run610 revoked-token evidence is retained in
  `central-auth-explicit-api-checkpoint.md`; it no longer describes the viable
  canonical native store observed and server-verified during run613.
- Metadata-only API inventory: root OpenAI key shape present; separate Anthropic
  API key absent/invalid. Both funding and API server usability UNVERIFIED.
- `central-auth-route-readback.json` is actual candidate CLI output. Its
  `configured-unverified` fields deliberately do not persist model-acceptance
  claims. This evidence section records the real turn results separately.
- Eight focused existing credential/routing/adapter test files including FOUR new
  behavioral regressions: 463 passed, zero failed. No full local suite repeated.
- `ruff check .`, Windows-footgun checker, launcher bash syntax and
  `git diff --check` passed. Existing hosted CI and independent Athena review
  remain required on the published head.
- Root user service `hermes-gateway.service` active/running; gateway PID 1483330
  observed. No service restart, installation, root routing edit, credential edit,
  guest grant, or profile skills/plugins/cron/memory mutation performed by run613.

Release evidence must append the reviewed head, hosted CI URLs/conclusions,
merged/installed commit, exact activation readback, and service health after
external safe activation. Do not treat this pre-release evidence as completion.

## Exact-head review and CI follow-through

Athena independently approved production code at `a50f02a92528`, card
`t_d4d8be37`/run614 (independent DeepSeek lane), with release preconditions F1
(existing explicit fallback admission above) and F2 (canonical executable/symlink).
API failure may advance to another vendor's subscription or approved DeepSeek
leg after a real request; it does not change the selected vendor's mode or
automatically spend on a paid fallback. Missing-key API preflight is terminal.
API server funding/usability remains unverified until separately authorized.

Initial hosted CI failed before jobs on a GitHub internal error. Same-head
reopen recovered full existing CI run39800725940. Its slice4 exposed two old
synthetic-worker E2E fixtures which injected availability but did not establish
the new trust boundary. The first follow-up code change was that existing test
fixture: temporary root trust and a 0600 synthetic native token, never runner or
live auth. Production bytes stay identical to Athena's approved candidate.
The repaired three E2E tests plus four new policy regressions passed (7/7).
Slice6 separately reported runner shutdown (exit143), not an assertion; its
initiating signal and infrastructure attribution were not established.
No local broad suite or new test harness was added. Updated manifest includes
the repaired existing E2E file (21 listed files, self-excluded). The successor
still requires narrow independent exact-head confirmation and hosted green CI.
Athena confirmed that successor `798fed3d0ec` in `t_99becafe`/run617. Its hosted
run39801852705 passed slice4, then exposed a separate slice1 test assumption:
the native child HOME assertion hardcoded `/root` instead of the OS-account
home (`/home/runner` on GitHub). Repair card `t_2cd0a51f`/run616 makes that test
use an isolated account-home fixture distinct from inherited HOME and asserts
the child uses the former. The old assertion failed under this deterministic
fixture before correction; all 74 focused worker/E2E/policy tests then passed.
No production bytes changed. Slice6 again explicitly reported runner shutdown
(exit143), not a test assertion. Manifest remains 21 entries, regenerated after
the test/document edits. This successor also requires fresh independent review
and terminal hosted green CI; earlier approvals cannot substitute for it.
Hermes/default release continuation card `t_fa44c5ab` owns authorized merge,
installation and safe activation after those gates; it must never use an older
approval for changed production bytes.

## Diagnosed protocol fixture successor (t_5a066f08/run623)

Hermes diagnostic `t_3c619957`/run619 independently reproduced a repository
test-fixture hang on `a14e99f39f9`, after 44 protocol tests passed. The fixture
replaced `hermes_constants` with a mock but did not concretize
`get_default_hermes_root`. The new root-policy lookup consequently passed a
mock stream to PyYAML whose reads never reached EOF. All three inspected hosted
slice6 logs left this protocol file as the sole pending file. This establishes
a fixture defect, not the identity of the hosted shutdown signal's sender.

The narrow successor supplies a concrete isolated `tmp_path` root in the existing
server fixture. Production parsing, fail-closed policy, assertions, test selection,
and timeouts are unchanged. Existing canonical verification completed normally:
`HERMES_PYTHON=/usr/local/lib/hermes-agent/venv/bin/python scripts/run_tests.sh
tests/tui_gateway/test_protocol.py tests/hermes_cli/test_provider_access.py -q`
returned exit0: 95 protocol tests plus 4 policy tests passed, zero failures, no
retry reported. Focused ruff and `git diff --check` passed. No broad local suite
or new harness was run. The refreshed manifest lists 22 files, self-excluded.

Talos publishes an ordinary successor to the same PR21; production remains
byte-identical to approved `a50f02a92528`. Independent Athena confirmation and
genuine hosted green must bind that successor SHA before release. Release
`t_fa44c5ab` retains F1/F2 and all merge/install/activation ownership. This repair
does not change live configuration, credentials, installation, or services and
makes no paid API call. Missing Anthropic API credentials remain a fail-closed
capability status, not a subscription-release blocker.
