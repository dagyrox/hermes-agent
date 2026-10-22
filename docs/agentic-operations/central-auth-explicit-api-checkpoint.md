# Central authentication / explicit API mode — blocked checkpoint

Project: Agent Authentication / Shared Credentials.
Primary repository: dagyrox/hermes-agent.
Consumer repository: dagyrox/or7-platform, issues #259/#260.
Acting lane: Talos; card t_ba92d6df; run610; openai-codex/gpt-6.1-sol.
Work packet: Kanban card t_ba92d6df body.

## Verified baseline

Worktree: `/root/projects/hermes-agent-credential-unification/.worktrees/t_ba92d6df`.
Branch: `talos/issue259-central-auth-explicit-api`.
Fetched fork main and fast-forwarded from 3a906fc872c to
fb739936576acc4b54f0244978a0201f56c96483 before edits.
Installed `/usr/local/lib/hermes-agent` is clean at that same commit.
No production code, live routing configuration, credential files, or service
state was deliberately changed by this run. Native CLI may manage its own
refresh state during the explicitly bounded subscription probes.
This checkpoint is NOT an implementation or release-completion claim.

## Effective route / source matrix

| Lane | Current launch / credential authority | Gap |
| --- | --- | --- |
| Hermes/default | Configured openai-codex/gpt-6.1-sol; `/root/.hermes/auth.json` OAuth pool | Real no-tools subscription smoke passed |
| Talos | Canonical supervised chain active for Talos only: openai-codex/gpt-6.1-sol → native Claude Sonnet → DeepSeek Flash. Task model/provider overrides bypass this supervisor. Shared root Codex permitted. | Native fallback uses `/root/.claude/.credentials.json`, which has no access/refresh token |
| Hyperion | Profile model openai-codex/gpt-5.6-terra; shared root Codex permitted; no canonical role chain currently activated | Separate legacy native home contains expired, unrefreshable Claude session |
| Athena | Profile model anthropic/claude-opus-5; not in active canonical role map or root trusted-profile list | Direct Hermes Anthropic path, NOT proof of native CLI usage; profile token revoked |
| Apollo | Profile model anthropic/claude-opus-5; not in active canonical role map or root trusted-profile list | Same revoked token as Athena; not centralized |
| Native Claude fallback | `_claude_environment` only retains base environment; HOME/.claude owns authentication | Does not read Athena/Apollo token; root native auth empty |
| orion / athena-hermes | Existing profiles, not admitted as active canonical internal lanes by this task | No new grant; orion is retired, athena-hermes remains unused/unverified |
| guestwhatsapp / gf / other guests | Excluded from internal canonical routes and root shared trust | No credential grant or routing change |

Relevant implementation seams:
- `agent/credential_trust.py`: root-only trust gate; raw `anthropic` inheritance intentionally forbidden.
- `hermes_cli/kanban_db.py::_backend_failover_worker_argv`: per-task overrides bypass supervised canonical routes.
- `hermes_cli/backend_failover.py::resolve_role_chain`: frozen canonical chains; direct OpenAI/Anthropic API legs currently forbidden.
- `hermes_cli/backend_failover_worker.py::_claude_environment`, `_leg_available`, `_leg_command`: native auth environment / readiness / launch.
- `hermes_cli/runtime_provider.py::resolve_runtime_provider`: direct runtime resolution also needs policy coverage; worker-only changes are insufficient.

## Real subscription probes

All children received allowlisted environment without Kanban task/run/claim,
session, delegation or lifecycle ownership. No model-call tools were enabled.
No paid OpenAI or Anthropic API probe was run.

1. Existing Athena `CLAUDE_CODE_OAUTH_TOKEN` supplied only in native CLI subprocess
   environment. Athena and Apollo values compare equal in memory; values and
   fingerprints were never printed or written to artifacts.
   `claude -p` with model `claude-sonnet-5-5`, one turn, no tools,
   strict empty MCP configuration, no session persistence, disposable cwd:
   exit 1, `is_error=true`, result:
   `Failed to authenticate: OAuth token revoked. Please log in again or contact your administrator.`
   This is a server authentication failure, NOT a subscription quota result.
2. Root native path `/root/.claude/.credentials.json` has neither access nor
   refresh token. Only other discovered native credential source:
   `/root/.hermes/profiles/hyperion/home/.claude/.credentials.json`.
   It has an expired access token and refresh token. Native CLI single-turn
   attempt with that HOME and the same no-tools bounds returned exit 1,
   `is_error=true`, result:
   `Failed to authenticate: OAuth session expired and could not be refreshed`.
3. `python -m hermes_cli.backend_smoke --profile default --provider openai-codex --model gpt-6.1-sol --timeout-seconds 90`
   returned exit 0 and
   `{"model": "gpt-6.1-sol", "ok": true, "provider": "openai-codex", "returncode": 0}`.
   Existing helper requires exact BACKEND_SMOKE_OK response and zero tool calls.

## API source inventory (paths / presence only)

- OpenAI API: `/root/.hermes/.env` contains `OPENAI_API_KEY`.
  Root pool contains environment-reference entries, not a separate proven
  server credential. Funded status and API usability NOT VERIFIED.
- Anthropic API: `/root/.hermes/.env`, Athena `.env`, and Apollo `.env` do not
  contain `ANTHROPIC_API_KEY`. Historical pool references to that variable
  are NOT proof of an available credential. API usability NOT VERIFIED.
- No subscription token was supplied to a raw API, no API mode activated,
  no top-up/purchase/revocation/delete/new-account action performed.

## Blockers and deterministic continuation

Genuine operator gate: recover one viable native Claude subscription login;
all currently discovered candidates are missing, revoked, or unrefreshable.
Do not copy revoked tokens into a new canonical source. Supply a separate true
Anthropic API credential via the approved secret mechanism if Claude emergency
API mode must be usable; never place its value in board comments or artifacts.

Independent bounded Athena decision requested on card t_0c464834. Decision
must settle canonical native source and refresh owner, root-only vendor mode
control for new runs, trust/guest boundary, and direct-runtime enforcement.
No broad architecture reset or model-policy redesign requested.

After input/decision: reorient on card comments, confirm worktree ownership,
fetch fork main safely preserving this checkpoint, implement only the bounded
delta, add a few focused behavioral cases, use scripts/run_tests.sh, then exact-
head PR/CI/security review. Subscription remains the default; explicit API
selection/revert tested offline. Do not perform paid canaries or restart the
live gateway from a worker cgroup. No PR/commit/merge/install activation was
performed by this run.
