"""Exact-head GitHub acceptance for explicitly declared PR tasks.

Network work happens outside SQLite transactions. The lifecycle owner persists
receipts only after rechecking the captured run/status/contract under its lock.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Any
from urllib.parse import quote

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PR = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/pull/([1-9][0-9]*)")


class _EvidenceError(Exception):
    """Secret-safe acceptance failure attributed to one collection phase."""

    def __init__(self, phase: str, reason: str):
        super().__init__(reason)
        self.phase = phase
        self.reason = reason


def validate_contract(value: str | None) -> str:
    if value is None or value == "local-only":
        return "local-only"
    if not isinstance(value, str) or not (_REPO.fullmatch(value) or _PR.fullmatch(value)):
        raise ValueError("completion_contract must be local-only, OWNER/REPO, or an exact GitHub PR URL")
    return value


def _decode_api_output(raw: str, *, paginate: bool) -> Any:
    """Decode both gh <2.47 streamed pages and newer --slurp-shaped output."""
    decoder = json.JSONDecoder()
    values = []
    offset = 0
    while offset < len(raw):
        while offset < len(raw) and raw[offset].isspace():
            offset += 1
        if offset == len(raw):
            break
        value, offset = decoder.raw_decode(raw, offset)
        values.append(value)
    if not values:
        raise ValueError("GitHub returned no JSON evidence")
    if not paginate:
        if len(values) != 1:
            raise ValueError("GitHub returned multiple unpaginated values")
        return values[0]
    if all(isinstance(value, dict) and set(value) == {"__hermes_page"} for value in values):
        return [value["__hermes_page"] for value in values]
    if len(values) == 1 and isinstance(values[0], list):
        # gh >=2.47 with --slurp produced one array containing every page. Accept
        # that historical shape so recorded fixtures and mixed fleets stay valid.
        return values[0]
    return values


def _api(endpoint: str, *, query: str | None = None, paginate: bool = False) -> Any:
    command = ["gh", "api", endpoint, "--hostname", "github.com"]
    if query is not None:
        command += ["-f", "query=" + query]
    if paginate:
        # --slurp was added in gh 2.47. --jq is supported by older gh; wrapping
        # each page removes the single-page array ambiguity while preserving the
        # page's original JSON shape.
        command += ["--paginate", "--jq", "{__hermes_page: .}"]
    try:
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=30, check=True)
    except FileNotFoundError as exc:
        raise _EvidenceError("GitHub CLI", "gh executable is unavailable") from exc
    except subprocess.TimeoutExpired as exc:
        raise _EvidenceError("GitHub API", "gh request timed out") from exc
    except subprocess.CalledProcessError as exc:
        # stderr may contain credentials, host details, or response bodies. It is
        # deliberately neither interpolated nor chained into a durable receipt.
        raise _EvidenceError("GitHub API", "gh request exited non-zero") from None
    value: Any = _decode_api_output(result.stdout, paginate=paginate)
    for page in value if paginate else [value]:
        if isinstance(page, dict) and page.get("errors"):
            raise _EvidenceError("GitHub API", "GitHub returned incomplete evidence")
    return value


def _phase_api(phase: str, endpoint: str, *, query: str | None = None, paginate: bool = False) -> Any:
    try:
        return _api(endpoint, query=query, paginate=paginate)
    except _EvidenceError as exc:
        raise _EvidenceError(phase, exc.reason) from None
    except (ValueError, TypeError):
        raise _EvidenceError(phase, "GitHub returned malformed or incomplete JSON") from None


def _configured_required(repo: str) -> set[tuple[str, int | None]]:
    """Read the explicit per-repository CI policy from the active profile."""
    try:
        from hermes_cli.config_effective import load_user_config_effective
        cfg = load_user_config_effective(fail_closed=True)
    except Exception:
        raise _EvidenceError("required CI configuration", "config.yaml is unreadable") from None
    kanban = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    acceptance = kanban.get("pr_acceptance", {}) if isinstance(kanban, dict) else {}
    policies = acceptance.get("required_checks", {}) if isinstance(acceptance, dict) else {}
    if not isinstance(policies, dict):
        raise _EvidenceError("required CI configuration", "kanban.pr_acceptance.required_checks must be a mapping")
    entries = policies.get(repo, [])
    if not isinstance(entries, list):
        raise _EvidenceError("required CI configuration", f"the {repo} policy must be a list")
    required: set[tuple[str, int | None]] = set()
    for entry in entries:
        if isinstance(entry, str):
            context, app_id = entry.strip(), None
        elif isinstance(entry, dict):
            context, app_id = entry.get("context"), entry.get("app_id")
            context = context.strip() if isinstance(context, str) else ""
            if app_id is not None and (not isinstance(app_id, int) or isinstance(app_id, bool)):
                raise _EvidenceError("required CI configuration", f"the {repo} policy has a non-integer app_id")
        else:
            raise _EvidenceError("required CI configuration", f"the {repo} policy has an invalid check entry")
        if not context:
            raise _EvidenceError("required CI configuration", f"the {repo} policy has an empty check context")
        required.add((context, app_id))
    return required


def collect_acceptance(contract: str, published_pr: str | None) -> dict:
    receipt = {"ok": False, "classification": "missing", "head_sha": None,
               "pr_url": published_pr, "checks": [],
               "recovery": "Fix required failures, rerun infrastructure checks or wait, then retry completion. "
                           "Use kanban_block if human input is needed; receipts remain on the task event log."}
    try:
        declared = _PR.fullmatch(contract)
        url = contract if declared else published_pr
        match = _PR.fullmatch(url or "")
        if not match or (not declared and match[1] != contract) or (declared and published_pr and published_pr != contract):
            receipt["detail"] = "Supply metadata.published_pr matching the persisted completion contract."
            return receipt
        repo, number = match[1], int(match[2])
        receipt["pr_url"] = url
        owner, name = repo.split("/")
        query = '''{repository(owner:%s,name:%s){pullRequest(number:%d){headRefOid baseRefName state
            baseRef{branchProtectionRule{requiredStatusChecks{context app{databaseId}}}}}}}''' % (
                json.dumps(owner), json.dumps(name), number)
        pr = _phase_api("pull request metadata", "graphql", query=query)["data"]["repository"]["pullRequest"]
        sha, branch = pr["headRefOid"], pr["baseRefName"]
        receipt["head_sha"] = sha
        if not re.fullmatch(r"[0-9a-f]{40}", sha) or pr["state"] not in {"OPEN", "MERGED"}:
            raise ValueError("PR is closed or current head is unavailable")
        protection = (pr.get("baseRef") or {}).get("branchProtectionRule") or {}
        required = _configured_required(repo)
        required.update((r["context"], (r.get("app") or {}).get("databaseId"))
                        for r in protection.get("requiredStatusChecks", []))
        rules = _phase_api("repository rules", f"repos/{repo}/rules/branches/{quote(branch, safe='')}?per_page=100", paginate=True)
        for page in rules:
            for rule in page:
                if rule["type"] == "required_status_checks":
                    required.update((r["context"], r.get("integration_id"))
                                    for r in rule["parameters"]["required_status_checks"])
        receipt["required"] = [{"context": c, "app_id": a} for c, a in sorted(required, key=str)]
        if not required:
            receipt["detail"] = ("No required CI policy is configured for this repository. Configure GitHub protection/rules "
                                 "or kanban.pr_acceptance.required_checks; PR contracts never auto-pass zero checks.")
            return receipt
        pages = _phase_api("check runs", f"repos/{repo}/commits/{sha}/check-runs?per_page=100&filter=latest", paginate=True)
        runs = [run for page in pages for run in page["check_runs"]]
        if not pages or len({r["id"] for r in runs}) != pages[0]["total_count"]:
            raise ValueError("Incomplete check-run pagination")
        statuses = [{**s, "sha": sha} for page in _phase_api(
            "commit statuses", f"repos/{repo}/commits/{sha}/statuses?per_page=100", paginate=True) for s in page]
        outcomes = []
        for context, app_id in sorted(required, key=str):
            matching = [r for r in runs if r["name"] == context and
                        (app_id in (None, -1) or r["app"]["id"] == app_id)]
            # A legacy status can satisfy an unpinned context, but never a check pinned to an app.
            legacy = [s for s in statuses if s["context"] == context] if app_id in (None, -1) else []
            selected = matching + ([max(legacy, key=lambda s: s["id"])] if legacy else [])
            if not selected:
                outcomes.append("missing")
                receipt["checks"].append({"name": context, "classification": "missing", "head_sha": sha})
            for check in selected:
                is_run = "conclusion" in check
                outcome = check.get("conclusion") if is_run else check["state"]
                classification = _classify(check, sha, outcome, is_run)
                outcomes.append(classification)
                receipt["checks"].append({"name": context, "id": check["id"],
                    "url": check.get("html_url") or check.get("target_url"),
                    "head_sha": check.get("head_sha", check.get("sha")),
                    "classification": classification, "conclusion": outcome})
        # Re-read after all pages: old-head successes are never transferable.
        current = _phase_api("final pull request readback", f"repos/{repo}/pulls/{number}")
        if current["head"]["sha"] != sha or current["base"]["ref"] != branch or (current["state"] == "closed" and not current.get("merged")):
            receipt.update(classification="stale", detail="PR head/base changed while collecting evidence; retry.")
            return receipt
        receipt["classification"] = next((x for x in outcomes if x != "success"), "missing" if not outcomes else "success")
        receipt["ok"] = receipt["classification"] == "success"
        return receipt
    except _EvidenceError as exc:
        receipt.update(classification="infra", detail=f"GitHub acceptance phase '{exc.phase}' failed: {exc.reason}.")
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        receipt.update(classification="infra", detail="GitHub acceptance evidence was malformed or incomplete during evaluation.")
        return receipt


def _classify(check: dict, sha: str, outcome: str | None, is_run: bool) -> str:
    if check.get("head_sha", check.get("sha")) != sha:
        return "stale"
    if is_run and check.get("status") != "completed":
        return "pending"
    return {"success": "success", "failure": "failure", "error": "infra", "pending": "pending"}.get(outcome, "infra")
