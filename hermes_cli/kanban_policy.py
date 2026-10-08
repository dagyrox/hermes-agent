"""``hermes kanban policy ...`` — board-scoped PR acceptance policy."""
from __future__ import annotations

import argparse
import json

from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_pr_policy as policy
from hermes_cli.kanban_output import _err


def _dispatch_policy(args: argparse.Namespace) -> int:
    action = getattr(args, "policy_action", None) or "list-required-checks"
    if action in {"list-required-checks", "list"}:
        with kbc.connect_closing() as conn:
            rows = policy.list_required_checks(conn, getattr(args, "repo", None))
        if getattr(args, "json", False):
            print(json.dumps(rows, indent=2))
        elif not rows:
            print("(no board-scoped required checks configured)")
        else:
            for row in rows:
                suffix = f" (app_id={row['app_id']})" if row["app_id"] is not None else ""
                print(f"{row['repo']}: {row['context']}{suffix}")
        return 0
    with kbc.connect_closing() as conn:
        if action in {"add-required-check", "add"}:
            created = policy.add_required_check(conn, args.repo, args.context, args.app_id)
            verb = "Added" if created else "Already configured"
            print(f"{verb}: {args.repo} requires {args.context!r} on this board.")
            return 0
        if action in {"remove-required-check", "remove", "rm"}:
            removed = policy.remove_required_check(conn, args.repo, args.context, args.app_id)
            if not removed:
                return _err("kanban policy: matching required check was not configured")
            print(f"Removed: {args.repo} no longer requires {args.context!r} from board policy.")
            return 0
    return _err(f"kanban policy: unknown action {action!r}", 2)