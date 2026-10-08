"""Board-scoped required-check policy for exact-head PR acceptance."""
from __future__ import annotations

import re
import sqlite3

from hermes_cli.kanban_db_connect import write_txn

_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


def _validated(repo: str, context: str, app_id: int | None) -> tuple[str, str, int]:
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise ValueError("repository must be OWNER/REPO")
    clean = context.strip() if isinstance(context, str) else ""
    if not clean:
        raise ValueError("required check context must not be empty")
    if app_id is not None and (not isinstance(app_id, int) or isinstance(app_id, bool) or app_id < 1):
        raise ValueError("app_id must be a positive integer")
    return repo, clean, -1 if app_id is None else app_id


def required_checks(conn: sqlite3.Connection, repo: str) -> set[tuple[str, int | None]]:
    """Return the board policy for ``repo``; ``None`` denotes an unpinned app."""
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise ValueError("repository must be OWNER/REPO")
    rows = conn.execute(
        "SELECT context, app_id FROM kanban_pr_required_checks WHERE repo=? ORDER BY context, app_id",
        (repo,),
    ).fetchall()
    return {(str(row[0]), None if int(row[1]) == -1 else int(row[1])) for row in rows}


def list_required_checks(conn: sqlite3.Connection, repo: str | None = None) -> list[dict]:
    params: tuple[str, ...] = ()
    where = ""
    if repo is not None:
        if not _REPO.fullmatch(repo):
            raise ValueError("repository must be OWNER/REPO")
        where, params = " WHERE repo=?", (repo,)
    rows = conn.execute(
        "SELECT repo, context, app_id FROM kanban_pr_required_checks" + where + " ORDER BY repo, context, app_id",
        params,
    ).fetchall()
    return [
        {"repo": str(row[0]), "context": str(row[1]), "app_id": None if int(row[2]) == -1 else int(row[2])}
        for row in rows
    ]


def add_required_check(conn: sqlite3.Connection, repo: str, context: str, app_id: int | None = None) -> bool:
    repo, context, stored_app_id = _validated(repo, context, app_id)
    with write_txn(conn):
        cursor = conn.execute(
            "INSERT OR IGNORE INTO kanban_pr_required_checks(repo, context, app_id) VALUES (?, ?, ?)",
            (repo, context, stored_app_id),
        )
    return cursor.rowcount > 0


def remove_required_check(conn: sqlite3.Connection, repo: str, context: str, app_id: int | None = None) -> bool:
    repo, context, stored_app_id = _validated(repo, context, app_id)
    with write_txn(conn):
        cursor = conn.execute(
            "DELETE FROM kanban_pr_required_checks WHERE repo=? AND context=? AND app_id=?",
            (repo, context, stored_app_id),
        )
    return cursor.rowcount > 0