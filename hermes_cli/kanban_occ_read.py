"""Bounded, read-only projection of a Hermes Kanban board for OCC.

This module is the data half of the OCC machine contract
(``/api/plugins/kanban/occ/v1/{summary,tasks,task}``). It exists because the
browser-facing ``/board`` and ``/tasks/{id}`` endpoints are far too wide for a
server-to-server consumer: they open read-write connections, hydrate whole
boards, and ship comment bodies, event payloads, task bodies, PIDs and worker
metadata.

Everything here is deliberately narrow:

* **Read-only by construction.** The connection is opened with SQLite's
  ``mode=ro`` URI, ``query_only``, an explicit read transaction, and an
  authorizer that permits nothing but reads and a small PRAGMA allowlist. The
  board path is resolved from the *configured* board only — never from
  ``HERMES_KANBAN_DB``, the ``current`` pointer or the active-board
  contextvar — so a worker process cannot be confused into reading a different
  board than the one the operator configured.
* **Projection, not hydration.** Only allowlisted columns are selected. Task
  bodies, run summaries, run errors, event payloads, PIDs, workspace and
  branch paths are never read, so they cannot leak by accident.
* **Truthful nulls.** Several fields the visual design wants do not exist in
  the current schema (``updated_at``, ``status_entered_at``, ``scheduled_at``,
  ``reason_preview``, typed references). They are reported as null/empty with
  a documented reason rather than derived from bodies, payloads or guesses.
* **Malformed records survive.** A NUL-bearing status, a non-string status or
  a row whose id is not valid UTF-8 stays visible as ``unknown``/``invalid``.
  Nothing is dropped, coerced or repaired.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

#: Non-archived statuses, in the board's column order. ``archived`` is never
#: requested or counted by the OCC contract.
KNOWN_STATUSES: tuple[str, ...] = (
    "triage",
    "todo",
    "scheduled",
    "ready",
    "running",
    "blocked",
    "review",
    "done",
)

#: Every key ``counts_by_status`` always carries, zero-filled.
COUNT_KEYS: tuple[str, ...] = KNOWN_STATUSES + ("unknown", "invalid")

MAX_LIMIT = 100
DEFAULT_LIMIT = 50
MAX_Q_SCALARS = 80
MAX_CURSOR_BYTES = 512
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_EVENTS = 20
MAX_RUNS = 5
BUSY_TIMEOUT_MS = 2000

#: Orders the provider owns. OCC never sorts client-side, so every design sort
#: is expressed here with a deterministic task-id tiebreaker.
ORDERS: tuple[str, ...] = (
    "oldest_status",
    "heartbeat_attention",
    "priority_age",
    "scheduled",
    "updated_desc",
    "completed_desc",
)

#: An order is only valid for the statuses it was designed for. A request that
#: pairs e.g. ``heartbeat_attention`` with ``status=done`` is a caller bug and
#: gets a 400 rather than a silently different sequence.
_ORDER_STATUS_REQUIREMENTS: dict[str, frozenset[str]] = {
    "heartbeat_attention": frozenset({"running"}),
    "priority_age": frozenset({"ready"}),
    "completed_desc": frozenset({"done"}),
}

_STATUS_CLASSES = frozenset({"known", "unknown", "invalid", "unknown,invalid"})

#: Columns this module requires to exist. Anything else degrades to null.
_REQUIRED_TASK_COLUMNS = frozenset({"id", "title", "status", "created_at"})
_REQUIRED_EVENT_COLUMNS = frozenset({"id", "task_id", "kind", "created_at"})

_READ_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
        # Transaction control is required for the single-snapshot read
        # (BEGIN/COMMIT/ROLLBACK). It carries no write capability on its own:
        # every write statement is still denied below.
        sqlite3.SQLITE_TRANSACTION,
    }
    | ({sqlite3.SQLITE_SAVEPOINT} if hasattr(sqlite3, "SQLITE_SAVEPOINT") else set())
)
#: PRAGMAs permitted on the read connection. ``query_only``/``busy_timeout`` are
#: set by us at connection time; ``table_info`` is a pure schema read used for
#: the required-column check. Nothing here can write or change journal mode.
_ALLOWED_PRAGMAS = frozenset({"query_only", "busy_timeout", "table_info"})


class OccReadError(Exception):
    """Base class for contract errors. ``code`` is the wire error code."""

    code = "internal_error"
    status_code = 500


class InvalidRequest(OccReadError):
    """Caller-supplied input failed validation (400)."""

    code = "invalid_request"
    status_code = 400


class InvalidTaskId(OccReadError):
    code = "invalid_task_id"
    status_code = 400


class InvalidOrder(OccReadError):
    code = "invalid_order"
    status_code = 400


class CursorInvalid(OccReadError):
    code = "cursor_invalid"
    status_code = 400


class CursorStale(OccReadError):
    code = "cursor_stale"
    status_code = 409


class NotFound(OccReadError):
    code = "not_found"
    status_code = 404


class BoardUnavailable(OccReadError):
    code = "board_unavailable"
    status_code = 503


class SchemaUnsupported(OccReadError):
    code = "schema_unsupported"
    status_code = 503


class ResponseTooLarge(OccReadError):
    code = "response_too_large"
    status_code = 503


# ---------------------------------------------------------------------------
# Text and status classification
# ---------------------------------------------------------------------------


def _is_control(char: str) -> bool:
    """True for control, format, surrogate and line/paragraph separators.

    ``Cf`` covers bidi overrides and zero-width joiners; ``Cc`` covers NUL and
    friends; ``Zl``/``Zp`` are line/paragraph separators that would break a
    single-line row. None of them may appear in a "known" status.
    """
    return unicodedata.category(char) in {"Cc", "Cf", "Cs", "Zl", "Zp"}


def classify_status(value: object) -> str:
    """Classify a raw status cell as ``known``, ``unknown`` or ``invalid``.

    ``known`` is an exact match against :data:`KNOWN_STATUSES` — no trimming,
    no case folding, no normalisation. Anything else on a decodable string is
    ``unknown`` (a clean but unsupported status); anything that is not a
    non-empty string, or that carries control characters, is ``invalid``.
    Malformed values are *classified*, never dropped.
    """
    if not isinstance(value, str) or not value:
        return "invalid"
    if any(_is_control(char) for char in value):
        return "invalid"
    return "known" if value in KNOWN_STATUSES else "unknown"


def occ_fold(value: Optional[str]) -> Optional[str]:
    """Deterministic search folding: NFC then ``casefold``.

    Registered on every connection ``sqlite3`` so SQL and Python agree. SQLite's
    own ``lower()`` is ASCII-only and would make ``q`` matching locale- and
    build-dependent.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return unicodedata.normalize("NFC", value).casefold()


def normalize_q(raw: Optional[str]) -> Optional[str]:
    """Validate and normalise the ``q`` filter.

    Trimmed, NFC-normalised, at most :data:`MAX_Q_SCALARS` Unicode scalar
    values, and free of control characters. SQL wildcards are *not* special
    here: matching is a literal substring test, so ``%`` and ``_`` match
    themselves.
    """
    if raw is None:
        return None
    text = unicodedata.normalize("NFC", raw.strip())
    if not text:
        return None
    if any(_is_control(char) for char in text):
        raise InvalidRequest("q contains control characters")
    if len(text) > MAX_Q_SCALARS:
        raise InvalidRequest(f"q exceeds {MAX_Q_SCALARS} characters")
    return text


# ---------------------------------------------------------------------------
# Board resolution and read-only connection
# ---------------------------------------------------------------------------


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_board_db_path(board: str, *, kanban_home: Path) -> Path:
    """Return the ``kanban.db`` path for an *explicitly configured* board.

    ``default`` keeps the legacy ``<home>/kanban.db`` location; named boards
    live at ``<home>/kanban/boards/<slug>/kanban.db``. Deliberately independent
    of ``kanban_db.kanban_db_path`` so the ``HERMES_KANBAN_DB`` override, the
    ``current`` pointer file and the active-board contextvar cannot redirect a
    machine read.

    Containment is enforced on resolved paths so a symlinked board directory
    cannot walk the read outside the kanban home.
    """
    root = Path(kanban_home).resolve()
    if board == "default":
        candidate = root / "kanban.db"
    else:
        candidate = root / "kanban" / "boards" / board / "kanban.db"
    try:
        resolved = candidate.resolve()
    except OSError as exc:  # pragma: no cover - defensive
        raise BoardUnavailable("board path unresolvable") from exc
    if not _is_within(resolved, root):
        raise BoardUnavailable("board path escapes kanban home")
    return resolved


def _authorizer(action: int, arg1: Optional[str], arg2: Optional[str],
                _db: Optional[str], _trigger: Optional[str]) -> int:
    """SQLite authorizer: reads and a tiny PRAGMA allowlist, nothing else.

    This is the belt to ``query_only``'s braces. A future refactor that
    accidentally issues an UPDATE, an ATTACH or a ``VACUUM`` fails loudly here
    instead of mutating the operator's board.
    """
    if action in _READ_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA:
        if arg1 and arg1.lower() in _ALLOWED_PRAGMAS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def open_readonly_connection(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` for a bounded read-only session.

    ``mode=ro`` (never ``immutable=1``: the dispatcher writes concurrently and
    an immutable handle would return stale pages). ``isolation_level=None``
    plus an explicit ``BEGIN`` gives one snapshot for the whole request, which
    is what makes the count/page/cursor consistency guarantees real.
    """
    if not db_path.is_file():
        raise BoardUnavailable("board database missing")
    uri = db_path.as_uri() + "?mode=ro"
    try:
        conn = sqlite3.connect(
            uri,
            uri=True,
            isolation_level=None,
            timeout=BUSY_TIMEOUT_MS / 1000,
        )
    except sqlite3.Error as exc:
        raise BoardUnavailable("board database unreadable") from exc
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.create_function("occ_fold", 1, occ_fold, deterministic=True)
        conn.set_authorizer(_authorizer)
    except sqlite3.Error as exc:
        conn.close()
        raise BoardUnavailable("board database unreadable") from exc
    return conn


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _require_schema(conn: sqlite3.Connection) -> None:
    """Fail closed on a schema that cannot satisfy the contract.

    Missing *required* columns mean the DB is not a kanban board this build
    understands; serving partial garbage would be worse than 503.
    """
    if not _REQUIRED_TASK_COLUMNS <= _table_columns(conn, "tasks"):
        raise SchemaUnsupported("tasks schema unsupported")
    if not _REQUIRED_EVENT_COLUMNS <= _table_columns(conn, "task_events"):
        raise SchemaUnsupported("task_events schema unsupported")


# ---------------------------------------------------------------------------
# Filters and predicates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Filters:
    """The AND-combined filters shared by summary and list.

    One instance feeds one predicate builder, so summary counts and list rows
    cannot drift: the packet requires that a filtered summary reconcile with
    the filtered rows.
    """

    q: Optional[str] = None
    assignee: Optional[str] = None
    tenant: Optional[str] = None
    exclude_tenant: Optional[str] = None
    status: Optional[str] = None
    status_class: Optional[str] = None

    def canonical(self) -> str:
        """Stable serialisation used for cursor binding."""
        return json.dumps(
            {
                "q": self.q,
                "assignee": self.assignee,
                "tenant": self.tenant,
                "exclude_tenant": self.exclude_tenant,
                "status": self.status,
                "status_class": self.status_class,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def echo(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "assignee": self.assignee,
            "tenant": self.tenant,
            "exclude_tenant": self.exclude_tenant,
            "status": self.status,
            "status_class": self.status_class,
        }


def validate_status_filter(status: Optional[str]) -> Optional[str]:
    """A ``status`` filter is one of the eight non-archived statuses."""
    if status is None:
        return None
    if status not in KNOWN_STATUSES:
        raise InvalidRequest("status must be a non-archived known status")
    return status


def validate_status_class_filter(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value not in _STATUS_CLASSES:
        raise InvalidRequest("status_class must be known|unknown|invalid|unknown,invalid")
    return value


def build_predicates(filters: Filters) -> tuple[str, list[Any]]:
    """Build the single WHERE clause used by summary and list.

    Status matching never round-trips malformed data through SQL text
    comparison: the column is read as ``typeof()`` + ``CAST(... AS BLOB)`` so a
    BLOB status, an integer status or a NUL-bearing string is compared on its
    exact storage class. Archive exclusion is null-safe (``IS NOT`` rather than
    ``!=``) so a NULL status survives as an invalid row instead of vanishing.
    """
    clauses: list[str] = ["t.status IS NOT 'archived'"]
    params: list[Any] = []

    if filters.q:
        clauses.append(
            "(instr(occ_fold(t.title), ?) > 0 OR instr(occ_fold(t.id), ?) > 0)"
        )
        params.extend([filters.q, filters.q])
    if filters.assignee is not None:
        clauses.append("t.assignee = ?")
        params.append(filters.assignee)
    if filters.tenant is not None:
        clauses.append("t.tenant = ?")
        params.append(filters.tenant)
    if filters.exclude_tenant is not None:
        clauses.append("(t.tenant IS NULL OR t.tenant IS NOT ?)")
        params.append(filters.exclude_tenant)
    if filters.status is not None:
        clauses.append("(typeof(t.status) = 'text' AND CAST(t.status AS BLOB) = ?)")
        params.append(filters.status.encode("utf-8"))
    if filters.status_class is not None:
        # The class set is decided in Python (control characters and the
        # eight-status allowlist are not cheap SQL tests), so the SQL side only
        # narrows to "text" and lets the row filter finish the job. Excluding
        # non-text storage here would silently drop the invalid records this
        # contract exists to preserve, so the SQL predicate is permissive and
        # the Python mirror is authoritative.
        clauses.append("(typeof(t.status) = 'text' OR t.status IS NULL)")
        # no params: the exact membership test happens in Python

    return " AND ".join(clauses), params


def _row_matches_class_filter(status_class: str, allowed: Optional[str]) -> bool:
    if allowed is None:
        return True
    if allowed == "unknown,invalid":
        return status_class in {"unknown", "invalid"}
    return status_class == allowed


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

_PROCESS_CURSOR_KEY: Optional[bytes] = None
_KEY_LOCK = threading.Lock()


def _cursor_key() -> bytes:
    """Per-process HMAC key.

    Deliberately not persisted: a cursor from a previous process epoch must be
    rejected (409) rather than silently interpreted against a different
    snapshot. Rotating per process removes any need to store a secret.
    """
    global _PROCESS_CURSOR_KEY
    if _PROCESS_CURSOR_KEY is None:
        with _KEY_LOCK:
            if _PROCESS_CURSOR_KEY is None:
                _PROCESS_CURSOR_KEY = secrets.token_bytes(32)
    return _PROCESS_CURSOR_KEY


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def encode_cursor(payload: dict[str, Any]) -> str:
    """Sign and encode a cursor. Bound to the process key epoch."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    raw = body.encode("utf-8")
    mac = hmac.new(_cursor_key(), raw, hashlib.sha256).digest()
    token = _b64encode(raw) + "." + _b64encode(mac)
    if len(token.encode("ascii")) > MAX_CURSOR_BYTES:
        raise CursorInvalid("cursor too large")
    return token


def decode_cursor(token: str) -> dict[str, Any]:
    """Verify and decode a cursor.

    A tampered or malformed cursor is a caller error (400 ``cursor_invalid``);
    a *valid* cursor whose snapshot no longer matches is a state conflict
    (409 ``cursor_stale``), decided by the caller.
    """
    if not isinstance(token, str) or not token or len(token) > MAX_CURSOR_BYTES:
        raise CursorInvalid("cursor malformed")
    try:
        body_part, mac_part = token.split(".", 1)
        raw = _b64decode(body_part)
        mac = _b64decode(mac_part)
    except (ValueError, TypeError) as exc:
        raise CursorInvalid("cursor malformed") from exc
    expected = hmac.new(_cursor_key(), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        raise CursorInvalid("cursor signature mismatch")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CursorInvalid("cursor payload invalid") from exc
    if not isinstance(payload, dict):
        raise CursorInvalid("cursor payload invalid")
    return payload


# ---------------------------------------------------------------------------
# Row projection
# ---------------------------------------------------------------------------


def _decode_text(value: object) -> Optional[str]:
    """Decode a BLOB-or-text cell as strict UTF-8; ``None`` when undecodable.

    Selecting ``CAST(col AS BLOB)`` keeps SQLite from handing back a decoded
    ``str`` (which would substitute U+FFFD for bad bytes) or an ``int`` coerced
    to text. One bad row must not fail the whole page, so the failure is local.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        return value
    return None


#: "Latest recorded activity". There is no ``tasks.updated_at`` column and no
#: complete status-transition event log, so this is the newest timestamp the
#: schema actually keeps. Timestamps only — payloads are never read.
_UPDATED_AT_SQL = (
    "MAX(COALESCE(t.created_at, 0), COALESCE(t.started_at, 0), "
    "COALESCE(t.completed_at, 0), COALESCE(ev.last_event_at, 0))"
)

_ORDER_SQL: dict[str, str] = {
    "oldest_status": f"{_UPDATED_AT_SQL} ASC, k.id_blob ASC, t.rowid ASC",
    "heartbeat_attention": (
        "(t.last_heartbeat_at IS NULL) DESC, t.last_heartbeat_at ASC, "
        "k.id_blob ASC, t.rowid ASC"
    ),
    "priority_age": (
        "COALESCE(t.priority, 0) DESC, "
        f"{_UPDATED_AT_SQL} ASC, k.id_blob ASC, t.rowid ASC"
    ),
    "scheduled": f"{_UPDATED_AT_SQL} ASC, k.id_blob ASC, t.rowid ASC",
    "updated_desc": f"{_UPDATED_AT_SQL} DESC, k.id_blob ASC, t.rowid ASC",
    "completed_desc": (
        "(t.completed_at IS NULL) ASC, t.completed_at DESC, "
        "k.id_blob ASC, t.rowid ASC"
    ),
}


def _snapshot_key_expression(order: str) -> str:
    """The non-id sort columns, for the snapshot digest.

    The digest must cover the values that determine order *and* membership, so
    a heartbeat touch or a direct SQL status update invalidates the cursor even
    when no event row was written.
    """
    if order == "heartbeat_attention":
        return "COALESCE(t.last_heartbeat_at, -1)"
    if order == "priority_age":
        return f"COALESCE(t.priority, 0) || ':' || {_UPDATED_AT_SQL}"
    if order == "completed_desc":
        return "COALESCE(t.completed_at, -1)"
    return _UPDATED_AT_SQL


def validate_order(order: str, *, status: Optional[str],
                   status_class: Optional[str]) -> str:
    """Reject order/status pairs the design never asked for.

    The UI always pairs a status with its designed sort. Accepting a mismatched
    pair would return a defensible-looking but wrong sequence, so it is a 400.
    """
    if order not in ORDERS:
        raise InvalidOrder("unknown order")
    required = _ORDER_STATUS_REQUIREMENTS.get(order)
    if required is not None:
        if status is None or status not in required:
            raise InvalidOrder(f"order {order} requires status={sorted(required)[0]}")
    if order == "heartbeat_attention" and status_class is not None:
        raise InvalidOrder("order heartbeat_attention does not accept status_class")
    return order


_BASE_CTE = """
WITH ev AS (
    SELECT task_id, MAX(created_at) AS last_event_at
    FROM task_events
    GROUP BY task_id
),
k AS (
    SELECT id, typeof(id) AS id_type, CAST(id AS BLOB) AS id_blob,
           title, typeof(title) AS title_type, CAST(title AS BLOB) AS title_blob,
           status, typeof(status) AS status_type, CAST(status AS BLOB) AS status_blob,
           assignee, typeof(assignee) AS assignee_type,
           CAST(assignee AS BLOB) AS assignee_blob,
           tenant, typeof(tenant) AS tenant_type, CAST(tenant AS BLOB) AS tenant_blob
    FROM tasks
)
"""

_SELECT_ROW = """
SELECT
    k.id_type, k.id_blob,
    k.title_type, k.title_blob,
    k.status_type, k.status_blob,
    k.assignee_type, k.assignee_blob,
    k.tenant_type, k.tenant_blob,
    t.rowid AS rowid,
    t.priority AS priority,
    t.created_at AS created_at,
    t.started_at AS started_at,
    t.completed_at AS completed_at,
    t.last_heartbeat_at AS last_heartbeat_at,
    t.claim_expires AS claim_expires,
    t.current_run_id AS current_run_id,
    {updated_at} AS updated_at,
    (SELECT r.status FROM task_runs r WHERE r.id = t.current_run_id) AS current_run_status,
    (SELECT MIN(l.parent_id) FROM task_links l WHERE l.child_id = t.id) AS parent_id_blob,
    (SELECT COUNT(*) FROM task_links l WHERE l.parent_id = t.id) AS child_count,
    {snapshot_key} AS snapshot_key
FROM tasks t
JOIN k ON k.id IS t.id
LEFT JOIN ev ON ev.task_id = t.id
WHERE {where}
"""


def _project_row(row: sqlite3.Row, *, now: int, claim_stale_secs: int) -> dict[str, Any]:
    """Project one joined row into the wire DTO.

    Only allowlisted fields. No body, result, error, payload, PID, workspace,
    branch, session or model column is read at all.
    """
    id_text = _decode_text(row["id_blob"]) if row["id_type"] == "text" else None
    title_text = _decode_text(row["title_blob"]) if row["title_type"] == "text" else None
    assignee_text = (
        _decode_text(row["assignee_blob"]) if row["assignee_type"] == "text" else None
    )
    tenant_text = (
        _decode_text(row["tenant_blob"]) if row["tenant_type"] == "text" else None
    )

    if row["status_type"] == "text":
        decoded_status = _decode_text(row["status_blob"])
        status_class = classify_status(decoded_status)
        wire_status = decoded_status
    elif row["status_type"] == "blob":
        decoded_status = _decode_text(row["status_blob"])
        status_class = "invalid"
        wire_status = decoded_status
    else:
        # integer/real/NULL storage classes: no text to show, so the wire says
        # null and OCC renders "Invalid status".
        decoded_status = None
        status_class = "invalid"
        wire_status = None

    heartbeat = row["last_heartbeat_at"]
    running: Optional[dict[str, Any]] = None
    if status_class == "known" and decoded_status == "running":
        claim_expires = row["claim_expires"]
        run_id = row["current_run_id"]
        stale = (
            run_id is not None
            and row["current_run_status"] == "running"
            and claim_expires is not None
            and claim_expires < now
            and heartbeat is not None
            and (now - heartbeat) > claim_stale_secs
        )
        running = {
            "run_id": run_id if isinstance(run_id, int) else None,
            "claimed_at": row["started_at"],
            "last_heartbeat_at": heartbeat,
            "claim_stale": bool(stale),
        }

    priority = row["priority"]

    return {
        "id": id_text,
        "title": title_text,
        "status": wire_status,
        "status_class": status_class,
        "assignee": assignee_text,
        "tenant": tenant_text,
        "priority": priority if isinstance(priority, int) else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status_entered_at": None,
        "scheduled_at": None,
        "completed_at": row["completed_at"],
        "parent_id": _decode_text(row["parent_id_blob"]),
        "child_count": row["child_count"],
        "links": [],
        "reason_preview": None,
        "running": running,
    }


def _snapshot_digest(rows: list[dict[str, Any]]) -> str:
    """SHA-256 over the ordered (rowid, id, status, sort-key) sequence."""
    digest = hashlib.sha256()
    for entry in rows:
        digest.update(
            repr(
                (
                    entry["rowid"],
                    entry["id_key"],
                    entry["status_key"],
                    entry["snapshot_key"],
                )
            ).encode("utf-8", errors="surrogatepass")
        )
        digest.update(b"\x1e")
    return digest.hexdigest()


def _load_snapshot(conn: sqlite3.Connection, *, where: str, params: list[Any],
                   order: str) -> list[dict[str, Any]]:
    """Load the full filtered+ordered key sequence used for ordering and digest.

    This is intentionally a *key* query, not a row hydration: it selects the
    task id blob, the raw status, and the sort key only. At 2,000 tasks that is
    a few hundred kilobytes of transient Python objects and keeps offset paging
    exact, which is what the digest-based cursor contract requires.
    """
    sql = (
        _BASE_CTE
        + "SELECT t.rowid AS rowid, k.id_type, k.id_blob, k.status_type, k.status_blob, "
        + f"{_snapshot_key_expression(order)} AS snapshot_key "
        + "FROM tasks t JOIN k ON k.id IS t.id LEFT JOIN ev ON ev.task_id = t.id "
        + f"WHERE {where} ORDER BY {_ORDER_SQL[order]}"
    )
    out: list[dict[str, Any]] = []
    for row in conn.execute(sql, params):
        id_text = _decode_text(row["id_blob"]) if row["id_type"] == "text" else None
        if row["status_type"] == "text":
            status_text = _decode_text(row["status_blob"])
        elif row["status_type"] == "blob":
            status_text = _decode_text(row["status_blob"])
        else:
            status_text = None
        out.append(
            {
                "rowid": row["rowid"],
                "id_key": id_text,
                "id_raw": row["id_blob"],
                "status_class": classify_status(status_text),
                "status_key": status_text,
                "snapshot_key": row["snapshot_key"],
            }
        )
    return out


def _rowids_for_page(entries: list[dict[str, Any]], *, offset: int,
                     limit: int) -> list[int]:
    return [e["rowid"] for e in entries[offset: offset + limit]]


def _fetch_rows_by_rowid(conn: sqlite3.Connection, rowids: list[int],
                         order: str) -> list[sqlite3.Row]:
    if not rowids:
        return []
    placeholders = ",".join("?" for _ in rowids)
    sql = (
        _BASE_CTE
        + _SELECT_ROW.format(
            updated_at=_UPDATED_AT_SQL,
            snapshot_key=_snapshot_key_expression(order),
            where=f"t.rowid IN ({placeholders})",
        )
        + f" ORDER BY {_ORDER_SQL[order]}"
    )
    return list(conn.execute(sql, rowids))


# ---------------------------------------------------------------------------
# Public read operations
# ---------------------------------------------------------------------------


def _begin(conn: sqlite3.Connection) -> int:
    """Open one read transaction and return the single ``now`` for the request.

    ``now`` is captured inside the transaction so every relative age OCC
    renders is measured against the same instant the snapshot was read at.
    """
    conn.execute("BEGIN")
    import time

    return int(time.time())


def read_summary(conn: sqlite3.Connection, *, board: str, filters: Filters,
                 claim_stale_secs: int, now: int) -> dict[str, Any]:
    """Counts, filter facets and board identity for the status strip."""
    where, params = build_predicates(filters)

    latest_event_id = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS m FROM task_events"
    ).fetchone()["m"]

    counts = {key: 0 for key in COUNT_KEYS}
    invalid_records = 0
    total = 0
    entries = _load_snapshot(conn, where=where, params=params, order="updated_desc")
    for entry in entries:
        if not _row_matches_class_filter(entry["status_class"], filters.status_class):
            continue
        total += 1
        if entry["status_class"] == "known":
            counts[entry["status_key"]] += 1
        else:
            counts[entry["status_class"]] += 1
        if entry["id_raw"] is None or entry["id_key"] is None:
            invalid_records += 1

    assignees = [
        _decode_text(row["assignee_blob"])
        for row in conn.execute(
            _BASE_CTE
            + "SELECT DISTINCT k.assignee_blob AS assignee_blob FROM tasks t "
            "JOIN k ON k.id IS t.id "
            "WHERE t.status IS NOT 'archived' AND k.assignee_type = 'text' "
            "ORDER BY k.assignee_blob"
        )
    ]
    tenants = [
        _decode_text(row["tenant_blob"])
        for row in conn.execute(
            _BASE_CTE
            + "SELECT DISTINCT k.tenant_blob AS tenant_blob FROM tasks t "
            "JOIN k ON k.id IS t.id "
            "WHERE t.status IS NOT 'archived' AND k.tenant_type = 'text' "
            "ORDER BY k.tenant_blob"
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "board": board,
        "now": now,
        "latest_event_id": latest_event_id,
        "total_filtered": total,
        "invalid_records": invalid_records,
        "counts_by_status": counts,
        "assignees": [a for a in assignees if a],
        "tenants": [t for t in tenants if t],
        "filters_echo": filters.echo(),
    }


def read_tasks(conn: sqlite3.Connection, *, board: str, filters: Filters,
               order: str, limit: int, cursor: Optional[str],
               claim_stale_secs: int, now: int) -> dict[str, Any]:
    """One page of the filtered, ordered task list.

    Paging is offset-based *behind* a snapshot-digest cursor: the digest of the
    full ordered key sequence is verified before the offset is applied, so a
    board that changed between pages yields 409 instead of silently skipping or
    repeating a row.
    """
    where, params = build_predicates(filters)
    entries = _load_snapshot(conn, where=where, params=params, order=order)
    if filters.status_class is not None:
        entries = [
            e for e in entries
            if _row_matches_class_filter(e["status_class"], filters.status_class)
        ]
    digest = _snapshot_digest(entries)
    latest_event_id = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS m FROM task_events"
    ).fetchone()["m"]

    offset = 0
    if cursor:
        payload = decode_cursor(cursor)
        expected = {
            "b": board,
            "f": hashlib.sha256(filters.canonical().encode("utf-8")).hexdigest(),
            "o": order,
            "l": limit,
            "d": digest,
            "e": latest_event_id,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise CursorStale("board changed since cursor was issued")
        offset = payload.get("x")
        if not isinstance(offset, int) or offset < 0:
            raise CursorInvalid("cursor offset invalid")

    page_entries = entries[offset: offset + limit]
    rowids = [e["rowid"] for e in page_entries]
    rows = _fetch_rows_by_rowid(conn, rowids, order)
    tasks = [_project_row(row, now=now, claim_stale_secs=claim_stale_secs) for row in rows]

    invalid_records = sum(1 for e in entries if e["id_raw"] is None or e["id_key"] is None)

    next_cursor = None
    if offset + limit < len(entries):
        next_cursor = encode_cursor(
            {
                "b": board,
                "f": hashlib.sha256(filters.canonical().encode("utf-8")).hexdigest(),
                "o": order,
                "l": limit,
                "d": digest,
                "e": latest_event_id,
                "x": offset + limit,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "board": board,
        "now": now,
        "latest_event_id": latest_event_id,
        "order": order,
        "limit": limit,
        "offset": offset,
        "total": len(entries),
        "invalid_records": invalid_records,
        "next_cursor": next_cursor,
        "tasks": tasks,
        "filters_echo": filters.echo(),
    }


def read_task(conn: sqlite3.Connection, *, board: str, task_id: str,
              claim_stale_secs: int, now: int) -> dict[str, Any]:
    """Detail for exactly one task, with bounded event and run summaries.

    Event *kinds and timestamps* only — payloads are never selected. Run rows
    carry status/timing/outcome only: no summary text, no error text, no PID,
    no claim lock, no metadata blob.
    """
    rowid = conn.execute(
        "SELECT rowid FROM tasks WHERE typeof(id) = 'text' AND CAST(id AS BLOB) = ?",
        (task_id.encode("utf-8"),),
    ).fetchone()
    if rowid is None:
        raise NotFound("task not found")
    task_rowid = rowid["rowid"]
    rows = _fetch_rows_by_rowid(conn, [task_rowid], "updated_desc")
    if not rows:
        raise NotFound("task not found")
    task = _project_row(rows[0], now=now, claim_stale_secs=claim_stale_secs)

    def _count(sql: str) -> int:
        return int(conn.execute(sql, (task_id,)).fetchone()[0])

    counts = {
        "comments": _count("SELECT COUNT(*) FROM task_comments WHERE task_id = ?"),
        "events": _count("SELECT COUNT(*) FROM task_events WHERE task_id = ?"),
        "attachments": _count("SELECT COUNT(*) FROM task_attachments WHERE task_id = ?"),
        "children": int(
            conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE parent_id = ?", (task_id,)
            ).fetchone()[0]
        ),
        "parents": int(
            conn.execute(
                "SELECT COUNT(*) FROM task_links WHERE child_id = ?", (task_id,)
            ).fetchone()[0]
        ),
    }

    events = [
        {"kind": _decode_text(r["kind"]), "created_at": r["created_at"]}
        for r in conn.execute(
            "SELECT typeof(kind) AS kind_type, CAST(kind AS BLOB) AS kind_blob, "
            "created_at FROM task_events WHERE task_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (task_id, MAX_EVENTS),
        )
    ]

    runs = [
        {
            "id": r["id"],
            "status": _decode_text(r["status_blob"]),
            "started_at": r["started_at"],
            "ended_at": r["ended_at"],
            "last_heartbeat_at": r["last_heartbeat_at"],
            "outcome": _decode_text(r["outcome_blob"]),
        }
        for r in conn.execute(
            "SELECT id, typeof(status) AS status_type, CAST(status AS BLOB) AS status_blob, "
            "started_at, ended_at, last_heartbeat_at, "
            "typeof(outcome) AS outcome_type, CAST(outcome AS BLOB) AS outcome_blob "
            "FROM task_runs WHERE task_id = ? ORDER BY id DESC LIMIT ?",
            (task_id, MAX_RUNS),
        )
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "board": board,
        "now": now,
        "task": task,
        "counts": counts,
        "events": events,
        "runs": runs,
    }


def enforce_response_size(payload: Any) -> bytes:
    """Serialise once and refuse anything over the 1 MB contract cap."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_RESPONSE_BYTES:
        raise ResponseTooLarge("response exceeds 1 MiB")
    return body


def open_board(board: str, *, kanban_home: Path) -> sqlite3.Connection:
    """Resolve the configured board and open its read-only connection."""
    conn = open_readonly_connection(resolve_board_db_path(board, kanban_home=kanban_home))
    try:
        _require_schema(conn)
    except Exception:
        conn.close()
        raise
    return conn
