"""Durable state machine for gateway-owned cross-runtime failover turns.

This module deliberately contains no platform or runtime wiring.  It is the
transactional core that a gateway turn coordinator can use without mutating a
conversation's frozen prompt/tool/media snapshot between route legs.

Replay is fail closed: only an explicitly classified availability failure that
occurred before tool activity, output, approval, timeout, cancellation, or
refusal may advance to the next leg.  A process restart while an attempt is in
``started`` state requires manual reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Mapping, Sequence


class CoordinatorError(RuntimeError):
    """Base error for a rejected coordinator transition."""


class ReplayDenied(CoordinatorError):
    """The ledger cannot prove that replay is safe."""


class OwnershipLost(CoordinatorError):
    """The caller no longer owns this exact gateway run generation."""


_REPLAYABLE_AVAILABILITY_REASONS = frozenset(
    {
        "auth",
        "auth_permanent",
        "billing",
        "model_not_found",
        "overloaded",
        "rate_limit",
        "server_error",
        "upstream_rate_limit",
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenTurn:
    session_key: str
    session_id: str
    event_id: str
    generation: int
    prompt: Any
    tools: tuple[Any, ...]
    media: tuple[Any, ...]
    legs: tuple[Any, ...]
    prompt_digest: str
    tool_digest: str
    media_digest: str
    chain_digest: str

    @classmethod
    def create(
        cls,
        *,
        session_key: str,
        session_id: str,
        event_id: str,
        generation: int,
        prompt: Any,
        tools: Sequence[Any],
        media: Sequence[Any],
        legs: Sequence[Any],
    ) -> "FrozenTurn":
        if not session_key or not session_id or not event_id:
            raise ValueError("session, conversation, and event identities are required")
        if generation < 1:
            raise ValueError("generation must be positive")
        # Canonical JSON round-tripping both validates the snapshot's durable
        # representation and detaches every nested container from caller-owned
        # mutable input. A frozen dataclass with shallow tuples is insufficient:
        # mutating an inner dict after create() would otherwise make the stored
        # bytes disagree with the digests calculated here.
        frozen_prompt = json.loads(_canonical_json(prompt))
        frozen_tools = tuple(json.loads(_canonical_json(tools)))
        frozen_media = tuple(json.loads(_canonical_json(media)))
        frozen_legs = tuple(json.loads(_canonical_json(legs)))
        if not frozen_legs:
            raise ValueError("at least one route leg is required")
        return cls(
            session_key=session_key,
            session_id=session_id,
            event_id=event_id,
            generation=int(generation),
            prompt=frozen_prompt,
            tools=frozen_tools,
            media=frozen_media,
            legs=frozen_legs,
            prompt_digest=_digest(frozen_prompt),
            tool_digest=_digest(frozen_tools),
            media_digest=_digest(frozen_media),
            chain_digest=_digest(frozen_legs),
        )

    @property
    def coordinator_id(self) -> str:
        # One platform event may own only one coordinator. Generation remains
        # persisted and fenced, but excluding it here prevents a stale and a
        # newer generation for the same event from running side by side.
        identity = [self.session_key, self.session_id, self.event_id]
        return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class AttemptObservation:
    outcome: str
    reason: str
    response: str | None = None
    is_availability_failure: bool = False
    is_refusal: bool = False
    partial_output: bool = False
    tool_activity: bool = False
    pending_approval: bool = False
    timed_out: bool = False
    cancelled: bool = False

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "failure", "refusal"}:
            raise ValueError(f"unsupported attempt outcome {self.outcome!r}")
        if (
            self.is_availability_failure
            and self.reason not in _REPLAYABLE_AVAILABILITY_REASONS
        ):
            raise ValueError(
                f"{self.reason!r} is not a replayable availability failure"
            )
        adverse = (
            self.is_availability_failure
            or self.is_refusal
            or self.partial_output
            or self.tool_activity
            or self.pending_approval
            or self.timed_out
            or self.cancelled
        )
        if self.outcome == "success" and (not self.response or adverse):
            raise ValueError("successful attempts require an unambiguous final response")
        if self.outcome != "success" and self.response is not None:
            raise ValueError("failed attempts cannot carry a final response")
        if self.outcome == "refusal" and not self.is_refusal:
            raise ValueError("refusal outcome requires refusal evidence")
        if self.is_refusal and self.outcome != "refusal":
            raise ValueError("refusal evidence requires a refusal outcome")
        if self.is_availability_failure and self.outcome != "failure":
            raise ValueError("availability evidence requires a failure outcome")

    @classmethod
    def success(cls, response: str) -> "AttemptObservation":
        if not response:
            raise ValueError("successful attempts require a final response")
        return cls(outcome="success", reason="success", response=response)

    @classmethod
    def availability_failure(cls, reason: str) -> "AttemptObservation":
        if reason not in _REPLAYABLE_AVAILABILITY_REASONS:
            raise ValueError(f"{reason!r} is not a replayable availability failure")
        return cls(outcome="failure", reason=reason, is_availability_failure=True)

    @classmethod
    def refusal(cls, reason: str) -> "AttemptObservation":
        return cls(outcome="refusal", reason=reason, is_refusal=True)

    @classmethod
    def failure(cls, reason: str, **evidence: bool) -> "AttemptObservation":
        allowed = {
            "partial_output",
            "tool_activity",
            "pending_approval",
            "timed_out",
            "cancelled",
        }
        unknown = set(evidence) - allowed
        if unknown:
            raise TypeError(f"unsupported attempt evidence: {', '.join(sorted(unknown))}")
        return cls(outcome="failure", reason=reason, **evidence)

    @property
    def replayable(self) -> bool:
        blocked = (
            self.is_refusal
            or self.partial_output
            or self.tool_activity
            or self.pending_approval
            or self.timed_out
            or self.cancelled
        )
        return (
            self.outcome == "failure"
            and self.is_availability_failure
            and not blocked
        )


@dataclass(frozen=True)
class CoordinatorRecord:
    coordinator_id: str
    state: str
    reason: str | None
    owner: str
    generation: int
    prompt_digest: str
    tool_digest: str
    media_digest: str
    chain_digest: str
    current_leg: int
    delivery_state: str


@dataclass(frozen=True)
class AttemptOutcome:
    state: str
    reason: str
    replayable: bool


@dataclass(frozen=True)
class FrozenLegDispatch:
    """Exact canonical bytes authorized for one persisted route attempt."""

    prompt_json: str
    tools_json: str
    media_json: str
    leg_json: str
    prompt_digest: str
    tool_digest: str
    media_digest: str
    chain_digest: str


class GatewayFailoverLedger:
    """SQLite-backed coordinator with transactionally fenced transitions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS gateway_failover_turns (
                    coordinator_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    owner TEXT NOT NULL,
                    state TEXT NOT NULL,
                    reason TEXT,
                    prompt_json TEXT NOT NULL,
                    tools_json TEXT NOT NULL,
                    media_json TEXT NOT NULL,
                    legs_json TEXT NOT NULL,
                    prompt_digest TEXT NOT NULL,
                    tool_digest TEXT NOT NULL,
                    media_digest TEXT NOT NULL,
                    chain_digest TEXT NOT NULL,
                    current_leg INTEGER NOT NULL DEFAULT 0,
                    final_response TEXT,
                    delivery_state TEXT NOT NULL DEFAULT 'none',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(session_key, session_id, event_id, generation)
                );
                CREATE TABLE IF NOT EXISTS gateway_failover_attempts (
                    coordinator_id TEXT NOT NULL,
                    leg_index INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    outcome TEXT,
                    reason TEXT,
                    replayable INTEGER NOT NULL DEFAULT 0,
                    partial_output INTEGER NOT NULL DEFAULT 0,
                    tool_activity INTEGER NOT NULL DEFAULT 0,
                    pending_approval INTEGER NOT NULL DEFAULT 0,
                    timed_out INTEGER NOT NULL DEFAULT 0,
                    cancelled INTEGER NOT NULL DEFAULT 0,
                    started_at REAL NOT NULL,
                    finished_at REAL,
                    PRIMARY KEY(coordinator_id, leg_index),
                    FOREIGN KEY(coordinator_id) REFERENCES gateway_failover_turns(coordinator_id)
                );
                CREATE TABLE IF NOT EXISTS gateway_failover_tool_calls (
                    coordinator_id TEXT NOT NULL,
                    leg_index INTEGER NOT NULL,
                    call_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    result_digest TEXT,
                    is_error INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(coordinator_id, leg_index, call_id),
                    FOREIGN KEY(coordinator_id, leg_index)
                        REFERENCES gateway_failover_attempts(coordinator_id, leg_index)
                );
                CREATE TABLE IF NOT EXISTS gateway_failover_delivery_parts (
                    coordinator_id TEXT NOT NULL,
                    part_index INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    receipt TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(coordinator_id, part_index),
                    FOREIGN KEY(coordinator_id) REFERENCES gateway_failover_turns(coordinator_id)
                );
                """
            )

    @staticmethod
    def _record(row: sqlite3.Row) -> CoordinatorRecord:
        return CoordinatorRecord(
            coordinator_id=row["coordinator_id"],
            state=row["state"],
            reason=row["reason"],
            owner=row["owner"],
            generation=int(row["generation"]),
            prompt_digest=row["prompt_digest"],
            tool_digest=row["tool_digest"],
            media_digest=row["media_digest"],
            chain_digest=row["chain_digest"],
            current_leg=int(row["current_leg"]),
            delivery_state=row["delivery_state"],
        )

    @staticmethod
    def _row(conn: sqlite3.Connection, coordinator_id: str) -> sqlite3.Row:
        row = conn.execute(
            "SELECT * FROM gateway_failover_turns WHERE coordinator_id=?",
            (coordinator_id,),
        ).fetchone()
        if row is None:
            raise CoordinatorError("unknown coordinator")
        return row

    @staticmethod
    def _fence(row: sqlite3.Row, *, owner: str, generation: int) -> None:
        if int(row["generation"]) != int(generation):
            raise OwnershipLost("gateway run generation changed")
        if row["owner"] != owner:
            raise OwnershipLost("gateway coordinator owner changed")

    def plan(self, turn: FrozenTurn, *, owner: str) -> CoordinatorRecord:
        if not owner:
            raise ValueError("owner is required")
        live_digests = (
            _digest(turn.prompt),
            _digest(turn.tools),
            _digest(turn.media),
            _digest(turn.legs),
        )
        frozen_digests = (
            turn.prompt_digest,
            turn.tool_digest,
            turn.media_digest,
            turn.chain_digest,
        )
        if live_digests != frozen_digests:
            raise ReplayDenied("frozen turn was mutated after creation")
        now = time.time()
        values = {
            "prompt_json": _canonical_json(turn.prompt),
            "tools_json": _canonical_json(turn.tools),
            "media_json": _canonical_json(turn.media),
            "legs_json": _canonical_json(turn.legs),
            "prompt_digest": turn.prompt_digest,
            "tool_digest": turn.tool_digest,
            "media_digest": turn.media_digest,
            "chain_digest": turn.chain_digest,
        }
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM gateway_failover_turns WHERE coordinator_id=?",
                (turn.coordinator_id,),
            ).fetchone()
            if existing is not None:
                if int(existing["generation"]) != turn.generation:
                    raise ReplayDenied("event is already bound to another generation")
                if any(existing[key] != value for key, value in values.items()):
                    raise ReplayDenied("frozen turn snapshot mismatch")
                if existing["owner"] != owner:
                    raise OwnershipLost("gateway coordinator owner changed")
                return self._record(existing)
            conn.execute(
                """INSERT INTO gateway_failover_turns (
                    coordinator_id, session_key, session_id, event_id, generation,
                    owner, state, prompt_json, tools_json, media_json, legs_json,
                    prompt_digest, tool_digest, media_digest, chain_digest,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    turn.coordinator_id,
                    turn.session_key,
                    turn.session_id,
                    turn.event_id,
                    turn.generation,
                    owner,
                    values["prompt_json"],
                    values["tools_json"],
                    values["media_json"],
                    values["legs_json"],
                    turn.prompt_digest,
                    turn.tool_digest,
                    turn.media_digest,
                    turn.chain_digest,
                    now,
                    now,
                ),
            )
            return self._record(self._row(conn, turn.coordinator_id))

    def claim(
        self,
        coordinator_id: str,
        *,
        owner: str,
        generation: int,
        previous_owner: str | None = None,
    ) -> CoordinatorRecord:
        """Claim a replay-safe plan or completed pending delivery after restart."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            if int(row["generation"]) != int(generation):
                raise OwnershipLost("gateway run generation changed")
            expected_owner = previous_owner or owner
            if row["owner"] != expected_owner:
                raise OwnershipLost("gateway coordinator owner changed")
            unfinished = conn.execute(
                "SELECT 1 FROM gateway_failover_attempts "
                "WHERE coordinator_id=? AND state='started' LIMIT 1",
                (coordinator_id,),
            ).fetchone()
            if unfinished:
                conn.execute(
                    "UPDATE gateway_failover_turns SET state='reconciliation_required', "
                    "reason='unfinished_attempt', updated_at=? WHERE coordinator_id=?",
                    (time.time(), coordinator_id),
                )
                # This terminal acknowledgement must survive the exception
                # used to stop the caller. sqlite3 context managers otherwise
                # roll back all writes when leaving through ReplayDenied.
                conn.commit()
                raise ReplayDenied("unfinished attempt requires reconciliation")
            claimable = row["state"] == "planned" or (
                row["state"] == "completed" and row["delivery_state"] == "pending"
            )
            if not claimable:
                raise ReplayDenied(f"cannot claim coordinator in {row['state']} state")
            conn.execute(
                "UPDATE gateway_failover_turns SET owner=?, updated_at=? WHERE coordinator_id=?",
                (owner, time.time(), coordinator_id),
            )
            return self._record(self._row(conn, coordinator_id))

    def start_leg(
        self,
        coordinator_id: str,
        leg_index: int,
        *,
        turn: FrozenTurn | None = None,
        owner: str,
        generation: int,
    ) -> FrozenLegDispatch:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if turn is None:
                raise ReplayDenied("frozen turn is required at dispatch")
            live_values = {
                "prompt_json": _canonical_json(turn.prompt),
                "tools_json": _canonical_json(turn.tools),
                "media_json": _canonical_json(turn.media),
                "legs_json": _canonical_json(turn.legs),
                "prompt_digest": turn.prompt_digest,
                "tool_digest": turn.tool_digest,
                "media_digest": turn.media_digest,
                "chain_digest": turn.chain_digest,
            }
            if turn.coordinator_id != coordinator_id or any(
                row[key] != value for key, value in live_values.items()
            ):
                raise ReplayDenied("frozen turn snapshot diverged before dispatch")
            if row["state"] == "cancelled":
                raise ReplayDenied("coordinator was cancelled")
            if row["state"] != "planned" or int(row["current_leg"]) != int(leg_index):
                raise ReplayDenied("leg is not reserved for this coordinator")
            legs = json.loads(row["legs_json"])
            if leg_index < 0 or leg_index >= len(legs):
                raise ReplayDenied("route leg is outside the frozen chain")
            try:
                conn.execute(
                    """INSERT INTO gateway_failover_attempts
                       (coordinator_id, leg_index, state, started_at)
                       VALUES (?, ?, 'started', ?)""",
                    (coordinator_id, int(leg_index), time.time()),
                )
            except sqlite3.IntegrityError as exc:
                raise ReplayDenied("route leg was already attempted") from exc
            conn.execute(
                "UPDATE gateway_failover_turns SET state='running', updated_at=? "
                "WHERE coordinator_id=?",
                (time.time(), coordinator_id),
            )
            return FrozenLegDispatch(
                prompt_json=row["prompt_json"],
                tools_json=row["tools_json"],
                media_json=row["media_json"],
                leg_json=_canonical_json(legs[leg_index]),
                prompt_digest=row["prompt_digest"],
                tool_digest=row["tool_digest"],
                media_digest=row["media_digest"],
                chain_digest=row["chain_digest"],
            )

    def finish_leg(
        self,
        coordinator_id: str,
        leg_index: int,
        observation: AttemptObservation,
        *,
        owner: str,
        generation: int,
    ) -> AttemptOutcome:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["state"] != "running" or int(row["current_leg"]) != int(leg_index):
                raise ReplayDenied("coordinator is no longer in running state")
            attempt = conn.execute(
                "SELECT state FROM gateway_failover_attempts "
                "WHERE coordinator_id=? AND leg_index=?",
                (coordinator_id, int(leg_index)),
            ).fetchone()
            if attempt is None or attempt["state"] != "started":
                raise ReplayDenied("attempt is not in started state")

            if observation.outcome == "success":
                unresolved = conn.execute(
                    """SELECT call_id FROM gateway_failover_tool_calls
                       WHERE coordinator_id=? AND leg_index=?
                         AND result_digest IS NULL LIMIT 1""",
                    (coordinator_id, int(leg_index)),
                ).fetchone()
                if unresolved is not None:
                    raise ReplayDenied(
                        "cannot complete a leg with an unresolved owner tool call"
                    )

            replayable = observation.replayable
            conn.execute(
                """UPDATE gateway_failover_attempts SET
                   state='finished', outcome=?, reason=?, replayable=?,
                   partial_output=?, tool_activity=?, pending_approval=?,
                   timed_out=?, cancelled=?, finished_at=?
                   WHERE coordinator_id=? AND leg_index=? AND state='started'""",
                (
                    observation.outcome,
                    observation.reason,
                    int(replayable),
                    int(observation.partial_output),
                    int(observation.tool_activity),
                    int(observation.pending_approval),
                    int(observation.timed_out),
                    int(observation.cancelled),
                    time.time(),
                    coordinator_id,
                    int(leg_index),
                ),
            )

            if observation.outcome == "success":
                state = "completed"
                reason = "success"
                delivery_state = "pending"
                next_index = int(leg_index)
                response = observation.response
            elif replayable:
                leg_count = len(json.loads(row["legs_json"]))
                next_index = int(leg_index) + 1
                if next_index < leg_count:
                    state = "planned"
                    reason = observation.reason
                else:
                    state = "stopped"
                    reason = "route_exhausted"
                delivery_state = "none"
                response = None
            else:
                state = "stopped"
                reason = observation.reason
                delivery_state = "none"
                next_index = int(leg_index)
                response = None

            conn.execute(
                """UPDATE gateway_failover_turns SET state=?, reason=?,
                   current_leg=?, final_response=?, delivery_state=?, updated_at=?
                   WHERE coordinator_id=?""",
                (
                    state,
                    reason,
                    next_index,
                    response,
                    delivery_state,
                    time.time(),
                    coordinator_id,
                ),
            )
            return AttemptOutcome(state=state, reason=reason, replayable=replayable)

    def next_leg(
        self, coordinator_id: str, *, owner: str, generation: int
    ) -> int:
        with self._lock, self._connect() as conn:
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["state"] != "planned":
                raise ReplayDenied(f"coordinator stopped in {row['state']} state")
            current_leg = int(row["current_leg"])
            if current_leg:
                prior = conn.execute(
                    "SELECT replayable FROM gateway_failover_attempts "
                    "WHERE coordinator_id=? AND leg_index=? AND state='finished'",
                    (coordinator_id, current_leg - 1),
                ).fetchone()
                if prior is None or not prior["replayable"]:
                    raise ReplayDenied("prior attempt does not permit replay")
            return current_leg

    def recover(
        self, coordinator_id: str, *, owner: str, generation: int
    ) -> CoordinatorRecord:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["state"] == "running":
                unfinished = conn.execute(
                    "SELECT 1 FROM gateway_failover_attempts "
                    "WHERE coordinator_id=? AND state='started' LIMIT 1",
                    (coordinator_id,),
                ).fetchone()
                if unfinished:
                    conn.execute(
                        "UPDATE gateway_failover_turns "
                        "SET state='reconciliation_required', reason='unfinished_attempt', "
                        "updated_at=? WHERE coordinator_id=?",
                        (time.time(), coordinator_id),
                    )
                    row = self._row(conn, coordinator_id)
            if row["delivery_state"] == "claimed":
                # The old owner crossed the send boundary but never recorded an
                # acknowledgement. Re-delivery could duplicate an accepted
                # platform message, so restart converts the claim into a
                # terminal reconciliation state rather than silently retrying.
                conn.execute(
                    "UPDATE gateway_failover_turns "
                    "SET delivery_state='ambiguous', reason='unfinished_delivery', "
                    "updated_at=? WHERE coordinator_id=?",
                    (time.time(), coordinator_id),
                )
                row = self._row(conn, coordinator_id)
            return self._record(row)

    def cancel(self, coordinator_id: str, *, owner: str, generation: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["state"] in {"completed", "stopped", "reconciliation_required"}:
                raise ReplayDenied(f"cannot cancel coordinator in {row['state']} state")
            conn.execute(
                "UPDATE gateway_failover_turns SET state='cancelled', reason='cancelled', "
                "updated_at=? WHERE coordinator_id=?",
                (time.time(), coordinator_id),
            )

    def claim_delivery(
        self, coordinator_id: str, *, owner: str, generation: int
    ) -> str | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["state"] != "completed":
                raise ReplayDenied("coordinator has no completed response")
            if row["delivery_state"] != "pending":
                return None
            cursor = conn.execute(
                "UPDATE gateway_failover_turns SET delivery_state='claimed', updated_at=? "
                "WHERE coordinator_id=? AND delivery_state='pending'",
                (time.time(), coordinator_id),
            )
            return row["final_response"] if cursor.rowcount == 1 else None

    def mark_delivered(
        self, coordinator_id: str, *, owner: str, generation: int
    ) -> None:
        self._finish_delivery(
            coordinator_id, "delivered", owner=owner, generation=generation
        )

    def mark_delivery_ambiguous(
        self, coordinator_id: str, *, owner: str, generation: int
    ) -> None:
        self._finish_delivery(
            coordinator_id, "ambiguous", owner=owner, generation=generation
        )

    def _finish_delivery(
        self,
        coordinator_id: str,
        delivery_state: str,
        *,
        owner: str,
        generation: int,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["delivery_state"] != "claimed":
                raise ReplayDenied("final delivery was not claimed")
            conn.execute(
                "UPDATE gateway_failover_turns SET delivery_state=?, updated_at=? "
                "WHERE coordinator_id=?",
                (delivery_state, time.time(), coordinator_id),
            )

    def debug_attempts(self, coordinator_id: str) -> list[Mapping[str, Any]]:
        """Return secret-free transition evidence for tests and diagnostics."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT leg_index, state, outcome, reason, replayable,
                   partial_output, tool_activity, pending_approval, timed_out, cancelled
                   FROM gateway_failover_attempts WHERE coordinator_id=?
                   ORDER BY leg_index""",
                (coordinator_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_tool_intent(
        self,
        coordinator_id: str,
        leg_index: int,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
        *,
        owner: str,
        generation: int,
    ) -> None:
        """Latch an authenticated tool intent before owner-side execution."""
        if not call_id or not name:
            raise ValueError("tool call id and name are required")
        arguments_json = _canonical_json(arguments)
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            attempt = conn.execute(
                "SELECT state FROM gateway_failover_attempts WHERE coordinator_id=? AND leg_index=?",
                (coordinator_id, int(leg_index)),
            ).fetchone()
            if attempt is None or attempt["state"] != "started":
                raise ReplayDenied("tool intent has no live attempt")
            existing = conn.execute(
                "SELECT name, arguments_json FROM gateway_failover_tool_calls "
                "WHERE coordinator_id=? AND leg_index=? AND call_id=?",
                (coordinator_id, int(leg_index), call_id),
            ).fetchone()
            if existing is not None:
                if existing["name"] != name or existing["arguments_json"] != arguments_json:
                    raise ReplayDenied("duplicate tool call id changed identity")
                return
            conn.execute(
                "INSERT INTO gateway_failover_tool_calls "
                "(coordinator_id, leg_index, call_id, name, arguments_json, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'intent', ?, ?)",
                (coordinator_id, int(leg_index), call_id, name, arguments_json, now, now),
            )

    def record_tool_result(
        self,
        coordinator_id: str,
        leg_index: int,
        call_id: str,
        content: str,
        is_error: bool,
        *,
        owner: str,
        generation: int,
    ) -> None:
        """Latch a result digest after owner persistence and before publication."""
        result_digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            existing = conn.execute(
                "SELECT state, result_digest, is_error FROM gateway_failover_tool_calls "
                "WHERE coordinator_id=? AND leg_index=? AND call_id=?",
                (coordinator_id, int(leg_index), call_id),
            ).fetchone()
            if existing is None:
                raise ReplayDenied("tool result has no durable intent")
            if existing["state"] == "result":
                if (
                    existing["result_digest"] != result_digest
                    or existing["is_error"] != int(bool(is_error))
                ):
                    raise ReplayDenied("duplicate tool result changed identity")
                return
            cursor = conn.execute(
                "UPDATE gateway_failover_tool_calls SET state='result', result_digest=?, "
                "is_error=?, updated_at=? WHERE coordinator_id=? AND leg_index=? "
                "AND call_id=? AND state IN ('intent', 'result')",
                (
                    result_digest,
                    int(bool(is_error)),
                    time.time(),
                    coordinator_id,
                    int(leg_index),
                    call_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ReplayDenied("tool result has no durable intent")

    def debug_tool_calls(self, coordinator_id: str) -> list[Mapping[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT leg_index, call_id, name, state, result_digest, is_error "
                "FROM gateway_failover_tool_calls WHERE coordinator_id=? "
                "ORDER BY leg_index, created_at",
                (coordinator_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def initialize_delivery_parts(
        self,
        coordinator_id: str,
        parts: Sequence[str],
        *,
        owner: str,
        generation: int,
    ) -> None:
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["state"] != "completed":
                raise ReplayDenied("delivery parts require a completed turn")
            existing = conn.execute(
                "SELECT part_index, content_digest FROM gateway_failover_delivery_parts "
                "WHERE coordinator_id=? ORDER BY part_index",
                (coordinator_id,),
            ).fetchall()
            expected = [hashlib.sha256(part.encode("utf-8")).hexdigest() for part in parts]
            if existing:
                if [r["content_digest"] for r in existing] != expected:
                    raise ReplayDenied("delivery parts changed after finalization")
                return
            conn.executemany(
                "INSERT INTO gateway_failover_delivery_parts "
                "(coordinator_id, part_index, content_digest, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(coordinator_id, index, digest, now, now) for index, digest in enumerate(expected)],
            )

    def record_delivery_receipt(
        self,
        coordinator_id: str,
        part_index: int,
        receipt: str,
        *,
        owner: str,
        generation: int,
    ) -> None:
        if not receipt:
            raise ValueError("delivery receipt is required")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = self._row(conn, coordinator_id)
            self._fence(row, owner=owner, generation=generation)
            if row["delivery_state"] != "claimed":
                raise ReplayDenied("delivery receipt requires a claimed final")
            cursor = conn.execute(
                "UPDATE gateway_failover_delivery_parts SET receipt=?, updated_at=? "
                "WHERE coordinator_id=? AND part_index=? AND (receipt IS NULL OR receipt=?)",
                (receipt, time.time(), coordinator_id, int(part_index), receipt),
            )
            if cursor.rowcount != 1:
                raise ReplayDenied("unknown delivery part or conflicting receipt")

    def delivery_receipts_complete(self, coordinator_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, SUM(CASE WHEN receipt IS NOT NULL THEN 1 ELSE 0 END) AS received "
                "FROM gateway_failover_delivery_parts WHERE coordinator_id=?",
                (coordinator_id,),
            ).fetchone()
        return bool(row and row["total"] and row["total"] == row["received"])

    def load_snapshot(self, coordinator_id: str) -> Mapping[str, Any]:
        """Read back the exact durable request buckets for diagnostics/tests."""
        with self._connect() as conn:
            row = self._row(conn, coordinator_id)
        return {
            "prompt": json.loads(row["prompt_json"]),
            "tools": json.loads(row["tools_json"]),
            "media": json.loads(row["media_json"]),
            "legs": json.loads(row["legs_json"]),
        }
