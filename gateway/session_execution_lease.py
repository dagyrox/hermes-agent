"""Canonical persistent execution lease for SessionDB-backed agent turns."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional


LEASE_TTL_SECONDS = 60.0
LEASE_HEARTBEAT_SECONDS = 20.0


class SessionExecutionConflict(RuntimeError):
    """Another owner already executes against this canonical session."""


@dataclass
class SessionExecutionLease:
    db: Any
    session_id: str
    owner_id: str
    _heartbeat_task: Optional[asyncio.Task[Any]] = None
    _released: bool = False
    _lost: bool = False
    _on_lost: Optional[Callable[[], None]] = None

    @classmethod
    async def acquire(
        cls,
        db: Any,
        session_id: str,
        *,
        owner_prefix: str,
        wait_timeout: float = 0.0,
        on_lost: Optional[Callable[[], None]] = None,
    ) -> "SessionExecutionLease":
        owner_id = f"{owner_prefix}:{uuid.uuid4().hex}"
        deadline = asyncio.get_running_loop().time() + max(0.0, wait_timeout)
        while True:
            acquired = await asyncio.to_thread(
                db.acquire_session_execution_lease,
                session_id,
                owner_id,
                ttl_seconds=LEASE_TTL_SECONDS,
            )
            if acquired:
                lease = cls(
                    db=db,
                    session_id=session_id,
                    owner_id=owner_id,
                    _on_lost=on_lost,
                )
                lease._heartbeat_task = asyncio.create_task(lease._heartbeat())
                return lease
            if asyncio.get_running_loop().time() >= deadline:
                raise SessionExecutionConflict(session_id)
            await asyncio.sleep(0.05)

    async def _heartbeat(self) -> None:
        try:
            while True:
                await asyncio.sleep(LEASE_HEARTBEAT_SECONDS)
                alive = await asyncio.to_thread(
                    self.db.heartbeat_session_execution_lease,
                    self.session_id,
                    self.owner_id,
                    ttl_seconds=LEASE_TTL_SECONDS,
                )
                if not alive:
                    self._lost = True
                    if self._on_lost is not None:
                        with contextlib.suppress(Exception):
                            self._on_lost()
                    return
        except asyncio.CancelledError:
            return

    async def still_owned(self) -> bool:
        if self._released or self._lost:
            return False
        row = await asyncio.to_thread(
            self.db.get_session_execution_lease, self.session_id
        )
        owned = bool(row and row.get("owner_id") == self.owner_id)
        if not owned:
            self._lost = True
        return owned

    async def release(self) -> bool:
        if self._released:
            return False
        self._released = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        return bool(await asyncio.to_thread(
            self.db.release_session_execution_lease, self.session_id, self.owner_id
        ))

    async def __aenter__(self) -> "SessionExecutionLease":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.release()
