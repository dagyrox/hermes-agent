"""Receipt-bound final delivery for gateway-owned failover turns.

The object intentionally does not send. Platform adapters claim the durable
parts before their first transport call, record each acknowledged part, and
complete only after every receipt is known. A claimed delivery recovered after
a restart is ambiguous and is never resent automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from gateway.backend_failover import GatewayFailoverLedger, ReplayDenied


@dataclass
class OwnerFinalDelivery:
    ledger: GatewayFailoverLedger
    coordinator_id: str
    owner: str
    generation: int
    payload: str
    parts: Sequence[str]
    obligation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.payload:
            raise ValueError("final delivery requires a non-empty payload")

    def claim(self, parts: Sequence[str] | None = None) -> tuple[str, ...]:
        if parts is not None:
            self.parts = tuple(parts)
        if not self.parts or any(
            not isinstance(part, str) or not part for part in self.parts
        ):
            raise ValueError("final delivery requires non-empty deterministic parts")
        self.ledger.initialize_delivery_parts(
            self.coordinator_id,
            self.parts,
            owner=self.owner,
            generation=self.generation,
        )
        claimed = self.ledger.claim_delivery(
            self.coordinator_id,
            owner=self.owner,
            generation=self.generation,
        )
        if claimed is None:
            return ()
        if claimed != self.payload:
            raise ReplayDenied("claimed final payload does not match delivery handle")
        return tuple(self.parts)

    def record_receipt(self, part_index: int, receipt: str) -> None:
        self.ledger.record_delivery_receipt(
            self.coordinator_id,
            part_index,
            receipt,
            owner=self.owner,
            generation=self.generation,
        )

    def complete(self) -> None:
        if not self.ledger.delivery_receipts_complete(self.coordinator_id):
            raise ReplayDenied("cannot complete delivery with missing part receipts")
        self.ledger.mark_delivered(
            self.coordinator_id,
            owner=self.owner,
            generation=self.generation,
        )

    def ambiguous(self) -> None:
        self.ledger.mark_delivery_ambiguous(
            self.coordinator_id,
            owner=self.owner,
            generation=self.generation,
        )

    def reconcile_after_restart(self) -> str:
        return self.ledger.recover(
            self.coordinator_id,
            owner=self.owner,
            generation=self.generation,
        ).delivery_state

    def state(self) -> str:
        return self.ledger.recover(
            self.coordinator_id,
            owner=self.owner,
            generation=self.generation,
        ).delivery_state
