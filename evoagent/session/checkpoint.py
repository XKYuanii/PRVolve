"""The append-only checkpoint log: one task's single source of truth.

A checkpoint here is not an overwritable row per node - it is an appended fact.
Node progress, model calls, tool calls and role traces all land in the same log,
so there is exactly one recovery mechanism and one place a run's history lives.
Task state, the run trace, the execution ledger and the report are folds over it
(see ``projections``); none of them is separately persisted.

A log with no store is a valid in-memory log, which is what one-off agent runs
(patch generation, evolution candidates) and tests use.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CheckpointKind:
    """The closed set of entry kinds. Projections switch on these."""

    TASK_CREATED = "task_created"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    AGENT_TRACE = "agent_trace"
    TASK_SUCCEEDED = "task_succeeded"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"


#: Kinds that decide what a resume re-runs, so they reach storage immediately.
#: Everything else is accounting: losing the tail of it in a crash costs nothing,
#: because the node it belonged to did not complete and will run again anyway.
DURABLE_KINDS = frozenset({
    CheckpointKind.TASK_CREATED, CheckpointKind.NODE_STARTED,
    CheckpointKind.NODE_COMPLETED, CheckpointKind.NODE_FAILED,
    CheckpointKind.TASK_SUCCEEDED, CheckpointKind.TASK_FAILED,
    CheckpointKind.TASK_CANCELLED,
})


@dataclass(frozen=True)
class Checkpoint:
    seq: int
    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq, "kind": self.kind,
            "payload": dict(self.payload), "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Checkpoint":
        return cls(
            int(value["seq"]), str(value["kind"]),
            dict(value.get("payload") or {}), str(value.get("created_at", "")),
        )


class CheckpointLog:
    """Ordered, append-only, thread-safe. Workers append concurrently.

    Accounting entries are buffered and flushed together with the next durable
    entry. Since a node's completion is itself durable, a node the log shows as
    completed always has all of its accounting stored with it.
    """

    def __init__(self, store=None, task_id: str = "", entries: Iterable[Checkpoint] = ()):
        # A store that cannot append gives an in-memory log rather than a crash:
        # replay and evaluation drive the reviewer with stub stores that only
        # supply task input, and they neither need nor want persistence.
        can_persist = bool(task_id) and hasattr(store, "append_checkpoints")
        self.store = store if can_persist else None
        self.task_id = task_id
        self._entries: List[Checkpoint] = list(entries)
        self._pending: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @classmethod
    def load(cls, store, task_id: str) -> "CheckpointLog":
        """Rebuild a log from persistence so a restarted worker resumes it."""
        reader = getattr(store, "load_checkpoints", None) if task_id else None
        entries = [Checkpoint.from_dict(item) for item in reader(task_id)] if reader else []
        return cls(store, task_id, entries)

    def append(self, kind: str, **payload) -> Checkpoint:
        with self._lock:
            entry = Checkpoint(len(self._entries) + 1, kind, payload, utc_now())
            self._entries.append(entry)
            if self.store is None:
                return entry
            self._pending.append(entry.to_dict())
            batch = self._take_pending() if kind in DURABLE_KINDS else []
        if batch:
            self.store.append_checkpoints(self.task_id, batch)
        return entry

    def flush(self) -> None:
        """Persist buffered accounting without a durable entry to ride on."""
        if self.store is None:
            return
        with self._lock:
            batch = self._take_pending()
        if batch:
            self.store.append_checkpoints(self.task_id, batch)

    def _take_pending(self) -> List[Dict[str, Any]]:
        batch, self._pending = self._pending, []
        return batch

    def entries(self, kind: str = "") -> List[Checkpoint]:
        with self._lock:
            values = list(self._entries)
        return [item for item in values if item.kind == kind] if kind else values

    def last(self, kind: str) -> Optional[Checkpoint]:
        for item in reversed(self.entries()):
            if item.kind == kind:
                return item
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
