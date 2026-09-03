"""Append-only event log: the single source of truth for one review run.

Every durable fact about a run - stage progress, model calls, tool calls, worker
assignments, published findings - is one appended event. Task state, the
execution ledger, the trace and the final report are folds over this log (see
``projections``); none of them is a separately persisted truth. Resuming a task
means replaying the fold and skipping the stages the log already completed.

A log with no store is a valid in-memory log, which is what one-off agent runs
(patch generation, evolution candidates) and tests use.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventKind:
    """The closed set of event kinds. Projections switch on these."""

    TASK_CREATED = "task_created"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    MODEL_CALL = "model_call"
    TOOL_CALL = "tool_call"
    AGENT_TRACE = "agent_trace"
    TASK_SUCCEEDED = "task_succeeded"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"


#: Kinds that decide what a resume re-runs, so they reach storage immediately.
#: Everything else is accounting: losing the tail of it in a crash costs nothing,
#: because the stage it belonged to did not complete and will run again anyway.
DURABLE_KINDS = frozenset({
    EventKind.TASK_CREATED, EventKind.STAGE_STARTED, EventKind.STAGE_COMPLETED,
    EventKind.STAGE_FAILED, EventKind.TASK_SUCCEEDED, EventKind.TASK_FAILED,
    EventKind.TASK_CANCELLED,
})


@dataclass(frozen=True)
class Event:
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
    def from_dict(cls, value: Dict[str, Any]) -> "Event":
        return cls(
            int(value["seq"]), str(value["kind"]),
            dict(value.get("payload") or {}), str(value.get("created_at", "")),
        )


class EventLog:
    """Ordered, append-only, thread-safe. Workers append concurrently.

    Accounting events are buffered and flushed together with the next durable
    event. Since a stage's completion is itself durable, a stage that the log
    shows as completed always has all of its accounting stored with it.
    """

    def __init__(self, store=None, task_id: str = "", events: Iterable[Event] = ()):
        # A store that cannot append events gives an in-memory log rather than a
        # crash: replay and evaluation drive the reviewer with stub stores that
        # only supply task input, and they neither need nor want persistence.
        can_persist = bool(task_id) and hasattr(store, "append_events")
        self.store = store if can_persist else None
        self.task_id = task_id
        self._events: List[Event] = list(events)
        self._pending: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @classmethod
    def load(cls, store, task_id: str) -> "EventLog":
        """Rebuild a log from persistence so a restarted worker resumes it."""
        reader = getattr(store, "load_events", None) if task_id else None
        events = [Event.from_dict(item) for item in reader(task_id)] if reader else []
        return cls(store, task_id, events)

    def append(self, kind: str, **payload) -> Event:
        with self._lock:
            event = Event(len(self._events) + 1, kind, payload, utc_now())
            self._events.append(event)
            if self.store is None:
                return event
            self._pending.append(event.to_dict())
            batch = self._take_pending() if kind in DURABLE_KINDS else []
        if batch:
            self.store.append_events(self.task_id, batch)
        return event

    def flush(self) -> None:
        """Persist buffered accounting events without a durable event to ride on."""
        if self.store is None:
            return
        with self._lock:
            batch = self._take_pending()
        if batch:
            self.store.append_events(self.task_id, batch)

    def _take_pending(self) -> List[Dict[str, Any]]:
        batch, self._pending = self._pending, []
        return batch

    def events(self, kind: str = "") -> List[Event]:
        with self._lock:
            values = list(self._events)
        return [item for item in values if item.kind == kind] if kind else values

    def last(self, kind: str) -> Optional[Event]:
        for item in reversed(self.events()):
            if item.kind == kind:
                return item
        return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)
