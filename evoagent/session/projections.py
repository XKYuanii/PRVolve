"""Pure folds over the event log. Nothing here reads or writes storage.

The pipeline advances through hierarchical stage names (``parse``, ``review``,
``review.scan``, ``review.work:security-1`` ...). The public task lifecycle is a
much coarser thing, so ``task_state`` and ``trace`` project the fine stage names
onto ``TaskState`` by their root segment. That is the only place the two
vocabularies meet - the log itself never stores a task state.
"""
from typing import Any, Dict, List, Optional

from ..core.models import TaskState, TraceEvent
from .events import EventKind, EventLog


#: Root stage name -> the public lifecycle state it reports as.
STAGE_STATES = (
    ("parse", TaskState.PLANNING),
    ("review", TaskState.EXECUTING),
    ("finalize", TaskState.REVIEWING),
)

#: Terminal event kind -> the public lifecycle state it reports as.
TERMINAL_STATES = {
    EventKind.TASK_SUCCEEDED: TaskState.SUCCESS,
    EventKind.TASK_FAILED: TaskState.FAILED,
    EventKind.TASK_CANCELLED: TaskState.CANCELLED,
}

COMPLETED = "completed"
FAILED = "failed"
RUNNING = "running"


def stage_state(stage: str) -> TaskState:
    """Map ``review.work:security-1`` onto the public ``EXECUTING`` state."""
    root = stage.split(".", 1)[0]
    for name, state in STAGE_STATES:
        if name == root:
            return state
    return TaskState.EXECUTING


def progress(log: EventLog) -> Dict[str, Dict[str, Any]]:
    """Stage name -> {status, attempt, output, error} for every stage seen."""
    result: Dict[str, Dict[str, Any]] = {}
    for item in log.events():
        stage = str(item.payload.get("stage", ""))
        if not stage:
            continue
        entry = result.setdefault(
            stage, {"status": RUNNING, "attempt": 0, "output": {}, "error": ""}
        )
        if item.kind == EventKind.STAGE_STARTED:
            entry["status"] = RUNNING
            entry["attempt"] = int(entry["attempt"]) + 1
        elif item.kind == EventKind.STAGE_COMPLETED:
            entry["status"] = COMPLETED
            entry["output"] = dict(item.payload.get("output") or {})
            entry["error"] = ""
        elif item.kind == EventKind.STAGE_FAILED:
            entry["status"] = FAILED
            entry["error"] = str(item.payload.get("error", ""))
    return result


def completed(log: EventLog, stage: str) -> bool:
    return progress(log).get(stage, {}).get("status") == COMPLETED


def stage_output(log: EventLog, stage: str) -> Optional[Dict[str, Any]]:
    """The output a completed stage recorded, or ``None`` if it never finished."""
    entry = progress(log).get(stage)
    return dict(entry["output"]) if entry and entry["status"] == COMPLETED else None


def task_state(log: EventLog) -> TaskState:
    state = TaskState.PENDING
    for item in log.events():
        if item.kind in TERMINAL_STATES:
            return TERMINAL_STATES[item.kind]
        if item.kind == EventKind.STAGE_STARTED:
            state = stage_state(str(item.payload.get("stage", "")))
    return state


def trace(log: EventLog) -> List[TraceEvent]:
    """One entry per public state change - what the task detail view shows."""
    events: List[TraceEvent] = []
    current = TaskState.PENDING
    for item in log.events():
        if item.kind == EventKind.STAGE_STARTED:
            target = stage_state(str(item.payload.get("stage", "")))
            message = str(item.payload.get("message", ""))
            if target == current:
                continue
        elif item.kind in TERMINAL_STATES:
            target = TERMINAL_STATES[item.kind]
            message = str(item.payload.get("message", ""))
        else:
            continue
        current = target
        events.append(
            TraceEvent(len(events) + 1, target, message, item.created_at)
        )
    return events


def report(log: EventLog) -> Optional[Dict[str, Any]]:
    """The published report, if the run reached a successful end."""
    event = log.last(EventKind.TASK_SUCCEEDED)
    return dict(event.payload.get("report") or {}) if event else None
