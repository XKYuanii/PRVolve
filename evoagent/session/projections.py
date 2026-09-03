"""Pure folds over the checkpoint log. Nothing here reads or writes storage.

The harness advances through hierarchical node names (``planning``,
``planning.scan``, ``executing.work:security-1`` ...). The public task lifecycle
is a much coarser thing, so ``task_state`` and ``trace`` project the fine node
names onto ``TaskState`` by their root segment. That is the only place the two
vocabularies meet - the log itself never stores a task state.
"""
from typing import Any, Dict, List, Optional

from ..core.models import TaskState, TraceEvent
from .checkpoint import CheckpointKind, CheckpointLog


#: The three harness nodes. They are named here because this module is where the
#: node vocabulary is translated for everything that reads it: the public task
#: lifecycle, the run trace, and the task read model.
PLANNING = "planning"
EXECUTING = "executing"
REVIEWING = "reviewing"

#: The node whose output carries the published report. Storing it once, in the
#: node that builds it, is why no terminal entry repeats it.
REPORT_NODE = REVIEWING

#: Root node name -> the public lifecycle state it reports as.
NODE_STATES = (
    (PLANNING, TaskState.PLANNING),
    (EXECUTING, TaskState.EXECUTING),
    (REVIEWING, TaskState.REVIEWING),
)

#: Terminal entry kind -> the public lifecycle state it reports as.
TERMINAL_STATES = {
    CheckpointKind.TASK_SUCCEEDED: TaskState.SUCCESS,
    CheckpointKind.TASK_FAILED: TaskState.FAILED,
    CheckpointKind.TASK_CANCELLED: TaskState.CANCELLED,
}

COMPLETED = "completed"
FAILED = "failed"
RUNNING = "running"


def node_state(node: str) -> TaskState:
    """Map ``executing.work:security-1`` onto the public ``EXECUTING`` state."""
    root = node.split(".", 1)[0]
    for name, state in NODE_STATES:
        if name == root:
            return state
    return TaskState.EXECUTING


def progress(log: CheckpointLog) -> Dict[str, Dict[str, Any]]:
    """Node name -> {status, attempt, output, error} for every node seen."""
    result: Dict[str, Dict[str, Any]] = {}
    for item in log.entries():
        node = str(item.payload.get("node", ""))
        if not node:
            continue
        entry = result.setdefault(
            node, {"status": RUNNING, "attempt": 0, "output": {}, "error": ""}
        )
        if item.kind == CheckpointKind.NODE_STARTED:
            entry["status"] = RUNNING
            entry["attempt"] = int(entry["attempt"]) + 1
        elif item.kind == CheckpointKind.NODE_COMPLETED:
            entry["status"] = COMPLETED
            entry["output"] = dict(item.payload.get("output") or {})
            entry["error"] = ""
        elif item.kind == CheckpointKind.NODE_FAILED:
            entry["status"] = FAILED
            entry["error"] = str(item.payload.get("error", ""))
    return result


def completed(log: CheckpointLog, node: str) -> bool:
    return progress(log).get(node, {}).get("status") == COMPLETED


def node_output(log: CheckpointLog, node: str) -> Optional[Dict[str, Any]]:
    """The output a completed node recorded, or ``None`` if it never finished."""
    entry = progress(log).get(node)
    return dict(entry["output"]) if entry and entry["status"] == COMPLETED else None


def task_state(log: CheckpointLog) -> TaskState:
    state = TaskState.PENDING
    for item in log.entries():
        if item.kind in TERMINAL_STATES:
            return TERMINAL_STATES[item.kind]
        if item.kind == CheckpointKind.NODE_STARTED:
            state = node_state(str(item.payload.get("node", "")))
    return state


def trace(log: CheckpointLog) -> List[TraceEvent]:
    """One entry per public state change - what the task detail view shows."""
    events: List[TraceEvent] = []
    current = TaskState.PENDING
    for item in log.entries():
        if item.kind == CheckpointKind.NODE_STARTED:
            target = node_state(str(item.payload.get("node", "")))
            message = str(item.payload.get("message", ""))
            if target == current:
                continue
        elif item.kind in TERMINAL_STATES:
            target = TERMINAL_STATES[item.kind]
            message = str(item.payload.get("message", ""))
        else:
            continue
        current = target
        events.append(TraceEvent(len(events) + 1, target, message, item.created_at))
    return events


def report(log: CheckpointLog) -> Optional[Dict[str, Any]]:
    """The published report, stored once by the node that built it."""
    stored = node_output(log, REPORT_NODE) or {}
    return dict(stored["report"]) if "report" in stored else None
