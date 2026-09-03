"""Node execution with budgets, retries, cancellation and checkpoint recovery.

A node is a named unit of work whose completion is a fact in the checkpoint log,
so re-running a task skips whatever the log already shows as done. Node names
are hierarchical (``executing.work:security-1``), which lets the harness and the
reviewer it drives record progress at their own granularity through one
mechanism - the reason there is no second checkpoint format anywhere.

The runtime never talks to a model. It schedules nodes; the model loop lives in
``agents.loop``.
"""
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Tuple

from ..errors import BudgetExceeded, TaskCancelled
from ..session import projections
from ..session.checkpoint import CheckpointKind, CheckpointLog


class AgentRuntime:
    def __init__(
        self, log: CheckpointLog, retries: int = 0, timeout_seconds: int = 0,
        cancel_check: Optional[Callable[[], bool]] = None,
        span_factory: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        non_retryable: Tuple[type, ...] = (ValueError, TaskCancelled, BudgetExceeded),
    ):
        self.log = log
        self.retries = max(0, retries)
        self.timeout_seconds = max(0, timeout_seconds)
        self.cancel_check = cancel_check
        self.span_factory = span_factory
        self.non_retryable = non_retryable
        self._started = time.monotonic()

    def guard(self, node: str = "") -> None:
        """Cancellation and wall-clock budget, checked at every node boundary.

        Cooperative, not preemptive: a handler already inside a model call, a
        scanner or a test subprocess runs to its own timeout first.
        """
        if self.cancel_check and self.cancel_check():
            raise TaskCancelled("Task was cancelled")
        if self.timeout_seconds and time.monotonic() - self._started > self.timeout_seconds:
            raise BudgetExceeded("task execution budget exceeded%s" % (
                " during %s" % node if node else ""
            ))

    def checkpoint(self, node: str) -> Optional[Dict[str, Any]]:
        """The stored output of a completed node, or None if it never finished."""
        return projections.node_output(self.log, node)

    def run(
        self, node: str, handler: Callable[[], Optional[Dict[str, Any]]],
        message: str = "", retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        cached = self.checkpoint(node)
        if cached is not None:
            return cached
        attempts = self.retries if retries is None else max(0, retries)
        for offset in range(attempts + 1):
            self.guard(node)
            self.log.append(CheckpointKind.NODE_STARTED, node=node, message=message)
            try:
                with self._span(node, offset + 1):
                    output = handler() or {}
                if not isinstance(output, dict):
                    raise TypeError("node %s must return a dict" % node)
                self.log.append(CheckpointKind.NODE_COMPLETED, node=node, output=output)
                return output
            except self.non_retryable as exc:
                self.log.append(
                    CheckpointKind.NODE_FAILED, node=node, error=str(exc)[:1000],
                    will_retry=False,
                )
                raise
            except Exception as exc:
                self.log.append(
                    CheckpointKind.NODE_FAILED, node=node, error=str(exc)[:1000],
                    will_retry=offset < attempts,
                )
                if offset >= attempts:
                    raise
        raise AssertionError("unreachable")

    def _span(self, node: str, attempt: int):
        if not self.span_factory:
            return nullcontext()
        return self.span_factory(
            "node.%s" % node,
            {"node": node, "attempt": attempt, "task_id": self.log.task_id},
        )
