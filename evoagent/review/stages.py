"""Resumable stage execution over one session log.

This is the whole of what used to be a node-graph engine. A stage is a named
unit of work whose completion is a fact in the log, so re-running a task skips
whatever the log already shows as done. Stage names are hierarchical
(``review.work:security-1``), which lets the pipeline and the reviewer it drives
record progress at their own granularity into the same log, under the same
resume rule - the reason there is no second checkpoint mechanism anywhere.
"""
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Tuple

from ..errors import BudgetExceeded, TaskCancelled
from ..session.events import EventKind, EventLog
from ..session import projections


class StageRunner:
    def __init__(
        self, log: EventLog, retries: int = 0, timeout_seconds: int = 0,
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

    def guard(self, stage: str = "") -> None:
        """Cancellation and wall-clock budget, checked at every stage boundary."""
        if self.cancel_check and self.cancel_check():
            raise TaskCancelled("Task was cancelled")
        if self.timeout_seconds and time.monotonic() - self._started > self.timeout_seconds:
            raise BudgetExceeded("task execution budget exceeded%s" % (
                " during %s" % stage if stage else ""
            ))

    def output(self, stage: str) -> Optional[Dict[str, Any]]:
        return projections.stage_output(self.log, stage)

    def run(
        self, stage: str, handler: Callable[[], Optional[Dict[str, Any]]],
        message: str = "", retries: Optional[int] = None,
    ) -> Dict[str, Any]:
        cached = self.output(stage)
        if cached is not None:
            return cached
        attempts = self.retries if retries is None else max(0, retries)
        for offset in range(attempts + 1):
            self.guard(stage)
            self.log.append(EventKind.STAGE_STARTED, stage=stage, message=message)
            try:
                with self._span(stage, offset + 1):
                    output = handler() or {}
                if not isinstance(output, dict):
                    raise TypeError("stage %s must return a dict" % stage)
                self.log.append(EventKind.STAGE_COMPLETED, stage=stage, output=output)
                return output
            except self.non_retryable as exc:
                self.log.append(
                    EventKind.STAGE_FAILED, stage=stage, error=str(exc)[:1000],
                    will_retry=False,
                )
                raise
            except Exception as exc:
                self.log.append(
                    EventKind.STAGE_FAILED, stage=stage, error=str(exc)[:1000],
                    will_retry=offset < attempts,
                )
                if offset >= attempts:
                    raise
        raise AssertionError("unreachable")

    def _span(self, stage: str, attempt: int):
        if not self.span_factory:
            return nullcontext()
        return self.span_factory(
            "stage.%s" % stage,
            {"stage": stage, "attempt": attempt, "task_id": self.log.task_id},
        )
