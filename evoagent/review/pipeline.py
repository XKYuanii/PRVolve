"""The review pipeline: three resumable stages over one session log.

This is the only orchestrator in the system. It owns the domain sequence - parse
the diff, review it, finalize the report - and nothing else: no node graph, no
state machine of its own, no checkpoint format. Progress is whatever the log
says, and the public task lifecycle is projected from it
(``session.projections``).

The reviewer it drives may itself be staged. When it is, it records its stages
into this same log through the same runner, which is why resuming a run that
died mid-review skips the individual workers that already finished.
"""
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from ..core.diff_parser import ParsedDiff, parse_unified_diff
from ..core.models import ChangedLine, Finding, ReviewReport, Severity, TaskState
from ..errors import TaskCancelled
from ..session.events import EventKind, EventLog
from .context import (
    FINALIZE, PARSE, REVIEW, ReviewOutcome, ReviewSession, outcome_to_stage,
)
from .stages import StageRunner


class ReviewPipeline:
    name = "evoagent-pipeline"
    stages = (PARSE, REVIEW, FINALIZE)

    def __init__(
        self, store, reviewer, timeout_seconds: int = 120, stage_retries: int = 2,
        observability=None,
    ):
        self.store = store
        self.reviewer = reviewer
        self.timeout_seconds = timeout_seconds
        self.stage_retries = stage_retries
        self.observability = observability

    def run(
        self, task_id: str, repository: str, pull_request: Optional[int], diff: str,
        tenant_id: str = "default",
    ) -> ReviewReport:
        task = self.store.get(task_id) or {}
        if task.get("state") == TaskState.SUCCESS.value and task.get("report"):
            return report_from_dict(task["report"])

        log = EventLog.load(self.store, task_id)
        runner = StageRunner(
            log, retries=self.stage_retries, timeout_seconds=self.timeout_seconds,
            cancel_check=lambda: self.store.is_cancelled(task_id),
            span_factory=self._span,
        )
        session = ReviewSession(
            task_id=task_id, repository=repository, pull_request=pull_request,
            tenant_id=tenant_id, diff=diff, parsed=ParsedDiff([], []),
            log=log, runner=runner, task_input=task.get("input") or {},
        )
        try:
            parsed = runner.run(
                PARSE, lambda: self._parse(diff),
                "Input accepted; preparing review plan", retries=0,
            )
            session.parsed = deserialize_parsed(parsed)
            reviewed = runner.run(
                REVIEW, lambda: self._review(session),
                "Reviewing %d changed files" % len(session.parsed.files),
            )
            finalized = runner.run(
                FINALIZE, lambda: self._finalize(session, reviewed),
                "Validating and ranking %d finding(s)" % len(reviewed["findings"]),
            )
            # The report lives in the finalize stage output and nowhere else;
            # the terminal event only marks the run done.
            log.append(EventKind.TASK_SUCCEEDED, message="Review completed")
            return report_from_dict(finalized["report"])
        except TaskCancelled as exc:
            log.append(EventKind.TASK_CANCELLED, message=str(exc))
            raise
        except Exception as exc:
            log.append(
                EventKind.TASK_FAILED, message="Review failed: %s" % exc,
                error=str(exc),
            )
            try:
                self.store.record_failure_case(
                    task_id, "execution_error", {"error": str(exc)[:1000]}
                )
            except Exception:
                pass
            raise

    def resume(
        self, task_id: str, repository: str, pull_request: Optional[int], diff: str,
        tenant_id: str = "default",
    ) -> ReviewReport:
        return self.run(task_id, repository, pull_request, diff, tenant_id)

    # -- stages ------------------------------------------------------------

    @staticmethod
    def _parse(diff: str) -> Dict[str, Any]:
        parsed = parse_unified_diff(diff)
        if not parsed.files and not parsed.added_lines:
            raise ValueError("diff does not contain a valid unified diff with added lines")
        return serialize_parsed(parsed)

    def _review(self, session: ReviewSession) -> Dict[str, Any]:
        staged = getattr(self.reviewer, "review_session", None)
        outcome = (
            staged(session) if staged
            else ReviewOutcome(self.reviewer.review(session.diff, session.parsed))
        )
        return outcome_to_stage(outcome)

    def _finalize(self, session: ReviewSession, reviewed: Dict[str, Any]) -> Dict[str, Any]:
        findings = [finding_from_dict(item) for item in reviewed["findings"]]
        summary = dict(reviewed.get("summary") or {})
        risk = risk_of(findings)
        execution = dict(summary.get("execution") or {})
        if summary:
            execution["gates"] = summary.get("gates") or {}
            execution["rejected_findings"] = summary.get("rejected_findings") or []
            execution["repository_context"] = summary.get("repository_context") or {}
        report = ReviewReport(
            repository=session.repository, pull_request=session.pull_request,
            summary=summarize(findings, len(session.parsed.files), risk), risk=risk,
            findings=findings,
            suggestions=[
                finding_from_dict(item)
                for item in summary.get("suggested_findings") or []
            ],
            files_reviewed=session.parsed.files, reviewer=self.reviewer.name,
            collaboration=dict(summary.get("collaboration") or {}),
            run_mode=dict(summary.get("run_mode") or {}),
            components=list(summary.get("components") or []),
            execution=execution,
        )
        return {"report": report.to_dict()}

    def _span(self, name: str, attributes: Dict[str, Any]):
        if not self.observability:
            return nullcontext()
        return self.observability.span(
            name, str(attributes.get("task_id", "")), **attributes
        )


# -- report helpers, shared with anything that reads a persisted report ----

def serialize_parsed(parsed: ParsedDiff) -> Dict[str, Any]:
    return {
        "files": parsed.files,
        "added_lines": [
            {"path": item.path, "line": item.line, "content": item.content}
            for item in parsed.added_lines
        ],
    }


def deserialize_parsed(value: Dict[str, Any]) -> ParsedDiff:
    return ParsedDiff(
        list(value["files"]), [ChangedLine(**item) for item in value["added_lines"]]
    )


def finding_from_dict(value: Dict[str, Any]) -> Finding:
    item = dict(value)
    item["severity"] = Severity(item["severity"])
    return Finding(**item)


def report_from_dict(value: Dict[str, Any]) -> ReviewReport:
    return ReviewReport(
        repository=value["repository"], pull_request=value.get("pull_request"),
        summary=value["summary"], risk=value["risk"],
        findings=[finding_from_dict(item) for item in value.get("findings", [])],
        suggestions=[finding_from_dict(item) for item in value.get("suggestions", [])],
        files_reviewed=list(value.get("files_reviewed", [])),
        reviewer=value.get("reviewer", "unknown"),
        collaboration=dict(value.get("collaboration", {})),
        run_mode=dict(value.get("run_mode", {})),
        components=list(value.get("components", [])),
        execution=dict(value.get("execution", {})),
    )


def risk_of(findings: List[Finding]) -> str:
    severities = {item.severity for item in findings}
    if Severity.CRITICAL in severities or Severity.HIGH in severities:
        return "high"
    if Severity.MEDIUM in severities:
        return "medium"
    return "low"


def summarize(findings: List[Finding], file_count: int, risk: str) -> str:
    if not findings:
        return (
            "Reviewed %d file(s); no actionable issue was detected in added lines."
            % file_count
        )
    return "Reviewed %d file(s); found %d actionable issue(s). Overall risk: %s." % (
        file_count, len(findings), risk,
    )
