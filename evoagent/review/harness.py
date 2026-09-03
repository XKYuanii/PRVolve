"""The review harness: three nodes, one checkpoint log.

    planning   parse the diff, run the deterministic scanners, prepare
               repository context, and have the Lead decompose the review
    executing  run the Workers in parallel, then the Lead's assess/revision
               loop until it converges or the revision budget runs out
    reviewing  Critic challenges the candidates, the Lead arbitrates, the gates
               decide what may publish, and the report is assembled

The boundaries are drawn by one rule: **work that takes part in a convergence
loop belongs to the node that owns the loop; a one-shot judgement outside it
belongs to the next node.** That is why revision sits in ``executing`` and
Critic sits in ``reviewing`` - and it is why each node carries real cost, which
is the only thing that makes a checkpoint between them worth taking.

The Lead appears in all three nodes. That is not a leak: nodes are the recovery
dimension, roles are the collaboration dimension, and the two are orthogonal.
Each Lead activation is a fresh, stateless ``AgentLoop`` whose context is passed
in explicitly, so the node boundaries fall cleanly between activations and
nothing has to be serialized mid-conversation to resume.
"""
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

from ..core.diff_parser import ParsedDiff, parse_unified_diff
from ..core.models import ChangedLine, Finding, ReviewReport, Severity, TaskState
from ..errors import TaskCancelled
from ..session.checkpoint import CheckpointKind, CheckpointLog
from .context import (
    EXECUTING, PLANNING, REVIEWING, ReviewOutcome, ReviewSession,
    finding_from_dict,
)
from .runtime import AgentRuntime


class ReviewHarness:
    name = "evoagent-harness"
    nodes = (PLANNING, EXECUTING, REVIEWING)

    def __init__(
        self, store, reviewer, timeout_seconds: int = 120, node_retries: int = 2,
        observability=None,
    ):
        self.store = store
        self.reviewer = reviewer
        self.timeout_seconds = timeout_seconds
        self.node_retries = node_retries
        self.observability = observability

    def run(
        self, task_id: str, repository: str, pull_request: Optional[int], diff: str,
        tenant_id: str = "default",
    ) -> ReviewReport:
        task = self.store.get(task_id) or {}
        if task.get("state") == TaskState.SUCCESS.value and task.get("report"):
            return report_from_dict(task["report"])

        log = CheckpointLog.load(self.store, task_id)
        runtime = AgentRuntime(
            log, retries=self.node_retries, timeout_seconds=self.timeout_seconds,
            cancel_check=lambda: self.store.is_cancelled(task_id),
            span_factory=self._span,
        )
        session = ReviewSession(
            task_id=task_id, repository=repository, pull_request=pull_request,
            tenant_id=tenant_id, diff=diff, parsed=ParsedDiff([], []),
            log=log, runtime=runtime, task_input=task.get("input") or {},
        )
        try:
            report = drive(session, self.reviewer, self._build_report)
            log.append(CheckpointKind.TASK_SUCCEEDED, message="Review completed")
            return report_from_dict(report)
        except TaskCancelled as exc:
            log.append(CheckpointKind.TASK_CANCELLED, message=str(exc))
            raise
        except Exception as exc:
            log.append(
                CheckpointKind.TASK_FAILED, message="Review failed: %s" % exc,
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

    def _build_report(
        self, session: ReviewSession, outcome: ReviewOutcome,
    ) -> Dict[str, Any]:
        return build_report(session, outcome, self.reviewer.name)

    def _span(self, name: str, attributes: Dict[str, Any]):
        if not self.observability:
            return nullcontext()
        return self.observability.span(
            name, str(attributes.get("task_id", "")), **attributes
        )


def drive(session: ReviewSession, reviewer, build) -> Dict[str, Any]:
    """Run the three nodes against ``reviewer`` and return the stored report.

    Shared by the harness and by callers that drive a reviewer directly
    (evaluation, Skill replay), so both produce an identically shaped log.
    """
    runtime = session.runtime
    # A reviewer that implements the three-node protocol drives all three
    # nodes; a plain Reviewer does all of its work inside executing.
    phased = hasattr(reviewer, "plan")

    planned = runtime.run(
        PLANNING, lambda: _planning(session, reviewer, phased),
        "Input accepted; preparing review plan", retries=0,
    )
    session.parsed = deserialize_parsed(planned["parsed"])

    executed = runtime.run(
        EXECUTING, lambda: _executing(session, reviewer, phased, planned),
        "Reviewing %d changed file(s)" % len(session.parsed.files),
    )
    reviewed = runtime.run(
        REVIEWING, lambda: _reviewing(session, reviewer, phased, planned, executed, build),
        "Validating and ranking candidate findings",
    )
    return reviewed["report"]


def _planning(session: ReviewSession, reviewer, phased: bool) -> Dict[str, Any]:
    parsed = parse_unified_diff(session.diff)
    if not parsed.files and not parsed.added_lines:
        raise ValueError("diff does not contain a valid unified diff with added lines")
    session.parsed = parsed
    output = {"parsed": serialize_parsed(parsed)}
    if phased:
        output.update(reviewer.plan(session))
    return output


def _executing(
    session: ReviewSession, reviewer, phased: bool, planned: Dict[str, Any],
) -> Dict[str, Any]:
    if phased:
        return reviewer.execute(session, planned)
    # A plain Reviewer has no phases: all of its work is this node.
    findings = reviewer.review(session.diff, session.parsed)
    return {"findings": [item.to_dict() for item in findings]}


def _reviewing(
    session: ReviewSession, reviewer, phased: bool, planned: Dict[str, Any],
    executed: Dict[str, Any], build,
) -> Dict[str, Any]:
    outcome = (
        reviewer.judge(session, planned, executed) if phased
        else ReviewOutcome([finding_from_dict(item) for item in executed["findings"]])
    )
    return {"report": build(session, outcome)}


# -- report assembly, shared with anything that reads a persisted report ---

def build_report(
    session: ReviewSession, outcome: ReviewOutcome, reviewer_name: str,
) -> Dict[str, Any]:
    summary = dict(outcome.summary or {})
    risk = risk_of(outcome.findings)
    execution = dict(summary.get("execution") or {})
    if summary:
        execution["gates"] = summary.get("gates") or {}
        execution["rejected_findings"] = summary.get("rejected_findings") or []
        execution["repository_context"] = summary.get("repository_context") or {}
    return ReviewReport(
        repository=session.repository, pull_request=session.pull_request,
        summary=summarize(outcome.findings, len(session.parsed.files), risk),
        risk=risk, findings=outcome.findings,
        suggestions=[
            finding_from_dict(item) for item in summary.get("suggested_findings") or []
        ],
        files_reviewed=session.parsed.files, reviewer=reviewer_name,
        collaboration=dict(summary.get("collaboration") or {}),
        run_mode=dict(summary.get("run_mode") or {}),
        components=list(summary.get("components") or []),
        execution=execution,
    ).to_dict()


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
