"""The contract between the harness and whichever reviewer it drives.

A reviewer that only needs the diff implements ``review``. A reviewer that runs
the full agent protocol implements ``plan``/``execute``/``judge``, one per
harness node, and gets the session: the same checkpoint log and the same runtime
the harness uses, so its own sub-nodes are recorded the same way and resume
under the same rule.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.diff_parser import ParsedDiff
from ..core.models import Finding, Severity
from ..session import projections
from ..session.checkpoint import CheckpointLog
from .runtime import AgentRuntime


#: Re-exported so review code names nodes without importing projections; the
#: single definition lives there, next to the lifecycle mapping that reads it.
PLANNING = projections.PLANNING
EXECUTING = projections.EXECUTING
REVIEWING = projections.REVIEWING


@dataclass
class ReviewSession:
    task_id: str
    repository: str
    pull_request: Optional[int]
    tenant_id: str
    diff: str
    parsed: ParsedDiff
    log: CheckpointLog
    runtime: AgentRuntime
    task_input: Dict[str, Any] = field(default_factory=dict)

    def sub(self, node: str, name: str) -> str:
        """Namespace a reviewer sub-node under the harness node that owns it."""
        return "%s.%s" % (node, name)


@dataclass
class ReviewOutcome:
    """What a reviewer produces: the findings plus how it got there."""

    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def finding_from_dict(value: Dict[str, Any]) -> Finding:
    item = dict(value)
    item["severity"] = Severity(item["severity"])
    return Finding(**item)


def summary_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Invert the merge the harness does when it builds a report.

    The report is the one stored form; a caller that only has a task id gets its
    run summary back from there rather than from a second stored copy.
    """
    execution = dict(report.get("execution") or {})
    return {
        "run_mode": dict(report.get("run_mode") or {}),
        "components": list(report.get("components") or []),
        "collaboration": dict(report.get("collaboration") or {}),
        "suggested_findings": list(report.get("suggestions") or []),
        "execution": execution,
        "gates": dict(execution.get("gates") or {}),
        "rejected_findings": list(execution.get("rejected_findings") or []),
        "repository_context": dict(execution.get("repository_context") or {}),
        "context_management": dict(execution.get("context_management") or {}),
    }


def outcome_from_report(report: Dict[str, Any]) -> ReviewOutcome:
    return ReviewOutcome(
        [finding_from_dict(item) for item in report.get("findings") or []],
        summary_from_report(report),
    )
