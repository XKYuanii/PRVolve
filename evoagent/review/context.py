"""The contract between the pipeline and whichever reviewer it drives.

A reviewer that only needs the diff implements ``review``. A reviewer that runs
its own resumable sub-stages - the agentic one - implements ``review_session``
and gets the session: the same log and the same stage runner the pipeline uses,
so its progress is recorded the same way and resumes under the same rule.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from typing import Any as _Any  # noqa: F401  (kept for the type alias below)

from ..core.diff_parser import ParsedDiff
from ..core.models import Finding, Severity
from ..session import projections
from ..session.events import EventLog
from .stages import StageRunner


#: Re-exported so review code names stages without importing projections; the
#: single definition lives there, next to the lifecycle mapping that reads it.
PARSE, REVIEW, FINALIZE = projections.PARSE, projections.REVIEW, projections.FINALIZE


@dataclass
class ReviewSession:
    task_id: str
    repository: str
    pull_request: Optional[int]
    tenant_id: str
    diff: str
    parsed: ParsedDiff
    log: EventLog
    runner: StageRunner
    task_input: Dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str) -> str:
        """Namespace a reviewer stage under the pipeline stage that owns it."""
        return "%s.%s" % (REVIEW, name)


@dataclass
class ReviewOutcome:
    """What a reviewer produces: the findings plus how it got there."""

    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


def outcome_to_stage(outcome: "ReviewOutcome") -> Dict[str, Any]:
    """The one stored form of a reviewer's result."""
    return {
        "findings": [item.to_dict() for item in outcome.findings],
        "summary": outcome.summary,
    }


def outcome_from_stage(stored: Dict[str, Any]) -> "ReviewOutcome":
    findings = []
    for value in stored.get("findings") or []:
        item = dict(value)
        item["severity"] = Severity(item["severity"])
        findings.append(Finding(**item))
    return ReviewOutcome(findings, dict(stored.get("summary") or {}))
