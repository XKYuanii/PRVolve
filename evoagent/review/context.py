"""The contract between the pipeline and whichever reviewer it drives.

A reviewer that only needs the diff implements ``review``. A reviewer that runs
its own resumable sub-stages - the agentic one - implements ``review_session``
and gets the session: the same log and the same stage runner the pipeline uses,
so its progress is recorded the same way and resumes under the same rule.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.diff_parser import ParsedDiff
from ..core.models import Finding
from ..session.events import EventLog
from .stages import StageRunner


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
        return "review.%s" % name


@dataclass
class ReviewOutcome:
    """What a reviewer produces: the findings plus how it got there."""

    findings: List[Finding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
