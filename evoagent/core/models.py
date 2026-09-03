from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskState(str, Enum):
    PENDING = "PENDING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REVIEWING = "REVIEWING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComponentKind(str, Enum):
    """Product-level component taxonomy; names describe behavior, not branding."""

    LLM_AGENT = "llm-agent"
    TOOL_SCANNER = "tool-scanner"
    GATE = "gate"


@dataclass
class ChangedLine:
    path: str
    line: int
    content: str


@dataclass
class Finding:
    rule_id: str
    severity: Severity
    title: str
    explanation: str
    path: str
    line: int
    evidence: str
    fix: str
    test: str
    confidence: float = 0.8
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    call_chain: List[Dict[str, Any]] = field(default_factory=list)
    source: str = "unknown"
    gate: Dict[str, Any] = field(default_factory=dict)
    original_rule_id: str = ""
    disposition: str = "candidate"

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["severity"] = self.severity.value
        return value


@dataclass
class ReviewReport:
    repository: str
    pull_request: Optional[int]
    summary: str
    risk: str
    findings: List[Finding] = field(default_factory=list)
    suggestions: List[Finding] = field(default_factory=list)
    files_reviewed: List[str] = field(default_factory=list)
    reviewer: str = "local-rules"
    collaboration: Dict[str, Any] = field(default_factory=dict)
    run_mode: Dict[str, Any] = field(default_factory=dict)
    components: List[Dict[str, Any]] = field(default_factory=list)
    execution: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repository": self.repository,
            "pull_request": self.pull_request,
            "summary": self.summary,
            "risk": self.risk,
            "findings": [item.to_dict() for item in self.findings],
            "suggestions": [item.to_dict() for item in self.suggestions],
            "files_reviewed": self.files_reviewed,
            "reviewer": self.reviewer,
            "collaboration": self.collaboration,
            "run_mode": self.run_mode,
            "components": self.components,
            "execution": self.execution,
        }


@dataclass
class TraceEvent:
    step: int
    state: TaskState
    message: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["state"] = self.state.value
        return value
