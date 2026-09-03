"""Agentic execution mode and product component taxonomy."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Optional

from ..core.models import ComponentKind


class RunMode(str, Enum):
    AGENTIC = "agentic"

    @classmethod
    def parse(cls, value: Optional[str], default: "RunMode" = None) -> "RunMode":
        fallback = default or cls.AGENTIC
        if value is None or not str(value).strip():
            return fallback
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError("mode must be agentic") from exc


@dataclass(frozen=True)
class ModeResolution:
    requested: RunMode
    effective: RunMode
    model_configured: bool
    fallback_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested": self.requested.value,
            "effective": self.effective.value,
            "model_configured": self.model_configured,
            "fallback_reason": self.fallback_reason,
        }


def resolve_mode(requested: Optional[str], model_configured: bool) -> ModeResolution:
    selected = RunMode.parse(requested, RunMode.AGENTIC)
    unavailable = "" if model_configured else "Agentic review requires a configured model."
    return ModeResolution(selected, selected, model_configured, unavailable)


def component(kind: ComponentKind, name: str, enabled: bool = True, **detail) -> dict:
    return {"kind": kind.value, "name": name, "enabled": bool(enabled), **detail}


def public_taxonomy() -> Dict[str, Any]:
    return {
        "component_kinds": {
            ComponentKind.LLM_AGENT.value: (
                "Reasons from a goal, autonomously selects tools or stops, and may revise."
            ),
            ComponentKind.TOOL_SCANNER.value: (
                "Produces facts through rules, AST, Semgrep, code search or command execution."
            ),
            ComponentKind.GATE.value: (
                "Validates format, evidence, confidence and release eligibility."
            ),
        },
        "run_modes": {
            RunMode.AGENTIC.value: "A Lead LLM delegates to two specialists and a blind Critic worker.",
        },
    }
