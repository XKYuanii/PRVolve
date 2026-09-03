"""Per-run accounting for real model and tool calls, folded from the event log.

The ledger is not a second store: ``record_model``/``record_tool``/``trace``
append events, and ``summary`` folds them back. A resumed run therefore keeps a
continuous cost and trace timeline for free, with no restore step, because the
earlier calls are already in the log it was constructed over.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .checkpoint import Checkpoint, CheckpointKind, CheckpointLog, utc_now


@dataclass
class ModelCall:
    role: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int
    ok: bool
    error: str = ""


@dataclass
class ToolCall:
    role: str
    tool: str
    arguments: Dict[str, Any]
    ok: bool
    duration_ms: int
    result_preview: str = ""
    error: str = ""


def _elapsed_ms(since: str, until: str = "") -> int:
    try:
        start = datetime.fromisoformat(since)
        end = datetime.fromisoformat(until or utc_now())
    except (TypeError, ValueError):
        return 0
    return max(0, int((end - start).total_seconds() * 1000))


class ExecutionLedger:
    def __init__(
        self, mode: str, input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0, log: Optional[CheckpointLog] = None,
    ):
        self.mode = mode
        self.input_cost_per_million = max(0.0, input_cost_per_million)
        self.output_cost_per_million = max(0.0, output_cost_per_million)
        self.log = log if log is not None else CheckpointLog()
        events = self.log.entries()
        self.started_at = events[0].created_at if events else utc_now()

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            input_tokens * self.input_cost_per_million / 1_000_000
            + output_tokens * self.output_cost_per_million / 1_000_000,
            8,
        )

    def record_model(
        self, role: str, provider: str, model: str, usage: Dict[str, Any],
        duration_ms: int, ok: bool = True, error: str = "",
    ) -> None:
        input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        reported_cost = usage.get("cost")
        cost = (
            float(reported_cost) if reported_cost is not None
            else self.estimate_cost(input_tokens, output_tokens)
        )
        self.log.append(CheckpointKind.MODEL_CALL, **asdict(ModelCall(
            role, provider, model, input_tokens, output_tokens, round(cost, 8),
            int(duration_ms), bool(ok), str(error)[:1000],
        )))

    def record_tool(
        self, role: str, tool: str, arguments: Dict[str, Any], ok: bool,
        duration_ms: int, result: Any = "", error: str = "",
    ) -> None:
        self.log.append(CheckpointKind.TOOL_CALL, **asdict(ToolCall(
            role, tool, dict(arguments), bool(ok), int(duration_ms),
            str(result)[:1000], str(error)[:1000],
        )))

    def trace(self, role: str, event: str, **detail) -> None:
        self.log.append(CheckpointKind.AGENT_TRACE, role=role, event=event, detail=detail)

    def tokens_used(self, role: str = "") -> int:
        return sum(
            int(item.payload["input_tokens"]) + int(item.payload["output_tokens"])
            for item in self.log.entries(CheckpointKind.MODEL_CALL)
            if not role or item.payload.get("role") == role
        )

    def model_call_count(self) -> int:
        return len(self.log.entries(CheckpointKind.MODEL_CALL))

    def summary(self, include_trace: bool = True) -> Dict[str, Any]:
        models: List[Dict[str, Any]] = []
        tools: List[Dict[str, Any]] = []
        traces: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.log.entries():
            if item.kind == CheckpointKind.MODEL_CALL:
                models.append(dict(item.payload))
            elif item.kind == CheckpointKind.TOOL_CALL:
                tools.append(dict(item.payload))
            elif item.kind == CheckpointKind.AGENT_TRACE and include_trace:
                role_traces = traces.setdefault(str(item.payload.get("role", "")), [])
                role_traces.append(self._trace_entry(item, len(role_traces)))
        return {
            "mode": self.mode,
            "llm_calls": len(models),
            "tool_calls": len(tools),
            "input_tokens": sum(item["input_tokens"] for item in models),
            "output_tokens": sum(item["output_tokens"] for item in models),
            "total_tokens": sum(
                item["input_tokens"] + item["output_tokens"] for item in models
            ),
            "cost_usd": round(sum(item["cost_usd"] for item in models), 8),
            "duration_ms": _elapsed_ms(self.started_at),
            "failed_model_calls": sum(not item["ok"] for item in models),
            "failed_tool_calls": sum(not item["ok"] for item in tools),
            "model_call_log": models,
            "tool_call_log": tools,
            "agent_traces": traces,
        }

    def _trace_entry(self, item: Checkpoint, index: int) -> Dict[str, Any]:
        return {
            "sequence": index + 1,
            "event": item.payload.get("event", ""),
            "elapsed_ms": _elapsed_ms(self.started_at, item.created_at),
            **dict(item.payload.get("detail") or {}),
        }
