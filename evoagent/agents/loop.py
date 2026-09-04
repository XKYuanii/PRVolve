"""The agent loop: one role, one tool registry, one budget, one JSON action.

This is the only place a model is driven in a tool-use cycle. It is deliberately
domain-free - it never sees a ``Finding``, a diff or a task id, and it never
persists anything. Callers supply the prompt, the tools, the budgets and an
optional validator for the final action; everything the run produced comes back
in the returned action.

Durability is not this layer's job. A single role run is bounded to a handful of
steps and tens of seconds, so a crash re-runs the role rather than restoring it;
resumable progress lives one layer up, in ``review.pipeline``.
"""
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from ..errors import BudgetExceeded
from ..llm.context import ContextManager, estimate_tokens
from ..session.ledger import ExecutionLedger
from ..tools.registry import ToolRegistry


REFUTATION_PROOF_KINDS = {
    "type_constraint", "assertion", "exhaustive_paths", "executable_test",
    "documented_contract",
}


def refutation_supported(
    proof_kind: str, supporting_evidence_ids, evidence: Dict[str, dict],
) -> bool:
    """Require evidence shaped like the claimed proof, not an arbitrary tool call."""
    refs = [
        evidence[str(value)] for value in supporting_evidence_ids or []
        if str(value) in evidence
    ]
    if not refs:
        return False
    tools = {str(item.get("tool") or "").lower() for item in refs}
    text = " ".join(
        json.dumps(item.get("output"), ensure_ascii=False, default=str).lower()
        for item in refs
    )
    kind = str(proof_kind or "").strip().lower()
    if kind == "type_constraint":
        return bool(
            tools.intersection({"read_file", "symbol", "ast_analyze", "search_repository"})
            and (
                re.search(r"\b(?:optional|nonnull|non-null|never|literal|protocol)\b", text)
                or re.search(r"\bis\s+(?:a\s+)?(?:bool|str|int|float|list|dict|tuple)\b", text)
                or re.search(r"[:>]\s*(?:bool|str|int|float|list|dict|tuple)\b", text)
            )
        )
    if kind == "assertion":
        return bool(
            tools.intersection({"read_file", "search_repository", "ast_analyze", "test"})
            and ("assert" in text or '"passed": true' in text)
        )
    if kind == "exhaustive_paths":
        return bool(
            len(refs) >= 2
            and tools.intersection({"symbol", "ast_analyze"})
            and tools.intersection({"search_repository", "read_file"})
        )
    if kind == "executable_test":
        return bool(
            tools.intersection({"test", "run_repository_checks"})
            and ('"passed": true' in text or '"returncode": 0' in text)
        )
    if kind == "documented_contract":
        return bool(
            tools.intersection({
                "read_file", "search_repository", "read_project_controls", "git_context",
            })
            and re.search(
                r"\b(?:contract|must|shall|guarantee(?:d|s)?|always|never|non-null)\b",
                text,
            )
        )
    return False


def collect_evidence(observations: List[dict]) -> Dict[str, dict]:
    values = {}
    for item in observations:
        result = item.get("result")
        if isinstance(result, dict) and result.get("evidence_id"):
            output = result.get("output")
            values[str(result["evidence_id"])] = {
                "evidence_id": result["evidence_id"],
                "tool": result.get("tool", item.get("tool", "")),
                "origin": str(
                    item.get("origin")
                    or (
                        "protocol" if item.get("tool") == "protocol-requirement"
                        else "automatic-preflight"
                        if int(item.get("step", 0)) == 0 else "agent-tool"
                    )
                ),
                "output_preview": json.dumps(
                    output, ensure_ascii=False, default=str
                )[:2000],
                # Gate decisions must inspect structured facts. A truncated JSON
                # preview is for display only and may not be parseable.
                "output": output,
            }
    return values


class AgentLoop:
    def __init__(
        self, name: str, prompt: str, client,
        token_budget: int, time_budget: int, max_steps: int = 4,
        context_manager: Optional[ContextManager] = None,
        working_memory_supplier=None, observation_sink=None,
        minimum_tool_calls: int = 0,
        final_action_validator: Optional[Callable[[Dict[str, Any]], str]] = None,
    ):
        self.name = name
        self.prompt = prompt
        self.client = client
        self.token_budget = token_budget
        self.time_budget = time_budget
        self.max_steps = max_steps
        self.context_manager = context_manager or ContextManager()
        self.working_memory_supplier = working_memory_supplier
        self.observation_sink = observation_sink
        self.minimum_tool_calls = max(0, int(minimum_tool_calls))
        self.final_action_validator = final_action_validator

    def run(
        self, user_context: str, tools: ToolRegistry, ledger: ExecutionLedger,
        initial_observations: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        observations: List[dict] = list(initial_observations or [])
        nudged = False
        correction_requested = False
        invalid_action_retried = False
        starting_tokens = ledger.tokens_used(self.name)
        ledger.trace(
            self.name, "started", token_budget=self.token_budget,
            time_budget_seconds=self.time_budget, tools=tools.names(),
        )
        for step in range(1, self.max_steps + 1):
            elapsed = time.monotonic() - started
            used = ledger.tokens_used(self.name) - starting_tokens
            if elapsed >= self.time_budget or used >= self.token_budget:
                ledger.trace(self.name, "budget_exhausted", step=step, tokens_used=used)
                raise BudgetExceeded("%s budget exhausted" % self.name)
            output_allowance = self.context_manager.output_token_limit(
                self.prompt, min(4000, max(256, self.token_budget - used))
            )
            tool_catalog = tools.catalog()
            output_allowance = min(
                output_allowance,
                max(
                    128,
                    self.context_manager.context_window_tokens
                    - estimate_tokens(self.prompt)
                    - estimate_tokens(tool_catalog)
                    - 900,
                ),
            )
            current_context = user_context
            if self.working_memory_supplier is not None:
                try:
                    working = self.working_memory_supplier()
                    if working:
                        task_context = json.loads(user_context)
                        task_context["working_memory"] = working
                        current_context = json.dumps(task_context, ensure_ascii=False)
                except Exception as exc:
                    ledger.trace(
                        self.name, "working_memory_unavailable", error=str(exc)[:500],
                    )
            managed, context_stats = self.context_manager.build_managed_context(
                current_context, tool_catalog, observations,
                max(0, self.token_budget - used),
                max(0, int(self.time_budget - elapsed)),
                system_prompt=self.prompt, max_output_tokens=output_allowance,
            )
            ledger.trace(
                self.name, "context_prepared", step=step,
                estimated_input_tokens=context_stats["estimated_input_tokens_after"],
                input_token_limit=context_stats["input_token_limit"],
                observations_summarized=context_stats["observations"]["summarized"],
                observations_dropped=context_stats["observations"]["dropped"],
            )
            action = self.client.complete_json(
                self.name, self.prompt,
                json.dumps(managed, ensure_ascii=False, default=str),
                ledger, max_tokens=output_allowance,
            )
            kind = str(action.get("action", "")).strip().lower()
            ledger.trace(
                self.name, "autonomous_decision", step=step, action=kind,
                tool=str(action.get("tool", "")), reason=str(action.get("reason", ""))[:500],
            )
            if kind == "final":
                # Only the role's own tool calls count. Preflight observations
                # (step 0) are handed in by the system, so counting them let a
                # role satisfy "verify before concluding" without verifying
                # anything - the opposite of what the rule is for.
                successful_tools = sum(
                    bool(item.get("ok")) and int(item.get("step", 0)) >= 1
                    for item in observations
                )
                if successful_tools < self.minimum_tool_calls and not nudged:
                    nudged = True
                    observation = {
                        "step": step, "tool": "protocol-requirement", "ok": False,
                        "error": (
                            "Repository context is available. Use an authorized factual tool "
                            "before returning a final answer."
                        ),
                    }
                    observations.append(observation)
                    ledger.trace(
                        self.name, "minimum_tool_calls_not_met", step=step,
                        required=self.minimum_tool_calls, completed=successful_tools,
                    )
                    continue
                finding_resolutions = {
                    str(item.get("evidence_id", ""))
                    for item in action.get("evidence_resolutions") or []
                    if isinstance(item, dict)
                    and str(item.get("status", "")).strip().lower() == "finding"
                    and str(item.get("evidence_id", "")).strip()
                }
                finding_citations = {
                    str(value)
                    for finding in action.get("findings") or []
                    if isinstance(finding, dict)
                    for value in finding.get("evidence_ids") or []
                }
                uncited_findings = sorted(finding_resolutions - finding_citations)
                current_tokens = ledger.tokens_used(self.name) - starting_tokens
                can_retry_finding = (
                    step < self.max_steps - 1
                    and current_tokens < int(self.token_budget * 0.7)
                )
                if (
                    uncited_findings and not correction_requested
                    and can_retry_finding
                ):
                    correction_requested = True
                    observations.append({
                        "step": step, "tool": "protocol-requirement", "ok": False,
                        "error": (
                            "An evidence_resolution with status finding must have a "
                            "corresponding structured Finding that cites the same evidence_id. "
                            "Return the missing Finding. Inconsistent: "
                            + ", ".join(uncited_findings)
                        ),
                    })
                    ledger.trace(
                        self.name, "finding_resolution_without_finding", step=step,
                        evidence_ids=uncited_findings,
                    )
                    continue
                if uncited_findings:
                    # Preserve a repeated positive signal without publishing an
                    # unvalidated comment. Lead can route this high-risk unresolved
                    # hypothesis through the existing revision path.
                    hypotheses = list(action.get("hypotheses") or [])
                    for evidence_id in uncited_findings:
                        resolution = next(
                            item for item in action.get("evidence_resolutions") or []
                            if isinstance(item, dict)
                            and str(item.get("evidence_id", "")) == evidence_id
                        )
                        explanation = str(
                            resolution.get("explanation")
                            or "The Worker identified positive evidence but did not return "
                            "a valid structured Finding."
                        )
                        resolution["status"] = "unresolved"
                        resolution["explanation"] = explanation
                        resolution["required_proof"] = (
                            "Return a structured Finding anchored to an exact added line, or "
                            "provide invariant-level evidence that refutes this signal."
                        )
                        supporting = [
                            str(value)
                            for value in resolution.get("supporting_evidence_ids") or []
                        ]
                        if evidence_id not in supporting:
                            supporting.append(evidence_id)
                        resolution["supporting_evidence_ids"] = supporting
                        hypotheses.append({
                            "hypothesis_id": "deferred-finding:%s" % evidence_id,
                            "claim": explanation,
                            "status": "unresolved", "risk_level": "high",
                            "explanation": explanation,
                            "required_proof": resolution["required_proof"],
                            "supporting_evidence_ids": supporting,
                        })
                    action["hypotheses"] = hypotheses
                    ledger.trace(
                        self.name, "finding_resolution_deferred", step=step,
                        evidence_ids=uncited_findings,
                    )
                required_evidence = {}
                for item in observations:
                    result = item.get("result")
                    if not isinstance(result, dict):
                        continue
                    output = result.get("output")
                    if isinstance(output, dict) and output.get("requires_resolution"):
                        evidence_id = str(result.get("evidence_id", ""))
                        if evidence_id:
                            required_evidence[evidence_id] = str(
                                output.get("resolution_question", "Resolve the counterexample.")
                            )
                evidence = collect_evidence(observations)
                successful_evidence = set(evidence)
                resolved = set()
                deferred = set()
                for item in action.get("evidence_resolutions") or []:
                    if not isinstance(item, dict):
                        continue
                    evidence_id = str(item.get("evidence_id", ""))
                    supporting = {
                        str(value) for value in item.get("supporting_evidence_ids") or []
                    }
                    status = str(item.get("status", "")).strip().lower()
                    if (
                        status == "refuted"
                        and str(item.get("explanation", "")).strip()
                        and str(item.get("proof_kind", "")).strip().lower()
                        in REFUTATION_PROOF_KINDS
                        and refutation_supported(
                            str(item.get("proof_kind", "")),
                            [value for value in supporting if value != evidence_id],
                            evidence,
                        )
                    ):
                        resolved.add(evidence_id)
                    if (
                        status == "unresolved"
                        and str(item.get("explanation", "")).strip()
                        and str(item.get("required_proof", "")).strip()
                        and any(value in successful_evidence for value in supporting)
                    ):
                        deferred.add(evidence_id)
                for finding in action.get("findings") or []:
                    if isinstance(finding, dict):
                        resolved.update(str(value) for value in finding.get("evidence_ids") or [])
                unresolved = sorted(set(required_evidence) - resolved - deferred)
                if unresolved:
                    observations.append({
                        "step": step, "tool": "protocol-requirement", "ok": False,
                        "error": (
                            "Resolve each fixed counterexample before finishing. Return a finding "
                            "that cites its evidence_id; refute it with an allowed proof_kind "
                            "and independent repository evidence; or mark it unresolved with "
                            "required_proof and supporting evidence. Missing: "
                            + ", ".join(unresolved)
                        ),
                    })
                    ledger.trace(
                        self.name, "counterexample_resolution_missing", step=step,
                        unresolved=unresolved,
                    )
                    continue
                if self.final_action_validator is not None:
                    candidate = dict(action)
                    candidate["_observations"] = observations
                    validation_error = str(
                        self.final_action_validator(candidate) or ""
                    ).strip()
                    action.update({
                        key: value for key, value in candidate.items()
                        if not str(key).startswith("_")
                    })
                    if validation_error:
                        correction_requested = True
                        positive_evidence = sorted({
                            str(item.get("evidence_id", ""))
                            for item in action.get("evidence_resolutions") or []
                            if isinstance(item, dict)
                            and str(item.get("status", "")).strip().lower() == "finding"
                            and str(item.get("evidence_id", "")).strip()
                        })
                        observation = {
                            "step": step, "tool": "protocol-requirement", "ok": False,
                            "error": validation_error[:4000],
                        }
                        if positive_evidence:
                            pending_id = "protocol-finding:" + positive_evidence[0]
                            observation["result"] = {
                                "evidence_id": pending_id,
                                "tool": "protocol-requirement",
                                "output": {
                                    "requires_resolution": True,
                                    "resolution_question": (
                                        "The prior final action positively identified a defect but "
                                        "failed output validation. Return a corrected Finding citing "
                                        "this protocol evidence, or explicitly refute it with new "
                                        "successful repository evidence. Original evidence: "
                                        + ", ".join(positive_evidence)
                                    ),
                                },
                            }
                            observation["error"] += (
                                " The prior positive Finding remains pending as %s; do not silently "
                                "drop it." % pending_id
                            )
                        observations.append(observation)
                        ledger.trace(
                            self.name, "final_action_validation_failed", step=step,
                            error=validation_error[:1000],
                        )
                        continue
                action["_observations"] = observations
                action["_steps"] = step
                ledger.trace(self.name, "finished", step=step)
                return action
            if kind != "tool":
                # A syntactically valid JSON object can still omit the tiny
                # action envelope. Repair that protocol slip once, on demand;
                # normal runs pay no extra model call and repeated failures
                # still fail closed.
                if not invalid_action_retried and step < self.max_steps:
                    invalid_action_retried = True
                    observations.append({
                        "step": step, "tool": "protocol-requirement", "ok": False,
                        "error": (
                            "Invalid action envelope. Return exactly one action: "
                            "a tool action with action/tool/arguments, or a final "
                            "action with action='final' and the role's required fields."
                        ),
                    })
                    ledger.trace(
                        self.name, "invalid_action_retry", step=step,
                    )
                    continue
                raise ValueError("%s returned an invalid action" % self.name)
            tool_name = str(action.get("tool", ""))
            arguments = action.get("arguments") or {}
            try:
                value = tools.invoke(tool_name, arguments)
                observation = {
                    "step": step, "tool": tool_name, "ok": True, "result": value,
                }
            except Exception as exc:
                observation = {
                    "step": step, "tool": tool_name, "ok": False,
                    "error": str(exc)[:1000],
                }
            observations.append(observation)
            if self.observation_sink is not None:
                try:
                    self.observation_sink(self.name, observation)
                except Exception as exc:
                    ledger.trace(
                        self.name, "working_memory_write_failed", error=str(exc)[:500],
                    )
            ledger.trace(
                self.name, "tool_observation", step=step, tool=tool_name,
                ok=observation["ok"],
            )
        ledger.trace(self.name, "budget_exhausted", budget="steps")
        raise BudgetExceeded("%s step budget exhausted" % self.name)
