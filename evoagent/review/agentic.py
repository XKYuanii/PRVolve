"""The hierarchical review: a Lead, bounded Workers, a Critic, one log.

The reviewer implements the harness protocol as three methods, one per node:

    plan     deterministic scanners, repository context, Lead decomposition
    execute  Workers in parallel, then the Lead assess/revision loop
    judge    Critic challenge, Lead arbitration, gates, run summary

Each records its own sub-nodes (``executing.work:security-1``) through the same
``AgentRuntime`` and the same checkpoint log the harness uses, so a restarted
worker re-reads whatever finished and re-runs only what did not - down to the
individual assignment, which is where a review actually spends its money.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..agents.loop import AgentLoop
from ..agents.parsing import parse_findings, worker_final_validation_error
from ..agents.prompts import (
    CRITIC_PROMPT, LEAD_PROMPT, RELIABILITY_PROMPT, ROLE_PERMISSIONS,
    RULE_ID_GUIDANCE, SECURITY_PROMPT,
)
from ..core.diff_parser import ParsedDiff
from ..core.gates import FindingGate
from ..core.models import ComponentKind, Finding
from ..core.modes import component, resolve_mode
from ..llm.context import ContextManager
from ..session.checkpoint import CheckpointLog
from ..session.ledger import ExecutionLedger
from ..tools.registry import AgentTool
from ..tools.repository import RepositoryToolSuite
from .context import (
    EXECUTING, PLANNING, REVIEWING, ReviewOutcome, ReviewSession,
    outcome_from_report, summary_from_report,
)
from .merge import (
    apply_critic, apply_lead_final, attach_diff_ast_evidence, candidates_from,
    merge_findings, normalize_delegations, normalize_revision_requests,
    partition_publication, public_decision, restore_findings, scanner_name,
    skill_tool_permissions,
)
from .preflight import repository_preflight
from .reviewers import LocalRuleReviewer, Reviewer
from .runtime import AgentRuntime


@dataclass
class RunContext:
    """Per-run setup. Derived from task input, so it is rebuilt on every node
    and on every resume rather than checkpointed."""

    resolution: Any
    ledger: ExecutionLedger
    root: str
    suite: RepositoryToolSuite
    enabled: Set[str]
    workers: List[str]
    scanners: List[Reviewer]
    skills: Dict[str, Any] = field(default_factory=dict)
    requested_skills: List[str] = field(default_factory=list)


class AgenticReviewer(Reviewer):
    """Runs the Lead/Worker/Critic protocol as resumable nodes on the log."""

    name = "mode-router"

    def __init__(
        self, store, llm_client=None,
        default_token_budget: int = 8000, default_time_budget: int = 60,
        input_cost_per_million: float = 0.0, output_cost_per_million: float = 0.0,
        enabled_roles: Optional[Set[str]] = None,
        scanners: Optional[List[Reviewer]] = None,
        scanner_provider=None,
        review_test_command: str = "",
        prompt_overlay: str = "",
        structured_config: Optional[Dict[str, Any]] = None,
        memory_manager=None,
        context_manager: Optional[ContextManager] = None,
        skill_provider=None,
    ):
        self.store = store
        self.client = llm_client
        self.default_token_budget = default_token_budget
        self.default_time_budget = default_time_budget
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million
        self.enabled_roles = enabled_roles or {
            "lead", "security", "correctness-reliability", "critic"
        }
        self.rules = LocalRuleReviewer()
        self.scanners = list(scanners or [])
        self.scanner_provider = scanner_provider
        self.review_test_command = review_test_command
        self.prompt_overlay = str(prompt_overlay or "").strip()
        self.structured_config = dict(structured_config or {})
        self.memory_manager = memory_manager
        self.context_manager = context_manager or ContextManager()
        self.skill_provider = skill_provider
        if self.structured_config:
            self.prompt_overlay += "\nStructured runtime policy:\n" + json.dumps(
                self.structured_config, ensure_ascii=False, sort_keys=True
            )
        self.gate = FindingGate()
        self._memory_scopes: Dict[str, tuple] = {}
        self._memory_scope_lock = threading.Lock()

    def _token_budget(self, role: str) -> int:
        raw = (self.structured_config.get("budget_parameters") or {}).get(
            role, self.default_token_budget
        )
        try:
            return max(256, min(int(raw), self.default_token_budget * 4))
        except (TypeError, ValueError):
            return self.default_token_budget

    def _max_revision_rounds(self) -> int:
        raw = self.structured_config.get("max_revision_rounds", 2)
        try:
            return max(0, min(int(raw), 2))
        except (TypeError, ValueError):
            return 2

    def review(self, diff: str, parsed: ParsedDiff) -> List[Finding]:
        raise RuntimeError(
            "agentic review requires the plan/execute/judge protocol and a model"
        )

    # -- the three harness nodes -------------------------------------------

    def plan(self, session: ReviewSession) -> Dict[str, Any]:
        """planning: deterministic scan, repository context, Lead decomposition."""
        with self._memory_scope(session):
            run = self._open_run(session)
            self.context_manager.begin(session.task_id)
            memory_context = self._recall(session, run.ledger)
            scanned = session.runtime.run(
                session.sub(PLANNING, "scan"),
                lambda: self._scan(session, run.ledger, run.scanners),
                "Running deterministic scanners",
            )
            delegated = session.runtime.run(
                session.sub(PLANNING, "delegate"),
                lambda: self._delegate(
                    session, run, scanned["findings"], memory_context,
                ),
                "Lead is decomposing the review",
            )
            return {
                "scanner_findings": scanned["findings"],
                "scanner_components": scanned["components"],
                "delegations": delegated["delegations"],
                "lead_delegation": delegated["decision"],
                "memory_context": memory_context,
                "run_mode": run.resolution.to_dict(),
                "repository_context": {
                    "available": run.suite.repository_available,
                    "root_supplied": bool(run.root),
                },
                "context_management": self.context_manager.summary(session.task_id),
            }

    def execute(self, session: ReviewSession, planned: Dict[str, Any]) -> Dict[str, Any]:
        """executing: Workers in parallel, then the Lead's assess/revision loop."""
        with self._memory_scope(session):
            run = self._open_run(session)
            self.context_manager.restore(
                session.task_id, planned.get("context_management")
            )
            memory_context = planned["memory_context"]
            delegations = planned["delegations"]
            rule_findings = restore_findings(planned["scanner_findings"])

            worker_results = self._run_assignments(
                session, run, delegations, planned["scanner_findings"], 0,
                memory_context,
            )
            max_rounds = self._max_revision_rounds()
            assessments: List[dict] = []
            revision_results: Dict[str, dict] = {}
            stop_reason = ""
            critic_objective = ""
            for index in range(max_rounds + 1):
                candidates = candidates_from(rule_findings, worker_results)
                assessment = session.runtime.run(
                    session.sub(EXECUTING, "assess-%d" % index),
                    lambda candidates=candidates, index=index: {
                        "decision": public_decision(self._assess(
                            session, run, delegations, worker_results, candidates,
                            index, max_rounds, memory_context,
                        ))
                    },
                    "Lead is assessing worker output (round %d)" % index,
                )["decision"]
                assessments.append(assessment)
                critic_objective = str(assessment.get("critic_objective", ""))
                requests = normalize_revision_requests(
                    assessment.get("revision_requests"), delegations,
                )
                if not requests or index >= max_rounds:
                    if requests:
                        stop_reason = (
                            "revision-skipped-stability-profile" if max_rounds == 0
                            else "revision-budget-exhausted"
                        )
                    break
                revisions = []
                for request in requests:
                    original = next(
                        item for item in delegations
                        if item["assignment_id"] == request["assignment_id"]
                    )
                    revision = dict(original)
                    revision["run_id"] = "%d:%s" % (index + 1, request["assignment_id"])
                    revision["revision_round"] = index + 1
                    revision["lead_feedback"] = request["guidance"]
                    revision["required_evidence"] = request["required_evidence"]
                    revisions.append(revision)
                revised = self._run_assignments(
                    session, run, revisions, planned["scanner_findings"], index + 1,
                    memory_context,
                )
                for revision in revisions:
                    key = revision["run_id"]
                    result = revised[key]
                    revision_results[key] = result
                    worker_results[revision["assignment_id"]] = result
                    run.ledger.trace(
                        "lead-session", "revision_completed",
                        assignment_id=revision["assignment_id"],
                        worker=revision["worker"], round=index + 1,
                        status=result["status"],
                    )
            return {
                "worker_results": worker_results,
                "lead_assessments": assessments,
                "revision_results": revision_results,
                "stop_reason": stop_reason or "lead-final",
                "critic_objective": critic_objective,
                "context_management": self.context_manager.summary(session.task_id),
            }

    def judge(
        self, session: ReviewSession, planned: Dict[str, Any], executed: Dict[str, Any],
    ) -> ReviewOutcome:
        """reviewing: Critic challenge, Lead arbitration, gates, run summary."""
        with self._memory_scope(session):
            run = self._open_run(session)
            self.context_manager.restore(
                session.task_id, executed.get("context_management")
            )
            memory_context = planned["memory_context"]
            rule_findings = restore_findings(planned["scanner_findings"])
            worker_results = executed["worker_results"]
            candidates = candidates_from(rule_findings, worker_results)
            before_critic = len(candidates)

            if "critic" in run.enabled and candidates:
                judged = session.runtime.run(
                    session.sub(REVIEWING, "critic"),
                    lambda: self._critic(
                        session, run, candidates, executed["critic_objective"],
                        memory_context,
                    ),
                    "Critic is challenging %d candidate finding(s)" % len(candidates),
                )
                candidates = restore_findings(judged["candidates"])
                critic_decisions = judged["decisions"]
            else:
                critic_decisions = [
                    {"finding_index": index, "accepted": True, "objections": []}
                    for index in range(len(candidates))
                ]

            arbitrated = session.runtime.run(
                session.sub(REVIEWING, "arbitrate"),
                lambda: {"decision": public_decision(self._arbitrate(
                    session, run, candidates, critic_decisions, worker_results,
                    memory_context,
                ))},
                "Lead is arbitrating the final finding set",
            )
            lead_final = arbitrated["decision"]
            lead_accepted = apply_lead_final(lead_final, candidates)
            accepted, suggestions, publication_decisions = partition_publication(
                rule_findings, candidates, lead_accepted, critic_decisions,
                run.suite.repository_available,
                critic_required="critic" in run.enabled,
                publish_unverified_suggestions=bool(
                    self.structured_config.get("publish_unverified_suggestions", True)
                ),
            )
            collaboration = self._collaboration(
                run, planned, executed, lead_final, publication_decisions,
                critic_decisions,
                len(rule_findings), before_critic, len(accepted), suggestions,
            )
            gated = self.gate.apply(accepted, session.parsed)
            run.ledger.trace("evidence-gate", "completed", **gated.checks)
            self._persist_task_memory(
                session.task_id, session.tenant_id, session.repository, accepted,
                gated, session.parsed.files, collaboration,
            )
            execution = run.ledger.summary()
            execution["context_management"] = self.context_manager.summary(session.task_id)
            summary = {
                "run_mode": planned["run_mode"],
                "components": planned["scanner_components"] + self._role_components(run) + [
                    component(ComponentKind.GATE, "finding-format-gate"),
                    component(ComponentKind.GATE, "evidence-gate"),
                    component(ComponentKind.GATE, "confidence-gate"),
                    component(ComponentKind.GATE, "release-gate"),
                ],
                "execution": execution,
                "collaboration": collaboration,
                "suggested_findings": [item.to_dict() for item in suggestions],
                "gates": gated.checks,
                "rejected_findings": gated.rejected,
                "repository_context": planned["repository_context"],
                "context_management": execution["context_management"],
            }
            return ReviewOutcome(gated.accepted, summary)

    # -- standalone entry, for evaluation and Skill replay ------------------

    def review_outcome(
        self, task_id: str, diff: str, parsed: ParsedDiff,
        repository: str = "", tenant_id: str = "default",
    ) -> ReviewOutcome:
        """Run the same three nodes without a harness, on a session of our own."""
        from .harness import build_report, drive

        session = self.open_session(task_id, diff, parsed, repository, tenant_id)
        report = drive(
            session, self,
            lambda run_session, outcome: build_report(run_session, outcome, self.name),
        )
        return outcome_from_report(report)

    def review_with_context(
        self, task_id: str, diff: str, parsed: ParsedDiff,
        repository: str = "", tenant_id: str = "default",
    ) -> List[Finding]:
        return self.review_outcome(task_id, diff, parsed, repository, tenant_id).findings

    def open_session(
        self, task_id: str, diff: str, parsed: ParsedDiff,
        repository: str = "", tenant_id: str = "default",
    ) -> ReviewSession:
        log = CheckpointLog.load(self.store, task_id)
        task = (self.store.get(task_id, tenant_id) or {}) if task_id else {}
        return ReviewSession(
            task_id=task_id, repository=repository, pull_request=None,
            tenant_id=tenant_id, diff=diff, parsed=parsed, log=log,
            runtime=AgentRuntime(log), task_input=task.get("input") or {},
        )

    def collaboration_summary(self, task_id: str) -> dict:
        """Project the run summary out of the log; never a cached side table."""
        if not task_id:
            return {}
        stored = AgentRuntime(CheckpointLog.load(self.store, task_id)).checkpoint(REVIEWING)
        report = (stored or {}).get("report")
        return summary_from_report(report) if report else {}

    # -- per-run setup, rebuilt on every node and on every resume -----------

    def _open_run(self, session: ReviewSession) -> "RunContext":
        task_input = session.task_input
        resolution = resolve_mode(task_input.get("mode"), self.client is not None)
        if self.client is None:
            raise RuntimeError("agentic review requires a configured model")
        ledger = ExecutionLedger(
            resolution.effective.value, self.input_cost_per_million,
            self.output_cost_per_million, log=session.log,
        )
        root = str(task_input.get("repository_root") or "")
        if not root and os.path.isdir(session.repository):
            root = session.repository
        enabled = set(task_input.get("enabled_agents") or self.enabled_roles)
        if "lead" not in enabled:
            raise ValueError("agentic mode requires the lead Agent")
        skills = {
            skill.name: skill
            for skill in (
                list(self.skill_provider(session.tenant_id)) if self.skill_provider else []
            )
        }
        requested = [str(value) for value in task_input.get("enabled_skills") or []]
        unknown = set(requested).difference(skills)
        if unknown:
            raise ValueError(
                "unknown enabled Agent Skill(s): %s" % ", ".join(sorted(unknown))
            )
        return RunContext(
            resolution=resolution, ledger=ledger, root=root,
            suite=RepositoryToolSuite(
                root, session.diff, session.parsed, ledger, self.review_test_command
            ),
            enabled=enabled,
            workers=[
                name for name in ("security", "correctness-reliability")
                if name in enabled
            ],
            scanners=self.scanners + (
                list(self.scanner_provider(session.tenant_id))
                if self.scanner_provider else []
            ),
            skills=skills, requested_skills=requested,
        )

    @contextmanager
    def _memory_scope(self, session: ReviewSession):
        """Bind the tenant/repository the role memory hooks write under."""
        with self._memory_scope_lock:
            self._memory_scopes[session.task_id] = (session.tenant_id, session.repository)
        try:
            yield
        finally:
            # Do not leave a stale binding behind when setup, a model call or a
            # gate raises. Working observations keep their TTL, so a resumed
            # task can still use them.
            with self._memory_scope_lock:
                self._memory_scopes.pop(session.task_id, None)

    def _recall(self, session: ReviewSession, ledger: ExecutionLedger) -> Dict[str, Any]:
        memory_query = self.context_manager.memory_query(session.diff, session.parsed.files)
        try:
            recalled = (
                self.memory_manager.recall(
                    session.tenant_id, session.repository, memory_query
                )
                if self.memory_manager is not None else []
            )
        except Exception as exc:
            recalled = []
            ledger.trace("context-manager", "memory_recall_failed", error=str(exc)[:1000])
        self.context_manager.record_memory_recall(session.task_id, memory_query, recalled)
        ledger.trace(
            "context-manager", "memory_recalled", count=len(recalled),
            repository=session.repository, tenant_id=session.tenant_id,
        )
        return self.context_manager.format_memories(recalled) or {
            "trust": "untrusted historical hints; verify with current diff or tools",
            "items": [],
        }

    def _role_components(self, run: "RunContext") -> List[dict]:
        return [
            component(
                ComponentKind.LLM_AGENT, name,
                token_budget=self._token_budget(name),
                time_budget_seconds=self.default_time_budget,
                tool_permissions=sorted(ROLE_PERMISSIONS[name]),
            )
            for name in ("lead", "security", "correctness-reliability", "critic")
            if name in run.enabled
        ]

    @staticmethod
    def _collaboration(
        run, planned, executed, lead_final, publication_decisions, critic_decisions,
        scanner_count, before_critic, accepted_count, suggestions,
    ) -> Dict[str, Any]:
        delegations = planned["delegations"]
        return {
            "protocol": "lead-workers",
            "roles": [
                name for name in ("lead", "security", "correctness-reliability", "critic")
                if name in run.enabled
            ],
            "lead": {
                "delegation": planned["lead_delegation"],
                "assessments": executed["lead_assessments"],
                "final": lead_final,
            },
            "assignments": delegations,
            "agent_skills": sorted({
                name for assignment in delegations
                for name in assignment.get("skills") or []
            }),
            "worker_results": list(executed["worker_results"].values()),
            "revision_results": list(executed["revision_results"].values()),
            "scanner_findings": scanner_count,
            "candidate_findings_before_critic": before_critic,
            "accepted_findings": accepted_count,
            "suggested_findings": [item.to_dict() for item in suggestions],
            "suggestion_count": len(suggestions),
            "publication_decisions": publication_decisions,
            "critic_decisions": critic_decisions,
            "stop_reason": executed["stop_reason"],
        }

    # -- node bodies -------------------------------------------------------

    def _scan(self, session: ReviewSession, ledger, scanners) -> Dict[str, Any]:
        findings, components = self._scan_findings(
            session.diff, session.parsed, ledger, scanners
        )
        return {
            "findings": [item.to_dict() for item in findings],
            "components": components,
        }

    def _delegate(
        self, session, run, scanner_findings, memory_context,
    ) -> Dict[str, Any]:
        suite, ledger = run.suite, run.ledger
        worker_roles, available_skills = run.workers, run.skills
        requested_skills = run.requested_skills
        decision = self._run_lead(
            "delegate", {
                **self._model_diff(
                    session.diff, session.task_id, "lead:delegate",
                    focus_files=session.parsed.files,
                ),
                "changed_files": session.parsed.files,
                "enabled_workers": worker_roles,
                "available_agent_skills": [
                    available_skills[name].catalog_entry()
                    for name in sorted(available_skills)
                ],
                "requested_agent_skills": requested_skills,
                "scanner_findings": scanner_findings,
                "repository_context_available": suite.repository_available,
                "recalled_memory": memory_context,
            }, suite, ledger, session.task_id,
        )
        delegations = normalize_delegations(
            decision.get("delegations"), worker_roles, session.parsed.files,
            set(available_skills), requested_skills,
        )
        if suite.repository_available:
            requirement = (
                "Use repository tools to verify relevant types, preconditions, callers "
                "or tests before returning final."
            )
            for assignment in delegations:
                if requirement not in assignment["required_evidence"]:
                    assignment["required_evidence"].append(requirement)
        for assignment in delegations:
            ledger.trace(
                "lead-session", "assignment_created",
                assignment_id=assignment["assignment_id"],
                worker=assignment["worker"],
                objective=assignment["objective"][:500],
            )
        return {"delegations": delegations, "decision": public_decision(decision)}

    def _critic(
        self, session, run, candidates, objective, memory_context,
    ) -> Dict[str, Any]:
        result = self._run_critic(
            session.diff, candidates, objective, run.suite, run.ledger,
            session.task_id, memory_context,
        )
        kept, decisions = apply_critic(result, candidates)
        return {
            "candidates": [item.to_dict() for item in kept],
            "decisions": decisions,
        }

    def _assess(
        self, session, run, delegations, worker_results, candidates,
        index, max_rounds, memory_context,
    ) -> Dict[str, Any]:
        return self._run_lead(
            "assess-workers", {
                **self._model_diff(
                    session.diff, session.task_id, "lead:assess-workers",
                    focus_files=[item.path for item in candidates],
                ),
                "assignments": delegations,
                "worker_results": list(worker_results.values()),
                "candidate_findings": [item.to_dict() for item in candidates],
                "revision_round": index,
                "remaining_revision_rounds": max_rounds - index,
                "recalled_memory": memory_context,
            }, run.suite, run.ledger, session.task_id,
        )

    def _arbitrate(
        self, session, run, candidates, critic_decisions, worker_results,
        memory_context,
    ) -> Dict[str, Any]:
        decision = self._run_lead(
            "finalize", {
                **self._model_diff(
                    session.diff, session.task_id, "lead:finalize",
                    focus_files=[item.path for item in candidates],
                    focus_locations=[(item.path, item.line) for item in candidates],
                ),
                "candidate_findings": [
                    {"finding_index": index, **item.to_dict()}
                    for index, item in enumerate(candidates)
                ],
                "critic_decisions": critic_decisions,
                "worker_results": list(worker_results.values()),
                "instruction": (
                    "Return the indices that should be published. Resolve critic objections "
                    "explicitly and prefer changed-line tool evidence."
                ),
                "recalled_memory": memory_context,
            }, run.suite, run.ledger, session.task_id,
        )
        if "accepted_finding_indices" not in decision:
            decision["accepted_finding_indices"] = [
                int(item["finding_index"])
                for item in critic_decisions if item.get("accepted")
            ]
        return decision

    def _run_assignments(
        self, session: ReviewSession, run, assignments, scanner_findings,
        revision_round, memory_context,
    ) -> Dict[str, dict]:
        """One sub-node per assignment: a restart re-runs only the worker that died."""
        results: Dict[str, dict] = {}
        pending = []
        for assignment in assignments:
            run_id = str(assignment.get("run_id") or assignment["assignment_id"])
            done = session.runtime.checkpoint(session.sub(EXECUTING, "work:%s" % run_id))
            if done is not None:
                results[run_id] = done["result"]
            else:
                pending.append((run_id, assignment))
        if not pending:
            return results

        def execute(run_id, assignment):
            return session.runtime.run(
                session.sub(EXECUTING, "work:%s" % run_id),
                lambda: {"result": self._worker_result(
                    session, run, assignment, run_id, scanner_findings,
                    revision_round, memory_context,
                )},
                "%s is working on %s" % (assignment["worker"], run_id),
                retries=0,
            )

        with ThreadPoolExecutor(max_workers=max(1, len(pending))) as pool:
            futures = {
                pool.submit(execute, run_id, assignment): run_id
                for run_id, assignment in pending
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()["result"]
        return results

    def _worker_result(
        self, session, run, assignment, run_id, scanner_findings,
        revision_round, memory_context,
    ) -> dict:
        """Always returns a result record: a failed worker is data, not an outage."""
        parsed, ledger, available_skills = session.parsed, run.ledger, run.skills
        validated_skill_names = {
            name for name in assignment.get("skills") or []
            if name in (available_skills or {})
            and available_skills[name].source == "evolved-db"
        }
        try:
            raw_result = self._run_worker(
                session, run, assignment, scanner_findings, memory_context,
            )
            findings = parse_findings(
                raw_result, parsed, assignment["worker"], validated_skill_names,
            )
            result = {
                "assignment_id": assignment["assignment_id"],
                "run_id": run_id, "worker": assignment["worker"],
                "revision_round": revision_round, "status": "completed",
                "findings": [item.to_dict() for item in findings], "error": "",
                "evidence_resolutions": [
                    {
                        "evidence_id": str(item.get("evidence_id", ""))[:200],
                        "status": str(item.get("status", ""))[:40],
                        "explanation": str(item.get("explanation", ""))[:2000],
                        "supporting_evidence_ids": [
                            str(value)[:200]
                            for value in item.get("supporting_evidence_ids") or []
                        ][:20],
                    }
                    for item in raw_result.get("evidence_resolutions") or []
                    if isinstance(item, dict)
                ][:20],
            }
        except Exception as exc:
            result = {
                "assignment_id": assignment["assignment_id"],
                "run_id": run_id, "worker": assignment["worker"],
                "revision_round": revision_round, "status": "failed",
                "findings": [], "error": str(exc)[:1000],
            }
        ledger.trace(
            "lead-session", "worker_reported",
            assignment_id=assignment["assignment_id"], run_id=run_id,
            worker=assignment["worker"], status=result["status"],
            findings=len(result["findings"]), revision_round=revision_round,
        )
        return result

    def _run_worker(
        self, session, run, assignment, scanner_findings, memory_context,
    ):
        parsed, suite, ledger = session.parsed, run.suite, run.ledger
        available_skills = run.skills
        worker = assignment["worker"]
        selected_skills = [
            available_skills[name]
            for name in assignment.get("skills") or []
            if name in (available_skills or {})
        ]
        prompt = SECURITY_PROMPT if worker == "security" else (
            RELIABILITY_PROMPT + "\n" + SECURITY_PROMPT.split("Final action:", 1)[-1]
        )
        prompt += RULE_ID_GUIDANCE
        if self.prompt_overlay:
            prompt += "\nActive validated prompt overlay:\n" + self.prompt_overlay
        if selected_skills:
            prompt += "\n\nActive Agent Skills:\n" + "\n\n".join(
                "<agent-skill name=\"%s\">\n%s\n</agent-skill>" % (
                    skill.name, skill.instructions,
                ) for skill in selected_skills
            )
        working_memory_supplier, observation_sink = self._memory_hooks(
            session.task_id, worker
        )
        role = AgentLoop(
            worker, prompt, self.client,
            self._token_budget(worker), self.default_time_budget,
            context_manager=self.context_manager,
            working_memory_supplier=working_memory_supplier,
            observation_sink=observation_sink,
            minimum_tool_calls=int(suite.repository_available),
            final_action_validator=lambda action: worker_final_validation_error(
                action, parsed,
            ),
        )
        context = {
            "lead_assignment": assignment,
            "lead_feedback": assignment.get("lead_feedback", ""),
            **self._model_diff(
                session.diff, session.task_id, "%s:assignment" % worker,
                focus_files=assignment.get("files") or parsed.files,
                risk_domains=assignment.get("risk_domains") or (),
            ),
            "changed_files": parsed.files,
            "scoreable_added_lines": [
                {"path": item.path, "line": item.line, "content": item.content}
                for item in parsed.added_lines[:200]
                if item.path in (assignment.get("files") or parsed.files)
            ],
            "scanner_findings": scanner_findings,
            "repository_context_available": suite.repository_available,
            "recalled_memory": memory_context or {"items": []},
            "active_agent_skills": [skill.runtime_entry() for skill in selected_skills],
            "instruction": (
                "Report only to the Lead. Return final findings with exact changed-line "
                "evidence and address every required_evidence item. A Finding path/line "
                "must come from scoreable_added_lines; read_file start_line does not "
                "renumber its content."
            ),
        }
        tools = suite.registry(worker, skill_tool_permissions(worker, selected_skills))
        if any(skill.resource_paths for skill in selected_skills):
            by_name = {skill.name: skill for skill in selected_skills}

            def read_skill_resource(skill: str, path: str):
                selected = by_name.get(skill)
                if selected is None:
                    raise PermissionError("Agent Skill was not selected for this assignment")
                return {
                    "skill": skill, "path": path,
                    "content": selected.read_resource(path),
                }

            tools.register(AgentTool(
                "read_skill_resource",
                "Read one supporting text resource from an active Agent Skill.",
                {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"}, "path": {"type": "string"},
                    },
                    "required": ["skill", "path"], "additionalProperties": False,
                },
                read_skill_resource,
            ))
        initial_observations = repository_preflight(
            assignment, parsed, tools,
            repository_available=suite.repository_available,
        )
        return role.run(
            json.dumps(context, ensure_ascii=False),
            tools, ledger, initial_observations=initial_observations,
        )

    def _scan_findings(self, diff, parsed, ledger, scanners=None):
        started = time.monotonic()
        findings = self.rules.review(diff, parsed)
        ledger.record_tool(
            "agentic-scanner", "local-rule-scanner", {"added_lines": len(parsed.added_lines)},
            True, int((time.monotonic() - started) * 1000),
            {"findings": len(findings)},
        )
        scanners = list(scanners or [])
        for scanner in scanners:
            scanner_started = time.monotonic()
            name = scanner_name(scanner.name)
            try:
                scanned = scanner.review(diff, parsed)
            except Exception as exc:
                ledger.record_tool(
                    "agentic-scanner", name,
                    {"added_lines": len(parsed.added_lines)}, False,
                    int((time.monotonic() - scanner_started) * 1000), error=str(exc),
                )
                continue
            ledger.record_tool(
                "agentic-scanner", name, {"added_lines": len(parsed.added_lines)},
                True, int((time.monotonic() - scanner_started) * 1000),
                {"findings": len(scanned)},
            )
            for finding in scanned:
                if not finding.evidence_refs:
                    finding.evidence_refs = [{
                        "evidence_id": "scanner:%s:%s:%s" % (
                            finding.rule_id, finding.path, finding.line
                        ),
                        "tool": "declarative-scanner", "scanner": name,
                    }]
                if finding.source == "unknown":
                    finding.source = "declarative-scanner:%s" % name
            findings.extend(scanned)
        findings = merge_findings(findings)
        ast_scans = attach_diff_ast_evidence(findings, parsed, ledger)
        return findings, [
            component(ComponentKind.TOOL_SCANNER, "local-rule-scanner"),
        ] + [
            component(ComponentKind.TOOL_SCANNER, scanner_name(item.name))
            for item in scanners
        ] + ([component(ComponentKind.TOOL_SCANNER, "diff-ast-analyze")] if ast_scans else [])

    def _model_diff(
        self, diff, context_key, label, focus_files=(), risk_domains=(),
        focus_locations=(),
    ):
        compressed = self.context_manager.compress_diff(
            diff, context_key, label, focus_files=focus_files,
            risk_domains=risk_domains, focus_locations=focus_locations,
        )
        return {
            "diff": self.context_manager.render_diff_view(compressed),
            "diff_context": self.context_manager.diff_metadata(compressed),
        }

    def _memory_hooks(self, task_id, role):
        """Return task-scoped Working Memory read/write hooks for one role loop."""
        def scope():
            with self._memory_scope_lock:
                return self._memory_scopes.get(task_id)

        def supplier():
            values = scope()
            if self.memory_manager is None or not values:
                return None
            tenant_id, repository = values
            # Lead is the authorized coordination point. Workers and Critic
            # only see their own transient observations, preserving the
            # hierarchy and Critic's independent review boundary.
            memories = self.memory_manager.recall_working(
                tenant_id, repository, task_id, limit=12,
                agent="" if role == "lead" else role,
            )
            if not memories:
                return None
            context = self.context_manager.format_memories(memories)
            context["trust"] = (
                "untrusted, task-scoped tool observations; verify before making a claim"
            )
            return context

        def sink(agent, observation):
            values = scope()
            if self.memory_manager is not None and values:
                self.memory_manager.remember_observation(
                    values[0], values[1], task_id, agent, observation,
                )

        return supplier, sink

    def _persist_task_memory(
        self, task_id, tenant_id, repository, findings, gated, files, collaboration,
    ):
        """Turn verified decisions into reusable episodes and clear Working Memory."""
        if self.memory_manager is None:
            return
        try:
            for finding in findings:
                gate = getattr(finding, "gate", {}) or {}
                self.memory_manager.remember_finding(
                    tenant_id, repository, task_id, finding.to_dict(),
                    bool(gate.get("passed")), gate.get("reasons") or (),
                )
            accepted = [
                {
                    "rule_id": item.rule_id, "path": item.path, "line": item.line,
                    "severity": item.severity.value, "confidence": item.confidence,
                }
                for item in gated.accepted[:50]
            ]
            self.memory_manager.consolidate_task(tenant_id, repository, task_id, {
                "schema_version": 1, "files": list(files)[:100],
                "accepted_findings": accepted,
                "rejected_findings": list(gated.rejected)[:50],
                "gate_checks": dict(gated.checks),
                "agent_roles": list(collaboration.get("roles") or []),
            })
        except Exception:
            # Memory must enrich a review, not turn a completed review into a failure.
            return

    def _run_lead(self, phase, payload, suite, ledger, context_key=""):
        working_memory_supplier, observation_sink = self._memory_hooks(context_key, "lead")
        role = AgentLoop(
            "lead", LEAD_PROMPT + (
                ("\nActive validated prompt overlay:\n" + self.prompt_overlay)
                if self.prompt_overlay else ""
            ), self.client, self._token_budget("lead"), self.default_time_budget,
            context_manager=self.context_manager,
            working_memory_supplier=working_memory_supplier,
            observation_sink=observation_sink,
        )
        context = {"phase": phase, **payload}
        ledger.trace("lead-session", "lead_activated", phase=phase)
        result = role.run(
            json.dumps(context, ensure_ascii=False),
            suite.registry("lead", ROLE_PERMISSIONS["lead"]), ledger,
        )
        ledger.trace("lead-session", "lead_completed", phase=phase)
        return result

    def _run_critic(
        self, diff, candidates, objective, suite, ledger, context_key="",
        memory_context=None,
    ):
        blinded = [
            {
                "finding_index": index, "rule_id": item.rule_id,
                "severity": item.severity.value, "title": item.title,
                "explanation": item.explanation, "path": item.path,
                "line": item.line, "evidence": item.evidence,
                "evidence_refs": item.evidence_refs, "call_chain": item.call_chain,
                "fix": item.fix, "test": item.test, "confidence": item.confidence,
            }
            for index, item in enumerate(candidates)
        ]
        working_memory_supplier, observation_sink = self._memory_hooks(context_key, "critic")
        role = AgentLoop(
            "critic", CRITIC_PROMPT + (
                ("\nActive validated prompt overlay:\n" + self.prompt_overlay)
                if self.prompt_overlay else ""
            ), self.client, self._token_budget("critic"), self.default_time_budget,
            context_manager=self.context_manager,
            working_memory_supplier=working_memory_supplier,
            observation_sink=observation_sink,
            minimum_tool_calls=int(bool(candidates)),
        )
        tools = suite.registry("critic", ROLE_PERMISSIONS["critic"])
        initial_observations = []
        seen_locations = set()
        for index, item in enumerate(candidates):
            location = (item.path, item.line)
            if location in seen_locations or len(seen_locations) >= 12:
                continue
            seen_locations.add(location)
            for tool_name, arguments in (
                ("changed_line", {"path": item.path, "line": item.line}),
                ("read_file", {
                    "path": item.path,
                    "start_line": max(1, item.line - 12),
                    "end_line": item.line + 12,
                }),
            ):
                if tool_name not in tools.names():
                    continue
                if tool_name == "read_file" and not suite.repository_available:
                    continue
                try:
                    value = tools.invoke(tool_name, arguments)
                    initial_observations.append({
                        "step": 0, "tool": tool_name, "ok": True,
                        "result": value,
                        "reason": "candidate %d exact-location verification" % index,
                    })
                except Exception as exc:
                    initial_observations.append({
                        "step": 0, "tool": tool_name, "ok": False,
                        "error": str(exc)[:1000],
                        "reason": "candidate %d exact-location verification" % index,
                    })
        return role.run(
            json.dumps({
                "lead_assignment": objective or (
                    "Blindly challenge every candidate and report explicit decisions."
                ),
                **self._model_diff(
                    diff, context_key, "critic:blind-review",
                    focus_files=[item.path for item in candidates],
                    focus_locations=[(item.path, item.line) for item in candidates],
                ),
                "candidates": blinded,
                "recalled_memory": memory_context or {"items": []},
            }, ensure_ascii=False),
            tools, ledger, initial_observations=initial_observations,
        )
