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
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..agents.loop import AgentLoop, collect_evidence
from ..agents.parsing import (
    assignment_requirements, downgrade_unsupported_conclusions, normalize_hypotheses,
    normalize_requirement_resolutions, parse_findings,
    worker_final_validation_error, worker_handoffs,
)
from ..agents.prompts import (
    CRITIC_PROMPT, LEAD_PROMPT, RELIABILITY_PROMPT, ROLE_PERMISSIONS,
    RULE_ID_GUIDANCE, SECURITY_PROMPT,
)
from ..core.diff_parser import ParsedDiff
from ..core.gates import FindingGate
from ..core.finding_policy import (
    claim_specific_high_risk_evidence_refs, repository_evidence_refs,
)
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
    partition_publication, public_decision, resolve_lead_reviews, restore_findings, scanner_name,
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
                    # Scanner output remains a deterministic publication
                    # baseline, but does not seed the Agent's hypothesis search.
                    session, run, [], memory_context,
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
                session, run, delegations, [], 0,
                memory_context,
            )
            worker_history = [dict(item) for item in worker_results.values()]
            max_rounds = self._max_revision_rounds()
            assessments: List[dict] = []
            revision_results: Dict[str, dict] = {}
            stop_reason = ""
            critic_objective = ""
            pre_critic_review: Dict[str, Any] = {}
            revision_performed = False
            for index in range(max_rounds + 1):
                candidates = candidates_from(rule_findings, worker_results)
                critic_preview = []
                if (
                    index == 0 and max_rounds > 0
                    and "critic" in run.enabled and candidates
                ):
                    preview = session.runtime.run(
                        session.sub(EXECUTING, "critic-preview"),
                        lambda candidates=candidates: self._critic(
                            session, run, candidates,
                            "Identify exact missing publication premises for a possible "
                            "targeted Worker evidence revision.",
                            memory_context,
                        ),
                        "Critic is identifying candidate proof gaps",
                    )
                    candidates = restore_findings(preview["candidates"])
                    critic_preview = list(preview["decisions"])
                    pre_critic_review = {
                        "candidates": [item.to_dict() for item in candidates],
                        "decisions": critic_preview,
                    }
                assessment = session.runtime.run(
                    session.sub(EXECUTING, "assess-%d" % index),
                    lambda candidates=candidates, index=index: {
                        "decision": public_decision(self._assess(
                            session, run, delegations, worker_results, candidates,
                            index, max_rounds, memory_context, critic_preview,
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
                    revision["handoff_ids"] = request.get("handoff_ids") or []
                    revision["evidence_targets"] = request.get("evidence_targets") or []
                    revision["prior_worker_result"] = worker_results.get(
                        request["assignment_id"], {}
                    )
                    revisions.append(revision)
                revised = self._run_assignments(
                    session, run, revisions, [], index + 1,
                    memory_context,
                )
                revision_performed = True
                for revision in revisions:
                    key = revision["run_id"]
                    result = self._merge_worker_revision(
                        worker_results.get(revision["assignment_id"], {}),
                        revised[key],
                    )
                    revision_results[key] = result
                    worker_history.append(dict(result))
                    worker_results[revision["assignment_id"]] = result
                    run.ledger.trace(
                        "lead-session", "revision_completed",
                        assignment_id=revision["assignment_id"],
                        worker=revision["worker"], round=index + 1,
                        status=result["status"],
                    )
            return {
                "worker_results": worker_results,
                "worker_history": worker_history,
                "lead_assessments": assessments,
                "revision_results": revision_results,
                "stop_reason": stop_reason or "lead-final",
                "critic_objective": critic_objective,
                "pre_critic_review": (
                    {} if revision_performed else pre_critic_review
                ),
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

            reusable_critic = executed.get("pre_critic_review") or {}
            if reusable_critic and candidates:
                candidates = restore_findings(
                    reusable_critic.get("candidates") or []
                )
                critic_decisions = list(
                    reusable_critic.get("decisions") or []
                )
            elif "critic" in run.enabled and candidates:
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
                )), "candidates": [item.to_dict() for item in candidates]},
                "Lead is arbitrating the final finding set",
            )
            lead_final = arbitrated["decision"]
            candidates = restore_findings(arbitrated.get("candidates") or [
                item.to_dict() for item in candidates
            ])
            lead_accepted = apply_lead_final(lead_final, candidates)
            accepted, suggestions, publication_decisions = partition_publication(
                rule_findings, candidates, lead_accepted, critic_decisions,
                run.suite.repository_available,
                critic_required="critic" in run.enabled,
                publish_unverified_suggestions=bool(
                    self.structured_config.get("publish_unverified_suggestions", True)
                ),
                lead_reviews=lead_final.get("publication_reviews") or [],
            )
            collaboration = self._collaboration(
                run, planned, executed, lead_final, publication_decisions,
                critic_decisions, planned["scanner_findings"],
                before_critic, len(accepted), suggestions,
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
        scanner_findings, before_critic, accepted_count, suggestions,
    ) -> Dict[str, Any]:
        delegations = planned["delegations"]
        public_worker = lambda item: {
            key: value for key, value in dict(item).items()
            if not str(key).startswith("_")
        }
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
            "worker_results": [
                public_worker(item) for item in executed["worker_results"].values()
            ],
            "worker_history": [
                public_worker(item) for item in executed.get("worker_history") or []
            ],
            "revision_results": [
                public_worker(item) for item in executed["revision_results"].values()
            ],
            "scanner_findings": len(scanner_findings),
            "scanner_finding_details": list(scanner_findings),
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
        try:
            result = self._run_critic(
                session.diff, candidates, objective, run.suite, run.ledger,
                session.task_id, memory_context,
            )
        except Exception as exc:
            # A malformed/timeout Critic response must not erase the Worker
            # audit trail or fail the whole review. Fail closed: no model
            # candidate is publication-ready without an explicit decision.
            run.ledger.trace(
                "critic", "critic_failed_closed", error=str(exc)[:1000],
                candidates=len(candidates),
            )
            return {
                "candidates": [item.to_dict() for item in candidates],
                "decisions": [{
                    "finding_index": index, "accepted": False,
                    "publication_ready": False,
                    "introduced_by_diff": False, "reproducible": False,
                    "evidence_sufficient": False,
                    "would_comment_on_real_pr": False,
                    "recommended_confidence_adjustment": 0.0,
                    "objections": [
                        "Critic failed closed: " + str(exc)[:500]
                    ],
                } for index, _item in enumerate(candidates)],
            }
        kept, decisions = apply_critic(result, candidates)
        return {
            "candidates": [item.to_dict() for item in kept],
            "decisions": decisions,
        }

    def _assess(
        self, session, run, delegations, worker_results, candidates,
        index, max_rounds, memory_context, critic_preview=None,
    ) -> Dict[str, Any]:
        critic_targets = self._pending_critic_review_items(
            candidates, critic_preview or [], worker_results,
        )
        decision = self._run_lead(
            "assess-workers", {
                **self._model_diff(
                    session.diff, session.task_id, "lead:assess-workers",
                    focus_files=[item.path for item in candidates],
                ),
                "assignments": delegations,
                "worker_results": list(worker_results.values()),
                "candidate_findings": [item.to_dict() for item in candidates],
                "pre_revision_critic": critic_targets,
                "revision_round": index,
                "remaining_revision_rounds": max_rounds - index,
                "recalled_memory": memory_context,
            }, run.suite, run.ledger, session.task_id,
        )
        return self._complete_assessment_protocol(
            decision, delegations, worker_results, max_rounds - index,
            candidates, critic_preview,
        )

    @staticmethod
    def _merge_worker_revision(previous: dict, current: dict) -> dict:
        """Make evidence revisions monotonic unless they prove a refutation.

        A revision is a refinement of the same assignment, not a fresh vote.
        Model truncation or an unresolved evidence mission must not erase an
        already structured candidate. A proof-backed refuted hypothesis at the
        candidate's exact location is the explicit removal path.
        """
        if not previous:
            return current
        merged = dict(current)
        if current.get("status") == "failed":
            merged["hypotheses"] = list(previous.get("hypotheses") or [])
            merged["requirement_resolutions"] = list(
                previous.get("requirement_resolutions") or []
            )
            merged["handoffs"] = list(previous.get("handoffs") or [])
        refuted_locations = {
            str(item.get("location") or "").strip()
            for item in current.get("hypotheses") or []
            if isinstance(item, dict)
            and item.get("status") == "refuted"
            and item.get("proof_kind")
        }
        prior_findings = [
            item for item in previous.get("findings") or []
            if "%s:%s" % (item.get("path", ""), item.get("line", ""))
            not in refuted_locations
        ]
        merged["findings"] = [
            item.to_dict() for item in merge_findings(restore_findings(
                prior_findings + list(current.get("findings") or [])
            ))
        ]

        def merge_evidence(key):
            by_id = {}
            for item in list(previous.get(key) or []) + list(current.get(key) or []):
                if not isinstance(item, dict):
                    continue
                identity = str(item.get("evidence_id") or "")
                if identity:
                    by_id[identity] = item
            return list(by_id.values())[:40]

        merged["evidence_inventory"] = merge_evidence("evidence_inventory")
        merged["_evidence_records"] = merge_evidence("_evidence_records")
        return merged

    @staticmethod
    def _pending_worker_review_items(worker_results) -> List[dict]:
        """Return risks that still need routing or a bounded evidence pass.

        High-risk unknowns always qualify. A normal-risk unknown qualifies only
        when the Worker anchored it to a concrete changed location; that keeps a
        generic ``protocol-fallback`` from spending a revision while preserving
        a real, testable hypothesis. Structured Findings also get one evidence
        pass when they cite source context but lack behavioral/cross-call proof.
        """
        handled = {
            str(value)
            for result in worker_results.values()
            if result.get("status") == "completed"
            for value in result.get("handled_handoff_ids") or []
        }
        values, seen = [], set()
        for result in worker_results.values():
            for item in result.get("handoffs") or []:
                item_id = str(item.get("handoff_id") or "")
                if not item_id or item_id in handled or item_id in seen:
                    continue
                seen.add(item_id)
                values.append(dict(item))
            for hypothesis in result.get("hypotheses") or []:
                if hypothesis.get("status") != "unresolved":
                    continue
                location = str(hypothesis.get("location") or "").strip()
                location_path = location.rsplit(":", 1)[0].replace("\\", "/").lower()
                if any(
                    part in location_path
                    for part in ("/test/", "/tests/", "tests/", "test_")
                ):
                    continue
                concrete_location = bool(re.match(r"^.+:\d+$", location))
                high_risk = hypothesis.get("risk_level") == "high"
                if not high_risk and (
                    not concrete_location
                    or not result.get("repository_context_available", False)
                ):
                    continue
                item_id = "%s:%s" % (
                    result.get("assignment_id", ""),
                    hypothesis.get("hypothesis_id", "hypothesis"),
                )
                if item_id in handled or item_id in seen:
                    continue
                seen.add(item_id)
                target_worker = str(hypothesis.get("target_worker") or "")
                domain = str(hypothesis.get("domain") or "").lower()
                if not target_worker and domain in {
                    "correctness", "reliability", "correctness-reliability",
                }:
                    target_worker = "correctness-reliability"
                if not target_worker:
                    target_worker = str(result.get("worker") or "")
                values.append({
                    "handoff_id": item_id,
                    "source_assignment_id": result.get("assignment_id", ""),
                    "source_worker": result.get("worker", ""),
                    "target_worker": target_worker,
                    "claim": hypothesis.get("claim", ""),
                    "location": hypothesis.get("location", ""),
                    "explanation": hypothesis.get("explanation", ""),
                    "required_proof": hypothesis.get("required_proof", ""),
                    "supporting_evidence_ids": hypothesis.get(
                        "supporting_evidence_ids", []
                    ),
                    "origin": hypothesis.get("origin", "worker"),
                    "kind": (
                        "high-risk-unresolved" if high_risk
                        else "evidence-gap-unresolved"
                    ),
                })
            for finding in restore_findings(result.get("findings") or []):
                if (
                    not result.get("repository_context_available", False)
                    or str(finding.source or "").startswith("agent-skill:")
                ):
                    continue
                repository_refs = repository_evidence_refs(finding)
                claim_refs = claim_specific_high_risk_evidence_refs(finding)
                if repository_refs and claim_refs:
                    continue
                item_id = "%s:evidence:%s:%s:%s" % (
                    result.get("assignment_id", ""), finding.path,
                    finding.line, finding.rule_id,
                )
                if item_id in handled or item_id in seen:
                    continue
                seen.add(item_id)
                missing = []
                if not repository_refs:
                    missing.append("repository trigger/reachability evidence")
                if not claim_refs:
                    missing.append("behavioral witness or corroborated cross-call path")
                values.append({
                    "handoff_id": item_id,
                    "source_assignment_id": result.get("assignment_id", ""),
                    "source_worker": result.get("worker", ""),
                    "target_worker": result.get("worker", ""),
                    "claim": "%s. %s" % (finding.title, finding.explanation),
                    "location": "%s:%s" % (finding.path, finding.line),
                    "explanation": (
                        "The Worker produced a Finding, but its proof bundle is not yet "
                        "strong enough for blind publication review."
                    ),
                    "required_proof": "Obtain " + " and ".join(missing) + ".",
                    "supporting_evidence_ids": [
                        str(item.get("evidence_id"))
                        for item in finding.evidence_refs
                        if isinstance(item, dict) and item.get("evidence_id")
                    ],
                    "origin": "evidence-gap",
                    "kind": "evidence-gap-finding",
                })
        priority = {
            "evidence-gap-finding": 0,
            "high-risk-unresolved": 1,
            "evidence-gap-unresolved": 2,
        }
        selected, selected_keys, per_worker = [], set(), {}
        for item in sorted(
            values,
            key=lambda value: (
                priority.get(str(value.get("kind") or ""), 0),
                str(value.get("location") or ""),
                0 if value.get("target_worker") == "correctness-reliability" else 1,
                str(value.get("handoff_id") or ""),
            ),
        ):
            worker = str(item.get("target_worker") or "")
            location = str(item.get("location") or "").strip()
            key = location or (worker, str(item.get("handoff_id") or ""))
            if key in selected_keys or per_worker.get(worker, 0) >= 2:
                continue
            selected_keys.add(key)
            per_worker[worker] = per_worker.get(worker, 0) + 1
            selected.append(item)
        return selected

    @staticmethod
    def _target_assignment(item, delegations) -> Optional[dict]:
        targets = [
            assignment for assignment in delegations
            if assignment.get("worker") == item.get("target_worker")
        ]
        if not targets:
            return None
        location = str(item.get("location") or "")
        path = location.rsplit(":", 1)[0] if ":" in location else location
        for assignment in targets:
            if path and path in (assignment.get("files") or []):
                return assignment
        return targets[0]

    @staticmethod
    def _pending_critic_review_items(candidates, critic_decisions, worker_results):
        """Turn one or two exact Critic gaps into optional revision targets."""
        ownership = []
        for result in worker_results.values():
            for finding in result.get("findings") or []:
                if not isinstance(finding, dict):
                    continue
                ownership.append((
                    str(finding.get("path") or ""),
                    int(finding.get("line") or 0),
                    str(result.get("assignment_id") or ""),
                    str(result.get("worker") or ""),
                ))
        items = []
        for decision in critic_decisions or []:
            if decision.get("verdict") != "inconclusive":
                continue
            try:
                index = int(decision.get("finding_index"))
                finding = candidates[index]
            except (IndexError, TypeError, ValueError):
                continue
            if finding.severity.value not in {"high", "critical"}:
                continue
            missing_obligations = [
                dict(item) for item in decision.get("missing_proof") or []
                if isinstance(item, dict)
                and str(item.get("obligation") or "").strip()
            ]
            missing_premises = [
                dict(item) for item in decision.get("missing_premises") or []
                if isinstance(item, dict)
                and str(item.get("premise") or "").strip()
            ]
            missing = missing_premises or missing_obligations
            if not 1 <= len(missing) <= 2:
                continue
            owner = next(
                (
                    value for value in ownership
                    if value[0] == finding.path and value[1] == finding.line
                ),
                None,
            )
            if owner is None:
                continue
            supporting = list(dict.fromkeys(
                [
                    str(ref.get("evidence_id"))
                    for ref in finding.evidence_refs
                    if isinstance(ref, dict) and ref.get("evidence_id")
                ] + [
                    str(evidence_id)
                    for state in decision.get("proof_state") or []
                    if isinstance(state, dict)
                    for evidence_id in state.get("supporting_evidence_ids") or []
                    if str(evidence_id).strip()
                ]
            ))[:8]
            item_id = "critic:%d:%s:%d" % (index, finding.path, finding.line)
            items.append({
                "handoff_id": item_id,
                "source_assignment_id": owner[2],
                "source_worker": owner[3],
                "target_worker": owner[3],
                "claim": "%s. %s" % (finding.title, finding.explanation),
                "location": "%s:%d" % (finding.path, finding.line),
                "explanation": (
                    "The Critic preserved the candidate but identified a bounded proof gap."
                ),
                "required_proof": " ".join(
                    str(item.get("required_proof") or "").strip()
                    for item in missing
                    if str(item.get("required_proof") or "").strip()
                )[:1000],
                "supporting_evidence_ids": supporting,
                "proof_state": list(decision.get("proof_state") or [])[:6],
                "missing_obligations": [
                    str(item.get("obligation"))[:80]
                    for item in missing_obligations
                ],
                "missing_premises": missing_premises,
                "origin": "critic-proof-gap",
                "kind": "critic-proof-gap",
                "severity": finding.severity.value,
            })
        return items

    @classmethod
    def _complete_assessment_protocol(
        cls, raw, delegations, worker_results, remaining_rounds,
        candidates=None, critic_decisions=None,
    ) -> Dict[str, Any]:
        """Make Lead handling of handoffs explicit without adding another role."""
        decision = dict(raw or {})
        pending = cls._pending_critic_review_items(
            candidates or [], critic_decisions or [], worker_results,
        ) + cls._pending_worker_review_items(worker_results)
        pending_by_id = {
            str(item.get("handoff_id")): item for item in pending
            if item.get("handoff_id")
        }
        # Lead prose can be useful, but a revision tied to protocol items is
        # rebuilt below from the bounded queue. This prevents the model from
        # copying every hypothesis into one unexecutable mega-assignment.
        requests = []
        if remaining_rounds > 0:
            for item in decision.get("revision_requests") or []:
                if not isinstance(item, dict) or item.get("handoff_ids"):
                    continue
                guidance = str(item.get("guidance") or "").strip()
                if not guidance:
                    continue
                requests.append({
                    "assignment_id": str(item.get("assignment_id") or "")[:100],
                    "worker": str(item.get("worker") or "")[:100],
                    "guidance": guidance[:600],
                    "required_evidence": [
                        str(value)[:200]
                        for value in item.get("required_evidence") or []
                        if str(value).strip()
                    ][:2],
                    "handoff_ids": [], "evidence_targets": [],
                })
        explicitly_requested_assignments = {item["assignment_id"] for item in requests}
        provided = {}
        for request in decision.get("revision_requests") or []:
            if not isinstance(request, dict) or not str(request.get("guidance") or "").strip():
                continue
            for item_id in request.get("handoff_ids") or []:
                if str(item_id) in pending_by_id:
                    provided[str(item_id)] = {
                        "handoff_id": str(item_id), "action": "revise",
                        "target_assignment_id": str(request.get("assignment_id") or ""),
                        "reason": str(request["guidance"])[:1000], "source": "lead",
                    }
        for item in decision.get("handoff_decisions") or []:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("handoff_id") or "")
            action = str(item.get("action") or "").strip().lower()
            reason = str(item.get("reason") or "").strip()
            if item_id in pending_by_id and action in {"revise", "defer"} and reason:
                provided[item_id] = {
                    "handoff_id": item_id,
                    "action": action,
                    "target_assignment_id": str(
                        item.get("target_assignment_id") or ""
                    )[:100],
                    "reason": reason[:1000],
                    "source": "lead",
                }

        def ensure_request(review_item, target, item_id):
            request = next(
                (
                    item for item in requests
                    if str(item.get("assignment_id") or "")
                    == target["assignment_id"]
                ),
                None,
            )
            if request is None:
                request = {
                    "assignment_id": target["assignment_id"],
                    "worker": target["worker"],
                    "guidance": "",
                    "required_evidence": [],
                    "handoff_ids": [],
                    "evidence_targets": [],
                }
                requests.append(request)
            handoff_ids = [
                str(value) for value in request.get("handoff_ids") or []
                if str(value).strip()
            ]
            if item_id not in handoff_ids:
                handoff_ids.append(item_id)
            request["handoff_ids"] = handoff_ids[:20]
            guidance = (
                "Resolve Worker evidence target %s: %s %s Evidence still needed: %s "
                "Build a three-part proof: allowed trigger, unguarded reachability, and "
                "deterministic failure/wrong-result contract. Preserve the prior conclusion "
                "unless invariant-level counter-evidence refutes it."
                % (
                    item_id, str(review_item.get("claim") or "").strip(),
                    str(review_item.get("explanation") or "").strip(),
                    str(review_item.get("required_proof") or "").strip(),
                )
            ).strip()
            if review_item.get("origin") == "protocol-downgrade":
                guidance += (
                    " The protocol rejected the prior certainty. If the described failure "
                    "path is reachable, return a structured Finding anchored to an exact added "
                    "line. Refute it only with invariant-level counter-evidence."
                )
            prior_guidance = str(request.get("guidance") or "").strip()
            if guidance not in prior_guidance:
                request["guidance"] = (
                    (prior_guidance + "\n" + guidance).strip()[:2000]
                )
            required = [
                str(value) for value in request.get("required_evidence") or []
                if str(value).strip()
            ]
            required_item = "Resolve evidence target %s: %s" % (
                item_id,
                str(review_item.get("required_proof") or review_item.get("claim") or "")
            )
            if required_item not in required:
                required.append(required_item)
            request["required_evidence"] = required[:20]
            location = str(review_item.get("location") or "").strip()
            path, separator, rendered_line = location.rpartition(":")
            if separator and path and rendered_line.isdigit():
                targets = [
                    dict(value) for value in request.get("evidence_targets") or []
                    if isinstance(value, dict)
                ]
                if not any(
                    str(value.get("evidence_target_id") or "") == item_id
                    for value in targets
                ):
                    targets.append({
                        "evidence_target_id": item_id,
                        "path": path,
                        "line": int(rendered_line),
                        "claim": str(review_item.get("claim") or "")[:2000],
                        "required_proof": str(
                            review_item.get("required_proof") or ""
                        )[:1000],
                        "kind": str(review_item.get("kind") or "")[:100],
                        "supporting_evidence_ids": [
                            str(value)[:200]
                            for value in review_item.get(
                                "supporting_evidence_ids"
                            ) or []
                            if str(value).strip()
                        ][:8],
                        "proof_state": [
                            dict(value)
                            for value in review_item.get("proof_state") or []
                            if isinstance(value, dict)
                        ][:6],
                        "missing_obligations": [
                            str(value)[:80]
                            for value in review_item.get(
                                "missing_obligations"
                            ) or []
                            if str(value).strip()
                        ][:6],
                        "missing_premises": [
                            dict(value)
                            for value in review_item.get("missing_premises") or []
                            if isinstance(value, dict)
                        ][:2],
                    })
                request["evidence_targets"] = targets[:12]

        routed = []
        for item_id, item in pending_by_id.items():
            target = cls._target_assignment(item, delegations)
            current = provided.get(item_id)
            if current and current["action"] == "defer":
                routed.append(current)
                continue
            explicitly_requested = bool(current and current["action"] == "revise") or bool(
                target is not None and target["assignment_id"] in explicitly_requested_assignments
            )
            if remaining_rounds > 0 and target is not None and explicitly_requested:
                ensure_request(item, target, item_id)
                routed.append({
                    "handoff_id": item_id,
                    "action": "revise",
                    "target_assignment_id": target["assignment_id"],
                    "reason": (
                        current["reason"] if current
                        else "Lead requested revision of this assignment."
                    ),
                    "source": "lead",
                })
            else:
                reason = (
                    "No revision round remains; preserve this item as unresolved."
                    if remaining_rounds <= 0
                    else "Lead did not request revision; preserve the evidence gap."
                    if target is not None else "No enabled assignment exists for target Worker %s."
                    % item.get("target_worker", "")
                )
                routed.append({
                    "handoff_id": item_id,
                    "action": "defer",
                    "target_assignment_id": "",
                    "reason": current["reason"] if current else reason,
                    "source": current["source"] if current else "protocol-guard",
                })
        decision["revision_requests"] = requests
        decision["handoff_decisions"] = routed
        return decision

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
                "instruction": (
                    "Return publishable indices. For an inconclusive Critic verdict you may "
                    "inspect the missing premise with repository tools and provide evidence_reviews "
                    "to complete its proof. Selection alone cannot waive missing evidence. "
                    "Do not repeat completed reviews or reopen verified counter-proofs."
                ),
                "recalled_memory": memory_context,
            }, run.suite, run.ledger, session.task_id,
        )
        if "accepted_finding_indices" not in decision:
            decision["accepted_finding_indices"] = [
                int(item["finding_index"])
                for item in critic_decisions if item.get("accepted")
            ]
        decision["publication_reviews"] = resolve_lead_reviews(
            decision, candidates, critic_decisions,
        )
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
            hypotheses = normalize_hypotheses(raw_result.get("hypotheses"))
            requirement_resolutions = normalize_requirement_resolutions(
                raw_result.get("requirement_resolutions")
            )
            evidence_records = list(collect_evidence(
                raw_result.get("_observations") or []
            ).values())
            evidence_inventory = [
                {
                    "evidence_id": item.get("evidence_id", ""),
                    "tool": item.get("tool", ""),
                    "origin": item.get("origin", ""),
                    "output_preview": item.get("output_preview", ""),
                }
                for item in evidence_records
            ]
            result = {
                "assignment_id": assignment["assignment_id"],
                "run_id": run_id, "worker": assignment["worker"],
                "revision_round": revision_round, "status": "completed",
                "repository_context_available": bool(run.suite.repository_available),
                "findings": [item.to_dict() for item in findings], "error": "",
                "assignment_requirements": assignment_requirements(assignment),
                "requirement_resolutions": requirement_resolutions,
                "hypotheses": hypotheses,
                "handoffs": worker_handoffs(
                    hypotheses, requirement_resolutions, assignment,
                ),
                "handled_handoff_ids": [
                    str(value)[:200]
                    for value in assignment.get("handoff_ids") or []
                    if str(value).strip()
                ][:20],
                "evidence_inventory": evidence_inventory,
                # Kept private for the next revision invocation. Public reports
                # expose only evidence_inventory and Finding citations.
                "_evidence_records": evidence_records[:40],
                "protocol_downgrades": list(
                    raw_result.get("protocol_downgrades") or []
                )[:30],
                "evidence_resolutions": [
                    {
                        "evidence_id": str(item.get("evidence_id", ""))[:200],
                        "status": str(item.get("status", ""))[:40],
                        "explanation": str(item.get("explanation", ""))[:2000],
                        "proof_kind": str(item.get("proof_kind", ""))[:80],
                        "required_proof": str(item.get("required_proof", ""))[:1000],
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
                "repository_context_available": bool(run.suite.repository_available),
                "findings": [], "error": str(exc)[:1000],
                "assignment_requirements": assignment_requirements(assignment),
                "requirement_resolutions": [], "hypotheses": [],
                "handoffs": [], "handled_handoff_ids": [],
                "evidence_inventory": [], "_evidence_records": [],
                "evidence_resolutions": [],
                "protocol_downgrades": [],
            }
        ledger.trace(
            "lead-session", "worker_reported",
            assignment_id=assignment["assignment_id"], run_id=run_id,
            worker=assignment["worker"], status=result["status"],
            findings=len(result["findings"]), revision_round=revision_round,
        )
        return result

    @staticmethod
    def _worker_validator(parsed, assignment, repository_available):
        def validate(action):
            downgrades = downgrade_unsupported_conclusions(action)
            if downgrades:
                action.setdefault("protocol_downgrades", []).extend(downgrades)
            error = worker_final_validation_error(
                action, parsed, demand_hypotheses=True,
                assignment=assignment,
                repository_available=repository_available,
            )
            if error and not (action.get("findings") or action.get("hypotheses")):
                action["hypotheses"] = [{
                    "hypothesis_id": "protocol-fallback",
                    "claim": (
                        "Worker returned no traceable conclusion for assignment: "
                        + str(assignment.get("objective") or "assigned review")
                    )[:2000],
                    "status": "unresolved", "risk_level": "normal",
                    "explanation": (
                        "The final response contained neither a Finding nor a settled "
                        "hypothesis; the protocol preserves this as unknown, not safe."
                    ),
                    "required_proof": (
                        "Re-run the assigned investigation and provide a Finding, invariant-level "
                        "refutation, or explicit domain handoff."
                    ),
                    "origin": "protocol-fallback",
                }]
                action["requirement_resolutions"] = [{
                    "requirement_id": item["requirement_id"],
                    "status": "unresolved",
                    "explanation": (
                        "The Worker did not return a traceable answer to this Lead requirement."
                    ),
                    "required_proof": item["question"],
                } for item in assignment_requirements(assignment)]
                error = worker_final_validation_error(
                    action, parsed, demand_hypotheses=True,
                    assignment=assignment,
                    repository_available=repository_available,
                )
            return error

        return validate

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
            # Repository preflight already supplies factual observations. The
            # Worker remains free to call a tool for a missing proof, but a
            # completed answer is not rejected merely to manufacture a turn.
            minimum_tool_calls=0,
            final_action_validator=self._worker_validator(
                parsed, assignment, suite.repository_available,
            ),
        )
        prior_worker_result = assignment.get("prior_worker_result") or {}
        evidence_target_ids = {
            str(value)
            for target in assignment.get("evidence_targets") or []
            if isinstance(target, dict)
            for value in target.get("supporting_evidence_ids") or []
            if str(value).strip()
        }
        prior_hypotheses = list(prior_worker_result.get("hypotheses") or [])
        focused_hypotheses = [
            item for item in prior_hypotheses
            if any(
                str(target.get("evidence_target_id") or "").endswith(
                    ":" + str(item.get("hypothesis_id") or "")
                )
                for target in assignment.get("evidence_targets") or []
                if isinstance(target, dict)
            )
        ] or prior_hypotheses[:4]
        context = {
            "lead_assignment": {
                key: value for key, value in assignment.items()
                if key != "prior_worker_result"
            },
            "assignment_requirements": assignment_requirements(assignment),
            "lead_feedback": assignment.get("lead_feedback", ""),
            "evidence_mission": list(assignment.get("evidence_targets") or []),
            "prior_worker_result": ({
                "findings": [
                    {
                        key: value for key, value in item.items()
                        if key != "evidence_refs"
                    }
                    for item in prior_worker_result.get("findings") or []
                    if isinstance(item, dict)
                ],
                "hypotheses": focused_hypotheses[:4],
                "requirement_resolutions": list(
                    prior_worker_result.get("requirement_resolutions") or []
                ),
                "evidence_inventory": [{
                    "evidence_id": item.get("evidence_id", ""),
                    "tool": item.get("tool", ""),
                    "origin": item.get("origin", ""),
                    "output_preview": str(item.get("output_preview", ""))[:500],
                } for item in prior_worker_result.get("evidence_inventory") or []
                if str(item.get("evidence_id") or "") in evidence_target_ids
                ][:8],
            } if prior_worker_result else {}),
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
                "evidence and address every assignment_requirements ID exactly once. "
                "Use a handoff hypothesis for a credible risk owned by the other Worker. "
                "On an evidence revision, preserve the prior risk and cite prior evidence IDs "
                "that remain valid while filling the trigger/reachability/failure gaps. "
                "A Finding path/line "
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
        initial_observations = []
        for item in prior_worker_result.get("_evidence_records") or []:
            if not isinstance(item, dict) or not item.get("evidence_id"):
                continue
            if str(item.get("evidence_id") or "") not in evidence_target_ids:
                continue
            initial_observations.append({
                "step": 0, "tool": item.get("tool", "prior-worker-evidence"),
                "ok": True, "origin": "prior-worker",
                "result": {
                    "evidence_id": item.get("evidence_id"),
                    "tool": item.get("tool", ""),
                    "output": item.get("output"),
                },
                "reason": "evidence carried forward from the prior Worker pass",
            })
        initial_observations.extend(repository_preflight(
            assignment, parsed, tools,
            repository_available=suite.repository_available,
        ))
        return role.run(
            json.dumps(context, ensure_ascii=False),
            tools, ledger, initial_observations=initial_observations,
        )

    def _scan_findings(self, diff, parsed, ledger, scanners=None):
        started = time.monotonic()
        findings = self.rules.review(diff, parsed)
        for finding in findings:
            for evidence in finding.evidence_refs:
                if isinstance(evidence, dict):
                    evidence.setdefault("origin", "deterministic-scanner")
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
                        "origin": "deterministic-scanner",
                    }]
                else:
                    for evidence in finding.evidence_refs:
                        if isinstance(evidence, dict):
                            evidence.setdefault("origin", "deterministic-scanner")
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
            # Candidate-specific preflight is supplied below. Let the Critic
            # request more evidence only when it judges that evidence missing.
            minimum_tool_calls=0,
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
            symbols = list(dict.fromkeys(
                str(value.get("symbol") or "").strip()
                for value in item.call_chain or []
                if isinstance(value, dict) and str(value.get("symbol") or "").strip()
            ))[:2]
            evidence_calls = []
            if "locate_tests" in tools.names() and suite.repository_available:
                evidence_calls.append(("locate_tests", {
                    "path": item.path,
                    "symbol": symbols[0] if symbols else "",
                }))
            if "symbol" in tools.names() and suite.repository_available:
                evidence_calls.extend(
                    ("symbol", {"name": symbol_name}) for symbol_name in symbols
                )
            for tool_name, arguments in evidence_calls:
                try:
                    value = tools.invoke(tool_name, arguments)
                    initial_observations.append({
                        "step": 0, "tool": tool_name, "ok": True,
                        "result": value,
                        "reason": "candidate %d trigger and reachability preflight" % index,
                    })
                except Exception as exc:
                    initial_observations.append({
                        "step": 0, "tool": tool_name, "ok": False,
                        "error": str(exc)[:1000],
                        "reason": "candidate %d trigger and reachability preflight" % index,
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
