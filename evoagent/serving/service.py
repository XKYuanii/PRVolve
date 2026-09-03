import hashlib
import uuid
from typing import Any, Dict, Optional

from ..review.agentic import AgenticReviewer
from ..serving.auth import AuthManager
from ..config import Settings
from ..llm.context import ContextManager
from ..evolution.engine import EvolutionEngine
from ..evolution.candidates import RootCauseEvolutionGenerator
from ..review.fixer import SafeFixer
from ..review.patching import SuggestionOnlyFixer, VerifiedPatchFixer
from ..serving.github import GitHubAppAuthenticator, GitHubClient
from ..review.pipeline import ReviewPipeline
from ..serving.metrics import metrics
from ..llm.client import JsonChatClient
from ..core.modes import RunMode
from ..agents.memory import MemoryManager
from ..core.models import TaskState
from ..session.events import EventKind, EventLog
from ..serving.observability import AlertManager, Observability
from ..store.postgres import create_store
from ..serving.report import to_markdown
from ..review.reviewers import (
    OpenAICompatibleReviewer, ReliabilityRuleReviewer, SecurityRuleReviewer,
)
from ..tools.skills import AgentSkill, SkillRegistry
from ..evolution.skills import AgentSkillReplayReviewer, SkillEvolutionEngine, validate_artifact
from ..serving.task_queue import PermanentTaskError, TaskQueue
from ..serving.rollout import ReleaseManager
from ..eval.verifier import RepairVerifier


class ReviewService:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.validate_evolution()
        self.llm_config = settings.resolved_llm()
        self.store = create_store(settings.database_url, settings.db_path)
        self.memory = MemoryManager(
            self.store, settings.memory_enabled, settings.memory_recall_limit,
            settings.memory_working_ttl_seconds,
        )
        self.context_manager = ContextManager(
            settings.agent_context_window_tokens,
            settings.agent_context_input_tokens,
            settings.context_diff_token_budget,
            settings.context_observation_token_budget,
            settings.context_recent_observations,
            settings.context_map_chunk_tokens,
        )
        self.observability = Observability(settings.otel_service_name, settings.otel_endpoint)
        self.registry = SkillRegistry(settings.skills_dir)
        self.registry.register(
            "security-review", SecurityRuleReviewer(),
            "1.0.0", "Security, injection and secret detection",
        )
        self.registry.register(
            "reliability-review", ReliabilityRuleReviewer(),
            "1.0.0", "Reliability and observability review",
        )
        if self.llm_config:
            active = self.store.get_active_skill_version("llm-review")
            self.registry.register(
                "llm-review",
                self._build_llm_reviewer(active["prompt"] if active else ""),
                "1.0.0", "Context-aware AI code review via %s" % self.llm_config["provider"],
            )
        self.registry.reload()
        self.chat_client = (
            JsonChatClient(
                str(self.llm_config["base_url"]), str(self.llm_config["api_key"]),
                str(self.llm_config["model"]), str(self.llm_config["provider"]),
                settings.timeout_seconds, dict(self.llm_config.get("headers") or {}),
            ) if self.llm_config else None
        )
        self.reviewer = self._build_mode_router()
        self.harness = ReviewPipeline(
            self.store, self.reviewer, settings.timeout_seconds,
            observability=self.observability,
        )
        self.github = GitHubClient(settings.github_token)
        repair_verifier = RepairVerifier(
            settings.repair_test_command, settings.repair_verify_timeout_seconds
        )
        self.fixer = (
            VerifiedPatchFixer(self.chat_client, repair_verifier)
            if self.chat_client else SuggestionOnlyFixer()
        )
        self.auth = AuthManager(
            self.store, settings.auth_secret, settings.session_ttl_seconds,
            settings.bootstrap_admin_username, settings.bootstrap_admin_password,
            settings.default_tenant_id,
        )
        self.releases = ReleaseManager(self.store)
        self.alerts = AlertManager(
            self.store, settings.alert_failure_rate, settings.alert_min_samples
        )
        self.evolution = EvolutionEngine(
            self.store,
            reviewer_factory=self._build_llm_reviewer if self.llm_config else None,
            min_cases=settings.eval_min_cases,
            max_cases=settings.eval_max_cases,
            min_improvement=settings.eval_min_improvement,
            min_holdout_cases=settings.eval_min_holdout_cases,
            max_metric_regression=settings.eval_max_metric_regression,
            candidate_generator=(
                RootCauseEvolutionGenerator(self.chat_client)
                if self.chat_client else None
            ),
        )
        self.skill_evolution = SkillEvolutionEngine(
            self.store,
            reviewer_factory=(lambda artifact: AgentSkillReplayReviewer(
                    artifact, self.chat_client,
                    settings.agent_token_budget,
                    settings.agent_time_budget_seconds,
                )) if self.chat_client else None,
            min_cases=settings.eval_min_cases,
            max_cases=settings.eval_max_cases,
            min_improvement=settings.eval_min_improvement,
            min_holdout_cases=settings.eval_min_holdout_cases,
            max_metric_regression=settings.eval_max_metric_regression,
        )
        self.queue = TaskQueue(
            self._process_queued, settings.async_workers, settings.redis_url,
            settings.queue_max_attempts, settings.queue_lease_seconds,
            self._on_dead_letter,
        )

    def _build_llm_reviewer(self, prompt: str = "") -> OpenAICompatibleReviewer:
        if not self.llm_config:
            raise RuntimeError("no LLM provider is configured")
        return OpenAICompatibleReviewer(
            str(self.llm_config["base_url"]),
            str(self.llm_config["api_key"]),
            str(self.llm_config["model"]),
            self.settings.timeout_seconds,
            system_prompt=prompt,
            provider=str(self.llm_config["provider"]),
            extra_headers=dict(self.llm_config.get("headers") or {}),
        )

    def _build_mode_router(self) -> AgenticReviewer:
        enabled = {
            item.strip() for item in self.settings.enabled_agents.split(",") if item.strip()
        }
        unknown = enabled.difference({
            "lead", "security", "correctness-reliability", "critic"
        })
        if unknown:
            raise ValueError("unsupported enabled Agent role(s): %s" % ", ".join(sorted(unknown)))
        active_prompt = self.store.get_active_skill_version("llm-review")
        structured_config = {}
        if active_prompt:
            run = next((
                item for item in self.store.list_evolution_runs(200)
                if int(item.get("candidate_version") or 0) == int(active_prompt["version"])
                and (item.get("metrics") or {}).get("structured_candidate")
            ), None)
            if run:
                structured_config = (
                    run["metrics"]["structured_candidate"].get("candidate") or {}
                )
        return AgenticReviewer(
            self.store, self.chat_client,
            self.settings.agent_token_budget,
            self.settings.agent_time_budget_seconds,
            self.settings.llm_input_cost_per_million,
            self.settings.llm_output_cost_per_million,
            enabled,
            [
                item for item in self.registry.reviewers()
                if not isinstance(item, OpenAICompatibleReviewer)
            ],
            None,
            self.settings.repair_test_command,
            active_prompt["prompt"] if active_prompt else "",
            structured_config,
            self.memory,
            self.context_manager,
            self._active_agent_skills,
        )

    def _run_review(
        self, task_id: str, repository: str, pull_request: Optional[int],
        diff: str, tenant_id: str,
    ):
        return self.harness.run(task_id, repository, pull_request, diff, tenant_id)

    def reload_skills(self) -> list:
        if self.llm_config:
            active = self.store.get_active_skill_version("llm-review")
            self.registry.register(
                "llm-review",
                self._build_llm_reviewer(active["prompt"] if active else ""),
                "1.0.0", "Context-aware AI code review via %s" % self.llm_config["provider"],
            )
        self.registry.reload()
        skills = self.registry.list()
        self.reviewer = self._build_mode_router()
        self.harness = ReviewPipeline(
            self.store, self.reviewer, self.settings.timeout_seconds,
            observability=self.observability,
        )
        return skills

    def _active_agent_skills(self, tenant_id: str) -> list:
        values = {skill.name: skill for skill in self.registry.agent_skills()}
        for version in self.store.list_active_skill_artifacts(tenant_id):
            artifact = validate_artifact(version["artifact"], version["skill_name"])
            values[version["skill_name"]] = AgentSkill.from_artifact(
                artifact, str(version["version"])
            )
        return [values[name] for name in sorted(values)]

    def list_skills(self, tenant_id: str) -> list:
        scanners = [item for item in self.registry.list() if item.get("kind") == "scanner"]
        return scanners + [{
            "name": skill.name, "version": skill.version,
            "description": skill.description, "source": skill.source,
            "kind": "agent-skill", "sandboxed": False,
            "permissions": list(skill.allowed_tools),
            "content_sha256": skill.content_sha256,
            "resources": list(skill.resource_paths),
        } for skill in self._active_agent_skills(tenant_id)]

    def _validate_review(self, repository: str, diff: str) -> None:
        if not repository or len(repository) > 250:
            raise ValueError("repository is required and must be at most 250 characters")
        size = len(diff.encode("utf-8"))
        if size == 0:
            raise ValueError("diff is required")
        if size > self.settings.max_diff_bytes:
            raise ValueError("diff exceeds maximum size of %d bytes" % self.settings.max_diff_bytes)

    def _require_agentic_model(self) -> None:
        if self.chat_client is None:
            raise RuntimeError("agentic review requires a configured model")

    def _create_task(
        self, repository: str, diff: str, pull_request: Optional[int], source: str,
        tenant_id: str = "default", repository_root: str = "",
        enabled_agents: Optional[list] = None,
        enabled_skills: Optional[list] = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        encoded = diff.encode("utf-8")
        assignment = self.releases.assignment(tenant_id, "llm-review", task_id)
        self.store.create(task_id, repository, pull_request, {
            "source": source, "diff_bytes": len(encoded), "diff_sha256": hashlib.sha256(encoded).hexdigest(),
            "release_lane": assignment["lane"], "shadow": assignment["shadow"],
            "mode": RunMode.AGENTIC.value,
            "repository_root": repository_root,
            "enabled_agents": enabled_agents or [],
            "enabled_skills": enabled_skills or [],
        }, tenant_id)
        self.store.save_task_payload(task_id, diff)
        return task_id

    def _create_deferred_task(
        self, repository: str, pull_request: Optional[int], source: str,
        tenant_id: str, payload: Dict[str, Any],
    ) -> str:
        task_id = str(uuid.uuid4())
        assignment = self.releases.assignment(tenant_id, "llm-review", task_id)
        self.store.create(task_id, repository, pull_request, {
            "source": source, "diff_pending": True,
            "release_lane": assignment["lane"], "shadow": assignment["shadow"],
            **payload,
            "mode": RunMode.AGENTIC.value,
        }, tenant_id)
        return task_id

    def create_review(
        self, repository: str, diff: str, pull_request: Optional[int] = None,
        source: str = "api", tenant_id: str = "default", mode: str = "",
        repository_root: str = "", enabled_agents: Optional[list] = None,
        enabled_skills: Optional[list] = None,
    ) -> Dict[str, Any]:
        self._validate_review(repository, diff)
        RunMode.parse(mode) if mode else None
        self._require_agentic_model()
        self._validate_repository_root(repository_root)
        self._validate_enabled_agents(enabled_agents)
        self._validate_enabled_skills(enabled_skills, tenant_id)
        self._authorize_repository(tenant_id, repository)
        task_id = self._create_task(
            repository, diff, pull_request, source, tenant_id,
            repository_root, enabled_agents, enabled_skills,
        )
        try:
            with self.observability.span(
                "review", task_id, task_id=task_id, tenant_id=tenant_id,
                repository=repository,
            ), metrics.timer("review_duration"):
                report = self._run_review(
                    task_id, repository, pull_request, diff, tenant_id
                )
            metrics.inc("reviews_total")
            lane = (self.store.get(task_id, tenant_id).get("input") or {}).get(
                "release_lane", "stable"
            )
            self.releases.observe(tenant_id, "llm-review", False, lane)
            return {"task_id": task_id, "state": "SUCCESS", "report": report.to_dict()}
        except Exception:
            task = self.store.get(task_id, tenant_id) or {}
            lane = (task.get("input") or {}).get("release_lane", "stable")
            self.releases.observe(tenant_id, "llm-review", True, lane)
            self.alerts.evaluate(tenant_id)
            raise

    def enqueue_review(
        self, repository: str, diff: str, pull_request: Optional[int] = None,
        source: str = "api", github_issue_url: str = "", installation_id: Optional[int] = None,
        tenant_id: str = "default", mode: str = "", repository_root: str = "",
        enabled_agents: Optional[list] = None,
        enabled_skills: Optional[list] = None,
    ) -> Dict[str, Any]:
        self._validate_review(repository, diff)
        RunMode.parse(mode) if mode else None
        self._require_agentic_model()
        self._validate_repository_root(repository_root)
        self._validate_enabled_agents(enabled_agents)
        self._validate_enabled_skills(enabled_skills, tenant_id)
        self._authorize_repository(tenant_id, repository)
        task_id = self._create_task(
            repository, diff, pull_request, source, tenant_id,
            repository_root, enabled_agents, enabled_skills,
        )
        self.queue.submit({
            "task_id": task_id, "repository": repository, "pull_request": pull_request,
            "github_issue_url": github_issue_url, "installation_id": installation_id,
            "tenant_id": tenant_id,
        }, message_id=task_id)
        metrics.inc("reviews_enqueued_total")
        return {"task_id": task_id, "state": "PENDING", "queue": self.queue.backend}

    def _process_queued(self, payload: Dict[str, Any]) -> None:
        task_id = payload["task_id"]
        task = self.store.get(task_id)
        if not task:
            raise PermanentTaskError("task record no longer exists")
        tenant_id = payload.get("tenant_id") or task.get("tenant_id") or "default"
        diff = self.store.get_task_payload(task_id)
        if diff is None and payload.get("diff_url"):
            client = (
                self.github_client_for_installation(payload.get("installation_id"))
                if payload.get("installation_id") else self.github
            )
            client.ensure_repository_access(payload["repository"])
            diff = client.fetch_diff(payload["diff_url"])
            self._validate_review(payload["repository"], diff)
            encoded = diff.encode("utf-8")
            self.store.save_task_payload(task_id, diff)
            self.store.update_task_input(task_id, {
                "diff_pending": False, "diff_bytes": len(encoded),
                "diff_sha256": hashlib.sha256(encoded).hexdigest(),
            })
        if diff is None:
            raise PermanentTaskError("task payload no longer exists")
        try:
            with self.observability.span(
                "review.async", task_id, task_id=task_id, tenant_id=tenant_id,
            ), metrics.timer("review_duration"):
                report = self._run_review(
                    task_id, payload["repository"], payload.get("pull_request"), diff,
                    tenant_id,
                )
            metrics.inc("reviews_total")
            lane = (task.get("input") or {}).get("release_lane", "stable")
            self.releases.observe(tenant_id, "llm-review", False, lane)
            if payload.get("github_issue_url") and self.settings.auto_post_review:
                client = self.github_client_for_installation(payload.get("installation_id"))
                client.upsert_comment(
                    payload["github_issue_url"], to_markdown(report.to_dict()),
                    "<!-- evoagent-review:%s -->" % task_id,
                )
        except Exception:
            metrics.inc("reviews_failed_total")
            lane = (task.get("input") or {}).get("release_lane", "stable")
            self.releases.observe(tenant_id, "llm-review", True, lane)
            self.alerts.evaluate(tenant_id)
            raise

    def _on_dead_letter(self, payload: Dict[str, Any], error: str) -> None:
        task_id = payload.get("task_id", "")
        tenant_id = payload.get("tenant_id", "default")
        task = self.store.get(task_id, tenant_id) if task_id else None
        if task and task.get("state") not in {
            TaskState.SUCCESS.value, TaskState.FAILED.value, TaskState.CANCELLED.value,
        }:
            EventLog.load(self.store, task_id).append(
                EventKind.TASK_FAILED,
                message="Task entered the dead-letter queue: %s" % error,
                error=error,
            )
        self.store.create_alert(
            tenant_id, "dlq:%s" % (task_id or "unknown"), "critical",
            "Task %s entered the dead-letter queue: %s" % (task_id, error),
        )
        metrics.inc("dead_letters_total")

    def handle_github_pull_request(
        self, payload: Dict[str, Any], delivery_id: str,
        payload_sha256: str, tenant_id: str = "",
    ) -> Dict[str, Any]:
        installation_id = (payload.get("installation") or {}).get("id")
        tenant_id = tenant_id or (
            self.store.installation_tenant(installation_id) if installation_id else None
        ) or self.settings.default_tenant_id
        if not self.store.claim_webhook(
            delivery_id, tenant_id, "pull_request", payload_sha256
        ):
            existing = self.store.get_webhook(delivery_id) or {}
            return {
                "duplicate": True, "task_id": existing.get("task_id"),
                "state": "PENDING" if existing.get("task_id") else "ACCEPTED",
            }
        action = payload.get("action")
        if action not in {"opened", "reopened", "synchronize"}:
            self.store.complete_webhook(delivery_id, None)
            return {"ignored": True, "reason": "unsupported pull_request action: %s" % action}
        self._require_agentic_model()
        pull = payload.get("pull_request") or {}
        repository = (payload.get("repository") or {}).get("full_name", "")
        number = payload.get("number")
        diff_url = pull.get("diff_url")
        if not repository or not isinstance(number, int) or not diff_url:
            raise ValueError("invalid GitHub pull_request payload")
        self._authorize_repository(tenant_id, repository)
        task_id = self._create_deferred_task(
            repository, number, "github-webhook", tenant_id,
            {"diff_url": diff_url},
        )
        self.queue.submit({
            "task_id": task_id, "repository": repository, "pull_request": number,
            "github_issue_url": pull.get("issue_url", ""),
            "installation_id": installation_id, "tenant_id": tenant_id,
            "diff_url": diff_url,
        }, message_id=task_id)
        metrics.inc("reviews_enqueued_total")
        result = {"task_id": task_id, "state": "PENDING", "queue": self.queue.backend}
        self.store.complete_webhook(delivery_id, result["task_id"])
        result["will_post_to_github"] = self.settings.auto_post_review
        return result

    def github_client_for_installation(self, installation_id: Optional[int] = None) -> GitHubClient:
        if installation_id is None:
            return self.github
        if not self.settings.github_app_id or not self.settings.github_private_key_path:
            raise ValueError("GitHub App credentials are not configured")
        token = GitHubAppAuthenticator(
            self.settings.github_app_id, self.settings.github_private_key_path
        ).installation_token(installation_id)
        return GitHubClient(token)

    def create_fix(
        self, task_id: str, installation_id: Optional[int] = None,
        tenant_id: Optional[str] = None,
    ) -> dict:
        task = self.store.get(task_id, tenant_id)
        if not task or not task.get("report"):
            raise ValueError("completed task not found")
        if task.get("pull_request") is None:
            raise ValueError("fix commits require a GitHub pull request task")
        actual_tenant = task.get("tenant_id") or tenant_id or "default"
        if not self.store.repository_allowed(actual_tenant, task["repository"], True):
            raise PermissionError("automatic repair is not enabled for this repository")
        result = self.fixer.create_fix_commits(
            self.github_client_for_installation(installation_id),
            task["repository"], task["pull_request"], task["report"],
        )
        metrics.inc("fix_runs_total")
        return result

    def record_feedback(
        self, task_id: str, category: str, finding: Optional[dict], note: str,
        tenant_id: Optional[str] = None,
    ) -> dict:
        task = self.store.get(task_id, tenant_id)
        if not task:
            raise ValueError("task not found")
        if task.get("state") != "SUCCESS" or not task.get("report"):
            raise ValueError("feedback requires a completed review task")
        if category not in {"false_positive", "missed_issue", "bad_fix", "accepted"}:
            raise ValueError("unsupported feedback category")
        self.store.record_failure_case(task_id, category, {"finding": finding, "note": note[:2000]})
        self.memory.remember_feedback(
            task.get("tenant_id") or tenant_id or "default", task["repository"],
            task_id, category, finding, note[:2000],
        )
        metrics.inc("feedback_total")
        return {"recorded": True, "category": category}

    def resume_task(self, task_id: str, tenant_id: Optional[str] = None) -> dict:
        task = self.store.get(task_id, tenant_id)
        if not task:
            raise ValueError("task not found")
        if task["state"] == "SUCCESS":
            return {"task_id": task_id, "state": "SUCCESS", "report": task["report"]}
        diff = self.store.get_task_payload(task_id)
        if diff is None:
            raise ValueError("task payload is no longer available")
        self.queue.submit({
            "task_id": task_id, "repository": task["repository"],
            "pull_request": task.get("pull_request"),
            "tenant_id": task.get("tenant_id", "default"),
        }, message_id=task_id)
        return {"task_id": task_id, "state": "PENDING", "resumed": True}

    def cancel_task(self, task_id: str, tenant_id: Optional[str] = None) -> bool:
        return self.store.request_cancel(task_id, tenant_id)

    def _authorize_repository(self, tenant_id: str, repository: str) -> None:
        if not self.store.repository_allowed(tenant_id, repository):
            raise PermissionError("repository is not authorized for this tenant")

    @staticmethod
    def _validate_repository_root(repository_root: str) -> None:
        if not repository_root:
            return
        import os
        if not os.path.isabs(repository_root) or not os.path.isdir(repository_root):
            raise ValueError("repository_root must be an existing absolute directory")

    @staticmethod
    def _validate_enabled_agents(enabled_agents: Optional[list]) -> None:
        if enabled_agents is None:
            return
        allowed = {"lead", "security", "correctness-reliability", "critic"}
        unknown = set(enabled_agents).difference(allowed)
        if unknown:
            raise ValueError("unsupported enabled Agent role(s): %s" % ", ".join(sorted(unknown)))

    def _validate_enabled_skills(
        self, enabled_skills: Optional[list], tenant_id: str,
    ) -> None:
        if enabled_skills is None:
            return
        if not all(isinstance(item, str) for item in enabled_skills):
            raise ValueError("enabled_skills must contain Agent Skill names")
        available = {skill.name for skill in self._active_agent_skills(tenant_id)}
        unknown = set(enabled_skills).difference(available)
        if unknown:
            raise ValueError(
                "unknown enabled Agent Skill(s): %s" % ", ".join(sorted(unknown))
            )
