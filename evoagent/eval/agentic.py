"""Product-backed agentic evaluation suite for labelled PRs."""
from collections import Counter
import hashlib
import json
import random
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..review.agentic import AgenticReviewer
from ..review.merge import merge_findings
from ..eval.benchmark import ContextRuleReviewer
from ..eval.harness import (
    RULE_TO_CWE,
    EndToEndEvaluationHarness,
    dataset_fingerprint,
    one_to_one_match,
    one_to_one_target_match,
)
from ..core.finding_policy import normalize_rule_id
from ..llm.client import JsonChatClient
from ..core.models import Finding, Severity


REQUIRED_ARMS = (
    "multi-llm-no-critic", "full-agentic",
)

ARM_TOPOLOGY = {
    "multi-llm-no-critic": {
        "mode": "agentic",
        "roles": ("lead", "security", "correctness-reliability"),
    },
    "full-agentic": {
        "mode": "agentic",
        "roles": ("lead", "security", "correctness-reliability", "critic"),
    },
}


def validate_real_dataset(cases: List[dict], minimum_cases: int = 300) -> Dict[str, Any]:
    repositories_by_split = {}
    source_kinds = set()
    for case in cases:
        split = str(case.get("split", ""))
        if split not in {"train", "validation", "holdout"}:
            raise ValueError("every case must use train, validation or holdout split")
        repositories_by_split.setdefault(split, set()).add(str(case.get("repository", "")))
        source_kinds.add(str((case.get("source") or {}).get("kind", "unknown")))
        if not isinstance(case.get("expected_findings"), list):
            raise ValueError("every real PR must include human expected_findings")
        for finding in case["expected_findings"]:
            if "should_comment" not in finding:
                raise ValueError("human labels must include should_comment")
            if not all(key in finding for key in ("severity", "path")):
                raise ValueError("human labels require severity and path")
    overlaps = {}
    splits = sorted(repositories_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            shared = repositories_by_split[left].intersection(repositories_by_split[right])
            if shared:
                overlaps["%s:%s" % (left, right)] = sorted(shared)
    public_or_historical = source_kinds.issubset({
        "public-github-pr", "private-historical-pr",
    }) and bool(source_kinds)
    gates = {
        "minimum_cases": len(cases) >= minimum_cases,
        "real_provenance": public_or_historical,
        "repository_isolation": not overlaps,
        "train_present": bool(repositories_by_split.get("train")),
        "validation_present": bool(repositories_by_split.get("validation")),
        "hidden_holdout_present": bool(repositories_by_split.get("holdout")),
    }
    return {
        "ready": all(gates.values()), "gates": gates, "cases": len(cases),
        "minimum_required": int(minimum_cases),
        "repositories": len({str(case.get("repository")) for case in cases}),
        "repositories_by_split": {
            key: len(value) for key, value in repositories_by_split.items()
        },
        "repository_overlap": overlaps, "source_kinds": sorted(source_kinds),
        "dataset_sha256": dataset_fingerprint(cases),
    }


class _EvaluationTaskStore:
    """Minimal task input provider used by AgenticReviewer during replay."""

    def __init__(self, task_input: dict):
        self.task_input = dict(task_input)

    def get(self, _task_id: str, _tenant_id: Optional[str] = None) -> dict:
        return {"input": dict(self.task_input)}


class ProductArmReviewer:
    """Run one ablation arm through the product AgenticReviewer."""

    def __init__(
        self, arm: str, client: JsonChatClient, total_token_budget: int,
        total_time_budget_seconds: int = 120, max_revision_rounds: int = 0,
    ):
        if arm not in ARM_TOPOLOGY:
            raise ValueError("unknown evaluation arm: %s" % arm)
        topology = ARM_TOPOLOGY[arm]
        roles = tuple(topology["roles"])
        llm_role_count = max(1, len(roles))
        if total_token_budget < 256 * llm_role_count:
            raise ValueError(
                "%s requires at least %d total tokens" % (arm, 256 * llm_role_count)
            )
        if total_time_budget_seconds < llm_role_count:
            raise ValueError("total_time_budget_seconds is too small for %s" % arm)
        per_role_tokens = max(256, total_token_budget // llm_role_count)
        per_role_seconds = max(1, total_time_budget_seconds // llm_role_count)
        enabled = set(roles)
        task_input = {
            "mode": topology["mode"],
            "enabled_agents": sorted(enabled),
        }
        self.arm = arm
        self.name = arm
        self.client = client
        self.total_token_budget = int(total_token_budget)
        self.total_time_budget_seconds = int(total_time_budget_seconds)
        self.per_role_token_budget = per_role_tokens
        self.per_role_time_budget_seconds = per_role_seconds
        self.expected_roles = roles
        self.store = _EvaluationTaskStore(task_input)
        # The default keeps evaluation arms on the stable one-pass profile, so
        # published baselines stay comparable. Raising it is how a run measures
        # what the Lead's targeted revision guidance is worth on recall; the
        # value used is recorded in evaluation_config().
        structured_config = {
            "max_revision_rounds": max(0, int(max_revision_rounds)),
            "publish_unverified_suggestions": False,
        }
        self.router = AgenticReviewer(
            self.store, client,
            default_token_budget=per_role_tokens,
            default_time_budget=per_role_seconds,
            enabled_roles=enabled,
            scanners=[ContextRuleReviewer()],
            structured_config=structured_config,
        )
        self._sequence = 0
        self._last_summary: Dict[str, Any] = {}

    def review(self, diff: str, parsed) -> list:
        return self.review_case({"diff": diff, "repository": ""}, parsed)

    def review_case(self, case: dict, parsed) -> list:
        self._sequence += 1
        task_id = "evaluation:%s:%d" % (self.arm, self._sequence)
        repository_root = str(case.get("repository_root") or "")
        outcome = self.router.review_outcome(
            task_id, case["diff"], parsed,
            repository=repository_root or str(case.get("repository") or ""),
        )
        self._last_summary = outcome.summary
        self._validate_execution()
        return outcome.findings

    def _validate_execution(self) -> None:
        execution = self._last_summary.get("execution") or {}
        calls = execution.get("model_call_log") or []
        actual = Counter(
            str(item.get("role")) for item in calls if bool(item.get("ok", True))
        )
        required = set(self.expected_roles)
        if self.arm == "full-agentic":
            collaboration = self._last_summary.get("collaboration") or {}
            proposed = int(
                collaboration.get("candidate_findings_before_critic", 0) or 0
            )
            if proposed == 0:
                required.discard("critic")
            critic_failed_closed = any(
                item.get("event") == "critic_failed_closed"
                for items in (execution.get("agent_traces") or {}).values()
                for item in items
            )
            if critic_failed_closed:
                # The product reviewer already quarantined every model finding.
                # Preserve that safe result instead of turning it into a harness
                # failure merely because the Critic transport never succeeded.
                required.discard("critic")
        missing = sorted(role for role in required if actual[role] < 1)
        if missing:
            raise RuntimeError(
                "%s completed without successful LLM role(s): %s"
                % (self.arm, ", ".join(missing))
            )

    def evaluation_execution(self) -> dict:
        return dict(self._last_summary.get("execution") or {})

    def evaluation_summary(self) -> dict:
        summary = self._last_summary
        collaboration = summary.get("collaboration") or {}
        execution = summary.get("execution") or {}
        lead = collaboration.get("lead") or {}
        protocol_events = {
            "minimum_tool_calls_not_met", "counterexample_resolution_missing",
            "finding_resolution_without_finding", "finding_resolution_deferred",
            "final_action_validation_failed",
            "budget_exhausted",
            "critic_failed_closed",
        }
        return {
            "gates": dict(summary.get("gates") or {}),
            "rejected_findings": list(summary.get("rejected_findings") or []),
            "candidate_findings_before_critic": int(
                collaboration.get("candidate_findings_before_critic", 0) or 0
            ),
            "accepted_findings": int(collaboration.get("accepted_findings", 0) or 0),
            "suggestion_count": int(collaboration.get("suggestion_count", 0) or 0),
            "suggested_findings": list(summary.get("suggested_findings") or []),
            "repository_context": dict(summary.get("repository_context") or {}),
            "assignments": list(collaboration.get("assignments") or []),
            "worker_results": list(collaboration.get("worker_results") or []),
            "worker_history": list(collaboration.get("worker_history") or []),
            "revision_results": list(collaboration.get("revision_results") or []),
            "lead_assessments": list(lead.get("assessments") or []),
            "scanner_findings": int(collaboration.get("scanner_findings", 0) or 0),
            "scanner_finding_details": list(
                collaboration.get("scanner_finding_details") or []
            ),
            "tool_call_log": list(execution.get("tool_call_log") or []),
            "protocol_events": [
                {"role": role, **item}
                for role, items in (execution.get("agent_traces") or {}).items()
                for item in items
                if item.get("event") in protocol_events
            ],
            "publication_decisions": list(
                collaboration.get("publication_decisions") or []
            ),
            "critic_decisions": list(collaboration.get("critic_decisions") or []),
            "lead_final": dict(lead.get("final") or {}),
            "stop_reason": str(collaboration.get("stop_reason") or ""),
        }

    def evaluation_config(self) -> dict:
        catalog = []
        for family, rules in (
            ("local", self.router.rules.RULES),
            ("diff", self.router.rules.DIFF_RULES),
            ("context", ContextRuleReviewer.RULES),
        ):
            for rule in rules:
                values = rule if isinstance(rule, tuple) else (rule,)
                catalog.append([
                    family,
                    *[
                        str(getattr(value, "value", getattr(value, "pattern", value)))
                        for value in values[:3]
                    ],
                ])
        catalog_sha256 = hashlib.sha256(json.dumps(
            catalog, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return {
            "arm": self.arm,
            "mode": ARM_TOPOLOGY[self.arm]["mode"],
            "roles": list(self.expected_roles),
            "deterministic_rules": (
                len(self.router.rules.RULES)
                + len(self.router.rules.DIFF_RULES)
                + len(ContextRuleReviewer.RULES)
            ),
            "scanner_findings_seed_agents": False,
            "scanner_orchestration": "parallel-unseeded-publication-safety-net",
            "scanner_catalog_policy": "freeze-per-evaluation",
            "scanner_catalog_sha256": catalog_sha256,
            "max_revision_rounds": self.router._max_revision_rounds(),
            "publish_unverified_suggestions": False,
            "total_token_budget_per_pr": self.total_token_budget,
            "per_role_token_budget": self.per_role_token_budget,
            "total_time_budget_seconds_per_pr": self.total_time_budget_seconds,
            "per_role_time_budget_seconds": self.per_role_time_budget_seconds,
        }


def product_reviewer_factories(
    client: JsonChatClient, total_time_budget_seconds: int = 120,
) -> Dict[str, Callable[[str, int], ProductArmReviewer]]:
    """Create the two agentic topology arms with one shared model client."""

    def build(arm: str, model: str, token_budget: int) -> ProductArmReviewer:
        if str(client.model) != str(model):
            raise ValueError(
                "evaluation model %s does not match client model %s"
                % (model, client.model)
            )
        return ProductArmReviewer(
            arm, client, token_budget, total_time_budget_seconds,
        )

    return {
        arm: (
            lambda model, budget, selected=arm: build(selected, model, budget)
        )
        for arm in REQUIRED_ARMS
    }


class ProductionEvaluationHarness(EndToEndEvaluationHarness):
    ADJUDICATION_VERDICTS = frozenset({
        "required", "optional", "invalid", "duplicate",
    })
    SUGGESTION_VERDICTS = ADJUDICATION_VERDICTS

    @staticmethod
    def _restore_findings(values):
        findings = []
        for value in values or []:
            try:
                item = dict(value)
                item["severity"] = Severity(str(item["severity"]))
                findings.append(Finding(**item))
            except (KeyError, TypeError, ValueError):
                continue
        return findings

    def _score_capability_lanes(self, result, expected, agentic_summary):
        """Keep model ability separate from the scanner-backed product result."""
        worker_findings = merge_findings(self._restore_findings([
            finding
            for worker in agentic_summary.get("worker_results") or []
            if isinstance(worker, dict)
            for finding in worker.get("findings") or []
        ]))
        scanner_findings = merge_findings(self._restore_findings(
            agentic_summary.get("scanner_finding_details") or []
        ))
        # Score what survived ALL gates, using final taxonomy and locations.
        # A partition verdict is not a published finding; a raw scanner hit
        # that never reaches output is not a publication rescue either.
        published = self._restore_findings(result.get("predicted_findings") or [])
        scanner_sources = ("local-rule-scanner", "declarative-scanner:")
        published_worker_findings = [
            finding for finding in published
            if not str(finding.source or "").startswith(scanner_sources)
        ]
        published_scanner_findings = [
            finding for finding in published
            if str(finding.source or "").startswith(scanner_sources)
        ]

        worker_strict = one_to_one_match(
            expected, worker_findings, self.line_tolerance
        )
        worker_targets = one_to_one_target_match(
            expected, worker_findings
        )
        published_worker_strict = one_to_one_match(
            expected, published_worker_findings, self.line_tolerance
        )
        published_worker_targets = one_to_one_target_match(
            expected, published_worker_findings
        )
        scanner_strict = one_to_one_match(
            expected, scanner_findings, self.line_tolerance
        )
        scanner_targets = one_to_one_target_match(
            expected, scanner_findings
        )
        worker_strict_expected = {item.expected_index for item in worker_strict}
        worker_target_expected = {item.expected_index for item in worker_targets}
        published_worker_strict_expected = {
            item.expected_index for item in published_worker_strict
        }
        published_worker_target_expected = {
            item.expected_index for item in published_worker_targets
        }
        scanner_strict_expected = {item.expected_index for item in scanner_strict}
        scanner_target_expected = {item.expected_index for item in scanner_targets}
        published_scanner_strict_expected = {
            item.expected_index for item in one_to_one_match(
                expected, published_scanner_findings, self.line_tolerance,
            )
        }
        published_scanner_target_expected = {
            item.expected_index for item in one_to_one_target_match(
                expected, published_scanner_findings,
            )
        }

        result.update({
            "worker_formal_predictions": len(worker_findings),
            "worker_formal_strict_tp": len(worker_strict),
            "worker_formal_target_tp": len(worker_targets),
            "worker_published_predictions": len(published_worker_findings),
            "worker_published_strict_tp": len(published_worker_strict),
            "worker_published_target_tp": len(published_worker_targets),
            "scanner_predictions": len(scanner_findings),
            "scanner_strict_tp": len(scanner_strict),
            "scanner_target_tp": len(scanner_targets),
            "scanner_unique_strict_tp": len(
                scanner_strict_expected - worker_strict_expected
            ),
            "scanner_unique_target_tp": len(
                scanner_target_expected - worker_target_expected
            ),
            "scanner_publication_rescue_strict_tp": len(
                published_scanner_strict_expected - published_worker_strict_expected
            ),
            "scanner_publication_rescue_target_tp": len(
                published_scanner_target_expected - published_worker_target_expected
            ),
        })

    @staticmethod
    def _targeted_labels(case):
        return str((case.get("source") or {}).get("label_completeness", "")) == (
            "targeted-review-comments"
        )

    @classmethod
    def _finding_judgments(cls, case, field, label):
        judgments = []
        for index, raw in enumerate(case.get(field) or []):
            if not isinstance(raw, dict):
                raise ValueError("%s judgment %d must be an object" % (label, index))
            verdict = str(raw.get("verdict", "")).strip().lower()
            if verdict not in cls.ADJUDICATION_VERDICTS:
                raise ValueError(
                    "%s judgment %d has invalid verdict: %s"
                    % (label, index, verdict)
                )
            try:
                line = int(raw.get("line", raw.get("start_line", 0)))
                end_line = int(raw.get("end_line", line))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "%s judgment %d has an invalid line" % (label, index)
                ) from exc
            if line < 1 or end_line < line or not str(raw.get("path", "")).strip():
                raise ValueError(
                    "%s judgment %d requires a valid path and line range"
                    % (label, index)
                )
            if not str(raw.get("rule_id", raw.get("cwe", ""))).strip():
                raise ValueError(
                    "%s judgment %d requires rule_id or cwe" % (label, index)
                )
            rule_id = normalize_rule_id(str(raw.get("rule_id", raw.get("cwe", ""))))
            judgments.append({
                "path": str(raw.get("path", "")),
                "start_line": line,
                "end_line": end_line,
                "rule_id": rule_id,
                "cwe": str(raw.get("cwe") or RULE_TO_CWE.get(rule_id, rule_id)),
                "severity": str(raw.get("severity", "medium")),
                "verdict": verdict,
                "note": str(raw.get("note", ""))[:2000],
            })
        return judgments

    @classmethod
    def _suggestion_judgments(cls, case):
        return cls._finding_judgments(
            case, "suggestion_judgments", "suggestion",
        )

    @classmethod
    def _formal_judgments(cls, case):
        return cls._finding_judgments(case, "formal_judgments", "formal")

    def _score_formal(self, result, case, findings):
        targeted = self._targeted_labels(case)
        labelled_prediction_indices = {
            int(item["predicted_index"])
            for item in result.get("target_matches") or []
        }
        adjudication = [
            {
                "finding_index": int(item["predicted_index"]),
                "verdict": "required",
                "source": "expected_findings",
                "expected_index": int(item["expected_index"]),
                "note": (
                    "covered a trusted labelled review target; taxonomy is scored "
                    "separately"
                ),
            }
            for item in result.get("target_matches") or []
        ]
        remaining = [
            (index, finding) for index, finding in enumerate(findings)
            if index not in labelled_prediction_indices
        ]
        judgments = self._formal_judgments(case) if targeted else []
        judgment_matches = one_to_one_match(
            judgments, [item[1] for item in remaining], self.line_tolerance
        )
        verdict_counts = Counter()
        judged_remaining = set()
        for match in judgment_matches:
            original_index = remaining[match.predicted_index][0]
            judgment = judgments[match.expected_index]
            judged_remaining.add(match.predicted_index)
            verdict_counts[judgment["verdict"]] += 1
            adjudication.append({
                "finding_index": original_index,
                "verdict": judgment["verdict"],
                "source": "formal_judgments",
                "judgment_index": match.expected_index,
                "note": judgment["note"],
            })
        for remaining_index, (original_index, _finding) in enumerate(remaining):
            if targeted and remaining_index not in judged_remaining:
                adjudication.append({
                    "finding_index": original_index,
                    "verdict": "unjudged",
                    "source": "none",
                })

        if targeted:
            required = int(verdict_counts["required"])
            optional = int(verdict_counts["optional"])
            invalid = int(verdict_counts["invalid"])
            duplicate = int(verdict_counts["duplicate"])
            unjudged = len(remaining) - len(judgment_matches)
            adjudicated = len(labelled_prediction_indices) + len(judgment_matches)
        else:
            required = optional = duplicate = unjudged = 0
            invalid = len(remaining)
            adjudicated = len(findings)

        result.update({
            "invalid_comments": invalid,
            "formal_label_gap_required": required,
            "formal_optional_findings": optional,
            "formal_invalid_findings": invalid,
            "formal_duplicate_findings": duplicate,
            "formal_unjudged_findings": unjudged,
            "formal_adjudicated": adjudicated,
            "formal_useful_findings": required + optional,
            "formal_adjudication": sorted(
                adjudication, key=lambda item: item["finding_index"]
            ),
        })
        return result

    def _score_suggestions(self, result, case, findings, suggested_findings):
        expected = [
            item for item in case["expected_findings"]
            if bool(item.get("should_comment", True))
        ]
        required_matches = one_to_one_match(
            expected, suggested_findings, self.line_tolerance
        )
        required_prediction_indices = {
            match.predicted_index for match in required_matches
        }
        formal_expected = {
            int(item["expected_index"])
            for item in result.get("target_matches") or []
        }
        adjudication = []
        for match in required_matches:
            duplicate_of_formal = match.expected_index in formal_expected
            adjudication.append({
                "suggestion_index": match.predicted_index,
                "verdict": "duplicate" if duplicate_of_formal else "required",
                "source": "expected_findings",
                "expected_index": match.expected_index,
                "note": (
                    "already covered by a confirmed finding"
                    if duplicate_of_formal else "recovers a required labelled finding"
                ),
            })

        remaining = [
            (index, finding) for index, finding in enumerate(suggested_findings)
            if index not in required_prediction_indices
        ]
        judgments = self._suggestion_judgments(case)
        judgment_matches = one_to_one_match(
            judgments, [item[1] for item in remaining], self.line_tolerance
        )
        judged_remaining = set()
        verdict_counts = Counter()
        for match in judgment_matches:
            original_index = remaining[match.predicted_index][0]
            judgment = judgments[match.expected_index]
            judged_remaining.add(match.predicted_index)
            verdict_counts[judgment["verdict"]] += 1
            adjudication.append({
                "suggestion_index": original_index,
                "verdict": judgment["verdict"],
                "source": "suggestion_judgments",
                "judgment_index": match.expected_index,
                "note": judgment["note"],
            })
        for remaining_index, (original_index, _finding) in enumerate(remaining):
            if remaining_index not in judged_remaining:
                adjudication.append({
                    "suggestion_index": original_index,
                    "verdict": "unjudged",
                    "source": "none",
                })

        combined = one_to_one_match(
            expected, list(findings) + list(suggested_findings), self.line_tolerance
        )
        required_tp = len(required_matches)
        required_recovery = sum(
            match.expected_index not in formal_expected for match in required_matches
        )
        formal_duplicates = required_tp - required_recovery
        optional = int(verdict_counts["optional"])
        label_gap = int(verdict_counts["required"])
        invalid = int(verdict_counts["invalid"])
        duplicate = int(verdict_counts["duplicate"]) + formal_duplicates
        unjudged = len(remaining) - len(judgment_matches)
        result.update({
            "suggestions": len(suggested_findings),
            "suggested_findings": [item.to_dict() for item in suggested_findings],
            "suggestion_tp": required_tp,
            "suggestion_fp": invalid,
            "suggestion_fn": len(expected) - required_tp,
            "suggestion_optional": optional,
            "suggestion_label_gap_required": label_gap,
            "suggestion_invalid": invalid,
            "suggestion_duplicate": duplicate,
            "suggestion_unjudged": unjudged,
            "suggestion_adjudicated": required_tp + len(judgment_matches),
            "suggestion_useful": required_recovery + label_gap + optional,
            "incremental_suggestion_tp": required_recovery,
            "combined_tp_after_verification": len(combined),
            "suggestion_adjudication": sorted(
                adjudication, key=lambda item: item["suggestion_index"]
            ),
        })
        return result

    def rescore_suggestions(self, result, case):
        """Re-score cached model output after adding human judgments, without new calls."""
        findings = merge_findings(
            self._restore_findings(result.get("predicted_findings") or [])
        )
        suggestions = self._restore_findings(result.get("suggested_findings") or [])
        return self._score_suggestions(result, case, findings, suggestions)

    def rescore_cached_result(self, result, case):
        """Re-score formal and suggestion layers after a dataset label revision."""
        expected = [
            item for item in case["expected_findings"]
            if bool(item.get("should_comment", True))
        ]
        findings = merge_findings(
            self._restore_findings(result.get("predicted_findings") or [])
        )
        suggestions = self._restore_findings(result.get("suggested_findings") or [])
        matches = one_to_one_match(expected, findings, self.line_tolerance)
        target_matches = one_to_one_target_match(expected, findings)
        targeted = self._targeted_labels(case)
        unmatched = len(findings) - len(matches)
        result.update({
            "expected": len(expected),
            "predicted": len(findings),
            "tp": len(matches),
            "fp": len(findings) - len(matches),
            "fn": len(expected) - len(matches),
            "target_tp": len(target_matches),
            "target_fn": len(expected) - len(target_matches),
            "taxonomy_hits": len(matches),
            "taxonomy_misses": len(target_matches) - len(matches),
            "severity_hits": 0,
            "high_total": sum(
                str(item["severity"]).lower() in {"high", "critical"}
                for item in expected
            ),
            "high_hits": 0,
            "clean_hit": not expected and not findings,
            "expected_findings": expected,
            "predicted_findings": [item.to_dict() for item in findings],
            "matches": [],
            "target_matches": [{
                "expected_index": match.expected_index,
                "predicted_index": match.predicted_index,
                "path": findings[match.predicted_index].path,
                "line": findings[match.predicted_index].line,
                "rule_id": findings[match.predicted_index].rule_id,
                "location_distance": match.location_distance,
            } for match in target_matches],
            "invalid_comments": 0 if targeted else unmatched,
            "formal_invalid_findings": 0 if targeted else unmatched,
            "formal_unjudged_findings": unmatched if targeted else 0,
            "label_completeness": (
                "targeted-review-comments" if targeted else "exhaustive-or-unspecified"
            ),
            "exact_location_hits": 0,
            "evidence_hits": 0,
            # Cached repair evidence is tied to the old truth set and cannot be
            # promoted safely without executing the repair verifier again.
            "repair_attempted": 0,
            "repair_passed": 0,
            "repair": [],
            "e2e_success": False,
        })
        for match in matches:
            truth = expected[match.expected_index]
            finding = findings[match.predicted_index]
            severity_hit = finding.severity.value == str(truth["severity"]).lower()
            high = str(truth["severity"]).lower() in {"high", "critical"}
            result["severity_hits"] += int(severity_hit)
            result["high_hits"] += int(high)
            result["exact_location_hits"] += int(match.location_distance == 0)
            result["evidence_hits"] += int(bool(
                finding.evidence_refs or finding.call_chain or finding.evidence.strip()
            ))
            result["matches"].append({
                "expected_index": match.expected_index,
                "predicted_index": match.predicted_index,
                "path": finding.path,
                "line": finding.line,
                "cwe": RULE_TO_CWE.get(finding.rule_id, finding.rule_id),
                "rule_id": finding.rule_id,
                "expected_severity": truth["severity"],
                "predicted_severity": finding.severity.value,
                "severity_hit": severity_hit,
                "location_distance": match.location_distance,
            })
        self._score_formal(result, case, findings)
        self._score_capability_lanes(
            result, expected, result.get("agentic_summary") or {}
        )
        return self._score_suggestions(result, case, findings, suggestions)

    def _run_case(self, reviewer, case):
        class RecordingReviewer:
            def __init__(self, delegate):
                self.delegate = delegate
                self.name = delegate.name
                self.findings = []

            def review(self, diff, parsed):
                self.findings = self.delegate.review(diff, parsed)
                return self.findings

            def review_case(self, case, parsed):
                method = getattr(self.delegate, "review_case", None)
                self.findings = (
                    method(case, parsed)
                    if method else self.delegate.review(case["diff"], parsed)
                )
                return self.findings

        recording = RecordingReviewer(reviewer)
        started = time.monotonic()
        result = super()._run_case(recording, case)
        targeted = self._targeted_labels(case)
        unmatched = int(result.get("fp", 0) or 0)
        result.update({
            "invalid_comments": 0 if targeted else unmatched,
            "formal_label_gap_required": 0,
            "formal_optional_findings": 0,
            "formal_invalid_findings": 0 if targeted else unmatched,
            "formal_duplicate_findings": 0,
            "formal_unjudged_findings": unmatched if targeted else 0,
            "formal_adjudicated": int(result.get("tp", 0)) + (0 if targeted else unmatched),
            "formal_useful_findings": 0,
            "formal_adjudication": [],
            "label_completeness": (
                "targeted-review-comments" if targeted else "exhaustive-or-unspecified"
            ),
            "exact_location_hits": 0,
            "evidence_hits": 0,
            "accepted_comments": int(case.get("accepted_comments", 0) or 0),
            "closed_comments": int(case.get("closed_comments", 0) or 0),
            "cost_usd": 0.0,
            "latency_ms": int((time.monotonic() - started) * 1000),
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model_roles": {},
            "agentic_summary": {},
            "worker_failures": 0,
            "degraded_execution": False,
            "suggestions": 0,
            "suggested_findings": [],
            "suggestion_tp": 0,
            "suggestion_fp": 0,
            "suggestion_fn": result["expected"],
            "suggestion_optional": 0,
            "suggestion_label_gap_required": 0,
            "suggestion_invalid": 0,
            "suggestion_duplicate": 0,
            "suggestion_unjudged": 0,
            "suggestion_adjudicated": 0,
            "suggestion_useful": 0,
            "suggestion_adjudication": [],
            "incremental_suggestion_tp": 0,
            "combined_tp_after_verification": result["tp"],
            "worker_formal_predictions": 0,
            "worker_formal_strict_tp": 0,
            "worker_formal_target_tp": 0,
            "worker_published_predictions": 0,
            "worker_published_strict_tp": 0,
            "worker_published_target_tp": 0,
            "scanner_predictions": 0,
            "scanner_strict_tp": 0,
            "scanner_target_tp": 0,
            "scanner_unique_strict_tp": 0,
            "scanner_unique_target_tp": 0,
            "scanner_publication_rescue_strict_tp": 0,
            "scanner_publication_rescue_target_tp": 0,
        })
        if result["execution_success"]:
            findings = recording.findings
            expected = [
                item for item in case["expected_findings"]
                if bool(item.get("should_comment", True))
            ]
            matches = one_to_one_match(
                expected, findings, self.line_tolerance
            )
            for match in matches:
                finding = findings[match.predicted_index]
                result["exact_location_hits"] += int(match.location_distance == 0)
                result["evidence_hits"] += int(bool(
                    finding.evidence_refs or finding.call_chain or finding.evidence.strip()
                ))
            self._score_formal(result, case, findings)
            summary_reader = getattr(reviewer, "evaluation_execution", None)
            if summary_reader:
                execution = summary_reader() or {}
                result["cost_usd"] = float(execution.get("cost_usd", 0) or 0)
                result["latency_ms"] = int(execution.get("duration_ms", result["latency_ms"]))
                result["llm_calls"] = int(execution.get("llm_calls", 0) or 0)
                result["input_tokens"] = int(execution.get("input_tokens", 0) or 0)
                result["output_tokens"] = int(execution.get("output_tokens", 0) or 0)
                result["total_tokens"] = int(execution.get("total_tokens", 0) or 0)
                result["model_roles"] = dict(Counter(
                    str(item.get("role"))
                    for item in execution.get("model_call_log") or []
                ))
            summary_reader = getattr(reviewer, "evaluation_summary", None)
            if summary_reader:
                result["agentic_summary"] = summary_reader() or {}
                self._score_capability_lanes(
                    result, expected, result["agentic_summary"]
                )
                failed_workers = [
                    item for item in result["agentic_summary"].get("worker_results") or []
                    if isinstance(item, dict) and item.get("status") == "failed"
                ]
                result["worker_failures"] = len(failed_workers)
                result["degraded_execution"] = bool(failed_workers)
                result["suggestions"] = int(
                    result["agentic_summary"].get("suggestion_count", 0) or 0
                )
                suggested_findings = self._restore_findings(
                    result["agentic_summary"].get("suggested_findings") or []
                )
                self._score_suggestions(
                    result, case, findings, suggested_findings
                )
        return result

    @staticmethod
    def _empty_totals():
        values = EndToEndEvaluationHarness._empty_totals()
        values.update({
            "invalid_comments": 0, "exact_location_hits": 0, "evidence_hits": 0,
            "accepted_comments": 0, "closed_comments": 0,
            "latency_ms": 0, "cost_microusd": 0,
            "llm_calls": 0, "input_tokens": 0,
            "output_tokens": 0, "total_tokens": 0, "suggestions": 0,
            "suggestion_tp": 0, "suggestion_fp": 0, "suggestion_fn": 0,
            "suggestion_optional": 0, "suggestion_label_gap_required": 0,
            "suggestion_invalid": 0, "suggestion_duplicate": 0,
            "suggestion_unjudged": 0, "suggestion_adjudicated": 0,
            "suggestion_useful": 0,
            "incremental_suggestion_tp": 0,
            "combined_tp_after_verification": 0,
            "clean_cases_without_suggestions": 0,
            "formal_label_gap_required": 0, "formal_optional_findings": 0,
            "formal_invalid_findings": 0, "formal_duplicate_findings": 0,
            "formal_unjudged_findings": 0, "formal_adjudicated": 0,
            "formal_useful_findings": 0,
            "targeted_label_cases": 0,
            "worker_failures": 0, "degraded_cases": 0,
            "worker_formal_predictions": 0,
            "worker_formal_strict_tp": 0,
            "worker_formal_target_tp": 0,
            "worker_published_predictions": 0,
            "worker_published_strict_tp": 0,
            "worker_published_target_tp": 0,
            "scanner_predictions": 0,
            "scanner_strict_tp": 0,
            "scanner_target_tp": 0,
            "scanner_unique_strict_tp": 0,
            "scanner_unique_target_tp": 0,
            "scanner_publication_rescue_strict_tp": 0,
            "scanner_publication_rescue_target_tp": 0,
        })
        return values

    @staticmethod
    def _accumulate(totals, result):
        EndToEndEvaluationHarness._accumulate(totals, result)
        for field in (
            "invalid_comments", "exact_location_hits", "evidence_hits",
            "accepted_comments", "closed_comments", "latency_ms",
            "llm_calls", "input_tokens", "output_tokens", "total_tokens",
            "suggestions",
            "suggestion_tp", "suggestion_fp", "suggestion_fn",
            "suggestion_optional", "suggestion_label_gap_required",
            "suggestion_invalid", "suggestion_duplicate", "suggestion_unjudged",
            "suggestion_adjudicated", "suggestion_useful",
            "incremental_suggestion_tp", "combined_tp_after_verification",
            "formal_label_gap_required", "formal_optional_findings",
            "formal_invalid_findings", "formal_duplicate_findings",
            "formal_unjudged_findings", "formal_adjudicated",
            "formal_useful_findings",
            "worker_failures",
            "worker_formal_predictions", "worker_formal_strict_tp",
            "worker_formal_target_tp", "worker_published_predictions",
            "worker_published_strict_tp", "worker_published_target_tp",
            "scanner_predictions",
            "scanner_strict_tp", "scanner_target_tp",
            "scanner_unique_strict_tp", "scanner_unique_target_tp",
            "scanner_publication_rescue_strict_tp",
            "scanner_publication_rescue_target_tp",
        ):
            totals[field] += int(result.get(field, 0))
        totals["degraded_cases"] += int(result.get("degraded_execution", False))
        totals["targeted_label_cases"] += int(
            result.get("label_completeness") == "targeted-review-comments"
        )
        totals["clean_cases_without_suggestions"] += int(
            result.get("expected", 0) == 0 and result.get("suggestions", 0) == 0
        )
        totals["cost_microusd"] += int(float(result.get("cost_usd", 0)) * 1_000_000)

    @staticmethod
    def _metrics(totals):
        values = EndToEndEvaluationHarness._metrics(totals)
        cases = totals["cases"] or 1
        tp = totals["tp"] or 1
        commented = totals["accepted_comments"] + totals["closed_comments"]
        expected_total = totals["tp"] + totals["fn"]
        formal_predictions = totals["tp"] + totals["fp"]
        expanded_tp = totals["target_tp"] + totals["formal_label_gap_required"]
        expanded_expected = expected_total + totals["formal_label_gap_required"]
        formal_complete = totals["formal_unjudged_findings"] == 0
        expanded_precision = (
            round(expanded_tp / formal_predictions, 4)
            if formal_predictions and formal_complete else None
        )
        expanded_recall = (
            round(expanded_tp / expanded_expected, 4)
            if expanded_expected else 1.0
        )
        expanded_f1 = (
            round(
                2 * expanded_precision * expanded_recall
                / (expanded_precision + expanded_recall),
                4,
            )
            if expanded_precision is not None
            and expanded_precision + expanded_recall > 0 else None
        )
        values.update({
            "invalid_comments_per_pr": round(totals["invalid_comments"] / cases, 4),
            "exact_line_accuracy": round(totals["exact_location_hits"] / tp, 4),
            "evidence_accuracy": round(totals["evidence_hits"] / tp, 4),
            "comment_acceptance_rate": round(
                totals["accepted_comments"] / commented, 4
            ) if commented else None,
            "average_cost_usd_per_pr": round(
                totals["cost_microusd"] / 1_000_000 / cases, 8
            ),
            "average_latency_ms_per_pr": round(totals["latency_ms"] / cases, 2),
            "average_llm_calls_per_pr": round(totals["llm_calls"] / cases, 4),
            "average_input_tokens_per_pr": round(totals["input_tokens"] / cases, 2),
            "average_output_tokens_per_pr": round(totals["output_tokens"] / cases, 2),
            "average_total_tokens_per_pr": round(totals["total_tokens"] / cases, 2),
            "average_suggestions_per_pr": round(totals["suggestions"] / cases, 4),
            "formal_label_gap_required": totals["formal_label_gap_required"],
            "formal_optional_findings": totals["formal_optional_findings"],
            "formal_invalid_findings": totals["formal_invalid_findings"],
            "formal_duplicate_findings": totals["formal_duplicate_findings"],
            "formal_unjudged_findings": totals["formal_unjudged_findings"],
            "formal_adjudication_coverage": round(
                totals["formal_adjudicated"] / max(1, formal_predictions), 4
            ),
            "precision_interpretation": (
                "not-estimable-until-unexpected-findings-are-adjudicated"
                if totals["formal_unjudged_findings"]
                else (
                    "human-adjudicated-targeted-labels"
                    if totals["targeted_label_cases"] else "exhaustive-labels"
                )
            ),
            "adjudicated_formal_precision": expanded_precision,
            "adjudicated_formal_utility_rate": round(
                (
                    totals["target_tp"] + totals["formal_label_gap_required"]
                    + totals["formal_optional_findings"]
                ) / totals["formal_adjudicated"], 4
            ) if totals["formal_adjudicated"] and formal_complete else None,
            "formal_nuisance_rate": round(
                (
                    totals["formal_invalid_findings"]
                    + totals["formal_duplicate_findings"]
                ) / totals["formal_adjudicated"], 4
            ) if totals["formal_adjudicated"] and formal_complete else None,
            "expanded_required_recall": expanded_recall,
            "expanded_required_f1": expanded_f1,
            "targeted_review_recall": round(
                totals["target_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "suggestion_utility_rate": round(
                totals["suggestion_useful"] / totals["suggestion_adjudicated"], 4
            ) if totals["suggestion_adjudicated"] else None,
            "suggestion_adjudication_coverage": round(
                totals["suggestion_adjudicated"] / totals["suggestions"], 4
            ) if totals["suggestions"] else 1.0,
            "suggestion_nuisance_rate": round(
                (totals["suggestion_invalid"] + totals["suggestion_duplicate"])
                / totals["suggestion_adjudicated"], 4
            ) if totals["suggestion_adjudicated"] else None,
            "strict_required_match_rate_per_suggestion": round(
                totals["suggestion_tp"] / totals["suggestions"], 4
            ) if totals["suggestions"] else 0.0,
            "strict_required_suggestion_recall": round(
                totals["suggestion_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "strict_clean_silence_rate": round(
                totals["clean_cases_without_suggestions"] / totals["clean_cases"], 4
            ) if totals["clean_cases"] else 1.0,
            "missed_finding_recovery_rate": round(
                totals["incremental_suggestion_tp"] / totals["fn"], 4
            ) if totals["fn"] else 1.0,
            "combined_recall_after_verification": round(
                totals["combined_tp_after_verification"] / expected_total, 4
            ) if expected_total else 1.0,
            "failure_rate": round(
                1 - totals["execution_successes"] / cases, 4
            ),
            "worker_failures": totals["worker_failures"],
            "degraded_execution_rate": round(totals["degraded_cases"] / cases, 4),
            "full_role_success_rate": round(1 - totals["degraded_cases"] / cases, 4),
            "worker_formal_strict_recall": round(
                totals["worker_formal_strict_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "worker_formal_target_recall": round(
                totals["worker_formal_target_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "worker_published_strict_recall": round(
                totals["worker_published_strict_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "worker_published_target_recall": round(
                totals["worker_published_target_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "scanner_strict_recall": round(
                totals["scanner_strict_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "scanner_target_recall": round(
                totals["scanner_target_tp"] / expected_total, 4
            ) if expected_total else 1.0,
            "scanner_unique_strict_recall": round(
                totals["scanner_unique_strict_tp"] / expected_total, 4
            ) if expected_total else 0.0,
            "scanner_unique_target_recall": round(
                totals["scanner_unique_target_tp"] / expected_total, 4
            ) if expected_total else 0.0,
            "scanner_publication_rescue_strict_recall": round(
                totals["scanner_publication_rescue_strict_tp"] / expected_total, 4
            ) if expected_total else 0.0,
            "scanner_publication_rescue_target_recall": round(
                totals["scanner_publication_rescue_target_tp"] / expected_total, 4
            ) if expected_total else 0.0,
        })
        return values


class FairAblationSuite:
    """Compare the full agentic topology with its no-critic variant."""

    def __init__(
        self, reviewer_factories: Mapping[str, Callable[[str, int], Any]],
        model: str, token_budget: int, require_production_ready: bool = True,
        bootstrap_iterations: int = 2000, bootstrap_seed: int = 20260819,
    ):
        missing = set(REQUIRED_ARMS).difference(reviewer_factories)
        if missing:
            raise ValueError("missing ablation arms: %s" % ", ".join(sorted(missing)))
        self.factories = reviewer_factories
        self.model = model
        self.token_budget = token_budget
        self.require_production_ready = bool(require_production_ready)
        self.bootstrap_iterations = max(200, int(bootstrap_iterations))
        self.bootstrap_seed = int(bootstrap_seed)

    @staticmethod
    def _role_totals(case_results: List[dict]) -> dict:
        totals = Counter()
        for case in case_results:
            totals.update(case.get("model_roles") or {})
        return dict(sorted(totals.items()))

    def _paired_delta(
        self, left: dict, right: dict, metric: str, seed_offset: int,
    ) -> dict:
        left_cases = left["case_results"]
        right_cases = right["case_results"]
        if [item["id"] for item in left_cases] != [item["id"] for item in right_cases]:
            raise ValueError("paired comparison requires identical ordered case ids")
        count = len(left_cases)
        if not count:
            return {"delta": 0.0, "ci95": [0.0, 0.0], "iterations": 0}
        rng = random.Random(self.bootstrap_seed + seed_offset)
        deltas = []
        for _ in range(self.bootstrap_iterations):
            left_totals = ProductionEvaluationHarness._empty_totals()
            right_totals = ProductionEvaluationHarness._empty_totals()
            for _sample in range(count):
                index = rng.randrange(count)
                ProductionEvaluationHarness._accumulate(left_totals, left_cases[index])
                ProductionEvaluationHarness._accumulate(right_totals, right_cases[index])
            left_value = ProductionEvaluationHarness._metrics(left_totals)[metric]
            right_value = ProductionEvaluationHarness._metrics(right_totals)[metric]
            deltas.append(float(right_value) - float(left_value))
        deltas.sort()
        lower = deltas[int((len(deltas) - 1) * 0.025)]
        upper = deltas[int((len(deltas) - 1) * 0.975)]
        point = float(right["metrics"][metric]) - float(left["metrics"][metric])
        return {
            "delta": round(point, 4),
            "ci95": [round(lower, 4), round(upper, 4)],
            "iterations": self.bootstrap_iterations,
        }

    def _comparison(self, left: dict, right: dict, seed_offset: int) -> dict:
        metrics = ("f1", "precision", "recall", "high_risk_recall")
        return {
            metric: self._paired_delta(left, right, metric, seed_offset + index)
            for index, metric in enumerate(metrics)
        }

    @staticmethod
    def _split_view(arm: dict, split: str) -> dict:
        return {
            "metrics": arm["by_split"][split],
            "case_results": [
                item for item in arm["case_results"] if item["split"] == split
            ],
        }

    def run(self, cases: List[dict]) -> Dict[str, Any]:
        readiness = validate_real_dataset(cases)
        if self.require_production_ready and not readiness["ready"]:
            raise ValueError("real PR dataset failed readiness gates: %s" % readiness["gates"])
        harness = ProductionEvaluationHarness()
        arms = {}
        for name in REQUIRED_ARMS:
            reviewer = self.factories[name](self.model, self.token_budget)
            arms[name] = harness.run(reviewer, cases, name)
            config_reader = getattr(reviewer, "evaluation_config", None)
            arms[name]["fairness"] = {
                "model": self.model, "token_budget_per_pr": self.token_budget,
                "product_runtime": type(getattr(reviewer, "router", reviewer)).__name__,
                "configuration": config_reader() if config_reader else {},
            }
            arms[name]["execution"] = {
                "model_role_calls": self._role_totals(arms[name]["case_results"]),
                "average_llm_calls_per_pr": arms[name]["metrics"]["average_llm_calls_per_pr"],
                "average_total_tokens_per_pr": arms[name]["metrics"]["average_total_tokens_per_pr"],
                "average_latency_ms_per_pr": arms[name]["metrics"]["average_latency_ms_per_pr"],
                "average_cost_usd_per_pr": arms[name]["metrics"]["average_cost_usd_per_pr"],
            }
        no_critic_holdout = self._split_view(
            arms["multi-llm-no-critic"], "holdout",
        )
        full_holdout = self._split_view(arms["full-agentic"], "holdout")
        candidate = full_holdout["metrics"]
        critic_comparison = self._comparison(
            no_critic_holdout, full_holdout, 200,
        )
        no_critic = no_critic_holdout["metrics"]
        critic_false_positive_non_regression = (
            candidate["invalid_comments_per_pr"]
            <= no_critic["invalid_comments_per_pr"]
        )
        critic_recall_non_regression = candidate["recall"] >= no_critic["recall"] - 0.01
        critic_statistically_positive = (
            critic_comparison["f1"]["ci95"][0] > 0
            or (
                critic_comparison["precision"]["ci95"][0] > 0
                and critic_recall_non_regression
            )
        )
        return {
            "schema_version": 4, "dataset": readiness, "arms": arms,
            "comparisons": {
                "scope": "hidden-holdout",
                "critic_vs_no_critic": critic_comparison,
            },
            "critic_gate": {
                "passed": bool(
                    readiness["ready"] and critic_statistically_positive
                    and critic_false_positive_non_regression
                    and critic_recall_non_regression
                ),
                "statistically_positive": critic_statistically_positive,
                "false_positive_non_regression": critic_false_positive_non_regression,
                "recall_non_regression_with_1pp_tolerance": critic_recall_non_regression,
                "production_dataset_ready": readiness["ready"],
                "decision": (
                    "keep-critic" if (
                        readiness["ready"] and critic_statistically_positive
                        and critic_false_positive_non_regression
                        and critic_recall_non_regression
                    ) else "critic-not-proven"
                ),
            },
            "claim_scope": (
                "Evidence applies to this labelled holdout and model version; "
                "it does not prove universal superiority."
            ),
        }
