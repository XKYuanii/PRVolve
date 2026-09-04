import json
import os
import tempfile
import unittest

from evoagent.review.agentic import AgenticReviewer
from evoagent.config import Settings
from evoagent.core.diff_parser import parse_unified_diff
from evoagent.eval.agentic import validate_real_dataset
from evoagent.evolution.candidates import RootCauseEvolutionGenerator
from evoagent.review.patching import apply_file_patch, parse_unified_patch
from evoagent.serving.service import ReviewService
from evoagent.store.sqlite import TaskStore
from evoagent.eval.verifier import RepairVerifier


DIFF = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+eval(user_input)\n"


class FakeChatClient:
    provider = "fake"
    model = "fake-model"

    def complete_json(self, role, system, user, ledger=None, max_tokens=None):
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 10, "completion_tokens": 5}, 1,
            )
        if role == "lead":
            managed = json.loads(user)
            task = json.loads(managed["task"])
            if task["phase"] == "delegate":
                return {
                    "action": "final", "delegations": [
                        {
                            "assignment_id": "security-1", "worker": "security",
                            "objective": "Trace user input", "files": ["app.py"],
                        },
                        {
                            "assignment_id": "reliability-1",
                            "worker": "correctness-reliability",
                            "objective": "Check failures", "files": ["app.py"],
                        },
                    ], "risk_level": "high",
                }
            if task["phase"] == "assess-workers":
                return {
                    "action": "final", "revision_requests": [],
                    "critic_objective": "Challenge every candidate.",
                }
            if task["phase"] == "finalize":
                return {
                    "action": "final",
                    "accepted_finding_indices": list(
                        range(len(task["candidate_findings"]))
                    ),
                    "confidence_adjustments": [],
                }
            raise AssertionError(task["phase"])
        if role == "security":
            return {
                "action": "final", "findings": [{
                    "rule_id": "SEC-EVAL", "severity": "critical",
                    "title": "Dynamic execution", "explanation": "User input reaches eval.",
                    "path": "app.py", "line": 1, "evidence": "eval(user_input)",
                    "call_chain": [{"path": "app.py", "line": 1, "symbol": "eval"}],
                    "fix": "Use a strict parser.", "test": "Pass malicious expressions.",
                    "confidence": 0.9,
                }],
            }
        if role == "correctness-reliability":
            return {"action": "final", "findings": []}
        if role == "critic":
            return {"action": "final", "decisions": [{
                "finding_index": 0, "accepted": True, "objections": [],
                "confidence_adjustment": 0.0,
            }]}
        if role == "evolution-root-cause":
            return {
                "clusters": [{"name": "weak evidence", "failure_case_ids": [1], "root_cause": "No call chain"}],
                "candidate": {
                    "prompt_additions": ["Require a call chain for high-risk claims."],
                    "few_shot_examples": [], "lead_delegation_rules": [],
                    "tool_selection_policy": [], "budget_parameters": {"critic": 2000},
                },
                "rationale": "Improve evidence quality.",
            }
        raise AssertionError(role)


class PhaseImplementationTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)

    def tearDown(self):
        try:
            os.unlink(self.path)
        except PermissionError:
            pass

    def settings(self):
        return Settings(
            host="127.0.0.1", port=8080, db_path=self.path, max_diff_bytes=10000,
            timeout_seconds=10, llm_base_url="", llm_api_key="",
            llm_model="", github_webhook_secret="", github_token="",
            auto_post_review=False,
        )

    def test_agentic_mode_rejects_missing_model_configuration(self):
        service = ReviewService(self.settings())
        try:
            with self.assertRaisesRegex(RuntimeError, "requires a configured model"):
                service.create_review("org/repo", DIFF, mode="agentic")
        finally:
            service.queue.close()

    def test_agentic_mode_runs_exact_four_real_roles(self):
        store = TaskStore(self.path)
        store.create("task", "org/repo", 1, {
            "mode": "agentic",
            "enabled_agents": ["lead", "security", "correctness-reliability", "critic"],
        })
        reviewer = AgenticReviewer(store, FakeChatClient())
        parsed = parse_unified_diff(DIFF)
        findings = reviewer.review_with_context("task", DIFF, parsed, "org/repo")
        summary = reviewer.collaboration_summary("task")
        self.assertEqual(["SEC-EVAL"], [item.rule_id for item in findings])
        self.assertEqual(7, summary["execution"]["llm_calls"])
        self.assertEqual(
            ["lead", "security", "correctness-reliability", "critic"],
            summary["collaboration"]["roles"],
        )
        for role in summary["collaboration"]["roles"]:
            decisions = [
                item for item in summary["execution"]["agent_traces"][role]
                if item["event"] == "autonomous_decision"
            ]
            self.assertTrue(decisions)

    def test_structured_evolution_candidate_has_diff_and_usage(self):
        generated = RootCauseEvolutionGenerator(FakeChatClient()).generate([
            {"id": 1, "category": "false_positive", "payload": {"note": "weak"}}
        ], "Review diff JSON severity fix test.")
        self.assertIn("Require a call chain", generated["candidate_prompt"])
        self.assertIn("candidate-prompt", generated["change_diff"])
        self.assertEqual(1, generated["generation"]["llm_calls"])

    def test_unified_patch_is_path_bounded_and_context_checked(self):
        patch = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n-value = eval(raw)\n+value = parse(raw)\n return value\n"
        parsed = parse_unified_patch(patch, ["app.py"])
        self.assertEqual("value = parse(raw)\nreturn value\n", apply_file_patch(
            "value = eval(raw)\nreturn value\n", parsed[0]
        ))
        with self.assertRaises(ValueError):
            parse_unified_patch(patch.replace("app.py", "../app.py"), ["../app.py"])

    def test_fix_success_requires_before_after_test_evidence(self):
        self.assertFalse(RepairVerifier.compare(
            {"passed": True, "checks": []}, {"passed": True, "checks": []}
        )["passed"])
        result = RepairVerifier.compare(
            {"passed": True, "checks": [{"name": "repository-tests", "passed": True}]},
            {"passed": True, "checks": [{"name": "repository-tests", "passed": True}]},
        )
        self.assertTrue(result["passed"])

    def test_real_dataset_gate_requires_300_repository_isolated_prs(self):
        cases = []
        for index in range(300):
            split = "train" if index < 100 else "validation" if index < 200 else "holdout"
            cases.append({
                "id": "pr-%d" % index, "repository": "%s/repo-%d" % (split, index),
                "pull_request": index + 1, "split": split,
                "source": {"kind": "public-github-pr"}, "diff": DIFF,
                "expected_findings": [{
                    "severity": "critical", "path": "app.py", "start_line": 1,
                    "should_comment": True,
                }],
            })
        self.assertTrue(validate_real_dataset(cases)["ready"])


if __name__ == "__main__":
    unittest.main()
