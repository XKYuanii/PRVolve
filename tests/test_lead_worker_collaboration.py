import json
import os
import tempfile
import unittest

from evoagent.review.agentic import AgenticReviewer
from evoagent.core.diff_parser import parse_unified_diff
from evoagent.agents.memory import MemoryManager
from evoagent.session.checkpoint import CheckpointLog
from evoagent.session.projections import progress
from evoagent.store.sqlite import TaskStore


DIFF = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+eval(user_input)\n"


class HierarchicalClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.calls = []
        self.security_calls = 0

    def complete_json(self, role, _system, user, ledger=None, max_tokens=None):
        managed = json.loads(user)
        task = json.loads(managed["task"])
        self.calls.append((role, task.get("phase", "worker")))
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 10, "completion_tokens": 5}, 1,
            )
        if role == "lead":
            if task["phase"] == "delegate":
                return {
                    "action": "final", "delegations": [
                        {
                            "assignment_id": "security-1", "worker": "security",
                            "objective": "Trace the changed input into dangerous calls.",
                        },
                        {
                            "assignment_id": "reliability-1",
                            "worker": "correctness-reliability",
                            "objective": "Review failure and resource behavior.",
                        },
                    ],
                }
            if task["phase"] == "assess-workers" and task["revision_round"] == 0:
                return {
                    "action": "final", "revision_requests": [{
                        "assignment_id": "security-1", "worker": "security",
                        "guidance": "Add an exact changed-line finding for dynamic execution.",
                        "required_evidence": ["changed-line evidence"],
                    }], "critic_objective": "Challenge every proposed finding.",
                }
            if task["phase"] == "assess-workers":
                return {
                    "action": "final", "revision_requests": [],
                    "critic_objective": "Challenge every proposed finding.",
                }
            if task["phase"] == "finalize":
                return {
                    "action": "final",
                    "accepted_finding_indices": list(
                        range(len(task["candidate_findings"]))
                    ),
                    "confidence_adjustments": [],
                    "resolution_summary": "Workers supplied evidence and Critic approved.",
                }
        if role == "security":
            self.security_calls += 1
            if self.security_calls == 1:
                return {"action": "final", "findings": []}
            return {"action": "final", "findings": [{
                "rule_id": "SEC-LEAD-REVISION", "severity": "medium",
                "title": "Dynamic execution", "explanation": "Input is executed as code.",
                "path": "app.py", "line": 1, "evidence": "eval(user_input)",
                "fix": "Use a constrained parser.",
                "test": "Prove expressions are treated as data.", "confidence": 0.9,
            }]}
        if role == "correctness-reliability":
            return {"action": "final", "findings": []}
        if role == "critic":
            return {
                "action": "final", "decisions": [
                    {
                        "finding_index": index, "accepted": True,
                        "introduced_by_diff": True,
                        "reproducible": True,
                        "evidence_sufficient": True,
                        "would_comment_on_real_pr": True,
                        "objections": [], "confidence_adjustment": 0.0,
                    }
                    for index, _item in enumerate(task["candidates"])
                ],
            }
        raise AssertionError((role, task))


class LeadWorkerCollaborationTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.store.create("task", "org/repo", 1, {
            "mode": "agentic",
            "enabled_agents": [
                "lead", "security", "correctness-reliability", "critic",
            ],
        })

    def tearDown(self):
        os.unlink(self.path)

    def test_lead_delegates_requests_revision_and_synthesizes(self):
        client = HierarchicalClient()
        reviewer = AgenticReviewer(self.store, client)

        findings = reviewer.review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )
        summary = reviewer.collaboration_summary("task")

        self.assertNotIn("SEC-LEAD-REVISION", {item.rule_id for item in findings})
        self.assertIn(
            "SEC-LEAD-REVISION",
            {item["rule_id"] for item in summary["suggested_findings"]},
        )
        self.assertIn("SEC-EVAL", {item.rule_id for item in findings})
        self.assertEqual("lead-workers", summary["collaboration"]["protocol"])
        self.assertEqual(2, client.security_calls)
        self.assertEqual(1, len(summary["collaboration"]["revision_results"]))
        self.assertEqual("lead-final", summary["collaboration"]["stop_reason"])
        checkpoints = {
            item["event"]
            for item in summary["execution"]["agent_traces"]["lead-session"]
        }
        self.assertTrue({
            "assignment_created", "worker_reported", "revision_completed",
            "lead_activated", "lead_completed",
        }.issubset(checkpoints))
        nodes = progress(CheckpointLog.load(self.store, "task"))
        self.assertEqual("completed", nodes["planning.scan"]["status"])
        self.assertEqual("completed", nodes["planning.delegate"]["status"])
        self.assertEqual("completed", nodes["executing.work:security-1"]["status"])
        self.assertEqual("completed", nodes["reviewing.arbitrate"]["status"])

    def test_completed_session_resumes_without_repeating_agent_calls(self):
        first = HierarchicalClient()
        AgenticReviewer(self.store, first).review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )
        resumed_client = HierarchicalClient()
        resumed = AgenticReviewer(self.store, resumed_client)

        findings = resumed.review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )

        summary = resumed.collaboration_summary("task")
        self.assertNotIn("SEC-LEAD-REVISION", {item.rule_id for item in findings})
        self.assertIn(
            "SEC-LEAD-REVISION",
            {item["rule_id"] for item in summary["suggested_findings"]},
        )
        self.assertEqual([], resumed_client.calls)
        self.assertGreater(
            resumed.collaboration_summary("task")["execution"]["llm_calls"], 0
        )

    def test_worker_killed_mid_run_resumes_without_repeating_finished_workers(self):
        class KillingClient(HierarchicalClient):
            def complete_json(self, role, system, user, ledger=None, max_tokens=None):
                task = json.loads(json.loads(user)["task"])
                assignment = task.get("lead_assignment")
                if isinstance(assignment, dict) and assignment["worker"] == (
                    "correctness-reliability"
                ):
                    # BaseException models the process dying rather than the
                    # worker reporting a failure, which is caught by design.
                    raise KeyboardInterrupt("worker process killed")
                return super().complete_json(role, system, user, ledger, max_tokens)

        killed = KillingClient()
        with self.assertRaises(KeyboardInterrupt):
            AgenticReviewer(self.store, killed).review_with_context(
                "task", DIFF, parse_unified_diff(DIFF), "org/repo"
            )
        nodes = progress(CheckpointLog.load(self.store, "task"))
        self.assertEqual("completed", nodes["executing.work:security-1"]["status"])
        self.assertEqual("running", nodes["executing.work:reliability-1"]["status"])

        resumed_client = HierarchicalClient()
        AgenticReviewer(self.store, resumed_client).review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )

        # The scan, the delegation and the finished security worker are read
        # back from the log; only the worker that died is started again. The
        # second security call is the Lead's revision request, a new stage.
        nodes = progress(CheckpointLog.load(self.store, "task"))
        self.assertNotIn(("lead", "delegate"), resumed_client.calls)
        self.assertEqual(1, nodes["planning.scan"]["attempt"])
        self.assertEqual(1, nodes["planning.delegate"]["attempt"])
        self.assertEqual(1, nodes["executing.work:security-1"]["attempt"])
        self.assertEqual(2, nodes["executing.work:reliability-1"]["attempt"])

    def test_gate_decisions_are_archived_for_future_agent_recall(self):
        memory = MemoryManager(self.store)
        reviewer = AgenticReviewer(self.store, HierarchicalClient(), memory_manager=memory)

        reviewer.review_with_context("task", DIFF, parse_unified_diff(DIFF), "org/repo")

        episodes = memory.recall("default", "org/repo", "SEC-EVAL")
        self.assertTrue(any(item["kind"] == "finding_approved" for item in episodes))
        self.assertTrue(any(item["kind"] == "task_summary" for item in episodes))
        unverified = memory.recall("default", "org/repo", "SEC-LEAD-REVISION")
        self.assertFalse(any(item["kind"] == "finding_approved" for item in unverified))


if __name__ == "__main__":
    unittest.main()
