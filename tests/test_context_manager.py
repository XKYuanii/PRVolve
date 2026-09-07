import json
import os
import tempfile
import unittest

from evoagent.review.agentic import AgenticReviewer
from evoagent.llm.context import ContextManager, estimate_tokens
from evoagent.core.diff_parser import parse_unified_diff
from evoagent.agents.memory import MemoryManager
from evoagent.store.sqlite import TaskStore


def large_diff():
    values = []
    for index in range(12):
        path = "src/module_%02d.py" % index
        content = "value_%d = transform(input_%d)" % (index, index)
        if index == 9:
            path = "src/auth.py"
            content = "result = eval(user_token)"
        values.extend([
            "--- a/%s" % path,
            "+++ b/%s" % path,
            "@@ -1,2 +1,3 @@ function_%d" % index,
            " context = True",
            "+%s" % content,
            "+audit(value_%d)" % index,
            *["+trace_%d_%d = normalize(value_%d)" % (index, line, index) for line in range(18)],
        ])
    return "\n".join(values) + "\n"


class CapturingClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.tasks = []

    def complete_json(self, role, _system, user, ledger=None, max_tokens=None):
        managed = json.loads(user)
        task = json.loads(managed["task"])
        self.tasks.append((role, task, managed))
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 20, "completion_tokens": 5}, 1,
            )
        if role == "lead" and task["phase"] == "delegate":
            return {"action": "final", "delegations": [{
                "assignment_id": "security-1", "worker": "security",
                "objective": "Review authentication and dynamic execution.",
                "files": ["src/auth.py"], "risk_domains": ["authorization"],
            }]}
        if role == "lead" and task["phase"] == "assess-workers":
            return {
                "action": "final", "revision_requests": [],
                "critic_objective": "Verify candidates.",
            }
        if role == "lead" and task["phase"] == "finalize":
            return {
                "action": "final", "accepted_finding_indices": [],
                "confidence_adjustments": [],
            }
        if role in {"security", "correctness-reliability"}:
            return {
                "action": "final", "findings": [],
                "hypotheses": [{
                    "hypothesis_id": "hyp-1",
                    "claim": "The compressed change may alter an assigned contract.",
                    "status": "unresolved",
                    "explanation": "This context fixture does not expose a repository checkout.",
                    "required_proof": "Inspect the full source and callers.",
                }],
            }
        if role == "critic":
            return {"action": "final", "decisions": []}
        raise AssertionError((role, task))


class ContextManagerTests(unittest.TestCase):
    def test_large_diff_uses_risk_ranked_hunk_map_reduce(self):
        manager = ContextManager(
            context_window_tokens=4096, input_token_budget=2500,
            diff_token_budget=900, map_chunk_tokens=160,
        )

        compressed = manager.compress_diff(
            large_diff(), "task", "security", focus_files=["src/auth.py"],
            risk_domains=["authorization", "token"],
        )

        self.assertEqual("semantic-diff-v1", compressed["format"])
        self.assertTrue(compressed["compression"]["applied"])
        self.assertGreater(compressed["compression"]["map_chunks"], 1)
        self.assertIn(
            "src/auth.py", {item["path"] for item in compressed["selected_hunks"]}
        )
        self.assertLess(compressed["compressed_estimated_tokens"], compressed["original_estimated_tokens"])
        self.assertLessEqual(estimate_tokens(compressed), 1000)

    def test_exact_candidate_location_outranks_unrelated_risky_hunk(self):
        diff = (
            "--- a/src/model.py\n+++ b/src/model.py\n"
            "@@ -1,2 +1,3 @@\n context = True\n+result = eval(user_input)\n"
            " return context\n"
            "@@ -199,2 +200,3 @@\n value = current\n+return value != other\n"
            " return value\n"
        )
        manager = ContextManager(
            context_window_tokens=2048, input_token_budget=800,
            diff_token_budget=340, map_chunk_tokens=160,
        )

        compressed = manager.compress_diff(
            diff, "task", "critic", focus_files=["src/model.py"],
            focus_locations=[("src/model.py", 201)],
        )

        self.assertIn("@@ -199,2 +200,3 @@", compressed["selected_hunks"][0]["header"])
        self.assertEqual(
            [{"path": "src/model.py", "line": 201}],
            compressed["focus"]["locations"],
        )

    def test_exact_candidate_line_survives_oversized_risky_hunk(self):
        body = [" context_%03d = value" % index for index in range(1, 90)]
        body[4] = "+timeout = 30"
        body[69] = "+result = records[0]"
        diff = (
            "--- a/src/model.py\n+++ b/src/model.py\n"
            "@@ -1,89 +1,89 @@\n" + "\n".join(body) + "\n"
        )
        manager = ContextManager(
            context_window_tokens=2048, input_token_budget=800,
            diff_token_budget=320,
        )

        compressed = manager.compress_diff(
            diff, focus_locations=[("src/model.py", 70)], max_tokens=320,
        )
        content = compressed["selected_hunks"][0]["content"]

        self.assertIn("result = records[0]", content)
        self.assertIn("timeout = 30", content)

    def test_exact_candidate_line_survives_final_payload_fit_at_minimum_budget(self):
        body = [" context_%03d = %s" % (index, "x" * 80) for index in range(120)]
        body[109] = "+focus_result = records[0]  # exact candidate location"
        diff = (
            "--- a/src/model.py\n+++ b/src/model.py\n"
            "@@ -1,120 +1,120 @@\n" + "\n".join(body) + "\n"
        )
        manager = ContextManager(
            context_window_tokens=1024, input_token_budget=512,
            diff_token_budget=256,
        )

        compressed = manager.compress_diff(
            diff, focus_locations=[("src/model.py", 110)], max_tokens=256,
        )

        self.assertIn(
            "focus_result = records[0]",
            compressed["selected_hunks"][0]["content"],
        )
        self.assertTrue(compressed["selected_hunks"][0]["contains_exact_focus"])
        self.assertTrue(compressed["compression"]["focus_locations_preserved"])


    def test_old_observations_are_summarized_and_evidence_is_retained(self):
        manager = ContextManager(
            observation_token_budget=350, recent_observations=1,
        )
        observations = [
            {
                "step": index, "tool": "read_file", "ok": True,
                "result": {
                    "evidence_id": "read_file:%d" % index,
                    "tool": "read_file",
                    "output": {"path": "app.py", "content": "x" * 1600},
                },
            }
            for index in range(1, 5)
        ]

        compact, stats = manager.compact_observations(observations, 350)

        self.assertGreater(stats["summarized"], 0)
        rendered = json.dumps(compact)
        self.assertIn("read_file:4", rendered)
        self.assertLess(estimate_tokens(compact), 500)

    def test_managed_context_is_trimmed_before_model_call(self):
        manager = ContextManager(
            context_window_tokens=3000, input_token_budget=1200,
            observation_token_budget=300, recent_observations=1,
        )
        task = json.dumps({
            "phase": "review", "diff": "a" * 12000,
            "candidate_findings": [
                {"finding_index": index, "explanation": "b" * 1200}
                for index in range(8)
            ],
        })

        managed, stats = manager.build_managed_context(
            task, [], [], 4000, 60, system_prompt="review", max_output_tokens=512,
        )

        self.assertLessEqual(stats["estimated_input_tokens_after"], stats["input_token_limit"])
        compact_task = json.loads(managed["task"])
        self.assertEqual(8, len(compact_task["candidate_findings"]))
        self.assertEqual(7, compact_task["candidate_findings"][7]["finding_index"])


class ContextMemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_review_recall_and_semantic_diff_reach_agent_context(self):
        diff = large_diff()
        self.store.create("task", "org/repo", 1, {
            "mode": "agentic", "enabled_agents": ["lead", "security"],
        }, "tenant-a")
        memory = MemoryManager(self.store)
        memory.remember(
            "tenant-a", "org/repo", "semantic", "review_feedback",
            "Dynamic execution in auth token handling was a confirmed missed issue.",
            importance=0.95,
        )
        client = CapturingClient()
        manager = ContextManager(
            context_window_tokens=6000, input_token_budget=4000,
            diff_token_budget=1200,
        )
        reviewer = AgenticReviewer(
            self.store, client, memory_manager=memory, context_manager=manager,
        )

        reviewer.review_with_context(
            "task", diff, parse_unified_diff(diff), "org/repo", "tenant-a",
        )
        summary = reviewer.collaboration_summary("task")

        self.assertTrue(client.tasks)
        for _role, task, managed in client.tasks:
            self.assertIsInstance(task["diff"], str)
            self.assertIn("semantic-diff-v1", task["diff"])
            self.assertEqual("semantic-diff-v1", task["diff_context"]["format"])
            self.assertTrue(task["recalled_memory"]["items"])
            self.assertLessEqual(
                estimate_tokens(managed), managed["context_policy"]["input_token_limit"],
            )
        context = summary["context_management"]
        self.assertTrue(context["semantic_compression_enabled"])
        self.assertGreater(context["compression_calls"], 0)
        self.assertEqual(1, context["memory_recall"]["recalled"])
        self.assertIn("context_management", summary["execution"])

    def test_assignment_filters_its_files_before_scoreable_line_limit(self):
        first = "\n".join("+value_%03d = %d" % (index, index) for index in range(205))
        diff = (
            "--- a/src/first.py\n+++ b/src/first.py\n@@ -0,0 +1,205 @@\n"
            + first
            + "\n--- a/src/later.py\n+++ b/src/later.py\n@@ -0,0 +1 @@\n"
            "+result = records[0]\n"
        )

        class LaterFileClient(CapturingClient):
            def complete_json(self, role, system, user, ledger=None, max_tokens=None):
                managed = json.loads(user)
                task = json.loads(managed["task"])
                if role == "lead" and task.get("phase") == "delegate":
                    self.tasks.append((role, task, managed))
                    if ledger:
                        ledger.record_model(role, self.provider, self.model, {}, 1)
                    return {"action": "final", "delegations": [{
                        "assignment_id": "reliability-1",
                        "worker": "correctness-reliability",
                        "objective": "Review the later file.",
                        "files": ["src/later.py"],
                    }]}
                return super().complete_json(
                    role, system, user, ledger=ledger, max_tokens=max_tokens,
                )

        self.store.create("later-task", "org/repo", 2, {"mode": "agentic"})
        client = LaterFileClient()
        AgenticReviewer(self.store, client).review_with_context(
            "later-task", diff, parse_unified_diff(diff), "org/repo",
        )
        worker_task = next(
            task for role, task, _managed in client.tasks
            if role == "correctness-reliability" and task.get("lead_assignment")
        )

        self.assertEqual([{
            "path": "src/later.py", "line": 1,
            "content": "result = records[0]",
        }], worker_task["scoreable_added_lines"])


if __name__ == "__main__":
    unittest.main()
