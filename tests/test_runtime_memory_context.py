import os
import tempfile
import unittest
import json

from evoagent.review.agentic import AgentLoop
from evoagent.llm.context import ContextManager
from evoagent.agents.memory import MemoryManager
from evoagent.errors import ToolProtocolError
from evoagent.review.runtime import AgentRuntime
from evoagent.session.checkpoint import CheckpointLog
from evoagent.session.projections import progress
from evoagent.tools.registry import AgentTool, ToolRegistry
from evoagent.store.sqlite import TaskStore, utc_now
from evoagent.session.ledger import ExecutionLedger


class RuntimeMemoryTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_completed_stage_is_read_back_instead_of_rerun(self):
        self.store.create("staged-task", "org/repo", 1, {})
        calls = []

        def drive():
            runner = AgentRuntime(CheckpointLog.load(self.store, "staged-task"))
            planned = runner.run(
                "parse", lambda: calls.append("parse") or {"value": 2},
            )
            return runner.run(
                "review",
                lambda: calls.append("review") or {"result": planned["value"] * 3},
            )

        self.assertEqual(6, drive()["result"])
        self.assertEqual(6, drive()["result"])
        self.assertEqual(["parse", "review"], calls)

    def test_failed_stage_is_retried_and_its_attempts_are_recorded(self):
        self.store.create("retry-task", "org/repo", 1, {})
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient provider failure")
            return {"ok": True}

        log = CheckpointLog.load(self.store, "retry-task")
        output = AgentRuntime(log, retries=2).run("review", flaky)

        self.assertEqual({"ok": True}, output)
        self.assertEqual(3, progress(log)["review"]["attempt"])

    def test_value_error_is_not_retried(self):
        self.store.create("invalid-task", "org/repo", 1, {})
        attempts = []

        def invalid():
            attempts.append(1)
            raise ValueError("diff is not a unified diff")

        log = CheckpointLog.load(self.store, "invalid-task")
        with self.assertRaises(ValueError):
            AgentRuntime(log, retries=2).run("parse", invalid)

        self.assertEqual(1, len(attempts))
        self.assertEqual("failed", progress(log)["parse"]["status"])

    def test_tool_registry_validates_arguments_before_invocation(self):
        calls = []
        registry = ToolRegistry([AgentTool(
            "lookup", "Lookup one key.",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"], "additionalProperties": False,
            },
            lambda key: calls.append(key) or "value:%s" % key,
        )])

        with self.assertRaisesRegex(ToolProtocolError, "unknown tool arguments"):
            registry.invoke("lookup", {"key": 7, "unexpected": True})

        self.assertEqual([], calls)
        self.assertEqual("value:x", registry.invoke("lookup", {"key": "x"}))
        self.assertEqual(["x"], calls)

    def test_memory_recall_is_repository_and_tenant_isolated(self):
        memory = MemoryManager(self.store, recall_limit=5)
        memory.remember(
            "tenant-a", "org/repo", "semantic", "review_feedback",
            "SEC-EVAL was a confirmed missed issue in authentication code",
            importance=0.9,
        )
        memory.remember(
            "tenant-b", "org/repo", "semantic", "review_feedback",
            "REL-DEBUG-PRINT was accepted", importance=0.9,
        )

        recalled = memory.recall(
            "tenant-a", "org/repo", "authentication SEC-EVAL security"
        )

        self.assertEqual(1, len(recalled))
        self.assertIn("SEC-EVAL", recalled[0]["content"])
        self.assertEqual([], memory.recall("tenant-a", "org/other", "SEC-EVAL"))

    def test_memory_recall_purges_expired_records(self):
        self.store.save_agent_memory({
            "id": "expired-memory", "tenant_id": "tenant-a",
            "repository": "org/repo", "task_id": "old-task", "agent": "agent",
            "scope": "working", "kind": "observation", "content": "expired content",
            "keywords": ["expired"], "metadata": {}, "importance": 0.5,
            "created_at": utc_now(), "expires_at": "2000-01-01T00:00:00+00:00",
        })

        MemoryManager(self.store).recall("tenant-a", "org/repo", "expired")

        self.assertEqual([], self.store.list_agent_memories(
            "tenant-a", "org/repo", ("working",), 10
        ))

    def test_tool_observation_is_written_and_visible_to_next_agent_step(self):
        memory = MemoryManager(self.store, recall_limit=8)
        seen = []

        class ToolThenFinalClient:
            provider = "fake"
            model = "fake"

            def complete_json(_self, _role, _prompt, user, ledger=None, max_tokens=None):
                managed = json.loads(user)
                task = json.loads(managed["task"])
                seen.append(task.get("working_memory"))
                if ledger:
                    ledger.record_model("security", "fake", "fake", {
                        "prompt_tokens": 10, "completion_tokens": 3,
                    }, 1)
                if len(seen) == 1:
                    return {"action": "tool", "tool": "lookup", "arguments": {"key": "auth"}}
                return {"action": "final", "findings": []}

        def supplier():
            values = memory.recall_working("tenant-a", "org/repo", "task")
            return ContextManager.format_memories(values) if values else None

        def sink(agent, observation):
            memory.remember_observation(
                "tenant-a", "org/repo", "task", agent, observation,
            )

        role = AgentLoop(
            "security", "Use facts.", ToolThenFinalClient(), 1000, 10,
            context_manager=ContextManager(), working_memory_supplier=supplier,
            observation_sink=sink,
        )
        registry = ToolRegistry([AgentTool(
            "lookup", "Lookup a fact.", {
                "type": "object", "properties": {"key": {"type": "string"}},
                "required": ["key"], "additionalProperties": False,
            }, lambda key: {"evidence_id": "lookup:%s" % key, "output": {"value": "allowed"}},
        )])

        role.run(json.dumps({"phase": "worker"}), registry, ExecutionLedger("agentic"))

        self.assertIsNone(seen[0])
        self.assertEqual("tool_observation", seen[1]["items"][0]["kind"])
        self.assertIn("lookup:auth", seen[1]["items"][0]["content"])

    def test_preflight_observations_do_not_satisfy_the_verify_first_rule(self):
        """A role must run a tool of its own before it may conclude."""
        seen = []

        class AlwaysFinalClient:
            provider = "fake"
            model = "fake"

            def complete_json(_self, _role, _prompt, user, ledger=None, max_tokens=None):
                managed = json.loads(user)
                seen.append([
                    item.get("tool") for item in managed.get("observations") or []
                ])
                return {"action": "final", "findings": []}

        role = AgentLoop(
            "security", "Report only verified defects.", AlwaysFinalClient(),
            1000, 10, context_manager=ContextManager(), minimum_tool_calls=1,
        )
        registry = ToolRegistry([AgentTool(
            "lookup", "Lookup a fact.", {
                "type": "object", "properties": {"key": {"type": "string"}},
                "required": ["key"], "additionalProperties": False,
            }, lambda key: {"evidence_id": "lookup:%s" % key, "output": "fact"},
        )])
        preflight = [{"step": 0, "tool": "read_file", "ok": True, "result": "source"}]

        role.run(
            json.dumps({"phase": "worker"}), registry, ExecutionLedger("agentic"),
            initial_observations=preflight,
        )

        # The system-supplied preflight result does not count, so the role is
        # asked once; it is then taken at its word rather than looped to death.
        self.assertEqual(2, len(seen))
        self.assertIn("protocol-requirement", seen[1])

    def test_task_consolidation_releases_working_memory_but_keeps_episode(self):
        memory = MemoryManager(self.store)
        memory.remember_observation("tenant-a", "org/repo", "task", "security", {
            "step": 1, "tool": "symbol", "ok": True,
            "result": {"evidence_id": "symbol:auth", "output": {"callers": ["api"]}},
        })

        memory.consolidate_task("tenant-a", "org/repo", "task", {"accepted_findings": []})

        self.assertEqual([], memory.recall_working("tenant-a", "org/repo", "task"))
        archived = memory.recall("tenant-a", "org/repo", "completed task")
        self.assertEqual("task_summary", archived[0]["kind"])

    def test_working_memory_is_isolated_by_agent_role(self):
        memory = MemoryManager(self.store)
        memory.remember_observation("tenant-a", "org/repo", "task", "security", {
            "step": 1, "tool": "lookup", "ok": True,
            "result": {"evidence_id": "security:auth", "output": "security fact"},
        })
        memory.remember_observation("tenant-a", "org/repo", "task", "correctness-reliability", {
            "step": 1, "tool": "lookup", "ok": True,
            "result": {"evidence_id": "correctness:auth", "output": "correctness fact"},
        })

        security = memory.recall_working(
            "tenant-a", "org/repo", "task", agent="security",
        )
        critic = memory.recall_working(
            "tenant-a", "org/repo", "task", agent="critic",
        )

        self.assertEqual(1, len(security))
        self.assertIn("security:auth", security[0]["content"])
        self.assertEqual([], critic)


if __name__ == "__main__":
    unittest.main()
