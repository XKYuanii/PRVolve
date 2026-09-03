import os
import tempfile
import unittest

from evoagent.core.models import TaskState
from evoagent.session.events import EventKind, EventLog
from evoagent.session.ledger import ExecutionLedger
from evoagent.session.projections import progress, report, task_state, trace
from evoagent.store.sqlite import TaskStore


class SessionLogTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.store.create("task", "org/repo", 1, {})

    def tearDown(self):
        os.unlink(self.path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(self.path + suffix):
                os.unlink(self.path + suffix)

    def test_accounting_is_persisted_with_the_stage_that_produced_it(self):
        log = EventLog.load(self.store, "task")
        log.append(EventKind.STAGE_STARTED, stage="review", message="reviewing")
        ledger = ExecutionLedger("agentic", log=log)
        ledger.record_model(
            "security", "fake", "fake", {"prompt_tokens": 10, "completion_tokens": 5}, 1
        )
        ledger.trace("security", "tool_observation", step=1)

        # Buffered until the stage completes, then stored together with it.
        self.assertEqual(1, len(self.store.load_events("task")))
        log.append(EventKind.STAGE_COMPLETED, stage="review", output={"findings": []})
        self.assertEqual(4, len(self.store.load_events("task")))

    def test_ledger_totals_continue_across_a_reload(self):
        log = EventLog.load(self.store, "task")
        first = ExecutionLedger("agentic", 1.0, 2.0, log=log)
        first.record_model(
            "lead", "fake", "fake", {"prompt_tokens": 1000, "completion_tokens": 500}, 5
        )
        log.flush()

        resumed = ExecutionLedger("agentic", 1.0, 2.0, log=EventLog.load(self.store, "task"))
        resumed.record_model(
            "security", "fake", "fake", {"prompt_tokens": 2000, "completion_tokens": 100}, 5
        )
        summary = resumed.summary()

        self.assertEqual(2, summary["llm_calls"])
        self.assertEqual(3600, summary["total_tokens"])
        self.assertEqual(2100, resumed.tokens_used("security"))
        self.assertEqual(1500, resumed.tokens_used("lead"))
        self.assertEqual(round(3000 / 1e6 + 600 * 2 / 1e6, 8), summary["cost_usd"])

    def test_state_trace_and_report_are_projections_of_one_log(self):
        log = EventLog.load(self.store, "task")
        log.append(EventKind.STAGE_STARTED, stage="parse", message="planning")
        log.append(EventKind.STAGE_COMPLETED, stage="parse", output={})
        log.append(EventKind.STAGE_STARTED, stage="review", message="reviewing")
        log.append(EventKind.STAGE_STARTED, stage="review.scan", message="scanning")
        log.append(EventKind.STAGE_COMPLETED, stage="review.scan", output={})
        log.append(EventKind.STAGE_STARTED, stage="finalize", message="ranking")
        log.append(EventKind.TASK_SUCCEEDED, message="Review completed", report={"risk": "low"})

        reloaded = EventLog.load(self.store, "task")
        self.assertEqual(TaskState.SUCCESS, task_state(reloaded))
        self.assertEqual(
            ["PLANNING", "EXECUTING", "REVIEWING", "SUCCESS"],
            [item.state.value for item in trace(reloaded)],
        )
        self.assertEqual({"risk": "low"}, report(reloaded))
        self.assertEqual("completed", progress(reloaded)["review.scan"]["status"])
        # The read model the API lists tasks from agrees with the projection.
        self.assertEqual("SUCCESS", self.store.get("task")["state"])

    def test_a_store_without_event_support_yields_an_in_memory_log(self):
        class InputOnlyStore:
            def get(self, _task_id, _tenant_id=None):
                return {"input": {}}

        log = EventLog(InputOnlyStore(), "task")
        log.append(EventKind.STAGE_COMPLETED, stage="review", output={})

        self.assertEqual(1, len(log.events()))
        self.assertEqual([], self.store.load_events("task"))


if __name__ == "__main__":
    unittest.main()
