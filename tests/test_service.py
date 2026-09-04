import os
import tempfile
import unittest

from evoagent.config import Settings
from evoagent.serving.service import ReviewService
from agentic_fake import enable_agentic_service


class ServiceTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.settings = Settings(
            host="127.0.0.1", port=8080, db_path=self.path, max_diff_bytes=10000,
            timeout_seconds=10, llm_base_url="", llm_api_key="", llm_model="",
            github_webhook_secret="", github_token="", auto_post_review=False,
        )

    def tearDown(self):
        os.unlink(self.path)

    def test_end_to_end_review(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(data)\n"
        service = enable_agentic_service(ReviewService(self.settings))
        result = service.create_review("org/repo", diff, 1)
        task = service.store.get(result["task_id"])
        service.queue.close()
        self.assertEqual("SUCCESS", result["state"])
        self.assertEqual("SEC-EVAL", result["report"]["findings"][0]["rule_id"])
        self.assertEqual("agentic", result["report"]["run_mode"]["effective"])
        self.assertEqual(
            ["lead", "security", "correctness-reliability", "critic"],
            result["report"]["collaboration"]["roles"],
        )
        self.assertEqual(6, result["report"]["execution"]["llm_calls"])
        self.assertGreater(result["report"]["execution"]["tool_calls"], 0)
        self.assertEqual([], task["collaboration"])

    def test_rejects_large_diff(self):
        service = enable_agentic_service(ReviewService(self.settings))
        with self.assertRaises(ValueError):
            service.create_review("org/repo", "x" * 10001)

    def test_completed_review_feedback_is_persisted_and_listed_per_task(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+eval(data)\n"
        service = enable_agentic_service(ReviewService(self.settings))
        result = service.create_review("org/repo", diff, 1)
        task_id = result["task_id"]

        feedback = service.record_feedback(
            task_id, "false_positive", result["report"]["findings"][0], "不是实际风险",
        )

        self.assertEqual({"recorded": True, "category": "false_positive"}, feedback)
        cases = service.store.list_task_failure_cases(task_id, "default")
        self.assertEqual(1, len(cases))
        self.assertEqual("false_positive", cases[0]["category"])
        self.assertEqual("SEC-EVAL", cases[0]["payload"]["finding"]["rule_id"])
        service.queue.close()


if __name__ == "__main__":
    unittest.main()
