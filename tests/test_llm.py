import json
import unittest
import urllib.error
from unittest import mock

from evoagent.llm.client import JsonChatClient
from evoagent.session.ledger import ExecutionLedger


class FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self.payload = json.dumps({
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        }).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class JsonChatClientTests(unittest.TestCase):
    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_truncated_critic_regenerates_from_task_once_with_same_output_cap(self, urlopen):
        urlopen.side_effect = [
            FakeResponse('{"action":"final","decisions":[', finish_reason="length"),
            FakeResponse('{"action":"final","decisions":[]}'),
        ]
        client = JsonChatClient("https://example.test", "secret", "model")
        ledger = ExecutionLedger("test")
        client.complete_json("critic", "role instructions", "original task", ledger, max_tokens=4000)
        retry = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual("role instructions", retry["messages"][0]["content"])
        self.assertIn("original task", retry["messages"][1]["content"])
        self.assertEqual(4000, retry["max_tokens"])
        self.assertEqual(2, ledger.summary()["llm_calls"])

    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_structured_output_repair_stops_after_one_failed_retry(self, urlopen):
        urlopen.return_value = FakeResponse('{"action":')
        client = JsonChatClient("https://example.test", "secret", "model")
        ledger = ExecutionLedger("test")
        with self.assertRaises(RuntimeError):
            client.complete_json("critic", "system", "task", ledger)
        self.assertEqual(2, urlopen.call_count)
        self.assertEqual(2, ledger.summary()["failed_model_calls"])

    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_invalid_structured_output_is_retried_and_accounted(self, urlopen):
        urlopen.side_effect = [FakeResponse('{"action":"final" "findings":[]}'), FakeResponse(
            '{"action":"final","findings":[]}'
        )]
        ledger = ExecutionLedger("test")
        client = JsonChatClient("https://example.test", "secret", "model")

        result = client.complete_json("lead", "system", "task", ledger)

        self.assertEqual("final", result["action"])
        self.assertEqual(2, urlopen.call_count)
        calls = ledger.summary()["model_call_log"]
        self.assertFalse(calls[0]["ok"])
        self.assertTrue(calls[1]["ok"])
        retry_payload = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(2, len(retry_payload["messages"]))
        self.assertIn("JSON syntax repair engine", retry_payload["messages"][0]["content"])
        self.assertIn("malformed JSON object", retry_payload["messages"][-1]["content"])

    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_request_timeout_is_bounded_by_call_deadline(self, urlopen):
        urlopen.return_value = FakeResponse('{"action":"final"}')
        client = JsonChatClient(
            "https://example.test", "secret", "model", timeout=60,
        )

        client.complete_json(
            "lead", "system", "task", timeout_seconds=0.2,
        )

        self.assertLessEqual(urlopen.call_args.kwargs["timeout"], 0.2)

    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_json_markdown_fence_is_removed_locally(self, urlopen):
        urlopen.return_value = FakeResponse('```json\n{"action":"final"}\n```')
        client = JsonChatClient("https://example.test", "secret", "model")

        self.assertEqual("final", client.complete_json("lead", "system", "task")["action"])
        self.assertEqual(1, urlopen.call_count)

    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_concatenated_json_objects_keep_first_action_and_are_audited(self, urlopen):
        urlopen.return_value = FakeResponse(
            '{"action":"final","findings":[]} {"duplicate":true}'
        )
        ledger = ExecutionLedger("test")
        client = JsonChatClient("https://example.test", "secret", "model")

        result = client.complete_json("lead", "system", "task", ledger)

        self.assertEqual("final", result["action"])
        self.assertEqual(1, urlopen.call_count)
        events = ledger.summary()["agent_traces"]["lead"]
        ignored = next(
            item for item in events
            if item["event"] == "structured_json_extra_values_ignored"
        )
        self.assertEqual(1, ignored["trailing_values"])

    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_json_with_trailing_prose_is_still_retried(self, urlopen):
        urlopen.side_effect = [
            FakeResponse('{"action":"final"} explanation'),
            FakeResponse('{"action":"final"}'),
        ]
        client = JsonChatClient("https://example.test", "secret", "model")

        self.assertEqual("final", client.complete_json("lead", "system", "task")["action"])
        self.assertEqual(2, urlopen.call_count)

    @mock.patch("evoagent.llm.client.time.sleep")
    @mock.patch("evoagent.llm.client.urllib.request.urlopen")
    def test_transient_transport_failure_is_retried_and_accounted(
        self, urlopen, sleep,
    ):
        urlopen.side_effect = [
            urllib.error.URLError("temporary reset"),
            FakeResponse('{"action":"final","findings":[]}'),
        ]
        ledger = ExecutionLedger("test")
        client = JsonChatClient("https://example.test", "secret", "model")

        result = client.complete_json("security", "system", "task", ledger)

        self.assertEqual("final", result["action"])
        self.assertEqual(2, urlopen.call_count)
        sleep.assert_called_once()
        calls = ledger.summary()["model_call_log"]
        self.assertFalse(calls[0]["ok"])
        self.assertTrue(calls[1]["ok"])


if __name__ == "__main__":
    unittest.main()
