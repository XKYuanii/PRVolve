import unittest

from evoagent.eval.adjudication import promote_confirmed_judgments


class DatasetAdjudicationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{
            "id": "pr-1",
            "repository": "owner/repo",
            "pull_request": 1,
            "split": "validation",
            "schema_version": 1,
            "diff": "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,2 @@\n+eval(value)\n+value.name\n",
            "expected_findings": [{
                "path": "app.py", "start_line": 1, "end_line": 1,
                "cwe": "CWE-95", "severity": "critical",
            }],
        }]

    def test_promotes_only_required_judgments_and_preserves_optional_labels(self):
        revised, manifest = promote_confirmed_judgments(self.cases, {
            "status": "confirmed",
            "reviewer": "owner",
            "reviewed_at": "2026-08-30",
            "reviewed_suggestions": 2,
            "cases": {"pr-1": [
                {
                    "path": "app.py", "line": 2, "rule_id": "CWE-476",
                    "severity": "high", "verdict": "required",
                },
                {
                    "path": "app.py", "line": 2, "rule_id": "CWE-561",
                    "severity": "low", "verdict": "optional",
                },
            ]},
        })

        self.assertEqual(1, self.cases[0]["schema_version"])
        self.assertEqual(2, revised[0]["schema_version"])
        self.assertEqual(2, len(revised[0]["expected_findings"]))
        self.assertTrue(all(
            item["should_comment"] for item in revised[0]["expected_findings"]
        ))
        self.assertEqual(2, len(revised[0]["suggestion_judgments"]))
        self.assertEqual(1, manifest["required_labels_added"])
        self.assertEqual({"optional": 1, "required": 1}, manifest["verdicts"])

    def test_rejects_unconfirmed_payload(self):
        with self.assertRaisesRegex(ValueError, "only confirmed"):
            promote_confirmed_judgments(self.cases, {
                "status": "provisional", "cases": {},
            })


if __name__ == "__main__":
    unittest.main()
