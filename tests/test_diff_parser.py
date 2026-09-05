import unittest

from evoagent.core.diff_parser import changed_block, parse_unified_diff


DIFF = """diff --git a/app.py b/app.py
index 123..456 100644
--- a/app.py
+++ b/app.py
@@ -2,3 +2,4 @@ def run():
 keep = True
-old = 1
+new = 2
+eval(user_input)
 tail = 3
"""


class DiffParserTests(unittest.TestCase):
    def test_edit_evidence_keeps_both_sides_and_all_added_lines(self):
        expected = {"before": "old = 1", "after": "new = 2\neval(user_input)", "complete": True}
        self.assertEqual(expected, changed_block(DIFF, "app.py", 3))
        self.assertEqual(expected, changed_block(DIFF, "app.py", 4))
        self.assertEqual({}, changed_block(DIFF, "app.py", 2))
        self.assertEqual({}, changed_block(DIFF, "other.py", 3))

    def test_edit_evidence_supports_reversed_patch_order_and_multiple_hunks(self):
        diff = (
            "--- a/lib.py\n+++ b/lib.py\n@@ -1 +1 @@\n"
            "+new_first()\n-old_first()\n"
            "@@ -10 +10 @@\n+new_second()\n-old_second()\n"
            "\\ No newline at end of file\n"
            "diff --git a/other.py b/other.py\n--- a/other.py\n+++ b/other.py\n"
            "@@ -0,0 +1 @@\n+created()\n"
        )
        self.assertEqual("old_first()", changed_block(diff, "lib.py", 1)["before"])
        self.assertEqual("old_second()", changed_block(diff, "lib.py", 10)["before"])
        self.assertEqual("new_second()", changed_block(diff, "lib.py", 10)["after"])
        self.assertEqual("", changed_block(diff, "other.py", 1)["before"])
        self.assertEqual({}, changed_block(diff, "other.py", 10))

    def test_large_edit_is_explicitly_incomplete(self):
        diff = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+" + "x" * 4001
        self.assertFalse(changed_block(diff, "a.py", 1)["complete"])

    def test_parses_added_line_numbers(self):
        parsed = parse_unified_diff(DIFF)
        self.assertEqual(["app.py"], parsed.files)
        self.assertEqual([(3, "new = 2"), (4, "eval(user_input)")], [(x.line, x.content) for x in parsed.added_lines])


if __name__ == "__main__":
    unittest.main()

