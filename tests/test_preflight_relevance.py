import tempfile
import unittest
from pathlib import Path

from evoagent.core.diff_parser import parse_unified_diff
from evoagent.review.preflight import code_identifiers, repository_preflight
from evoagent.tools.repository import RepositoryToolSuite


class PreflightRelevanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def preflight(self, path, line, content):
        diff = "--- a/%s\n+++ b/%s\n@@ -%d +%d @@\n-old_value\n+%s\n" % (
            path, path, line, line, content,
        )
        parsed = parse_unified_diff(diff)
        suite = RepositoryToolSuite(str(self.root), diff, parsed)
        observations = repository_preflight({"files": [path]}, parsed, suite.registry("correctness-reliability"))
        return observations

    def test_identifiers_exclude_comments_strings_keywords_and_keep_partial_code(self):
        names = code_identifiers('    for row in self.fetch_rows(): # Where does this come from?\n'
                                 '        emit(row, "unrelated_search_word")\n')
        self.assertIn("fetch_rows", names)
        self.assertNotIn("Where", names)
        self.assertNotIn("for", names)
        self.assertNotIn("unrelated_search_word", names)
        self.assertIn("fetch_rows", code_identifiers("value = fetch_rows(\n"))

    def test_identifiers_do_not_require_supported_ast_syntax(self):
        self.assertIn("convert_value", code_identifiers("def convert_value[T](item: T) -> T:\n"))

    def test_first_pass_carries_one_exact_old_new_delta(self):
        self.write("pkg/source.py", "result = consume(value)\n")

        observations = self.preflight(
            "pkg/source.py", 1, "result = consume(value)",
        )

        deltas = [
            item for item in observations
            if item["tool"] == "changed_line" and item["ok"] is True
        ]
        self.assertEqual(1, len(deltas))
        self.assertEqual("old_value", deltas[0]["result"]["output"]["change"]["before"])
        self.assertEqual(
            "result = consume(value)",
            deltas[0]["result"]["output"]["change"]["after"],
        )

    def test_context_ranking_and_whole_identifier_do_not_change_default_text_search(self):
        self.write("aaa.py", "filenames = []\nfilename = 1\n")
        self.write("pkg/source.py", "fileName = raw\n")
        self.write("tests/consumer.py", "consume(fileName)\n")
        suite = RepositoryToolSuite(str(self.root), "", parse_unified_diff(""))
        found = suite.search_repository("fileName", context_path="pkg/source.py", identifier=True)["output"]
        self.assertEqual(["pkg/source.py", "tests/consumer.py"], [item["path"] for item in found])
        self.assertIn("aaa.py", [item["path"] for item in suite.search_repository("filename")["output"]])
        with self.assertRaises(ValueError):
            suite.search_repository("not an identifier", identifier=True)

    def test_same_file_producer_is_read_instead_of_unrelated_cross_file_match(self):
        source = ["# filler"] * 110
        source[9:12] = ["def fetch_rows():", "    return [{'entry': 1}]", ""]
        source[99] = "entries = fetch_rows()"
        self.write("pkg/source.py", "\n".join(source))
        self.write("aaa.py", "def unrelated_entries():\n    return None\n")
        observations = self.preflight("pkg/source.py", 100, source[99])
        bodies = [o["result"]["output"] for o in observations if o["tool"] == "read_file"]
        self.assertEqual(2, len(bodies))
        self.assertEqual("pkg/source.py", bodies[-1]["path"])
        self.assertIn("return [{'entry': 1}]", bodies[-1]["content"])
        self.assertLessEqual(len([o for o in observations if o["tool"] == "search_repository"]), 2)

    def test_actual_test_call_beats_multiline_import_and_registry_reference(self):
        source = ["# filler"] * 130
        source[59:61] = ["def convert_value(value):", "    return value"]
        source[119] = "REGISTRY = {'convert': convert_value}"
        self.write("pkg/source.py", "\n".join(source))
        test = ["# filler"] * 90
        test[0:3] = ["from pkg.source import (", "    convert_value,", ")"]
        test[69:72] = ["def test_output():", "    result = convert_value(1)", "    assert result == 1"]
        self.write("tests/test_output.py", "\n".join(test))
        observations = self.preflight("pkg/source.py", 60, source[59])
        bodies = [o["result"]["output"] for o in observations if o["tool"] == "read_file"]
        self.assertEqual("tests/test_output.py", bodies[-1]["path"])
        self.assertIn("assert result == 1", bodies[-1]["content"])
        self.assertGreater(bodies[-1]["start_line"], 3)

    def test_comment_only_change_does_not_invent_code_searches(self):
        self.write("pkg/source.py", "# Where did this come from?\n")
        observations = self.preflight("pkg/source.py", 1, "# Where did this come from?")
        self.assertFalse(any(o["tool"] == "search_repository" for o in observations))

    def test_revision_does_not_force_another_preflight(self):
        class ForbiddenTools:
            def names(self):
                raise AssertionError("revision must keep tool use optional")

        self.assertEqual([], repository_preflight(
            {"prior_worker_result": {"hypotheses": []}}, parse_unified_diff(""), ForbiddenTools(),
        ))
