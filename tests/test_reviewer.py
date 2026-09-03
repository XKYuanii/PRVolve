import unittest

from evoagent.core.diff_parser import parse_unified_diff
from evoagent.review.reviewers import LocalRuleReviewer


class LocalReviewerTests(unittest.TestCase):
    def test_detects_security_findings_only_on_added_lines(self):
        diff = """--- a/app.py
+++ b/app.py
@@ -1,2 +1,3 @@
-eval(old_input)
+password = "super-secret"
+eval(user_input)
 safe = True
"""
        findings = LocalRuleReviewer().review(diff, parse_unified_diff(diff))
        self.assertEqual({"SEC-EVAL", "SEC-HARDCODED-SECRET"}, {item.rule_id for item in findings})
        self.assertTrue(all(item.line in {1, 2} for item in findings))

    def test_ignores_explicit_placeholder_secret_but_keeps_real_literal(self):
        diff = (
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,2 @@\n"
            "+token = 'test-placeholder'\n"
            "+api_key = 'production-secret-value'\n"
        )
        findings = LocalRuleReviewer().review(diff, parse_unified_diff(diff))

        self.assertEqual(1, len(findings))
        self.assertEqual("api_key = 'production-secret-value'", findings[0].evidence)

    def test_debug_print_rule_ignores_github_form_examples(self):
        diff = (
            "--- /dev/null\n+++ b/.github/DISCUSSION_TEMPLATE/questions.yml\n"
            "@@ -0,0 +1,2 @@\n+placeholder: |\n+  print(example)\n"
            "--- /dev/null\n+++ b/app.py\n@@ -0,0 +1 @@\n+print(secret)\n"
        )

        findings = LocalRuleReviewer().review(diff, parse_unified_diff(diff))

        self.assertEqual(1, len(findings))
        self.assertEqual("app.py", findings[0].path)
        self.assertEqual("REL-DEBUG-PRINT", findings[0].rule_id)

    def test_empty_except_rule_requires_a_pass_body(self):
        swallowed = (
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,2 @@\n"
            "+except Exception:\n+    pass\n"
        )
        cleanup_and_reraise = (
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,3 @@\n"
            "+except Exception:\n+    cleanup()\n+    raise\n"
        )

        swallowed_findings = LocalRuleReviewer().review(
            swallowed, parse_unified_diff(swallowed),
        )
        safe_findings = LocalRuleReviewer().review(
            cleanup_and_reraise, parse_unified_diff(cleanup_and_reraise),
        )

        self.assertEqual(
            ["REL-EMPTY-EXCEPT"],
            [item.rule_id for item in swallowed_findings],
        )
        self.assertEqual([], safe_findings)

    def test_detects_jinja_sandbox_downgrade_but_not_security_fix(self):
        regression = (
            "--- a/environment.py\n+++ b/environment.py\n@@ -1,2 +1,2 @@\n"
            "-from jinja2.sandbox import SandboxedEnvironment\n"
            "+from jinja2 import Environment\n"
            "-env = SandboxedEnvironment()\n+env = Environment()\n"
            "-return SandboxedEnvironment()\n+return Environment()\n"
        )
        security_fix = (
            "--- a/environment.py\n+++ b/environment.py\n@@ -1,2 +1,2 @@\n"
            "-from jinja2 import Environment\n"
            "+from jinja2.sandbox import SandboxedEnvironment\n"
            "-env = Environment()\n+env = SandboxedEnvironment()\n"
        )

        findings = LocalRuleReviewer().review(
            regression, parse_unified_diff(regression),
        )
        fixed_findings = LocalRuleReviewer().review(
            security_fix, parse_unified_diff(security_fix),
        )

        self.assertEqual(["SEC-JINJA-UNSANDBOXED"], [item.rule_id for item in findings])
        self.assertEqual(2, findings[0].line)
        self.assertEqual([], fixed_findings)

    def test_detects_insecure_jwt_default_but_not_secure_default(self):
        regression = (
            "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n"
            "-verify = options.get('verify_signature', True)\n"
            "+verify = options.get('verify_signature', False)\n"
        )
        security_fix = (
            "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n"
            "-verify = options.get('verify_signature', False)\n"
            "+verify = options.get('verify_signature', True)\n"
        )

        findings = LocalRuleReviewer().review(
            regression, parse_unified_diff(regression),
        )
        fixed_findings = LocalRuleReviewer().review(
            security_fix, parse_unified_diff(security_fix),
        )

        self.assertEqual(
            ["SEC-JWT-SIGNATURE-DISABLED"], [item.rule_id for item in findings],
        )
        self.assertEqual([], fixed_findings)

        test_fixture = regression.replace("a/auth.py", "a/tests/test_auth.py").replace(
            "b/auth.py", "b/tests/test_auth.py"
        )
        self.assertEqual([], LocalRuleReviewer().review(
            test_fixture, parse_unified_diff(test_fixture),
        ))

    def test_detects_removed_path_resolver_but_not_restored_resolver(self):
        regression = (
            "--- a/reader.go\n+++ b/reader.go\n@@ -1,4 +1,2 @@\n"
            "-fullPath, err := fr.resolveWorkspacePath(path)\n"
            "-if err != nil { return \"\", err }\n"
            "+fullPath := filepath.Join(fr.RepoDir, path)\n"
            " content, err := os.ReadFile(fullPath)\n"
        )
        security_fix = (
            "--- a/reader.go\n+++ b/reader.go\n@@ -1,2 +1,4 @@\n"
            "-fullPath := filepath.Join(fr.RepoDir, path)\n"
            "+fullPath, err := fr.resolveWorkspacePath(path)\n"
            "+if err != nil { return \"\", err }\n"
            " content, err := os.ReadFile(fullPath)\n"
        )

        findings = LocalRuleReviewer().review(
            regression, parse_unified_diff(regression),
        )
        fixed_findings = LocalRuleReviewer().review(
            security_fix, parse_unified_diff(security_fix),
        )

        self.assertEqual(["SEC-PATH-TRAVERSAL"], [item.rule_id for item in findings])
        self.assertEqual([], fixed_findings)

    def test_detects_actions_expression_in_run_but_not_env_indirection(self):
        regression = (
            "--- a/.github/workflows/check.yml\n"
            "+++ b/.github/workflows/check.yml\n"
            "@@ -1,3 +1,3 @@\n"
            " run: |\n"
            "-  check \"$VALUE\"\n"
            "+  check \"${{ steps.discover.outputs.value }}\"\n"
            " next: true\n"
        )
        security_fix = (
            "--- a/.github/workflows/check.yml\n"
            "+++ b/.github/workflows/check.yml\n"
            "@@ -1,3 +1,5 @@\n"
            "+env:\n"
            "+  VALUE: ${{ steps.discover.outputs.value }}\n"
            " run: |\n"
            "-  check \"${{ steps.discover.outputs.value }}\"\n"
            "+  check \"$VALUE\"\n"
            " next: true\n"
        )

        findings = LocalRuleReviewer().review(
            regression, parse_unified_diff(regression),
        )
        fixed_findings = LocalRuleReviewer().review(
            security_fix, parse_unified_diff(security_fix),
        )

        self.assertEqual(
            ["SEC-GHA-EXPRESSION-IN-SHELL"], [item.rule_id for item in findings],
        )
        self.assertEqual([], fixed_findings)


if __name__ == "__main__":
    unittest.main()
