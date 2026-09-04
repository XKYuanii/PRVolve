import json
import unittest

from evoagent.core.diff_parser import parse_unified_diff
from evoagent.agents.loop import AgentLoop
from evoagent.agents.parsing import (
    normalize_model_rule_id, parse_findings, worker_final_validation_error,
)
from evoagent.review.agentic import AgenticReviewer
from evoagent.review.merge import apply_critic, merge_findings, normalize_delegations
from evoagent.review.preflight import repository_preflight
from evoagent.eval.benchmark import ContextRuleReviewer
from evoagent.eval.harness import one_to_one_match
from evoagent.eval.agentic import (
    FairAblationSuite,
    ProductArmReviewer,
    ProductionEvaluationHarness,
    product_reviewer_factories,
)
from evoagent.core.models import Finding, Severity
from evoagent.review.reviewers import LocalRuleReviewer
from evoagent.tools.repository import RepositoryToolSuite
from evoagent.tools.registry import AgentTool, ToolRegistry
from evoagent.session.ledger import ExecutionLedger


DIFF = (
    "--- a/app.py\n"
    "+++ b/app.py\n"
    "@@ -0,0 +1 @@\n"
    "+value = open(base / user_path)\n"
)


class FakeClient:
    provider = "fake"
    model = "fake-model"

    def complete_json(self, role, _system, user, ledger=None, max_tokens=None):
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 10, "completion_tokens": 5}, 1,
            )
        if role == "lead":
            managed = json.loads(user)
            task = json.loads(managed["task"])
            if task["phase"] == "delegate":
                return {
                    "action": "final",
                    "delegations": [
                        {
                            "assignment_id": "security-1", "worker": "security",
                            "objective": "Review security",
                        },
                        {
                            "assignment_id": "reliability-1",
                            "worker": "correctness-reliability",
                            "objective": "Review correctness",
                        },
                    ],
                }
            if task["phase"] == "assess-workers":
                return {
                    "action": "final", "revision_requests": [],
                    "critic_objective": "Blindly verify every candidate.",
                }
            if task["phase"] == "finalize":
                return {
                    "action": "final",
                    "accepted_finding_indices": list(
                        range(len(task["candidate_findings"]))
                    ),
                    "confidence_adjustments": [],
                }
            raise AssertionError(task["phase"])
        if role in {"security", "correctness-reliability"}:
            return {"action": "final", "findings": []}
        if role == "critic":
            managed = json.loads(user)
            task = json.loads(managed["task"])
            return {
                "action": "final",
                "decisions": [
                    {
                        "finding_index": index,
                        "accepted": True,
                        "objections": [],
                        "confidence_adjustment": 0.0,
                    }
                    for index, _item in enumerate(task["candidates"])
                ],
            }
        raise AssertionError(role)


class AgenticEvaluationTests(unittest.TestCase):
    def test_repository_preflight_prioritizes_semantic_source_over_config(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return [
                    "read_file", "search_repository", "semantic_probe", "ast_analyze",
                ]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return {
                    "evidence_id": "%s:%d" % (name, len(self.calls)),
                    "tool": name, "output": dict(arguments),
                }

        parsed = parse_unified_diff(
            "--- a/.github/workflow.yml\n+++ b/.github/workflow.yml\n"
            "@@ -0,0 +1 @@\n+name: build\n"
            "--- a/src/app.py\n+++ b/src/app.py\n"
            "@@ -99,0 +100 @@\n+message = str(err).replace(inv_location, safe_url)\n"
        )
        tools = RecordingTools()

        observations = repository_preflight(
            {"files": parsed.files}, parsed, tools,
        )

        self.assertTrue(observations)
        self.assertEqual("read_file", tools.calls[0][0])
        self.assertEqual("src/app.py", tools.calls[0][1]["path"])
        queries = [
            arguments["query"] for name, arguments in tools.calls
            if name == "search_repository"
        ]
        self.assertIn("inv_location", queries)
        self.assertIn(
            ("semantic_probe", {"kind": "url-normalization-redaction"}), tools.calls
        )

    def test_url_normalization_probe_demonstrates_exact_replacement_gap(self):
        evidence = RepositoryToolSuite.semantic_probe("url-normalization-redaction")
        output = evidence["output"]

        self.assertFalse(output["exact_original_still_matches"])
        self.assertTrue(output["credentials_remaining"])
        self.assertFalse(output["network_used"])
        self.assertFalse(output["arbitrary_code_executed"])

    def test_semantic_preflight_runs_without_repository_checkout(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return [
                    "read_file", "search_repository", "semantic_probe", "ast_analyze",
                ]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return RepositoryToolSuite.semantic_probe(arguments["kind"])

        parsed = parse_unified_diff(
            "--- a/auth.py\n+++ b/auth.py\n"
            "@@ -9 +9 @@\n"
            "-verify_signature = options.get('verify_signature', True)\n"
            "+verify_signature = options.get('verify_signature', False)\n"
        )
        tools = RecordingTools()

        observations = repository_preflight(
            {"files": parsed.files}, parsed, tools, repository_available=False,
        )

        self.assertEqual(
            [("semantic_probe", {"kind": "security-control-default"})],
            tools.calls,
        )
        self.assertTrue(observations[0]["ok"])

    def test_workflow_expression_preflight_compares_safe_env_indirection(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["semantic_probe"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return RepositoryToolSuite.semantic_probe(arguments["kind"])

        parsed = parse_unified_diff(
            "--- a/.github/workflows/review.yml\n"
            "+++ b/.github/workflows/review.yml\n"
            "@@ -10 +10 @@\n"
            "-    --target \"$TARGET\"\n"
            "+    --target \"${{ steps.changed.outputs.target }}\"\n"
        )
        tools = RecordingTools()

        observations = repository_preflight(
            {"files": parsed.files}, parsed, tools, repository_available=False,
        )

        self.assertEqual(
            [("semantic_probe", {"kind": "github-actions-expression-shell"})],
            tools.calls,
        )
        self.assertTrue(observations[0]["ok"])

    def test_preflight_runs_fixed_python_runtime_contract_probes(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["semantic_probe"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return RepositoryToolSuite.semantic_probe(arguments["kind"])

        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,5 @@\n"
            "+for key in values:\n+    values.pop(key)\n"
            "+if Decimal('NaN').as_tuple().exponent >= 0:\n+    pass\n"
            "+memo.add(exc_value)\n"
        )
        tools = RecordingTools()

        observations = repository_preflight(
            {"files": parsed.files}, parsed, tools, repository_available=False,
        )

        self.assertEqual([
            ("semantic_probe", {"kind": "dict-mutation-during-iteration"}),
            ("semantic_probe", {"kind": "decimal-special-exponent"}),
            ("semantic_probe", {"kind": "unhashable-exception-membership"}),
        ], tools.calls)
        self.assertTrue(all(item["ok"] for item in observations))

    def test_repository_preflight_traces_template_environment_usage(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["read_file", "search_repository", "ast_analyze"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                if name == "search_repository" and arguments["query"] == "_inline_env":
                    return {
                        "evidence_id": "search:inline", "tool": name,
                        "output": [{
                            "path": "templates/message.py", "line": 17,
                            "content": "template = _inline_env.from_string(self.content_template)",
                        }],
                    }
                if name == "read_file" and arguments["path"] == "templates/message.py":
                    return {
                        "evidence_id": "read:message", "tool": name,
                        "output": {
                            "path": "templates/message.py", "start_line": 1,
                            "end_line": 47,
                            "content": "class MessageTemplate:\n    def render(self):\n        return _inline_env.from_string(self.content_template)\n",
                        },
                    }
                return {
                    "evidence_id": "%s:%d" % (name, len(self.calls)),
                    "tool": name, "output": dict(arguments),
                }

        parsed = parse_unified_diff(
            "--- a/templates/environment.py\n"
            "+++ b/templates/environment.py\n"
            "@@ -2,2 +2,2 @@\n"
            "-from jinja2.sandbox import SandboxedEnvironment\n"
            "+from jinja2 import Environment\n"
            "@@ -37 +37 @@\n"
            "-_inline_env = SandboxedEnvironment()\n"
            "+_inline_env = Environment()\n"
        )
        tools = RecordingTools()

        repository_preflight(
            {"files": parsed.files}, parsed, tools, repository_available=True,
        )

        queries = [
            arguments["query"] for name, arguments in tools.calls
            if name == "search_repository"
        ]
        self.assertIn("_inline_env", queries)
        self.assertIn("MessageTemplate", queries)
        self.assertTrue(any(
            name == "read_file" and arguments["path"] == "templates/message.py"
            for name, arguments in tools.calls
        ))

    def test_fixed_semantic_probes_cover_common_cross_line_contracts(self):
        serialization = RepositoryToolSuite.semantic_probe(
            "serialization-exclusion-update"
        )["output"]
        equality = RepositoryToolSuite.semantic_probe(
            "equality-negation-contract"
        )["output"]
        decorators = RepositoryToolSuite.semantic_probe("decorator-order")["output"]
        cycle = RepositoryToolSuite.semantic_probe("self-cycle-collection")["output"]
        alias = RepositoryToolSuite.semantic_probe(
            "alias-configuration-direction"
        )["output"]
        module_getattr = RepositoryToolSuite.semantic_probe(
            "module-getattr-alias-bypass"
        )["output"]
        path = RepositoryToolSuite.semantic_probe("path-containment")["output"]
        security_default = RepositoryToolSuite.semantic_probe(
            "security-control-default"
        )["output"]
        workflow_expression = RepositoryToolSuite.semantic_probe(
            "github-actions-expression-shell"
        )["output"]
        git_option = RepositoryToolSuite.semantic_probe(
            "git-option-normalization"
        )["output"]
        nullable_length = RepositoryToolSuite.semantic_probe(
            "nullable-length"
        )["output"]
        sentinel = RepositoryToolSuite.semantic_probe(
            "sentinel-error-propagation"
        )["output"]
        dict_mutation = RepositoryToolSuite.semantic_probe(
            "dict-mutation-during-iteration"
        )["output"]
        decimal_exponent = RepositoryToolSuite.semantic_probe(
            "decimal-special-exponent"
        )["output"]
        unhashable_exception = RepositoryToolSuite.semantic_probe(
            "unhashable-exception-membership"
        )["output"]
        missing_scandir = RepositoryToolSuite.semantic_probe(
            "scandir-missing-directory"
        )["output"]
        empty_netrc = RepositoryToolSuite.semantic_probe(
            "empty-netrc-credentials"
        )["output"]
        exception_cleanup = RepositoryToolSuite.semantic_probe(
            "exception-cleanup-state"
        )["output"]

        self.assertTrue(serialization["excluded_field_update_lost"])
        self.assertTrue(equality["contract_violated"])
        self.assertFalse(decorators["same_result"])
        self.assertTrue(cycle["collection_delayed_until_cyclic_gc"])
        self.assertEqual(
            [True, False, False],
            [item["conditions_diverge"] for item in alias["truth_table"]],
        )
        self.assertFalse(module_getattr["module_getattr_invoked"])
        self.assertFalse(module_getattr["deprecation_warning_path_reached"])
        self.assertTrue(path["parent_segments_escape_base"])
        self.assertFalse(path["filesystem_read"])
        self.assertTrue(security_default["security_control_disabled_by_default"])
        self.assertFalse(security_default["verification_branch_entered"])
        self.assertTrue(workflow_expression["direct_command_contains_attacker_text"])
        self.assertTrue(workflow_expression["shell_metacharacters_reach_direct_command"])
        self.assertTrue(
            workflow_expression[
                "environment_reference_keeps_value_out_of_command_text"
            ]
        )
        self.assertFalse(git_option["raw_check_blocks"])
        self.assertTrue(git_option["canonical_check_blocks"])
        self.assertTrue(git_option["dangerous_flag_emitted_after_raw_check"])
        self.assertTrue(nullable_length["raises_when_value_is_none"])
        self.assertTrue(sentinel["conversion_failed"])
        self.assertTrue(sentinel["returned_missing_sentinel"])
        self.assertTrue(dict_mutation["dict_size_change_raises"])
        self.assertTrue(all(
            item["comparison_with_zero_raises"]
            for item in decimal_exponent["values"]
        ))
        self.assertTrue(unhashable_exception["set_insertion_raises"])
        self.assertTrue(missing_scandir["scandir_open_raises"])
        self.assertTrue(empty_netrc["tuple_is_truthy"])
        self.assertFalse(empty_netrc["any_field_is_truthy"])
        self.assertTrue(empty_netrc["blank_credentials_pass_tuple_truthiness"])
        self.assertTrue(exception_cleanup["state_remains_installed"])
        self.assertEqual("RuntimeError", exception_cleanup["error_type"])
        self.assertTrue(all(
            item["arbitrary_code_executed"] is False
            for item in (
                serialization, equality, decorators, cycle, alias, path,
                security_default, workflow_expression, git_option,
                nullable_length, sentinel, module_getattr,
                dict_mutation, decimal_exponent, unhashable_exception,
                missing_scandir, empty_netrc, exception_cleanup,
            )
        ))

    def test_preflight_probes_exception_cleanup_only_without_added_handler(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["semantic_probe"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return RepositoryToolSuite.semantic_probe(arguments["kind"])

        risk = parse_unified_diff(
            "--- a/live.py\n+++ b/live.py\n@@ -1 +1 @@\n+self.refresh()\n"
        )
        clean = parse_unified_diff(
            "--- a/live.py\n+++ b/live.py\n@@ -1 +1,4 @@\n"
            "+try:\n+    self.refresh()\n+except Exception:\n+    self.stop()\n"
        )
        risk_tools = RecordingTools()
        clean_tools = RecordingTools()

        repository_preflight(
            {"files": risk.files}, risk, risk_tools, repository_available=False,
        )
        repository_preflight(
            {"files": clean.files}, clean, clean_tools, repository_available=False,
        )

        self.assertEqual(
            [("semantic_probe", {"kind": "exception-cleanup-state"})],
            risk_tools.calls,
        )
        self.assertEqual([], clean_tools.calls)

    def test_preflight_does_not_probe_guarded_decimal_or_hashed_exception_id(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["semantic_probe"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return RepositoryToolSuite.semantic_probe(arguments["kind"])

        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1,4 @@\n"
            "+exponent = value.as_tuple().exponent\n"
            "+if isinstance(exponent, int) and exponent >= 0:\n"
            "+    return int(value)\n"
            "+memo.add(id(exc_value))\n"
        )
        tools = RecordingTools()

        repository_preflight(
            {"files": parsed.files}, parsed, tools, repository_available=False,
        )

        self.assertEqual([], tools.calls)

    def test_preflight_probes_missing_scandir_and_blank_netrc_only_on_regressions(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["semantic_probe"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return RepositoryToolSuite.semantic_probe(arguments["kind"])

        risk = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1,2 @@\n"
            "+with os.scandir(path) as entries:\n+    consume(entries)\n"
            "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n+if _netrc:\n"
        )
        clean = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1,4 @@\n"
            "+try:\n+    entries = os.scandir(path)\n+except FileNotFoundError:\n+    return []\n"
            "--- a/auth.py\n+++ b/auth.py\n@@ -1 +1 @@\n"
            "+if _netrc and any(_netrc):\n"
        )
        risk_tools = RecordingTools()
        clean_tools = RecordingTools()

        repository_preflight(
            {"files": risk.files}, risk, risk_tools, repository_available=False,
        )
        repository_preflight(
            {"files": clean.files}, clean, clean_tools, repository_available=False,
        )

        self.assertIn(
            ("semantic_probe", {"kind": "scandir-missing-directory"}),
            risk_tools.calls,
        )
        self.assertIn(
            ("semantic_probe", {"kind": "empty-netrc-credentials"}),
            risk_tools.calls,
        )
        self.assertEqual([], clean_tools.calls)

    def test_repository_preflight_reads_distinct_regions_and_probes_nullable_len(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["read_file", "semantic_probe"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                if name == "semantic_probe":
                    return RepositoryToolSuite.semantic_probe(arguments["kind"])
                return {
                    "evidence_id": "read:%d" % len(self.calls),
                    "tool": name, "output": dict(arguments),
                }

        parsed = parse_unified_diff(
            "--- a/src/app.py\n+++ b/src/app.py\n"
            "@@ -9 +10 @@\n+normalized = value.strip()\n"
            "@@ -99 +100 @@\n+width = len(rule.subdomain)\n"
        )
        tools = RecordingTools()

        observations = repository_preflight(
            {"files": parsed.files}, parsed, tools,
        )

        reads = [arguments for name, arguments in tools.calls if name == "read_file"]
        self.assertEqual(2, len(reads))
        self.assertEqual({10, 100}, {item["end_line"] - 25 for item in reads})
        self.assertIn(
            ("semantic_probe", {"kind": "nullable-length"}), tools.calls,
        )
        self.assertTrue(all(item["ok"] for item in observations))

    def test_git_option_preflight_runs_fixed_normalization_probe(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["semantic_probe"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return RepositoryToolSuite.semantic_probe(arguments["kind"])

        parsed = parse_unified_diff(
            "--- a/git/cmd.py\n+++ b/git/cmd.py\n"
            "@@ -950,2 +950,3 @@\n"
            "+bare_unsafe_options = [item.lstrip('-') for item in unsafe_options]\n"
            "+if option.startswith(unsafe_option):\n"
            "+    raise UnsafeOptionError()\n"
        )
        tools = RecordingTools()

        observations = repository_preflight(
            {"files": parsed.files}, parsed, tools, repository_available=False,
        )

        self.assertEqual(
            [("semantic_probe", {"kind": "git-option-normalization"})],
            tools.calls,
        )
        self.assertTrue(observations[0]["ok"])

    def test_expected_finding_can_declare_review_taxonomy_aliases(self):
        finding = Finding(
            rule_id="CWE-200", severity=Severity.HIGH,
            title="Leak", explanation="Credentials remain visible.",
            path="app.py", line=5, evidence="replace(raw, safe)",
            fix="Redact normalized values.", test="Use an encoded password.",
        )
        expected = [{
            "cwe": "CWE-532", "acceptable_cwes": ["CWE-200", "CWE-522"],
            "path": "app.py", "start_line": 5, "end_line": 5,
            "severity": "high",
        }]

        self.assertEqual(1, len(one_to_one_match(expected, [finding])))

    def test_decorator_order_rule_maps_to_improper_behavior_order(self):
        finding = Finding(
            rule_id="DECORATOR-ORDER", severity=Severity.HIGH,
            title="Decorator order reversed", explanation="Order changes behavior.",
            path="mypyc/irbuild/function.py", line=506,
            evidence="decorated_func = classmethod(decorated_func)",
            fix="Preserve source order.", test="Cover both decorator orders.",
        )
        expected = [{
            "cwe": "CWE-696", "path": "mypyc/irbuild/function.py",
            "start_line": 506, "end_line": 520, "severity": "medium",
        }]

        self.assertEqual(1, len(one_to_one_match(expected, [finding])))

    def test_same_semantic_probe_and_location_are_deduplicated_across_roles(self):
        values = [
            Finding(
                rule_id=rule_id, severity=Severity.HIGH,
                title="Credential leak", explanation="Normalized URL leaks a password.",
                path="app.py", line=5, evidence="replace(raw, safe)",
                evidence_refs=[{
                    "evidence_id": "semantic_probe:test", "tool": "semantic_probe",
                    "output": {"kind": "url-normalization-redaction",
                               "arbitrary_code_executed": False},
                }],
                fix="Redact normalized values.", test="Use an encoded password.",
                source=source,
            )
            for rule_id, source in (
                ("CWE-200", "security"),
                ("CWE-522", "correctness-reliability"),
            )
        ]

        self.assertEqual(1, len(merge_findings(values)))

    def test_same_semantic_probe_deduplicates_adjacent_lines_of_one_defect(self):
        shared_ref = {
            "evidence_id": "semantic_probe:dict-mutation",
            "tool": "semantic_probe",
            "output": {
                "kind": "dict-mutation-during-iteration",
                "dict_size_change_raises": True,
                "arbitrary_code_executed": False,
            },
        }
        values = [
            Finding(
                rule_id="CWE-703", severity=Severity.MEDIUM,
                title="Dictionary changes size during iteration",
                explanation="The loop pops from the dictionary it iterates.",
                path="app.py", line=line, evidence=evidence,
                evidence_refs=[shared_ref], fix="Iterate over a copy.",
                test="Exercise one remaining key.", confidence=confidence,
                source=source,
            )
            for line, evidence, confidence, source in (
                (10, "for key in values:", 0.9, "security"),
                (12, "values.pop(key)", 0.95, "correctness-reliability"),
            )
        ]

        merged = merge_findings(values)

        self.assertEqual(1, len(merged))
        self.assertEqual(12, merged[0].line)

    def test_same_semantic_probe_deduplicates_taxonomy_variants_on_same_line(self):
        shared_ref = {
            "evidence_id": "semantic_probe:netrc",
            "tool": "semantic_probe",
            "output": {
                "kind": "empty-netrc-credentials",
                "blank_credentials_pass_tuple_truthiness": True,
                "arbitrary_code_executed": False,
            },
        }
        values = [
            Finding(
                rule_id=rule_id, severity=Severity.MEDIUM,
                title=title,
                explanation="A truthy blank tuple returns empty credentials.",
                path="auth.py", line=20, evidence="if credentials:",
                evidence_refs=[shared_ref], fix="Check any(credentials).",
                test="Cover blank credentials.", confidence=confidence,
                source=source,
            )
            for rule_id, title, confidence, source in (
                ("CWE-252", "Blank tuple is accepted", 0.9,
                 "correctness-reliability"),
                ("CWE-522", "Empty credentials are returned", 0.95, "security"),
            )
        ]

        merged = merge_findings(values)

        self.assertEqual(1, len(merged))
        self.assertEqual("CWE-522", merged[0].rule_id)

    def test_semantic_merge_prefers_the_claim_actually_proved_by_probe(self):
        shared_ref = {
            "evidence_id": "semantic_probe:git",
            "tool": "semantic_probe",
            "output": {
                "kind": "git-option-normalization",
                "dangerous_flag_emitted_after_raw_check": True,
                "arbitrary_code_executed": False,
            },
        }
        mixed = Finding(
            rule_id="CWE-184", severity=Severity.HIGH,
            title="Unsafe option bypass and false positive for --config-file",
            explanation=(
                "Underscore normalization bypasses the check, and prefix matching "
                "may incorrectly reject --config-file."
            ),
            path="git/cmd.py", line=959,
            evidence="if option.startswith(unsafe_option):",
            evidence_refs=[shared_ref], fix="Canonicalize names.",
            test="Cover upload_pack.", confidence=0.95,
            source="correctness-reliability",
        )
        proved = Finding(
            rule_id="CWE-184", severity=Severity.HIGH,
            title="Unsafe upload_pack option bypasses canonical check",
            explanation=(
                "The raw underscore name fails to match before canonicalization, "
                "so the dangerous upload-pack option bypasses the guard."
            ),
            path="git/cmd.py", line=959,
            evidence="if option.startswith(unsafe_option):",
            evidence_refs=[shared_ref], fix="Canonicalize names.",
            test="Cover upload_pack.", confidence=0.9, source="security",
        )

        merged = merge_findings([mixed, proved])

        self.assertEqual(1, len(merged))
        self.assertEqual("security", merged[0].source)
        self.assertIn("bypasses canonical", merged[0].title)

    def test_critic_recommends_but_does_not_double_apply_confidence_adjustment(self):
        candidate = Finding(
            rule_id="CWE-703", severity=Severity.MEDIUM,
            title="Missing directory escapes", explanation="The error propagates.",
            path="app.py", line=10, evidence="with os.scandir(path):",
            fix="Catch FileNotFoundError.", test="Cover a missing directory.",
            confidence=0.8, source="correctness-reliability",
        )
        result = {
            "decisions": [{
                "finding_index": 0,
                "accepted": True,
                "introduced_by_diff": True,
                "reproducible": True,
                "evidence_sufficient": True,
                "would_comment_on_real_pr": True,
                "confidence_adjustment": -0.05,
            }],
            "_observations": [],
        }

        candidates, decisions = apply_critic(
            result, [candidate],
        )

        self.assertEqual(0.8, candidates[0].confidence)
        self.assertEqual(-0.05, decisions[0]["recommended_confidence_adjustment"])
        self.assertTrue(decisions[0]["publication_ready"])

    def test_same_repository_evidence_and_similar_title_are_deduplicated(self):
        values = [
            Finding(
                rule_id=rule_id, severity=Severity.MEDIUM,
                title=title, explanation="Empty credentials are returned.",
                path="app.py", line=5, evidence="if credentials:",
                evidence_refs=[{
                    "evidence_id": "read_file:same", "tool": "read_file",
                    "output": {"path": "app.py", "content": "if credentials:"},
                }],
                fix="Reject empty credentials.", test="Cover an empty tuple.",
                source=source, confidence=confidence,
            )
            for rule_id, source, title, confidence in (
                (
                    "CWE-522", "security",
                    "get_auth returns empty credentials for default entry", 0.9,
                ),
                (
                    "CWE-252", "correctness-reliability",
                    "get_auth returns empty credentials when entry is blank", 0.95,
                ),
            )
        ]

        merged = merge_findings(values)

        self.assertEqual(1, len(merged))
        self.assertEqual("CWE-252", merged[0].rule_id)

    def test_distinct_claims_at_same_location_are_not_merged(self):
        values = [
            Finding(
                rule_id=rule_id, severity=Severity.MEDIUM,
                title=title, explanation="Repository-backed defect.",
                path="app.py", line=5, evidence="process(value)",
                evidence_refs=[{
                    "evidence_id": "read_file:same", "tool": "read_file",
                    "output": {"path": "app.py", "content": "process(value)"},
                }],
                fix="Fix it.", test="Cover it.", source=source,
            )
            for rule_id, source, title in (
                ("CWE-400", "security", "Unbounded input exhausts memory"),
                ("CWE-772", "correctness-reliability", "File handle is never closed"),
            )
        ]

        self.assertEqual(2, len(merge_findings(values)))

    def test_delegation_coverage_gate_assigns_every_production_source(self):
        delegations = normalize_delegations(
            [{
                "assignment_id": "correctness-1",
                "worker": "correctness-reliability",
                "files": ["uv.lock"],
            }],
            {"correctness-reliability"},
            ["uv.lock", ".github/workflow.yml", "sqlmodel/main.py", "tests/test_main.py"],
        )

        coverage = next(
            item for item in delegations
            if item["assignment_id"] == "correctness-source-coverage"
        )
        self.assertEqual(["sqlmodel/main.py"], coverage["files"])
        self.assertNotIn("tests/test_main.py", coverage["files"])
        self.assertNotIn("uv.lock", coverage["files"])

    def test_repository_role_cannot_finish_before_a_factual_tool_call(self):
        class SequencedClient:
            def __init__(self):
                self.actions = [
                    {"action": "final", "findings": []},
                    {"action": "tool", "tool": "read_file", "arguments": {}},
                    {"action": "final", "findings": []},
                ]

            def complete_json(self, *_args, **_kwargs):
                return self.actions.pop(0)

        registry = ToolRegistry([AgentTool(
            "read_file", "Read evidence.",
            {"type": "object", "properties": {}, "additionalProperties": False},
            lambda: {"evidence_id": "read_file:test", "output": "value"},
        )])
        result = AgentLoop(
            "correctness-reliability", "Review.", SequencedClient(),
            token_budget=4000, time_budget=30, minimum_tool_calls=1,
        ).run("{}", registry, ExecutionLedger("agentic"))

        self.assertEqual(3, result["_steps"])
        self.assertEqual("protocol-requirement", result["_observations"][0]["tool"])
        self.assertTrue(result["_observations"][1]["ok"])

    def test_worker_must_resolve_fixed_counterexample_before_finishing(self):
        class SequencedClient:
            def __init__(self):
                self.actions = [
                    {"action": "final", "findings": []},
                    {
                        "action": "final", "findings": [],
                        "evidence_resolutions": [{
                            "evidence_id": "semantic_probe:test",
                            "status": "refuted",
                            "explanation": "Repository type evidence proves None is unreachable.",
                            "supporting_evidence_ids": ["read_file:test"],
                        }],
                    },
                ]

            def complete_json(self, *_args, **_kwargs):
                return self.actions.pop(0)

        initial = [{
            "step": 0, "tool": "semantic_probe", "ok": True,
            "result": {
                "evidence_id": "semantic_probe:test", "tool": "semantic_probe",
                "output": {
                    "requires_resolution": True,
                    "resolution_question": "Explain the divergent branch.",
                },
            },
        }, {
            "step": 0, "tool": "read_file", "ok": True,
            "result": {
                "evidence_id": "read_file:test", "tool": "read_file",
                "output": {"path": "app.py", "content": "value is bool"},
            },
        }]

        result = AgentLoop(
            "correctness-reliability", "Review.", SequencedClient(),
            token_budget=4000, time_budget=30,
        ).run(
            "{}", ToolRegistry([]), ExecutionLedger("agentic"),
            initial_observations=initial,
        )

        self.assertEqual(2, result["_steps"])
        self.assertEqual("protocol-requirement", result["_observations"][-1]["tool"])
        self.assertIn("semantic_probe:test", result["_observations"][-1]["error"])

    def test_worker_finding_resolution_requires_structured_finding(self):
        class SequencedClient:
            def __init__(self):
                self.actions = [
                    {
                        "action": "final", "findings": [],
                        "evidence_resolutions": [{
                            "evidence_id": "read_file:defect",
                            "status": "finding",
                            "explanation": "The value is used before assignment.",
                        }],
                    },
                    {
                        "action": "final",
                        "findings": [{
                            "rule_id": "CWE-457", "severity": "high",
                            "title": "Value used before assignment",
                            "explanation": "The false branch leaves value undefined.",
                            "path": "app.py", "line": 2, "evidence": "consume(value)",
                            "evidence_ids": ["read_file:defect"],
                            "call_chain": [], "fix": "Initialize value before the branch.",
                            "test": "Exercise the false branch.", "confidence": 0.95,
                            "skill": "",
                        }],
                        "evidence_resolutions": [{
                            "evidence_id": "read_file:defect",
                            "status": "finding",
                            "explanation": "The value is used before assignment.",
                        }],
                    },
                ]

            def complete_json(self, *_args, **_kwargs):
                return self.actions.pop(0)

        result = AgentLoop(
            "correctness-reliability", "Review.", SequencedClient(),
            token_budget=4000, time_budget=30,
        ).run(
            "{}", ToolRegistry([]), ExecutionLedger("agentic"),
            initial_observations=[{
                "step": 0, "tool": "read_file", "ok": True,
                "result": {
                    "evidence_id": "read_file:defect", "tool": "read_file",
                    "output": {"path": "app.py", "content": "consume(value)"},
                },
            }],
        )

        self.assertEqual(2, result["_steps"])
        self.assertEqual(1, len(result["findings"]))
        self.assertEqual("protocol-requirement", result["_observations"][-1]["tool"])
        self.assertIn("read_file:defect", result["_observations"][-1]["error"])

    def test_worker_retries_when_final_finding_location_fails_validation(self):
        class SequencedClient:
            def __init__(self):
                self.actions = [
                    {
                        "action": "final",
                        "findings": [{
                            "path": "app.py", "line": 3,
                            "evidence_ids": ["read_file:defect"],
                        }],
                        "evidence_resolutions": [{
                            "evidence_id": "read_file:defect", "status": "finding",
                        }],
                    },
                    {
                        "action": "final",
                        "findings": [{
                            "path": "app.py", "line": 2,
                            "evidence_ids": [
                                "read_file:defect",
                                "protocol-finding:read_file:defect",
                            ],
                        }],
                        "evidence_resolutions": [{
                            "evidence_id": "read_file:defect", "status": "finding",
                        }, {
                            "evidence_id": "protocol-finding:read_file:defect",
                            "status": "finding",
                        }],
                    },
                ]

            def complete_json(self, *_args, **_kwargs):
                return self.actions.pop(0)

        valid = {("app.py", 2)}

        def validate(action):
            finding = action["findings"][0]
            if (finding["path"], finding["line"]) not in valid:
                return "Anchor the Finding to the exact added line app.py:2."
            return ""

        result = AgentLoop(
            "correctness-reliability", "Review.", SequencedClient(),
            token_budget=4000, time_budget=30,
            final_action_validator=validate,
        ).run("{}", ToolRegistry([]), ExecutionLedger("agentic"))

        self.assertEqual(2, result["_steps"])
        self.assertEqual(2, result["findings"][0]["line"])
        self.assertEqual("protocol-requirement", result["_observations"][-1]["tool"])
        self.assertIn("app.py:2", result["_observations"][-1]["error"])
        self.assertTrue(
            result["_observations"][-1]["result"]["output"]["requires_resolution"]
        )

    def test_model_rule_normalization_corrects_exception_cwe_252_only(self):
        raw = {
            "rule_id": "CWE-252", "title": "Decoder raises TypeError",
            "explanation": "A malformed value raises TypeError and crashes the request.",
            "evidence": "value = decode(raw)",
        }

        self.assertEqual("CWE-248", normalize_model_rule_id(raw))
        raw.update({
            "title": "Unchecked return status",
            "explanation": "The caller fails to inspect the return code.",
        })
        self.assertEqual("CWE-252", normalize_model_rule_id(raw))

    def test_model_rule_normalization_corrects_git_option_bypass_cwe_697(self):
        raw = {
            "rule_id": "CWE-697",
            "title": "Unsafe option false negative for underscore names",
            "explanation": (
                "The raw upload_pack name fails to match --upload-pack before "
                "canonical dash normalization, allowing a guard bypass."
            ),
            "evidence": "if option.startswith(unsafe_option):",
        }

        self.assertEqual("CWE-184", normalize_model_rule_id(raw))
        raw.update({
            "title": "Unrelated incorrect comparison",
            "explanation": "Two ordinary values compare incorrectly.",
        })
        self.assertEqual("CWE-697", normalize_model_rule_id(raw))

    def test_finding_location_recovers_only_from_unique_exact_added_evidence(self):
        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,2 @@\n"
            "+prepare()\n+consume(value)\n"
        )
        result = {"findings": [{
            "rule_id": "CWE-248", "severity": "medium",
            "title": "Invalid value crashes", "explanation": "The call raises.",
            "path": "app.py", "line": 99, "evidence": "consume(value)",
            "fix": "Validate value.", "test": "Exercise invalid value.",
        }]}

        findings = parse_findings(result, parsed, "correctness-reliability")

        self.assertEqual(1, len(findings))
        self.assertEqual(2, findings[0].line)

    def test_worker_validation_rejects_any_unscoreable_structured_finding(self):
        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+consume(value)\n"
        )
        result = {"findings": [{
            "rule_id": "CWE-248", "path": "app.py", "line": 99,
            "title": "Crash", "explanation": "The call crashes.",
            "evidence": "a different expression",
        }]}

        error = worker_final_validation_error(result, parsed)

        self.assertIn("Rejected locations: app.py:99", error)
        self.assertIn("app.py:1", error)

    def test_suggestion_metrics_measure_recovery_without_publishing_the_claim(self):
        suggestion = Finding(
            rule_id="CWE-502", severity=Severity.HIGH,
            title="Unsafe deserialization", explanation="Untrusted bytes are loaded.",
            path="app.py", line=1, evidence="pickle.loads(value)",
            fix="Use a safe format.", test="Reject a crafted payload.",
            source="security", disposition="suggestion",
        )

        class SuggestionOnlyReviewer:
            name = "suggestion-only"

            def review_case(self, _case, _parsed):
                return []

            def evaluation_execution(self):
                return {}

            def evaluation_summary(self):
                return {
                    "suggestion_count": 1,
                    "suggested_findings": [suggestion.to_dict()],
                }

        case = {
            "id": "suggestion-recovery", "repository": "repo", "pull_request": 1,
            "split": "validation", "source": {"kind": "synthetic-controlled"},
            "diff": (
                "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n"
                "+pickle.loads(value)\n"
            ),
            "expected_findings": [{
                "path": "app.py", "start_line": 1, "end_line": 1,
                "rule_id": "SEC-PICKLE-LOAD", "cwe": "CWE-502",
                "severity": "high", "should_comment": True,
            }],
        }

        report = ProductionEvaluationHarness().run(
            SuggestionOnlyReviewer(), [case], "suggestion-recovery"
        )
        metrics = report["metrics"]
        self.assertEqual(0, metrics["tp"])
        self.assertEqual(1, metrics["incremental_suggestion_tp"])
        self.assertEqual(1.0, metrics["missed_finding_recovery_rate"])
        self.assertEqual(1.0, metrics["combined_recall_after_verification"])
        self.assertEqual(1.0, metrics["suggestion_utility_rate"])

    def test_worker_failure_is_reported_as_degraded_execution(self):
        class DegradedReviewer:
            name = "degraded"

            def review_case(self, _case, _parsed):
                return []

            def evaluation_execution(self):
                return {}

            def evaluation_summary(self):
                return {
                    "worker_results": [{
                        "worker": "correctness-reliability",
                        "status": "failed",
                        "error": "budget exhausted",
                    }],
                }

        case = {
            "id": "degraded", "repository": "repo", "pull_request": 1,
            "split": "validation", "source": {"kind": "synthetic-controlled"},
            "diff": "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+value = 1\n",
            "expected_findings": [],
        }

        report = ProductionEvaluationHarness().run(
            DegradedReviewer(), [case], "degraded",
        )

        self.assertTrue(report["case_results"][0]["execution_success"])
        self.assertTrue(report["case_results"][0]["degraded_execution"])
        self.assertEqual(1, report["case_results"][0]["worker_failures"])
        self.assertEqual(0.0, report["metrics"]["full_role_success_rate"])

    def test_targeted_review_labels_do_not_call_unmatched_findings_invalid(self):
        finding = Finding(
            rule_id="CWE-754", severity=Severity.MEDIUM,
            title="Unexpected issue", explanation="A separate review candidate.",
            path="app.py", line=2, evidence="other()",
            fix="Fix it.", test="Test it.",
        )

        class FormalReviewer:
            name = "formal"

            def review_case(self, _case, _parsed):
                return [finding]

        case = {
            "id": "targeted", "repository": "repo", "pull_request": 1,
            "split": "validation",
            "source": {
                "kind": "public-github-pr",
                "label_completeness": "targeted-review-comments",
            },
            "diff": (
                "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,2 @@\n"
                "+expected()\n+other()\n"
            ),
            "expected_findings": [{
                "path": "app.py", "start_line": 1, "end_line": 1,
                "cwe": "CWE-476", "severity": "high", "should_comment": True,
            }],
        }

        metrics = ProductionEvaluationHarness().run(
            FormalReviewer(), [case], "targeted"
        )["metrics"]
        self.assertEqual(0, metrics["formal_invalid_findings"])
        self.assertEqual(1, metrics["formal_unjudged_findings"])
        self.assertEqual(0.0, metrics["invalid_comments_per_pr"])
        self.assertEqual(
            "not-estimable-until-unexpected-findings-are-adjudicated",
            metrics["precision_interpretation"],
        )

    def test_formal_judgments_make_targeted_precision_estimable(self):
        findings = [
            Finding(
                rule_id=rule_id, severity=Severity.MEDIUM,
                title=title, explanation="Adjudication fixture.",
                path="app.py", line=line, evidence="value_%d" % line,
                fix="Apply a focused fix.", test="Add a focused test.",
            )
            for line, rule_id, title in (
                (1, "CWE-476", "labelled"),
                (2, "CWE-400", "new required defect"),
                (3, "CWE-20", "invalid candidate"),
            )
        ]

        class FormalReviewer:
            name = "adjudicated-formal"

            def review_case(self, _case, _parsed):
                return findings

        case = {
            "id": "adjudicated-formal", "repository": "repo", "pull_request": 1,
            "split": "validation",
            "source": {
                "kind": "public-github-pr",
                "label_completeness": "targeted-review-comments",
            },
            "diff": (
                "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,3 @@\n"
                "+value_1\n+value_2\n+value_3\n"
            ),
            "expected_findings": [{
                "path": "app.py", "start_line": 1, "end_line": 1,
                "cwe": "CWE-476", "severity": "medium", "should_comment": True,
            }],
            "formal_judgments": [
                {
                    "path": "app.py", "line": 2, "rule_id": "CWE-400",
                    "verdict": "required",
                },
                {
                    "path": "app.py", "line": 3, "rule_id": "CWE-20",
                    "verdict": "invalid",
                },
            ],
        }

        report = ProductionEvaluationHarness().run(
            FormalReviewer(), [case], "adjudicated-formal"
        )
        metrics = report["metrics"]
        self.assertEqual(1, metrics["formal_label_gap_required"])
        self.assertEqual(1, metrics["formal_invalid_findings"])
        self.assertEqual(0, metrics["formal_unjudged_findings"])
        self.assertEqual(1.0, metrics["formal_adjudication_coverage"])
        self.assertEqual(0.6667, metrics["adjudicated_formal_precision"])
        self.assertEqual(0.6667, metrics["adjudicated_formal_utility_rate"])
        self.assertEqual(0.3333, metrics["formal_nuisance_rate"])
        self.assertEqual(1.0, metrics["expanded_required_recall"])
        self.assertEqual(0.8, metrics["expanded_required_f1"])
        self.assertEqual(
            "human-adjudicated-targeted-labels",
            metrics["precision_interpretation"],
        )

    def test_suggestion_utility_uses_only_adjudicated_optional_and_invalid_labels(self):
        suggestions = [
            Finding(
                rule_id=rule_id, severity=Severity.MEDIUM,
                title=verdict, explanation="Adjudication fixture.",
                path="app.py", line=line, evidence="value_%d" % line,
                fix="Apply a focused fix.", test="Add a focused test.",
                source="security", disposition="suggestion",
            )
            for line, rule_id, verdict in (
                (1, "CWE-561", "optional"),
                (2, "CWE-20", "invalid"),
                (3, "CWE-248", "duplicate"),
                (4, "CWE-999", "unjudged"),
            )
        ]

        class SuggestionReviewer:
            name = "adjudicated-suggestions"

            def review_case(self, _case, _parsed):
                return []

            def evaluation_execution(self):
                return {}

            def evaluation_summary(self):
                return {
                    "suggestion_count": len(suggestions),
                    "suggested_findings": [item.to_dict() for item in suggestions],
                }

        case = {
            "id": "adjudicated-suggestions", "repository": "repo", "pull_request": 1,
            "split": "validation", "source": {"kind": "synthetic-controlled"},
            "diff": (
                "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,4 @@\n"
                "+value_1\n+value_2\n+value_3\n+value_4\n"
            ),
            "expected_findings": [],
            "suggestion_judgments": [
                {"path": "app.py", "line": 1, "rule_id": "CWE-561", "verdict": "optional"},
                {"path": "app.py", "line": 2, "rule_id": "CWE-20", "verdict": "invalid"},
                {"path": "app.py", "line": 3, "rule_id": "CWE-248", "verdict": "duplicate"},
            ],
        }

        metrics = ProductionEvaluationHarness().run(
            SuggestionReviewer(), [case], "adjudicated-suggestions"
        )["metrics"]
        self.assertEqual(1, metrics["suggestion_optional"])
        self.assertEqual(1, metrics["suggestion_invalid"])
        self.assertEqual(1, metrics["suggestion_duplicate"])
        self.assertEqual(1, metrics["suggestion_unjudged"])
        self.assertEqual(0.3333, metrics["suggestion_utility_rate"])
        self.assertEqual(0.75, metrics["suggestion_adjudication_coverage"])
        self.assertEqual(0.6667, metrics["suggestion_nuisance_rate"])

    def test_cached_result_rescore_updates_formal_truth_after_label_revision(self):
        formal = Finding(
            rule_id="CWE-95", severity=Severity.CRITICAL,
            title="eval", explanation="Dynamic execution.", path="app.py", line=1,
            evidence="eval(value)", fix="Remove eval.", test="Add an injection test.",
        )
        suggestion = Finding(
            rule_id="CWE-476", severity=Severity.HIGH,
            title="none", explanation="None dereference.", path="app.py", line=2,
            evidence="value.name", fix="Guard value.", test="Add a None test.",
            disposition="suggestion",
        )
        case = {
            "expected_findings": [
                {
                    "path": "app.py", "start_line": 1, "end_line": 1,
                    "cwe": "CWE-95", "severity": "critical", "should_comment": True,
                },
                {
                    "path": "app.py", "start_line": 2, "end_line": 2,
                    "cwe": "CWE-476", "severity": "high", "should_comment": True,
                },
            ],
        }
        cached = {
            "predicted_findings": [formal.to_dict()],
            "suggested_findings": [suggestion.to_dict()],
            "matches": [],
        }

        rescored = ProductionEvaluationHarness().rescore_cached_result(cached, case)

        self.assertEqual((1, 0, 1), (rescored["tp"], rescored["fp"], rescored["fn"]))
        self.assertEqual(1, rescored["incremental_suggestion_tp"])
        self.assertEqual(2, rescored["combined_tp_after_verification"])
        self.assertEqual(2, len(rescored["expected_findings"]))

    def test_agentic_arms_share_stable_rules_and_real_role_topologies(self):
        self.assertEqual(
            18,
            len(LocalRuleReviewer.RULES)
            + len(LocalRuleReviewer.DIFF_RULES)
            + len(ContextRuleReviewer.RULES),
        )
        expected_calls = {
            "multi-llm-no-critic": {
                "lead": 3, "security": 1, "correctness-reliability": 1,
            },
            # Critic is asked once to verify with a tool of its own before it
            # may judge, so it costs two calls, not one.
            "full-agentic": {
                "lead": 3, "security": 1,
                "correctness-reliability": 1, "critic": 2,
            },
        }
        parsed = parse_unified_diff(DIFF)
        for arm, calls in expected_calls.items():
            reviewer = ProductArmReviewer(arm, FakeClient(), 4096, 40)
            findings = reviewer.review(DIFF, parsed)
            self.assertEqual(["SEC-PATH-TRAVERSAL"], [item.rule_id for item in findings])
            actual = {}
            execution = reviewer.evaluation_execution()
            for item in execution["model_call_log"]:
                actual[item["role"]] = actual.get(item["role"], 0) + 1
            self.assertEqual(calls, actual)
            self.assertEqual(18, reviewer.evaluation_config()["deterministic_rules"])
            self.assertEqual(0, reviewer.evaluation_config()["max_revision_rounds"])
            self.assertFalse(
                reviewer.evaluation_config()["publish_unverified_suggestions"]
            )
            if arm == "full-agentic":
                self.assertTrue(any(
                    item["role"] == "critic" and item["tool"] == "changed_line"
                    for item in execution["tool_call_log"]
                ))

    def test_weak_hash_rule_ignores_fixed_fixture_but_keeps_dynamic_input(self):
        diff = (
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,2 @@\n"
            "+fixture = hashlib.md5(b'fixture-id').hexdigest()\n"
            "+digest = hashlib.md5(value).hexdigest()\n"
        )
        findings = ContextRuleReviewer().review(diff, parse_unified_diff(diff))

        self.assertEqual(1, len(findings))
        self.assertEqual("digest = hashlib.md5(value).hexdigest()", findings[0].evidence)

    def test_unbounded_retry_rule_requires_no_visible_break(self):
        bounded_diff = (
            "--- /dev/null\n+++ b/parser.py\n@@ -0,0 +1,5 @@\n"
            "+while True:\n+    if exhausted():\n+        break\n"
            "+    consume()\n+return result\n"
        )
        retry_diff = (
            "--- /dev/null\n+++ b/retry.py\n@@ -0,0 +1,3 @@\n"
            "+while True:\n+    if send():\n+        return True\n"
        )

        bounded = ContextRuleReviewer().review(
            bounded_diff, parse_unified_diff(bounded_diff)
        )
        retry = ContextRuleReviewer().review(
            retry_diff, parse_unified_diff(retry_diff)
        )

        self.assertEqual([], bounded)
        self.assertEqual(["REL-UNBOUNDED-RETRY"], [item.rule_id for item in retry])

    def test_non_production_data_can_debug_but_cannot_prove_claims(self):
        cases = []
        for index, split in enumerate(("train", "validation", "holdout"), 1):
            cases.append({
                "id": "case-%d" % index,
                "repository": "repo-%d" % index,
                "pull_request": index,
                "split": split,
                "source": {"kind": "synthetic-controlled"},
                "diff": DIFF,
                "expected_findings": [{
                    "path": "app.py", "start_line": 1, "end_line": 1,
                    "rule_id": "SEC-PATH-TRAVERSAL", "cwe": "CWE-22",
                    "severity": "high", "should_comment": True,
                }],
            })
        suite = FairAblationSuite(
            product_reviewer_factories(FakeClient(), 40),
            "fake-model", 4096, require_production_ready=False,
            bootstrap_iterations=200,
        )
        report = suite.run(cases)
        self.assertFalse(report["dataset"]["ready"])
        self.assertFalse(report["critic_gate"]["passed"])
        self.assertEqual(
            {"lead": 9, "security": 3, "correctness-reliability": 3, "critic": 6},
            report["arms"]["full-agentic"]["execution"]["model_role_calls"],
        )


if __name__ == "__main__":
    unittest.main()
