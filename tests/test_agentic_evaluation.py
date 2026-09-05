import json
import unittest
from types import SimpleNamespace

from evoagent.core.diff_parser import parse_unified_diff
from evoagent.agents.loop import AgentLoop
from evoagent.agents.parsing import (
    normalize_model_rule_id, parse_findings, worker_final_validation_error,
)
from evoagent.review.agentic import AgenticReviewer
from evoagent.review.merge import (
    apply_critic, merge_findings, normalize_delegations, partition_publication,
)
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
        managed = json.loads(user)
        task = json.loads(managed["task"])
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 10, "completion_tokens": 5}, 1,
            )
        if role == "lead":
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
            observations = managed.get("observations") or []
            own_success = any(
                item.get("ok") and int(item.get("step") or 0) >= 1
                for item in observations
            )
            if task.get("repository_context_available") and not own_success:
                line = task["scoreable_added_lines"][0]
                return {
                    "action": "tool", "tool": "changed_line",
                    "arguments": {"path": line["path"], "line": line["line"]},
                    "reason": "Verify an assigned changed line.",
                }
            evidence_ids = []
            for item in observations:
                result = item.get("result")
                evidence_id = (
                    result.get("evidence_id") if isinstance(result, dict)
                    else item.get("evidence_id")
                )
                if item.get("ok") and evidence_id:
                    evidence_ids.append(str(evidence_id))
            support = evidence_ids[-1:] if evidence_ids else []
            return {
                "action": "final", "findings": [],
                "hypotheses": [{
                    "hypothesis_id": "hyp-1",
                    "claim": "The assigned change may violate a repository contract.",
                    "status": "unresolved",
                    "explanation": "The fixture records the investigated risk without inventing a defect.",
                    "required_proof": "Inspect domain-specific callers and tests.",
                    "supporting_evidence_ids": support,
                }],
                "requirement_resolutions": [{
                    "requirement_id": item["requirement_id"],
                    "status": "satisfied" if support else "unresolved",
                    "explanation": "The assigned repository evidence was inspected.",
                    "supporting_evidence_ids": support,
                    "required_proof": "Obtain repository context." if not support else "",
                } for item in task.get("assignment_requirements") or []],
            }
        if role == "critic":
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
    @staticmethod
    def _causal_delta():
        return {
            "trigger": "A supported input reaches the changed call.",
            "before": "The old branch handled the input.",
            "after": "The new branch passes it to the failing operation.",
            "failure": "The operation raises instead of returning a result.",
            "contract": "The existing caller expects a result for this input.",
            "premises": [{
                "premise": "The supported input reaches this branch.",
                "status": "verified",
                "evidence": "Repository source shows the caller and branch.",
            }],
        }

    def test_critic_protocol_failure_fails_closed_without_failing_review(self):
        class FailingCritic:
            @staticmethod
            def _run_critic(*_args, **_kwargs):
                raise ValueError("invalid critic JSON")

        candidate = Finding(
            rule_id="CWE-248", severity=Severity.MEDIUM,
            title="Candidate", explanation="May raise.", path="app.py", line=1,
            evidence="value['key']", fix="Guard the key.",
            test="Exercise a missing key.", confidence=0.9,
        )
        ledger = ExecutionLedger("agentic")

        result = AgenticReviewer._critic(
            FailingCritic(),
            SimpleNamespace(diff="", task_id="task"),
            SimpleNamespace(suite=object(), ledger=ledger),
            [candidate], "verify", {},
        )

        self.assertFalse(result["decisions"][0]["publication_ready"])
        self.assertIn("failed closed", result["decisions"][0]["objections"][0])

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
        self.assertNotIn("MessageTemplate", queries)  # no unrelated second-hop search
        self.assertLessEqual(len(queries), 2)
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
        empty_index = RepositoryToolSuite.semantic_probe(
            "empty-sequence-index"
        )["output"]
        missing_key = RepositoryToolSuite.semantic_probe(
            "missing-mapping-key"
        )["output"]
        truthiness = RepositoryToolSuite.semantic_probe(
            "truthiness-vs-none"
        )["output"]
        json_serialization = RepositoryToolSuite.semantic_probe(
            "json-serialization"
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
        self.assertTrue(all(
            item["negative_one_index_raises"] for item in empty_index["results"]
        ))
        self.assertTrue(missing_key["missing_key_subscript_raises"])
        self.assertTrue(any(
            item["branches_diverge"] for item in truthiness["values"]
        ))
        self.assertTrue(json_serialization["json_serialization_raises"])
        self.assertTrue(all(
            output["requires_resolution"] is False for output in (
                empty_index, missing_key,
                truthiness, json_serialization,
            )
        ))
        self.assertEqual("RuntimeError", exception_cleanup["error_type"])
        self.assertTrue(all(
            item["arbitrary_code_executed"] is False
            for item in (
                serialization, equality, decorators, cycle, alias, path,
                security_default, workflow_expression, git_option,
                nullable_length, sentinel, module_getattr,
                dict_mutation, decimal_exponent, unhashable_exception,
                missing_scandir, empty_netrc, exception_cleanup,
                empty_index, missing_key,
                truthiness, json_serialization,
            )
        ))

    def test_evidence_revision_preflight_prioritizes_target_and_runs_safe_probe(self):
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
            "@@ -9 +10 @@\n+safe = normalize(value)\n"
            "@@ -99 +100 @@\n+last = pattern[-1]\n"
        )
        tools = RecordingTools()

        repository_preflight({
            "files": parsed.files,
            "evidence_targets": [{
                "evidence_target_id": "correctness-1:hyp-1",
                "path": "src/app.py", "line": 100,
                "claim": "An empty string is indexed at -1.",
                "required_proof": "Show whether empty strings reach this indexing operation.",
            }],
        }, parsed, tools)

        reads = [arguments for name, arguments in tools.calls if name == "read_file"]
        self.assertEqual(100, reads[0]["end_line"] - 25)
        self.assertIn(
            ("semantic_probe", {"kind": "empty-sequence-index"}), tools.calls,
        )

    def test_evidence_revision_does_not_repeat_automatic_preflight(self):
        class RecordingTools:
            def __init__(self):
                self.calls = []

            def names(self):
                return ["read_file", "search_repository", "ast_analyze"]

            def invoke(self, name, arguments):
                self.calls.append((name, dict(arguments)))
                return {"evidence_id": name + ":1", "tool": name, "output": {}}

        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n+value = changed()\n"
        )
        tools = RecordingTools()

        observations = repository_preflight({
            "files": parsed.files,
            "prior_worker_result": {"hypotheses": [{"status": "unresolved"}]},
            "evidence_targets": [{
                "path": "app.py", "line": 1,
                "required_proof": "Trace a caller.",
            }],
        }, parsed, tools)

        self.assertEqual([], observations)
        self.assertEqual([], tools.calls)

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
                "causal_delta": self._causal_delta(),
            }],
            "_observations": [],
        }

        candidates, decisions = apply_critic(
            result, [candidate],
        )

        self.assertEqual(0.8, candidates[0].confidence)
        self.assertEqual(-0.05, decisions[0]["recommended_confidence_adjustment"])
        self.assertTrue(decisions[0]["publication_ready"])

    def test_critic_boolean_verdict_without_causal_proof_fails_closed(self):
        candidate = Finding(
            rule_id="CWE-703", severity=Severity.MEDIUM,
            title="Missing directory escapes", explanation="The error propagates.",
            path="app.py", line=10, evidence="with os.scandir(path):",
            fix="Catch FileNotFoundError.", test="Cover a missing directory.",
            confidence=0.9, source="correctness-reliability",
        )
        result = {
            "decisions": [{
                "finding_index": 0, "accepted": True,
                "introduced_by_diff": True, "reproducible": True,
                "evidence_sufficient": True,
                "would_comment_on_real_pr": True, "objections": [],
            }],
            "_observations": [],
        }

        _candidates, decisions = apply_critic(result, [candidate])

        self.assertFalse(decisions[0]["publication_ready"])
        self.assertIn(
            "critic omitted a complete before/after causal proof",
            decisions[0]["objections"],
        )
        self.assertIn(
            "critic did not verify the premises used by the causal proof",
            decisions[0]["objections"],
        )

    def test_critic_rejection_without_counter_proof_is_inconclusive(self):
        candidate = Finding(
            rule_id="CWE-248", severity=Severity.MEDIUM,
            title="Removed type guard can crash", explanation="A new type reaches lookup.",
            path="app.py", line=10, evidence="if isinstance(value, Node):",
            fix="Restore the narrow guard.", test="Cover the formerly supported type.",
            confidence=0.8, source="correctness-reliability",
        )
        result = {
            "decisions": [{
                "finding_index": 0, "accepted": False,
                "objections": ["The new type is safe."],
            }],
            "_observations": [],
        }

        _candidates, decisions = apply_critic(result, [candidate])

        self.assertFalse(decisions[0]["rejection_ready"])
        self.assertEqual("inconclusive", decisions[0]["verdict"])

    def test_critic_rejection_with_counter_proof_is_conclusive(self):
        candidate = Finding(
            rule_id="CWE-248", severity=Severity.MEDIUM,
            title="Removed type guard can crash", explanation="A new type reaches lookup.",
            path="app.py", line=10, evidence="if isinstance(value, Node):",
            fix="Restore the narrow guard.", test="Cover the formerly supported type.",
            confidence=0.8, source="correctness-reliability",
        )
        diff = "--- a/app.py\n+++ b/app.py\n@@ -10 +10 @@\n-if isinstance(value, Class):\n+if isinstance(value, Node):\n"
        changed = RepositoryToolSuite("", diff, parse_unified_diff(diff)).changed_line("app.py", 10)
        proof = self._causal_delta()
        proof.update({
            "code_before": changed["output"]["change"]["before"],
            "code_after": changed["output"]["change"]["after"],
            "before": "The caller rejects that type.",
            "after": "The caller rejects that type.",
        })
        result = {
            "decisions": [{
                "finding_index": 0, "accepted": False,
                "objections": ["The caller rejects that type before this branch."],
                "causal_delta": proof,
                "supporting_evidence_ids": [changed["evidence_id"]],
            }],
            "_observations": [{"tool": "changed_line", "ok": True, "result": changed}],
        }

        _candidates, decisions = apply_critic(result, [candidate])

        self.assertTrue(decisions[0]["rejection_ready"])
        self.assertEqual("rejected", decisions[0]["verdict"])

        # Copying new code into the old side (the observed false-refutation
        # failure), inventing a citation, or citing a failed tool cannot prove
        # that the defect was pre-existing.
        for fault in ("wrong_old_code", "unknown_citation", "failed_tool"):
            with self.subTest(fault=fault):
                broken = json.loads(json.dumps(result))
                if fault == "wrong_old_code":
                    broken["decisions"][0]["causal_delta"]["code_before"] = proof["code_after"]
                elif fault == "unknown_citation":
                    broken["decisions"][0]["supporting_evidence_ids"] = ["changed_line:invented"]
                else:
                    broken["_observations"][0]["ok"] = False
                _, verdicts = apply_critic(broken, [candidate])
                self.assertFalse(verdicts[0]["rejection_ready"])
                self.assertEqual("inconclusive", verdicts[0]["verdict"])

    def test_critic_cannot_accept_while_reporting_blocking_objections(self):
        candidate = Finding(
            rule_id="CWE-682", severity=Severity.MEDIUM,
            title="Candidate", explanation="The result may be wrong.",
            path="app.py", line=1, evidence="value = open(base / user_path)",
            fix="Preserve the prior behavior.", test="Cover the boundary input.",
            confidence=0.9, source="correctness-reliability",
        )
        result = {
            "decisions": [{
                "finding_index": 0, "accepted": True,
                "introduced_by_diff": True, "reproducible": True,
                "evidence_sufficient": True,
                "would_comment_on_real_pr": True,
                "objections": ["The supported-input contract is still missing."],
            }],
            "_observations": [],
        }

        _candidates, decisions = apply_critic(result, [candidate])

        self.assertFalse(decisions[0]["accepted"])
        self.assertFalse(decisions[0]["publication_ready"])
        self.assertEqual(1, len(decisions[0]["objections"]))

    def test_critic_can_correct_a_mismatched_cwe_without_rewriting_finding(self):
        candidate = Finding(
            rule_id="CWE-835", severity=Severity.MEDIUM,
            title="Empty filter masks every value",
            explanation="The computed mask is all false.",
            path="app.py", line=1, evidence="value = open(base / user_path)",
            fix="Preserve empty-filter behavior.", test="Cover an empty filter.",
            confidence=0.9, source="correctness-reliability",
        )
        result = {
            "decisions": [{
                "finding_index": 0, "accepted": True,
                "introduced_by_diff": True, "reproducible": True,
                "evidence_sufficient": True,
                "would_comment_on_real_pr": True, "objections": [],
                "corrected_rule_id": "CWE-682",
                "causal_delta": self._causal_delta(),
            }],
            "_observations": [],
        }

        candidates, decisions = apply_critic(result, [candidate])

        self.assertEqual("CWE-682", candidates[0].rule_id)
        self.assertEqual("CWE-835", candidates[0].original_rule_id)
        self.assertEqual("CWE-682", decisions[0]["corrected_rule_id"])

    def test_critic_cannot_rewrite_a_deterministic_scanner_rule(self):
        candidate = Finding(
            rule_id="COR-EMPTY-SEQUENCE-ACCESS", severity=Severity.MEDIUM,
            title="Empty sequence access", explanation="An empty value raises.",
            path="app.py", line=1, evidence="value[-1]",
            fix="Guard the empty value.", test="Cover an empty value.",
            confidence=0.98, source="local-rule-scanner",
        )
        result = {
            "decisions": [{
                "finding_index": 0, "accepted": True,
                "introduced_by_diff": True, "reproducible": True,
                "evidence_sufficient": True,
                "would_comment_on_real_pr": True, "objections": [],
                "corrected_rule_id": "CWE-476",
            }],
            "_observations": [],
        }

        candidates, decisions = apply_critic(result, [candidate])

        self.assertEqual("COR-EMPTY-SEQUENCE-ACCESS", candidates[0].rule_id)
        self.assertEqual("", candidates[0].original_rule_id)
        self.assertEqual("", decisions[0]["corrected_rule_id"])

    def test_scanner_and_worker_taxonomy_variants_merge_by_shared_behavior(self):
        probe = {
            "evidence_id": "semantic_probe:missing-key", "tool": "semantic_probe",
            "output": {
                "kind": "missing-mapping-key",
                "missing_key_subscript_raises": True,
                "arbitrary_code_executed": False,
            },
        }
        scanner = Finding(
            rule_id="COR-MISSING-MAPPING-GUARD", severity=Severity.MEDIUM,
            title="Missing mapping guard", explanation="A missing key raises KeyError.",
            path="app.py", line=7, evidence="value['key']",
            evidence_refs=[{"evidence_id": "local:1", "tool": "local-rule-scanner"}],
            fix="Keep optional access.", test="Omit the key.", confidence=0.98,
            source="local-rule-scanner",
        )
        taxonomy_variant = Finding(
            rule_id="CWE-476", severity=Severity.MEDIUM,
            title="Direct mapping access raises KeyError",
            explanation="The missing mapping key raises KeyError.",
            path="app.py", line=7, evidence="value['key']",
            evidence_refs=[{
                "evidence_id": "read_file:security", "tool": "read_file",
            }], fix="Guard access.", test="Omit the key.",
            confidence=0.9, source="security",
        )
        exact_confirmation = Finding(
            rule_id="COR-MISSING-MAPPING-GUARD", severity=Severity.MEDIUM,
            title="Direct mapping access raises KeyError",
            explanation="The missing mapping key is reachable.",
            path="app.py", line=7, evidence="value['key']",
            evidence_refs=[
                {"evidence_id": "read_file:correctness", "tool": "read_file"}, probe,
            ], fix="Keep optional access.", test="Omit the key.",
            confidence=0.9, source="correctness-reliability",
        )
        distinct_mechanism = Finding(
            rule_id="CWE-798", severity=Severity.HIGH,
            title="A credential is embedded in the expression",
            explanation="The literal credential is exposed.",
            path="app.py", line=7, evidence="value['key']",
            evidence_refs=[{
                "evidence_id": "read_file:secret", "tool": "read_file",
            }], fix="Load the credential securely.", test="Check configuration.",
            confidence=0.9, source="security",
        )

        merged = merge_findings([
            scanner, taxonomy_variant, exact_confirmation, distinct_mechanism,
        ])

        self.assertEqual(2, len(merged))
        scanner_result = next(
            item for item in merged
            if item.rule_id == "COR-MISSING-MAPPING-GUARD"
        )
        self.assertEqual("correctness-reliability", scanner_result.source)
        self.assertTrue(any(
            ref.get("tool") == "local-rule-scanner"
            for ref in scanner_result.evidence_refs
        ))
        self.assertIn("CWE-798", {item.rule_id for item in merged})

    def test_scanner_corroboration_does_not_bypass_worker_semantic_review(self):
        scanner_ref = {
            "evidence_id": "local-rule:guard", "tool": "local-rule-scanner",
        }
        scanner = Finding(
            rule_id="COR-MISSING-MAPPING-GUARD", severity=Severity.MEDIUM,
            title="Missing mapping guard", explanation="A missing key raises.",
            path="app.py", line=7, evidence="value['key']",
            evidence_refs=[scanner_ref], fix="Guard access.",
            test="Omit the key.", confidence=0.98, source="local-rule-scanner",
        )
        worker = Finding(
            rule_id="COR-MISSING-MAPPING-GUARD", severity=Severity.MEDIUM,
            title="Mapped value is required", explanation="The caller may omit key.",
            path="app.py", line=7, evidence="value['key']",
            evidence_refs=[{
                "evidence_id": "read_file:caller", "tool": "read_file",
                "output": {"path": "app.py", "content": "value['key']"},
            }], fix="Guard access.", test="Omit the key.",
            confidence=0.8, source="correctness-reliability",
        )
        candidates = merge_findings([scanner, worker])

        published, _suggestions, decisions = partition_publication(
            [scanner], candidates, [],
            [{"finding_index": 0, "publication_ready": False}],
            repository_available=True,
        )

        self.assertEqual("correctness-reliability", candidates[0].source)
        self.assertEqual([], published)
        self.assertEqual("rejected", decisions[0]["disposition"])

    def test_failed_closed_critic_does_not_erase_the_safe_review(self):
        reviewer = ProductArmReviewer.__new__(ProductArmReviewer)
        reviewer.arm = "full-agentic"
        reviewer.expected_roles = (
            "lead", "security", "correctness-reliability", "critic",
        )
        reviewer._last_summary = {
            "collaboration": {"candidate_findings_before_critic": 1},
            "execution": {
                "model_call_log": [
                    {"role": role, "ok": True}
                    for role in ("lead", "security", "correctness-reliability")
                ],
                "agent_traces": {
                    "critic": [{"event": "critic_failed_closed"}],
                },
            },
        }

        reviewer._validate_execution()

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

    def test_invalid_action_envelope_is_repaired_once_on_demand(self):
        class SequencedClient:
            def __init__(self):
                self.actions = [{}, {"action": "final", "decisions": []}]

            def complete_json(self, *_args, **_kwargs):
                return self.actions.pop(0)

        result = AgentLoop(
            "critic", "Review.", SequencedClient(),
            token_budget=4000, time_budget=30,
        ).run("{}", ToolRegistry([]), ExecutionLedger("agentic"))

        self.assertEqual(2, result["_steps"])
        self.assertEqual(1, len(result["_observations"]))
        self.assertIn("Invalid action envelope", result["_observations"][0]["error"])

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
                            "proof_kind": "type_constraint",
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

    def test_worker_may_preserve_a_fixed_counterexample_as_unresolved(self):
        class UnresolvedClient:
            def complete_json(self, *_args, **_kwargs):
                return {
                    "action": "final", "findings": [],
                    "evidence_resolutions": [{
                        "evidence_id": "semantic_probe:test",
                        "status": "unresolved",
                        "explanation": "The probe diverges but callers are not available.",
                        "required_proof": "Inspect every caller precondition.",
                        "supporting_evidence_ids": ["semantic_probe:test"],
                    }],
                }

        initial = [{
            "step": 0, "tool": "semantic_probe", "ok": True,
            "result": {
                "evidence_id": "semantic_probe:test", "tool": "semantic_probe",
                "output": {
                    "requires_resolution": True,
                    "resolution_question": "Explain the divergent branch.",
                },
            },
        }]

        result = AgentLoop(
            "correctness-reliability", "Review.", UnresolvedClient(),
            token_budget=4000, time_budget=30,
        ).run(
            "{}", ToolRegistry([]), ExecutionLedger("agentic"),
            initial_observations=initial,
        )

        self.assertEqual(1, result["_steps"])
        self.assertEqual(
            "unresolved", result["evidence_resolutions"][0]["status"]
        )

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

    def test_repeated_unstructured_positive_signal_is_deferred_to_lead(self):
        class RepeatedClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                return {
                    "action": "final", "findings": [],
                    "evidence_resolutions": [{
                        "evidence_id": "read_file:defect", "status": "finding",
                        "explanation": "The removed guard admits an invalid state.",
                    }],
                }

        result = AgentLoop(
            "correctness-reliability", "Review.", RepeatedClient(),
            token_budget=4000, time_budget=30,
        ).run(
            "{}", ToolRegistry([]), ExecutionLedger("agentic"),
            initial_observations=[{
                "step": 0, "tool": "read_file", "ok": True,
                "result": {
                    "evidence_id": "read_file:defect", "tool": "read_file",
                    "output": {"content": "transition(record)"},
                },
            }],
        )

        self.assertEqual(2, result["_steps"])
        self.assertEqual("unresolved", result["evidence_resolutions"][0]["status"])
        self.assertEqual("high", result["hypotheses"][0]["risk_level"])
        self.assertIn("removed guard", result["hypotheses"][0]["claim"])

    def test_late_unstructured_positive_signal_is_deferred_without_another_call(self):
        class LateSignalClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, *_args, **_kwargs):
                self.calls += 1
                if self.calls < 3:
                    return {"action": "tool", "tool": "lookup", "arguments": {}}
                return {
                    "action": "final", "findings": [],
                    "evidence_resolutions": [{
                        "evidence_id": "lookup:defect", "status": "finding",
                        "explanation": "The late probe found a reachable invalid state.",
                    }],
                }

        client = LateSignalClient()
        result = AgentLoop(
            "correctness-reliability", "Review.", client,
            token_budget=4000, time_budget=30,
        ).run(
            "{}", ToolRegistry([AgentTool(
                "lookup", "Look up evidence.",
                {"type": "object", "properties": {}, "additionalProperties": False},
                lambda: {
                    "evidence_id": "lookup:defect", "tool": "read_file",
                    "output": {"content": "invalid state"},
                },
            )]), ExecutionLedger("agentic"),
        )

        self.assertEqual(3, client.calls)
        self.assertEqual("unresolved", result["evidence_resolutions"][0]["status"])
        self.assertEqual("high", result["hypotheses"][0]["risk_level"])

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

    def test_model_rule_normalization_corrects_python_index_and_mask_taxonomy(self):
        python_index = {
            "rule_id": "CWE-787", "path": "pkg/module.py",
            "title": "Out-of-bounds write raises IndexError",
            "explanation": "A short list is indexed beyond its length.",
            "evidence": "values[i] = replacement",
        }
        native_index = dict(python_index, path="src/module.c")
        all_masked = {
            "rule_id": "CWE-252", "path": "reader.py",
            "title": "Empty filter masks all data",
            "explanation": "The wrong mask converts all values to NaN.",
            "evidence": "data = np.where(valid, data, np.nan)",
        }

        self.assertEqual("CWE-129", normalize_model_rule_id(python_index))
        self.assertEqual("CWE-787", normalize_model_rule_id(native_index))
        self.assertEqual("CWE-682", normalize_model_rule_id(all_masked))

    def test_model_rule_normalization_corrects_python_keyerror_taxonomy(self):
        raw = {
            "rule_id": "CWE-476", "path": "pkg/processor.py",
            "title": "Direct indexing of a missing key raises KeyError",
            "explanation": "The mapping no longer uses get().",
            "evidence": 'value = event["transaction"]',
        }

        self.assertEqual("CWE-248", normalize_model_rule_id(raw))
        raw.update({
            "title": "Optional object is None",
            "explanation": "Dereferencing the object raises AttributeError.",
            "evidence": "value = optional.name",
        })
        self.assertEqual("CWE-476", normalize_model_rule_id(raw))

    def test_scanner_findings_are_published_without_seeding_agents(self):
        class RecordingClient(FakeClient):
            def __init__(self):
                self.tasks = []

            def complete_json(self, role, system, user, ledger=None, max_tokens=None):
                managed = json.loads(user)
                self.tasks.append((role, json.loads(managed["task"])))
                return super().complete_json(
                    role, system, user, ledger=ledger, max_tokens=max_tokens,
                )

        diff = (
            "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"
            "-value = event.get(\"transaction\")\n"
            "+value = event[\"transaction\"]\n"
        )
        client = RecordingClient()
        reviewer = ProductArmReviewer("full-agentic", client, 4096)

        findings = reviewer.review(diff, parse_unified_diff(diff))

        self.assertEqual(1, reviewer.evaluation_summary()["scanner_findings"])
        self.assertEqual("local-rule-scanner", findings[0].source)
        seeded = [
            task["scanner_findings"]
            for _role, task in client.tasks if "scanner_findings" in task
        ]
        self.assertTrue(seeded)
        self.assertTrue(all(not values for values in seeded))

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

    def test_worker_refutation_requires_an_invariant_and_repository_evidence(self):
        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+consume(value)\n"
        )
        result = {
            "findings": [],
            "hypotheses": [{
                "hypothesis_id": "hyp-1", "claim": "value may be None",
                "status": "refuted",
                "explanation": "The repository type says value is always present.",
                "supporting_evidence_ids": ["read_file:type"],
            }],
            "_observations": [{
                "step": 1, "tool": "read_file", "ok": True,
                "result": {
                    "evidence_id": "read_file:type", "tool": "read_file",
                    "output": {"content": "value: str"},
                },
            }],
        }

        error = worker_final_validation_error(
            result, parsed, demand_hypotheses=True,
            assignment={"worker": "correctness-reliability"},
            repository_available=True,
        )
        self.assertIn("proof_kind", error)
        result["hypotheses"][0]["proof_kind"] = "type_constraint"
        self.assertEqual(
            "",
            worker_final_validation_error(
                result, parsed, demand_hypotheses=True,
                assignment={"worker": "correctness-reliability"},
                repository_available=True,
            ),
        )
        result["_observations"][0]["result"]["output"] = {
            "content": "consume(value)"
        }
        self.assertIn(
            "does not establish",
            worker_final_validation_error(
                result, parsed, demand_hypotheses=True,
                assignment={"worker": "correctness-reliability"},
                repository_available=True,
            ),
        )

    def test_worker_cannot_refute_a_risk_as_out_of_scope(self):
        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+consume(value)\n"
        )
        result = {
            "findings": [],
            "hypotheses": [{
                "hypothesis_id": "hyp-1", "claim": "value may be None",
                "status": "refuted", "proof_kind": "documented_contract",
                "explanation": "This is not a security issue; it belongs to correctness.",
                "supporting_evidence_ids": ["read_file:type"],
            }],
            "_observations": [{
                "step": 1, "tool": "read_file", "ok": True,
                "result": {
                    "evidence_id": "read_file:type", "tool": "read_file",
                    "output": {"content": "consume(value)"},
                },
            }],
        }

        error = worker_final_validation_error(
            result, parsed, demand_hypotheses=True,
            assignment={"worker": "security"}, repository_available=True,
        )

        self.assertIn("use handoff or unresolved", error)

    def test_worker_validator_downgrades_unsupported_refutation(self):
        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+consume(value)\n"
        )
        action = {
            "action": "final", "findings": [],
            "hypotheses": [{
                "hypothesis_id": "hyp-1", "claim": "value may be None",
                "status": "refuted", "proof_kind": "exhaustive_paths",
                "explanation": "One nearby read did not show a None value.",
                "supporting_evidence_ids": ["read_file:type"],
            }],
            "_observations": [{
                "step": 1, "tool": "read_file", "ok": True,
                "result": {
                    "evidence_id": "read_file:type", "tool": "read_file",
                    "output": {"content": "consume(value)"},
                },
            }],
        }

        error = AgenticReviewer._worker_validator(
            parsed, {"worker": "correctness-reliability"}, True,
        )(action)

        self.assertEqual("", error)
        self.assertEqual("unresolved", action["hypotheses"][0]["status"])
        self.assertEqual("high", action["hypotheses"][0]["risk_level"])
        self.assertEqual(
            "protocol-downgrade", action["hypotheses"][0]["origin"]
        )
        self.assertIn("does not establish", action["protocol_downgrades"][0]["reason"])

    def test_unresolved_is_a_legal_worker_completion(self):
        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+consume(value)\n"
        )
        assignment = {
            "worker": "correctness-reliability",
            "required_evidence": ["Verify the value contract."],
        }
        result = {
            "findings": [],
            "hypotheses": [{
                "hypothesis_id": "hyp-1", "claim": "value may be None",
                "status": "unresolved",
                "explanation": "The only visible annotation is optional.",
                "required_proof": "Enumerate all writers of value.",
                "supporting_evidence_ids": ["read_file:type"],
            }],
            "requirement_resolutions": [{
                "requirement_id": "req-1", "status": "satisfied",
                "explanation": "The visible value contract was inspected.",
                "supporting_evidence_ids": ["read_file:type"],
            }],
            "_observations": [{
                "step": 1, "tool": "read_file", "ok": True,
                "result": {
                    "evidence_id": "read_file:type", "tool": "read_file",
                    "output": {"content": "value: str | None"},
                },
            }],
        }

        self.assertEqual(
            "",
            worker_final_validation_error(
                result, parsed, demand_hypotheses=True,
                assignment=assignment, repository_available=True,
            ),
        )

    def test_every_lead_requirement_must_be_resolved(self):
        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+consume(value)\n"
        )
        result = {
            "findings": [],
            "hypotheses": [{
                "hypothesis_id": "hyp-1", "claim": "value may be None",
                "status": "unresolved", "explanation": "Writers are not visible.",
                "required_proof": "Inspect all writers.",
            }],
            "requirement_resolutions": [{
                "requirement_id": "req-1", "status": "unresolved",
                "explanation": "The type contract is unavailable.",
                "required_proof": "Inspect the declaration.",
            }],
        }
        assignment = {
            "worker": "correctness-reliability",
            "required_evidence": ["Verify types.", "Verify callers."],
        }

        error = worker_final_validation_error(
            result, parsed, demand_hypotheses=True,
            assignment=assignment, repository_available=False,
        )

        self.assertIn("Missing: req-2", error)

    def test_silent_worker_pass_is_preserved_as_protocol_unresolved(self):
        class SilentClient:
            def complete_json(self, *_args, **_kwargs):
                return {"action": "final", "findings": []}

        parsed = parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+consume(value)\n"
        )
        validator = AgenticReviewer._worker_validator(
            parsed, {
                "worker": "security", "objective": "Review trust boundaries.",
                "required_evidence": ["Trace untrusted input."],
            }, False,
        )

        result = AgentLoop(
            "security", "Review.", SilentClient(), 4000, 30,
            final_action_validator=validator,
        ).run("{}", ToolRegistry([]), ExecutionLedger("agentic"))

        self.assertEqual([], result["findings"])
        self.assertEqual("unresolved", result["hypotheses"][0]["status"])
        self.assertEqual("protocol-fallback", result["hypotheses"][0]["origin"])
        self.assertEqual(
            "unresolved", result["requirement_resolutions"][0]["status"]
        )

    def test_high_risk_unresolved_hypothesis_does_not_force_revision(self):
        delegations = [{
            "assignment_id": "security-1", "worker": "security",
            "files": ["app.py"],
        }]
        worker_results = {"security-1": {
            "assignment_id": "security-1", "worker": "security",
            "status": "completed", "handled_handoff_ids": [], "handoffs": [],
            "hypotheses": [{
                "hypothesis_id": "auth-boundary", "status": "unresolved",
                "risk_level": "high", "claim": "Authorization may be bypassed.",
                "location": "app.py:1", "required_proof": "Trace all callers.",
            }],
        }}

        decision = AgenticReviewer._complete_assessment_protocol(
            {"action": "final", "revision_requests": []},
            delegations, worker_results, remaining_rounds=1,
        )

        self.assertEqual([], decision["revision_requests"])
        self.assertEqual("defer", decision["handoff_decisions"][0]["action"])

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

    def test_evaluation_separates_worker_scanner_and_product_recall(self):
        worker = Finding(
            rule_id="CWE-248", severity=Severity.HIGH,
            title="Missing-key failure", explanation="The first lookup can fail.",
            path="app.py", line=1, evidence="mapping[first]",
            fix="Guard the lookup.", test="Cover a missing first key.",
            source="correctness-reliability",
        )
        worker_second = Finding(
            rule_id="CWE-129", severity=Severity.HIGH,
            title="Empty sequence", explanation="The second access can fail.",
            path="app.py", line=2, evidence="values[0]",
            fix="Guard the access.", test="Cover an empty sequence.",
            source="correctness-reliability",
        )
        scanner_first = Finding(
            rule_id="COR-MISSING-MAPPING-GUARD", severity=Severity.HIGH,
            title="Missing-key failure", explanation="The first lookup can fail.",
            path="app.py", line=1, evidence="mapping[first]",
            fix="Guard the lookup.", test="Cover a missing first key.",
            source="local-rule-scanner",
        )
        scanner_second = Finding(
            rule_id="COR-EMPTY-SEQUENCE-ACCESS", severity=Severity.HIGH,
            title="Empty sequence", explanation="The second access can fail.",
            path="app.py", line=2, evidence="values[0]",
            fix="Guard the access.", test="Cover an empty sequence.",
            source="local-rule-scanner",
        )

        class MixedReviewer:
            name = "mixed"

            def review_case(self, _case, _parsed):
                return [worker, scanner_second]

            def evaluation_summary(self):
                return {
                    "worker_results": [{"findings": [
                        worker.to_dict(), worker_second.to_dict(),
                    ]}],
                    "scanner_finding_details": [
                        scanner_first.to_dict(), scanner_second.to_dict(),
                    ],
                    "publication_decisions": [
                        {"source": "correctness-reliability",
                         "rule_id": "CWE-248", "disposition": "confirmed"},
                        {"source": "correctness-reliability",
                         "rule_id": "CWE-129", "disposition": "rejected"},
                    ],
                }

        case = {
            "id": "source-lanes", "repository": "repo", "pull_request": 1,
            "split": "validation", "source": {"kind": "synthetic-controlled"},
            "diff": (
                "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1,2 @@\n"
                "+mapping[first]\n+values[0]\n"
            ),
            "expected_findings": [
                {"path": "app.py", "start_line": 1, "end_line": 1,
                 "cwe": "CWE-248", "severity": "high"},
                {"path": "app.py", "start_line": 2, "end_line": 2,
                 "cwe": "CWE-129", "severity": "high"},
            ],
        }

        metrics = ProductionEvaluationHarness().run(
            MixedReviewer(), [case], "source-lanes"
        )["metrics"]

        self.assertEqual(1.0, metrics["worker_formal_strict_recall"])
        self.assertEqual(0.5, metrics["worker_published_strict_recall"])
        self.assertEqual(1.0, metrics["scanner_strict_recall"])
        self.assertEqual(0.0, metrics["scanner_unique_strict_recall"])
        self.assertEqual(0.5, metrics["scanner_publication_rescue_strict_recall"])
        self.assertEqual(1.0, metrics["recall"])
        self.assertEqual(1.0, metrics["target_detection_recall"])

    def test_cwe_mismatch_is_a_taxonomy_miss_not_a_target_miss(self):
        finding = Finding(
            rule_id="CWE-476", severity=Severity.HIGH,
            title="Missing-key failure",
            explanation="The removed guard allows a missing key lookup.",
            path="app.py", line=1, evidence="mapping[key]",
            fix="Restore the guard.", test="Cover a missing key.",
            source="correctness-reliability",
        )

        class WrongTaxonomyReviewer:
            name = "wrong-taxonomy"

            def review_case(self, _case, _parsed):
                return [finding]

        case = {
            "id": "taxonomy", "repository": "repo", "pull_request": 1,
            "split": "validation", "source": {"kind": "synthetic-controlled"},
            "diff": "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+mapping[key]\n",
            "expected_findings": [{
                "path": "app.py", "start_line": 1, "end_line": 1,
                "cwe": "CWE-248", "severity": "high",
            }],
        }

        metrics = ProductionEvaluationHarness().run(
            WrongTaxonomyReviewer(), [case], "taxonomy"
        )["metrics"]

        self.assertEqual(0, metrics["tp"])
        self.assertEqual(1.0, metrics["target_detection_recall"])
        self.assertEqual(1.0, metrics["targeted_review_recall"])
        self.assertEqual(0.0, metrics["taxonomy_accuracy_on_detected_targets"])
        self.assertEqual(1.0, metrics["adjudicated_formal_precision"])

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
            21,
            len(LocalRuleReviewer.RULES)
            + len(LocalRuleReviewer.DIFF_RULES)
            + len(ContextRuleReviewer.RULES),
        )
        expected_calls = {
            "multi-llm-no-critic": {
                "lead": 3, "security": 1, "correctness-reliability": 1,
            },
            "full-agentic": {
                "lead": 3, "security": 1,
                "correctness-reliability": 1, "critic": 1,
            },
        }
        parsed = parse_unified_diff(DIFF)
        for arm, calls in expected_calls.items():
            reviewer = ProductArmReviewer(
                arm, FakeClient(), 4096, 40, max_revision_rounds=0,
            )
            findings = reviewer.review(DIFF, parsed)
            self.assertEqual(["SEC-PATH-TRAVERSAL"], [item.rule_id for item in findings])
            actual = {}
            execution = reviewer.evaluation_execution()
            for item in execution["model_call_log"]:
                actual[item["role"]] = actual.get(item["role"], 0) + 1
            self.assertEqual(calls, actual)
            self.assertEqual(21, reviewer.evaluation_config()["deterministic_rules"])
            self.assertFalse(
                reviewer.evaluation_config()["scanner_findings_seed_agents"]
            )
            self.assertEqual(
                "parallel-unseeded-publication-safety-net",
                reviewer.evaluation_config()["scanner_orchestration"],
            )
            self.assertEqual(
                64, len(reviewer.evaluation_config()["scanner_catalog_sha256"])
            )
            self.assertEqual(0, reviewer.evaluation_config()["max_revision_rounds"])
            self.assertFalse(
                reviewer.evaluation_config()["publish_unverified_suggestions"]
            )
            evaluation = reviewer.evaluation_summary()
            self.assertEqual(
                evaluation["scanner_findings"],
                len(evaluation["scanner_finding_details"]),
            )
            self.assertTrue(evaluation["lead_assessments"])
            self.assertEqual([], evaluation["revision_results"])
            self.assertTrue(all(
                item["hypotheses"] for item in evaluation["worker_results"]
            ))
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
            {"lead": 9, "security": 3, "correctness-reliability": 3, "critic": 3},
            report["arms"]["full-agentic"]["execution"]["model_role_calls"],
        )


if __name__ == "__main__":
    unittest.main()
