import unittest

from evoagent.agents.loop import collect_evidence
from evoagent.review.merge import partition_publication
from evoagent.core.diff_parser import parse_unified_diff
from evoagent.core.finding_policy import normalize_rule_id
from evoagent.core.gates import FindingGate
from evoagent.core.models import Finding, Severity


DIFF = "--- a/app.py\n+++ b/app.py\n@@ -0,0 +1 @@\n+dangerous(value)\n"


def finding(rule_id="MODEL-CLAIM", source="security", severity=Severity.MEDIUM):
    return Finding(
        rule_id=rule_id,
        severity=severity,
        title="Potential issue",
        explanation="The added call may violate an unstated invariant.",
        path="app.py",
        line=1,
        evidence="dangerous(value)",
        fix="Validate the invariant before invoking the call.",
        test="Add a regression test for the invalid value.",
        confidence=0.9,
        source=source,
    )


class FindingPolicyTests(unittest.TestCase):
    def test_known_aliases_are_canonicalized_and_generic_ids_are_bucketed(self):
        self.assertEqual(
            "SEC-PICKLE-LOAD",
            normalize_rule_id("injection-pickle-untrusted"),
        )
        self.assertEqual("LLM-OTHER", normalize_rule_id("correctness-1"))
        self.assertEqual("LLM-OTHER", normalize_rule_id("SEC-001"))
        self.assertEqual("QUALITY-UNFINISHED", normalize_rule_id("quality_unfinished"))

    def test_deterministic_baseline_survives_lead_and_critic_rejection(self):
        baseline = finding("SEC-EVAL", "local-rule-scanner", Severity.CRITICAL)
        candidate = finding()

        published, suggestions, decisions = partition_publication(
            [baseline],
            [baseline, candidate],
            [],
            [
                {"finding_index": 0, "publication_ready": False},
                {"finding_index": 1, "publication_ready": False},
            ],
            repository_available=False,
        )

        self.assertEqual(["SEC-EVAL"], [item.rule_id for item in published])
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])
        self.assertEqual("rejected", decisions[1]["disposition"])

    def test_lead_selected_model_claim_without_repository_proof_is_a_suggestion(self):
        candidate = finding()

        published, suggestions, decisions = partition_publication(
            [],
            [candidate],
            [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=False,
        )

        self.assertEqual([], published)
        self.assertEqual(["MODEL-CLAIM"], [item.rule_id for item in suggestions])
        self.assertEqual("suggestion", decisions[0]["disposition"])
        self.assertIn("repository context is unavailable", decisions[0]["reasons"])

    def test_bounded_behavioral_probe_can_replace_repository_checkout(self):
        candidate = finding(rule_id="CWE-22", severity=Severity.HIGH)
        candidate.title = "Path traversal escapes repository containment"
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:path-containment",
            "tool": "semantic_probe",
            "output": {
                "kind": "path-containment",
                "parent_segments_escape_base": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=False,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])

    def test_path_probe_does_not_prove_symlink_error_branch_claim(self):
        candidate = finding(rule_id="CWE-22", severity=Severity.HIGH)
        candidate.title = "Symlink escape in EvalSymlinks IsNotExist branch"
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:path-containment",
            "tool": "semantic_probe",
            "output": {
                "kind": "path-containment",
                "parent_segments_escape_base": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=False,
            publish_unverified_suggestions=False,
        )

        self.assertEqual([], published)
        self.assertEqual([], suggestions)
        self.assertEqual("rejected", decisions[0]["disposition"])
        self.assertIn(
            "stability profile suppresses unverified suggestions",
            decisions[0]["reasons"],
        )

    def test_unrelated_semantic_probe_cannot_publish_high_risk_claim(self):
        candidate = finding(rule_id="CWE-347", severity=Severity.HIGH)
        candidate.title = "JWT signature verification disabled"
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:path-containment",
            "tool": "semantic_probe",
            "output": {
                "kind": "path-containment",
                "parent_segments_escape_base": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=False,
        )

        self.assertEqual([], published)
        self.assertEqual([candidate], suggestions)
        self.assertIn("repository context is unavailable", decisions[0]["reasons"])

    def test_workflow_shell_probe_supports_command_injection_claim(self):
        candidate = finding(rule_id="CWE-78", severity=Severity.CRITICAL)
        candidate.title = "GitHub Actions expression permits command injection"
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:github-actions-expression-shell",
            "tool": "semantic_probe",
            "output": {
                "kind": "github-actions-expression-shell",
                "shell_metacharacters_reach_direct_command": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=False,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])

    def test_git_option_probe_supports_normalization_bypass_claim(self):
        candidate = finding(rule_id="CWE-184", severity=Severity.HIGH)
        candidate.title = "Unsafe upload_pack kwarg bypasses canonical option check"
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:git-option-normalization",
            "tool": "semantic_probe",
            "output": {
                "kind": "git-option-normalization",
                "dangerous_flag_emitted_after_raw_check": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=False,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])

    def test_repository_backed_lead_and_critic_approved_claim_is_publishable(self):
        candidate = finding()
        candidate.evidence_refs = [{
            "evidence_id": "read_file:1234",
            "tool": "read_file",
            "output": {"path": "app.py", "content": "dangerous(value)"},
        }]

        published, suggestions, decisions = partition_publication(
            [],
            [candidate],
            [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual(["MODEL-CLAIM"], [item.rule_id for item in published])
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])

    def test_low_severity_model_claim_remains_advisory(self):
        candidate = finding(severity=Severity.LOW)
        candidate.evidence_refs = [{
            "evidence_id": "read_file:1234",
            "tool": "read_file",
            "output": {"path": "app.py", "content": "dangerous(value)"},
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([], published)
        self.assertEqual([candidate], suggestions)
        self.assertIn(
            "low-severity model finding remains advisory",
            decisions[0]["reasons"],
        )

    def test_low_confidence_model_claim_remains_advisory(self):
        candidate = finding()
        candidate.confidence = 0.79
        candidate.evidence_refs = [{
            "evidence_id": "read_file:1234",
            "tool": "read_file",
            "output": {"path": "app.py", "content": "dangerous(value)"},
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([], published)
        self.assertEqual([candidate], suggestions)
        self.assertIn(
            "model confidence below stable publication threshold 0.80",
            decisions[0]["reasons"],
        )

    def test_fully_verified_critic_can_publish_evidence_backed_point_six(self):
        candidate = finding(source="correctness-reliability")
        candidate.confidence = 0.6
        candidate.evidence_refs = [{
            "evidence_id": "read_file:proof",
            "tool": "read_file",
            "output": {
                "path": "app.py",
                "start_line": 1,
                "end_line": 4,
                "content": "dangerous(value)\nresult = consume(value)",
            },
        }]
        critic = {
            "finding_index": 0,
            "publication_ready": True,
            "introduced_by_diff": True,
            "reproducible": True,
            "evidence_sufficient": True,
            "would_comment_on_real_pr": True,
            "differential_causality": True,
            "premises_verified": True,
        }

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate], [critic], repository_available=True,
            publish_unverified_suggestions=False,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])

    def test_bare_critic_ready_flag_does_not_lower_confidence_threshold(self):
        candidate = finding(source="correctness-reliability")
        candidate.confidence = 0.7
        candidate.evidence_refs = [{
            "evidence_id": "read_file:proof",
            "tool": "read_file",
            "output": {"path": "app.py", "content": "dangerous(value)"},
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([], published)
        self.assertEqual([candidate], suggestions)
        self.assertIn(
            "model confidence below stable publication threshold 0.80",
            decisions[0]["reasons"],
        )

    def test_verified_review_survives_severity_and_tool_type_heuristics(self):
        critic = {"finding_index": 0, "publication_ready": True, **{
            key: True for key in (
                "introduced_by_diff", "reproducible", "evidence_sufficient",
                "would_comment_on_real_pr", "differential_causality", "premises_verified",
            )
        }}
        for severity in (Severity.LOW, Severity.CRITICAL):
            with self.subTest(severity=severity):
                candidate = finding(rule_id="CWE-95", severity=severity)
                candidate.evidence_refs = [{
                    "evidence_id": "read_file:contract", "tool": "read_file",
                    "output": {"path": "app.py", "content": "dangerous(value)"},
                }]
                published, suggestions, _ = partition_publication(
                    [], [candidate], [candidate], [critic], repository_available=True,
                    publish_unverified_suggestions=False,
                )
                self.assertEqual([], suggestions)
                self.assertEqual([candidate], FindingGate().apply(
                    published, parse_unified_diff(DIFF),
                ).accepted)
                self.assertEqual(severity, candidate.severity)

    def test_verified_review_cannot_bypass_location_or_release_requirements(self):
        critic = {"finding_index": 0, "publication_ready": True, **{
            key: True for key in (
                "introduced_by_diff", "reproducible", "evidence_sufficient",
                "would_comment_on_real_pr", "differential_causality", "premises_verified",
            )
        }}
        for field, value in (("line", 99), ("fix", ""), ("test", "")):
            with self.subTest(field=field):
                candidate = finding(severity=Severity.CRITICAL)
                setattr(candidate, field, value)
                candidate.evidence_refs = [{
                    "evidence_id": "read_file:contract", "tool": "read_file",
                    "output": {"path": "app.py", "content": "dangerous(value)"},
                }]
                published, _, _ = partition_publication(
                    [], [candidate], [candidate], [critic], repository_available=True,
                    publish_unverified_suggestions=False,
                )
                self.assertEqual([], FindingGate().apply(published, parse_unified_diff(DIFF)).accepted)

    def test_exact_stable_confidence_threshold_survives_float_subtraction(self):
        candidate = finding()
        candidate.confidence = 0.85 - 0.05
        candidate.evidence_refs = [{
            "evidence_id": "read_file:1234",
            "tool": "read_file",
            "output": {"path": "app.py", "content": "dangerous(value)"},
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])

    def test_verified_behavioral_claim_uses_evidence_backed_threshold(self):
        candidate = finding(
            rule_id="CWE-703", source="correctness-reliability",
            severity=Severity.MEDIUM,
        )
        candidate.title = "Missing directory error escapes scanner"
        candidate.explanation = "os.scandir raises FileNotFoundError before entry."
        candidate.confidence = 0.7
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:scandir",
            "tool": "semantic_probe",
            "output": {
                "kind": "scandir-missing-directory",
                "scandir_open_raises": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
            publish_unverified_suggestions=False,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual("confirmed", decisions[0]["disposition"])

    def test_behavioral_claim_below_evidence_backed_threshold_is_rejected(self):
        candidate = finding(rule_id="CWE-703", severity=Severity.MEDIUM)
        candidate.title = "Missing directory error escapes scanner"
        candidate.explanation = "os.scandir raises FileNotFoundError before entry."
        candidate.confidence = 0.69
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:scandir",
            "tool": "semantic_probe",
            "output": {
                "kind": "scandir-missing-directory",
                "scandir_open_raises": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, _suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
            publish_unverified_suggestions=False,
        )

        self.assertEqual([], published)
        self.assertIn(
            "model confidence below stable publication threshold 0.70",
            decisions[0]["reasons"],
        )

    def test_self_refuting_model_finding_is_not_published(self):
        candidate = finding(rule_id="CWE-248", source="correctness-reliability")
        candidate.title = "Guard raises TypeError"
        candidate.explanation = (
            "The isinstance guard short-circuits correctly, so no defect is introduced."
        )
        candidate.fix = "No fix needed."
        candidate.test = "No test needed."
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:decimal",
            "tool": "semantic_probe",
            "output": {
                "kind": "decimal-special-exponent",
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
            publish_unverified_suggestions=False,
        )

        self.assertEqual([], published)
        self.assertEqual([], suggestions)
        self.assertIn(
            "finding explicitly states that no actionable defect exists",
            decisions[0]["reasons"],
        )

    def test_same_line_repository_search_cannot_prove_high_risk_behavior(self):
        candidate = finding(severity=Severity.HIGH)
        candidate.evidence_refs = [{
            "evidence_id": "search_repository:contract",
            "tool": "search_repository",
            "output": [{"path": "app.py", "line": 2, "content": "value = None"}],
        }]
        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([], published)
        self.assertEqual([candidate], suggestions)
        self.assertIn(
            "high-risk claim lacks behavioral or cross-call evidence",
            decisions[0]["reasons"],
        )

    def test_repository_backed_correctness_claim_downgrades_unsupported_high_impact(self):
        candidate = finding(
            rule_id="CWE-703",
            source="correctness-reliability",
            severity=Severity.HIGH,
        )
        candidate.evidence_refs = [{
            "evidence_id": "read_file:state",
            "tool": "read_file",
            "output": {"path": "app.py", "content": "dangerous(value)"},
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
            publish_unverified_suggestions=False,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual(Severity.MEDIUM, candidate.severity)
        self.assertEqual("confirmed", decisions[0]["disposition"])
        self.assertEqual(
            {"from": "high", "to": "medium", "reason": (
                "correctness defect is repository-backed, but high impact is not "
                "supported by concrete outage, data-loss, or corruption evidence"
            )},
            decisions[0]["severity_adjustment"],
        )

    def test_behavioral_probe_proves_crash_but_not_high_impact(self):
        candidate = finding(
            rule_id="CWE-248",
            source="correctness-reliability",
            severity=Severity.HIGH,
        )
        candidate.title = "Unhashable exception crashes traceback rendering"
        candidate.explanation = "Putting the exception in a set raises TypeError."
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:unhashable",
            "tool": "semantic_probe",
            "output": {
                "kind": "unhashable-exception-membership",
                "set_insertion_raises": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, _suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
            publish_unverified_suggestions=False,
        )

        self.assertEqual([candidate], published)
        self.assertEqual(Severity.MEDIUM, candidate.severity)
        self.assertEqual("medium", decisions[0]["severity_adjustment"]["to"])

    def test_git_normalization_probe_does_not_prove_prefix_false_positive(self):
        candidate = finding(
            rule_id="CWE-697",
            source="correctness-reliability",
            severity=Severity.HIGH,
        )
        candidate.title = "Unsafe option check false positive for --config-file"
        candidate.explanation = "A prefix comparison may incorrectly reject it."
        candidate.evidence_refs = [{
            "evidence_id": "semantic_probe:git-option-normalization",
            "tool": "semantic_probe",
            "output": {
                "kind": "git-option-normalization",
                "dangerous_flag_emitted_after_raw_check": True,
                "arbitrary_code_executed": False,
            },
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([], published)
        self.assertEqual([candidate], suggestions)
        self.assertIn(
            "hypothetical rejection claim lacks behavioral or configured-value evidence",
            decisions[0]["reasons"],
        )

    def test_cross_call_repository_evidence_can_support_high_risk(self):
        candidate = finding(severity=Severity.HIGH)
        candidate.call_chain = [
            {"path": "app.py", "line": 1, "symbol": "dangerous"},
            {"path": "config.py", "line": 20, "symbol": "load_value"},
        ]
        candidate.evidence_refs = [{
            "evidence_id": "search_repository:contract",
            "tool": "search_repository",
            "output": [{
                "path": "config.py", "line": 20, "content": "value = None",
            }],
        }]
        published, suggestions, _decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        result = FindingGate().apply(published, parse_unified_diff(DIFF))

        self.assertEqual([], suggestions)
        self.assertEqual([candidate], result.accepted)
        self.assertTrue(candidate.gate["collaborative_repository_verification"])

    def test_lead_only_repository_search_cannot_support_high_risk(self):
        candidate = finding(severity=Severity.HIGH)
        candidate.evidence_refs = [{
            "evidence_id": "search_repository:contract",
            "tool": "search_repository",
            "output": [{"path": "app.py", "line": 2, "content": "value = None"}],
        }]
        candidate.gate = {
            "lead_selected": True,
            "critic_publication_ready": False,
            "publication_partition_passed": True,
        }

        result = FindingGate().apply([candidate], parse_unified_diff(DIFF))

        self.assertEqual([], result.accepted)
        self.assertIn(
            "evidence gate: high-risk finding requires verified scanner or tool evidence",
            result.rejected[0]["reasons"],
        )

    def test_empty_repository_search_is_not_publication_evidence(self):
        candidate = finding()
        candidate.evidence_refs = [{
            "evidence_id": "search_repository:empty",
            "tool": "search_repository",
            "output_preview": "[]",
        }]

        published, suggestions, decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([], published)
        self.assertEqual([candidate], suggestions)
        self.assertIn("no repository-backed tool evidence", decisions[0]["reasons"])

    def test_truncated_preview_does_not_discard_structured_repository_facts(self):
        output = [
            {"path": "app.py", "line": index + 1, "content": "value " + "x" * 200}
            for index in range(20)
        ]
        evidence = collect_evidence([{
            "tool": "search_repository", "ok": True,
            "result": {
                "evidence_id": "search_repository:large",
                "tool": "search_repository", "output": output,
            },
        }])
        candidate = finding()
        candidate.evidence_refs = [evidence["search_repository:large"]]

        published, suggestions, _decisions = partition_publication(
            [], [candidate], [candidate],
            [{"finding_index": 0, "publication_ready": True}],
            repository_available=True,
        )

        self.assertEqual([candidate], published)
        self.assertEqual([], suggestions)
        self.assertEqual(output, candidate.evidence_refs[0]["output"])

    def test_raw_high_risk_call_chain_does_not_replace_tool_evidence(self):
        candidate = finding(severity=Severity.HIGH)
        candidate.call_chain = [{"from": "input", "to": "dangerous"}]

        result = FindingGate().apply(
            [candidate], parse_unified_diff(DIFF)
        )

        self.assertEqual([], result.accepted)
        self.assertIn(
            "evidence gate: high-risk finding requires verified scanner or tool evidence",
            result.rejected[0]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
