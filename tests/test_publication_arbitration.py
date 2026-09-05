"""Publication regressions: optional arbitration, independent scanner, real provenance."""
import copy
import unittest
from dataclasses import replace

from evoagent.agents.prompts import ROLE_PERMISSIONS
from evoagent.core.diff_parser import parse_unified_diff
from evoagent.core.gates import FindingGate
from evoagent.core.models import Finding, Severity
from evoagent.eval.agentic import ProductionEvaluationHarness
from evoagent.review.merge import (
    candidates_from, partition_publication, resolve_lead_reviews,
)


class PublicationArbitrationTests(unittest.TestCase):
    def setUp(self):
        self.finding = Finding(
            rule_id="CWE-248", severity=Severity.HIGH, source="security",
            title="Missing key raises", explanation="An allowed empty mapping raises KeyError.",
            path="app.py", line=2, evidence="return value['key']",
            fix="Restore the optional lookup.", test="Pass an empty mapping.", confidence=0.8,
        )
        self.change = {
            "evidence_id": "changed_line:edit", "tool": "changed_line",
            "output": {"found": True, "path": "app.py", "line": 2,
                       "change": {"complete": True, "before": "return value.get('key')",
                                  "after": "return value['key']"}},
        }
        self.fact = {
            "evidence_id": "read_file:contract", "tool": "read_file",
            "output": {"path": "test_app.py", "start_line": 1,
                       "content": "def test_optional_key(): assert lookup({}) is None"},
        }
        self.result = {
            "accepted_finding_indices": [0],
            "_observations": [{"ok": True, "tool": ref["tool"], "result": ref}
                              for ref in (self.change, self.fact)],
            "evidence_reviews": [{
                "finding_index": 0,
                "supporting_evidence_ids": ["changed_line:edit", "read_file:contract"],
                "causal_delta": {
                    "trigger": "lookup({})", "before": "Returns None", "after": "Raises KeyError",
                    "failure": "Optional key lookup crashes", "contract": "Empty mappings are supported",
                    "code_before": "return value.get('key')", "code_after": "return value['key']",
                    "premises": [{"premise": "Empty mapping is allowed", "status": "verified",
                                  "evidence": "The existing optional-key test calls lookup({})",
                                  "supporting_evidence_ids": ["read_file:contract"]}],
                },
            }],
        }
        self.critic = [{"finding_index": 0, "verdict": "inconclusive", "publication_ready": False}]

    def test_grounded_lead_proof_completes_inconclusive_review_and_final_gate(self):
        reviews = resolve_lead_reviews(self.result, [self.finding], self.critic)
        self.assertEqual(1, len(reviews))
        published, _, decisions = partition_publication(
            [], [self.finding], [self.finding], self.critic, True,
            publish_unverified_suggestions=False, lead_reviews=reviews,
        )
        self.assertEqual([self.finding], published)
        self.assertEqual("lead", decisions[0]["proof_source"])
        gated = FindingGate().apply(published, parse_unified_diff(
            "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def lookup(value):\n"
            "-return value.get('key')\n+return value['key']\n"
        ))
        self.assertEqual(1, len(gated.accepted))
        self.assertEqual("lead", gated.accepted[0].gate["proof_source"])
        self.assertTrue(gated.accepted[0].gate["causal_proof_verified"])

    def test_proof_may_reuse_existing_candidate_facts_without_new_tools(self):
        self.finding.evidence_refs = [self.change, self.fact]
        self.result.pop("_observations")
        self.assertEqual(1, len(resolve_lead_reviews(self.result, [self.finding], self.critic)))

    def test_lead_selection_alone_cannot_publish(self):
        self.result.pop("evidence_reviews")
        reviews = resolve_lead_reviews(self.result, [self.finding], self.critic)
        self.assertEqual([], reviews)
        published, _, _ = partition_publication(
            [], [self.finding], [self.finding], self.critic, True,
            publish_unverified_suggestions=False, lead_reviews=reviews,
        )
        self.assertEqual([], published)

    def test_unverified_or_unbound_premise_wrong_edit_and_failed_tool_cannot_publish(self):
        for mutation in ("unverified", "missing_id", "invented_id", "wrong_edit", "failed_tool", "wrong_type"):
            with self.subTest(mutation=mutation):
                result = copy.deepcopy(self.result)
                proof = result["evidence_reviews"][0]["causal_delta"]
                if mutation == "unverified":
                    proof["premises"][0]["status"] = "unknown"
                elif mutation == "missing_id":
                    proof["premises"][0].pop("supporting_evidence_ids")
                elif mutation == "invented_id":
                    proof["premises"][0]["supporting_evidence_ids"] = ["read_file:invented"]
                elif mutation == "wrong_edit":
                    proof["code_before"] = proof["code_after"]
                elif mutation == "failed_tool":
                    result["_observations"][1]["ok"] = False
                else:
                    result["evidence_reviews"][0]["causal_delta"] = "not a proof"
                self.assertEqual([], resolve_lead_reviews(result, [self.finding], self.critic))

    def test_grounded_counterproof_cannot_be_overridden(self):
        self.critic[0].update(verdict="rejected", rejection_ready=True)
        self.assertEqual([], resolve_lead_reviews(self.result, [self.finding], self.critic))

    def test_scanner_claim_is_not_erased_by_overlapping_inconclusive_worker(self):
        scanner = replace(self.finding, source="local-rule-scanner", evidence_refs=[{
            "evidence_id": "local:guard", "tool": "local-rule-scanner",
        }])
        candidates = candidates_from([scanner], {"worker": {"findings": [self.finding.to_dict()]}})
        self.assertEqual(2, len(candidates))
        published, _, decisions = partition_publication(
            [scanner], candidates, [],
            [{"finding_index": index, "publication_ready": False} for index in (0, 1)], True,
            publish_unverified_suggestions=False,
        )
        self.assertEqual([scanner], published)
        self.assertEqual("rejected", decisions[1]["disposition"])
        self.assertEqual([], self.finding.evidence_refs)  # no borrowed scanner authority

    def test_scanner_own_counterproof_blocks_its_baseline(self):
        scanner = replace(self.finding, source="local-rule-scanner")
        published, _, _ = partition_publication(
            [scanner], [scanner], [], [{"finding_index": 0, "rejection_ready": True}], True,
        )
        self.assertEqual([], published)

    def test_confirmed_worker_and_scanner_deduplicate_after_review(self):
        scanner = replace(self.finding, source="local-rule-scanner")
        self.finding.evidence_refs = [self.fact]
        candidates = candidates_from([scanner], {"worker": {"findings": [self.finding.to_dict()]}})
        proof = {"finding_index": 1, "publication_ready": True, **dict.fromkeys((
            "introduced_by_diff", "reproducible", "evidence_sufficient",
            "would_comment_on_real_pr", "differential_causality", "premises_verified",
        ), True)}
        published, _, _ = partition_publication([scanner], candidates, [candidates[1]], [proof], True)
        self.assertEqual(1, len(published))
        self.assertEqual("security", published[0].source)

    def test_unpublished_scanner_is_not_counted_as_publication_rescue(self):
        result = {"predicted_findings": []}
        scanner = replace(self.finding, source="local-rule-scanner")
        ProductionEvaluationHarness()._score_capability_lanes(result, [{
            "path": "app.py", "start_line": 2, "end_line": 2, "cwe": "CWE-248",
        }], {"scanner_finding_details": [scanner.to_dict()]})
        self.assertEqual(1, result["scanner_strict_tp"])
        self.assertEqual(0, result["scanner_publication_rescue_strict_tp"])

    def test_lead_can_read_evidence_for_final_arbitration(self):
        self.assertTrue({"read_file", "changed_line", "search_repository", "symbol"}
                        .issubset(ROLE_PERMISSIONS["lead"]))
