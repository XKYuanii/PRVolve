import json
import os
import tempfile
import unittest

from evoagent.review.agentic import AgenticReviewer
from evoagent.core.diff_parser import parse_unified_diff
from evoagent.agents.memory import MemoryManager
from evoagent.session.checkpoint import CheckpointLog
from evoagent.session.projections import progress
from evoagent.store.sqlite import TaskStore


DIFF = "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+eval(user_input)\n"
HANDOFF_DIFF = (
    "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n"
    "-result = guarded_transition(record)\n+result = transition(record)\n"
)


class HierarchicalClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.calls = []
        self.security_calls = 0

    def complete_json(self, role, _system, user, ledger=None, max_tokens=None):
        managed = json.loads(user)
        task = json.loads(managed["task"])
        self.calls.append((role, task.get("phase", "worker")))
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 10, "completion_tokens": 5}, 1,
            )
        if role == "lead":
            if task["phase"] == "delegate":
                return {
                    "action": "final", "delegations": [
                        {
                            "assignment_id": "security-1", "worker": "security",
                            "objective": "Trace the changed input into dangerous calls.",
                        },
                        {
                            "assignment_id": "reliability-1",
                            "worker": "correctness-reliability",
                            "objective": "Review failure and resource behavior.",
                        },
                    ],
                }
            if task["phase"] == "assess-workers" and task["revision_round"] == 0:
                return {
                    "action": "final", "revision_requests": [{
                        "assignment_id": "security-1", "worker": "security",
                        "guidance": "Add an exact changed-line finding for dynamic execution.",
                        "required_evidence": ["changed-line evidence"],
                    }], "critic_objective": "Challenge every proposed finding.",
                }
            if task["phase"] == "assess-workers":
                return {
                    "action": "final", "revision_requests": [],
                    "critic_objective": "Challenge every proposed finding.",
                }
            if task["phase"] == "finalize":
                return {
                    "action": "final",
                    "accepted_finding_indices": list(
                        range(len(task["candidate_findings"]))
                    ),
                    "confidence_adjustments": [],
                    "resolution_summary": "Workers supplied evidence and Critic approved.",
                }
        if role == "security":
            self.security_calls += 1
            if self.security_calls == 1:
                return {
                    "action": "final", "findings": [],
                    "hypotheses": [{
                        "hypothesis_id": "hyp-1",
                        "claim": "Dynamic execution may cross a trust boundary.",
                        "status": "unresolved",
                        "explanation": "The initial pass needs an exact anchor.",
                        "required_proof": "Inspect the changed execution line.",
                    }],
                }
            return {"action": "final", "findings": [{
                "rule_id": "SEC-LEAD-REVISION", "severity": "medium",
                "title": "Dynamic execution", "explanation": "Input is executed as code.",
                "path": "app.py", "line": 1, "evidence": "eval(user_input)",
                "fix": "Use a constrained parser.",
                "test": "Prove expressions are treated as data.", "confidence": 0.9,
            }], "requirement_resolutions": [{
                "requirement_id": "req-1", "status": "finding",
                "explanation": "The requested changed-line defect is reported.",
            }]}
        if role == "correctness-reliability":
            return {
                "action": "final", "findings": [],
                "hypotheses": [{
                    "hypothesis_id": "hyp-1",
                    "claim": "Dynamic execution may also raise reliability failures.",
                    "status": "unresolved",
                    "explanation": "No repository context is available in this test.",
                    "required_proof": "Inspect exception handling around the call.",
                }],
            }
        if role == "critic":
            return {
                "action": "final", "decisions": [
                    {
                        "finding_index": index, "accepted": True,
                        "introduced_by_diff": True,
                        "reproducible": True,
                        "evidence_sufficient": True,
                        "would_comment_on_real_pr": True,
                        "objections": [], "confidence_adjustment": 0.0,
                    }
                    for index, _item in enumerate(task["candidates"])
                ],
            }
        raise AssertionError((role, task))


class CrossDomainHandoffClient:
    provider = "fake"
    model = "fake-model"

    def __init__(self):
        self.correctness_calls = 0

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
                return {"action": "final", "delegations": [{
                    "assignment_id": "security-1", "worker": "security",
                    "objective": "Review trust boundaries.", "files": ["app.py"],
                }, {
                    "assignment_id": "reliability-1",
                    "worker": "correctness-reliability",
                    "objective": "Review state transitions.", "files": ["app.py"],
                }]}
            if task["phase"] == "assess-workers":
                return {
                    "action": "final", "revision_requests": [],
                    "handoff_decisions": [{
                        "handoff_id": "security-1:state-transition", "action": "revise",
                        "reason": "Trace the state transition in the owning Worker.",
                    }], "critic_objective": "Verify candidates.",
                }
            if task["phase"] == "finalize":
                return {
                    "action": "final",
                    "accepted_finding_indices": list(
                        range(len(task["candidate_findings"]))
                    ),
                    "confidence_adjustments": [],
                }
        if role == "security":
            return {
                "action": "final", "findings": [],
                "hypotheses": [{
                    "hypothesis_id": "state-transition",
                    "claim": "Removing the guarded transition may admit an invalid state.",
                    "location": "app.py:1", "domain": "correctness-reliability",
                    "risk_level": "high", "status": "handoff",
                    "target_worker": "correctness-reliability",
                    "explanation": "The risk is credible but belongs to state semantics.",
                    "required_proof": "Compare transition preconditions with all callers.",
                }],
            }
        if role == "correctness-reliability":
            self.correctness_calls += 1
            if self.correctness_calls == 1:
                return {
                    "action": "final", "findings": [],
                    "hypotheses": [{
                        "hypothesis_id": "initial-state-review",
                        "claim": "The transition precondition may have changed.",
                        "status": "unresolved", "risk_level": "normal",
                        "explanation": "The initial assignment did not identify the removed guard.",
                        "required_proof": "Inspect the exact guard-removal hypothesis.",
                    }],
                }
            return {
                "action": "final",
                "findings": [{
                    "rule_id": "COR-INVALID-STATE", "severity": "medium",
                    "title": "Invalid state reaches transition",
                    "explanation": "The replacement bypasses the guarded state transition.",
                    "path": "app.py", "line": 1,
                    "evidence": "result = transition(record)",
                    "fix": "Restore the state precondition.",
                    "test": "Exercise a record rejected by guarded_transition.",
                    "confidence": 0.9,
                }],
                "requirement_resolutions": [{
                    "requirement_id": "req-1", "status": "finding",
                    "explanation": "The transferred guard-removal risk is reported.",
                }],
            }
        if role == "critic":
            return {
                "action": "final", "decisions": [{
                    "finding_index": index, "accepted": True,
                    "introduced_by_diff": True, "reproducible": True,
                    "evidence_sufficient": True, "would_comment_on_real_pr": True,
                    "objections": [], "confidence_adjustment": 0.0,
                } for index, _item in enumerate(task["candidates"])]
            }
        raise AssertionError((role, task))


class LeadWorkerCollaborationTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)
        self.store.create("task", "org/repo", 1, {
            "mode": "agentic",
            "enabled_agents": [
                "lead", "security", "correctness-reliability", "critic",
            ],
        })

    def tearDown(self):
        os.unlink(self.path)

    def test_lead_delegates_requests_revision_and_synthesizes(self):
        client = HierarchicalClient()
        reviewer = AgenticReviewer(self.store, client)

        findings = reviewer.review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )
        summary = reviewer.collaboration_summary("task")

        self.assertNotIn("SEC-LEAD-REVISION", {item.rule_id for item in findings})
        self.assertIn(
            "SEC-LEAD-REVISION",
            {item["rule_id"] for item in summary["suggested_findings"]},
        )
        self.assertIn("SEC-EVAL", {item.rule_id for item in findings})
        self.assertEqual("lead-workers", summary["collaboration"]["protocol"])
        self.assertEqual(2, client.security_calls)
        self.assertEqual(1, len(summary["collaboration"]["revision_results"]))
        self.assertEqual("lead-final", summary["collaboration"]["stop_reason"])
        checkpoints = {
            item["event"]
            for item in summary["execution"]["agent_traces"]["lead-session"]
        }
        self.assertTrue({
            "assignment_created", "worker_reported", "revision_completed",
            "lead_activated", "lead_completed",
        }.issubset(checkpoints))
        nodes = progress(CheckpointLog.load(self.store, "task"))
        self.assertEqual("completed", nodes["planning.scan"]["status"])
        self.assertEqual("completed", nodes["planning.delegate"]["status"])
        self.assertEqual("completed", nodes["executing.work:security-1"]["status"])
        self.assertEqual("completed", nodes["reviewing.arbitrate"]["status"])

    def test_cross_domain_handoff_uses_the_existing_lead_revision_path(self):
        client = CrossDomainHandoffClient()
        reviewer = AgenticReviewer(self.store, client)

        reviewer.review_with_context(
            "task", HANDOFF_DIFF, parse_unified_diff(HANDOFF_DIFF), "org/repo"
        )
        collaboration = reviewer.collaboration_summary("task")["collaboration"]

        security = next(
            item for item in collaboration["worker_results"]
            if item["worker"] == "security"
        )
        self.assertEqual(
            "security-1:state-transition", security["handoffs"][0]["handoff_id"]
        )
        first_assessment = collaboration["lead"]["assessments"][0]
        routed = first_assessment["handoff_decisions"][0]
        self.assertEqual("revise", routed["action"])
        self.assertEqual("reliability-1", routed["target_assignment_id"])
        self.assertEqual("lead", routed["source"])
        self.assertEqual(2, client.correctness_calls)
        self.assertEqual(1, len(collaboration["revision_results"]))
        self.assertEqual(3, len(collaboration["worker_history"]))
        self.assertTrue(any(
            item.get("handoffs") for item in collaboration["worker_history"]
        ))
        revision = collaboration["revision_results"][0]
        self.assertEqual(
            ["security-1:state-transition"], revision["handled_handoff_ids"]
        )
        self.assertEqual(
            ["COR-INVALID-STATE"],
            [item["rule_id"] for item in revision["findings"]],
        )

    def test_concrete_normal_unresolved_gets_one_targeted_evidence_revision(self):
        delegations = [{
            "assignment_id": "reliability-1",
            "worker": "correctness-reliability",
            "files": ["app.py"],
        }]
        worker_results = {"reliability-1": {
            "assignment_id": "reliability-1",
            "worker": "correctness-reliability",
            "status": "completed", "findings": [],
            "repository_context_available": True,
            "handled_handoff_ids": [], "handoffs": [],
            "hypotheses": [{
                "hypothesis_id": "empty-input", "status": "unresolved",
                "risk_level": "normal", "location": "app.py:7",
                "claim": "An empty value may reach the new index.",
                "explanation": "The input contract is not established.",
                "required_proof": "Find a caller that permits the empty value.",
            }],
        }}

        decision = AgenticReviewer._complete_assessment_protocol(
            {"revision_requests": [], "handoff_decisions": [{
                "handoff_id": "reliability-1:empty-input", "action": "revise",
                "reason": "Check the caller contract.",
            }]},
            delegations, worker_results, remaining_rounds=1,
        )

        self.assertEqual(1, len(decision["revision_requests"]))
        request = decision["revision_requests"][0]
        self.assertEqual(
            ["reliability-1:empty-input"], request["handoff_ids"]
        )
        self.assertEqual("app.py", request["evidence_targets"][0]["path"])
        self.assertEqual(7, request["evidence_targets"][0]["line"])
        self.assertIn("three-part proof", request["guidance"])

    def test_worker_finding_without_claim_specific_proof_gets_evidence_revision(self):
        delegations = [{
            "assignment_id": "reliability-1",
            "worker": "correctness-reliability",
            "files": ["app.py"],
        }]
        worker_results = {"reliability-1": {
            "assignment_id": "reliability-1",
            "worker": "correctness-reliability",
            "status": "completed", "handled_handoff_ids": [],
            "repository_context_available": True,
            "handoffs": [], "hypotheses": [],
            "findings": [{
                "rule_id": "CWE-248", "severity": "medium",
                "title": "Missing key raises", "explanation": "A key may be absent.",
                "path": "app.py", "line": 7, "evidence": "value['key']",
                "fix": "Use get.", "test": "Pass a missing key.",
                "confidence": 0.9, "source": "correctness-reliability",
                "evidence_refs": [], "call_chain": [],
            }],
        }}

        decision = AgenticReviewer._complete_assessment_protocol(
            {"revision_requests": [{
                "assignment_id": "reliability-1", "worker": "correctness-reliability",
                "guidance": "Check the missing-key contract.",
            }], "handoff_decisions": []},
            delegations, worker_results, remaining_rounds=1,
        )

        target = decision["revision_requests"][0]["evidence_targets"][0]
        self.assertEqual("evidence-gap-finding", target["kind"])
        self.assertIn("repository trigger", target["required_proof"])

    def test_evidence_revision_preserves_candidate_until_explicit_refutation(self):
        finding = {
            "rule_id": "CWE-248", "severity": "medium",
            "title": "Missing key raises", "explanation": "A key may be absent.",
            "path": "app.py", "line": 7, "evidence": "value['key']",
            "fix": "Use get.", "test": "Pass a missing key.",
            "confidence": 0.9, "source": "correctness-reliability",
            "evidence_refs": [], "call_chain": [],
        }
        previous = {"findings": [finding], "evidence_inventory": []}
        unresolved = {
            "findings": [], "hypotheses": [{
                "status": "unresolved", "location": "app.py:7",
            }], "evidence_inventory": [],
        }

        retained = AgenticReviewer._merge_worker_revision(previous, unresolved)
        self.assertEqual(["CWE-248"], [
            item["rule_id"] for item in retained["findings"]
        ])

        refuted = {
            "findings": [], "hypotheses": [{
                "status": "refuted", "location": "app.py:7",
                "proof_kind": "type_constraint",
            }], "evidence_inventory": [],
        }
        removed = AgenticReviewer._merge_worker_revision(previous, refuted)
        self.assertEqual([], removed["findings"])

    def test_evidence_queue_deduplicates_location_across_workers(self):
        worker_results = {
            "security-1": {
                "assignment_id": "security-1", "worker": "security",
                "status": "completed", "repository_context_available": True,
                "findings": [], "handoffs": [], "handled_handoff_ids": [],
                "hypotheses": [{
                    "hypothesis_id": "same-risk", "status": "unresolved",
                    "risk_level": "high", "location": "app.py:7",
                    "domain": "correctness-reliability",
                    "claim": "The new index may raise.",
                    "required_proof": "Trace callers.",
                }],
            },
            "correctness-1": {
                "assignment_id": "correctness-1",
                "worker": "correctness-reliability", "status": "completed",
                "repository_context_available": True, "findings": [],
                "handoffs": [], "handled_handoff_ids": [],
                "hypotheses": [{
                    "hypothesis_id": "same-risk", "status": "unresolved",
                    "risk_level": "high", "location": "app.py:7",
                    "domain": "correctness-reliability",
                    "claim": "The new index may raise.",
                    "required_proof": "Trace callers.",
                }],
            },
        }

        pending = AgenticReviewer._pending_worker_review_items(worker_results)

        self.assertEqual(1, len(pending))
        self.assertEqual("correctness-reliability", pending[0]["target_worker"])

    def test_completed_session_resumes_without_repeating_agent_calls(self):
        first = HierarchicalClient()
        AgenticReviewer(self.store, first).review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )
        resumed_client = HierarchicalClient()
        resumed = AgenticReviewer(self.store, resumed_client)

        findings = resumed.review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )

        summary = resumed.collaboration_summary("task")
        self.assertNotIn("SEC-LEAD-REVISION", {item.rule_id for item in findings})
        self.assertIn(
            "SEC-LEAD-REVISION",
            {item["rule_id"] for item in summary["suggested_findings"]},
        )
        self.assertEqual([], resumed_client.calls)
        self.assertGreater(
            resumed.collaboration_summary("task")["execution"]["llm_calls"], 0
        )

    def test_worker_killed_mid_run_resumes_without_repeating_finished_workers(self):
        class KillingClient(HierarchicalClient):
            def complete_json(self, role, system, user, ledger=None, max_tokens=None):
                task = json.loads(json.loads(user)["task"])
                assignment = task.get("lead_assignment")
                if isinstance(assignment, dict) and assignment["worker"] == (
                    "correctness-reliability"
                ):
                    # BaseException models the process dying rather than the
                    # worker reporting a failure, which is caught by design.
                    raise KeyboardInterrupt("worker process killed")
                return super().complete_json(role, system, user, ledger, max_tokens)

        killed = KillingClient()
        with self.assertRaises(KeyboardInterrupt):
            AgenticReviewer(self.store, killed).review_with_context(
                "task", DIFF, parse_unified_diff(DIFF), "org/repo"
            )
        nodes = progress(CheckpointLog.load(self.store, "task"))
        self.assertEqual("completed", nodes["executing.work:security-1"]["status"])
        self.assertEqual("running", nodes["executing.work:reliability-1"]["status"])

        resumed_client = HierarchicalClient()
        AgenticReviewer(self.store, resumed_client).review_with_context(
            "task", DIFF, parse_unified_diff(DIFF), "org/repo"
        )

        # The scan, the delegation and the finished security worker are read
        # back from the log; only the worker that died is started again. The
        # second security call is the Lead's revision request, a new stage.
        nodes = progress(CheckpointLog.load(self.store, "task"))
        self.assertNotIn(("lead", "delegate"), resumed_client.calls)
        self.assertEqual(1, nodes["planning.scan"]["attempt"])
        self.assertEqual(1, nodes["planning.delegate"]["attempt"])
        self.assertEqual(1, nodes["executing.work:security-1"]["attempt"])
        self.assertEqual(2, nodes["executing.work:reliability-1"]["attempt"])

    def test_gate_decisions_are_archived_for_future_agent_recall(self):
        memory = MemoryManager(self.store)
        reviewer = AgenticReviewer(self.store, HierarchicalClient(), memory_manager=memory)

        reviewer.review_with_context("task", DIFF, parse_unified_diff(DIFF), "org/repo")

        episodes = memory.recall("default", "org/repo", "SEC-EVAL")
        self.assertTrue(any(item["kind"] == "finding_approved" for item in episodes))
        self.assertTrue(any(item["kind"] == "task_summary" for item in episodes))
        unverified = memory.recall("default", "org/repo", "SEC-LEAD-REVISION")
        self.assertFalse(any(item["kind"] == "finding_approved" for item in unverified))


if __name__ == "__main__":
    unittest.main()
