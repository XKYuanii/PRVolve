import json


class FakeAgenticClient:
    provider = "fake"
    model = "fake-model"

    def complete_json(self, role, _system, user, ledger=None, max_tokens=None):
        if ledger:
            ledger.record_model(
                role, self.provider, self.model,
                {"prompt_tokens": 10, "completion_tokens": 5}, 1,
            )
        managed = json.loads(user)
        task = json.loads(managed["task"])
        if role == "lead":
            if task["phase"] == "delegate":
                skills = [
                    item["name"] for item in task.get("available_agent_skills") or []
                ]
                return {
                    "action": "final",
                    "delegations": [
                        {
                            "assignment_id": "security-1", "worker": "security",
                            "objective": "Review security",
                            "skills": skills,
                        },
                        {
                            "assignment_id": "reliability-1",
                            "worker": "correctness-reliability",
                            "objective": "Review correctness and reliability",
                            "skills": skills,
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
        if role == "correctness-reliability":
            instructions = "\n".join(
                item.get("instructions", "")
                for item in task.get("active_agent_skills") or []
            )
            rendered_diff = json.dumps(task.get("diff"), ensure_ascii=False)
            if "TODO" in instructions and "TODO" in rendered_diff:
                return {"action": "final", "findings": [{
                    "rule_id": "QUALITY-UNFINISHED", "severity": "low",
                    "title": "Unfinished production behavior",
                    "explanation": "The added TODO marks unfinished production behavior.",
                    "path": "a.py", "line": 2,
                    "evidence": "# TODO finish validation",
                    "fix": "Complete the behavior before merge.",
                    "test": "Add a regression test for the unfinished path.",
                    "confidence": 0.8,
                    "skill": (
                        task.get("active_agent_skills") or [{}]
                    )[0].get("name", ""),
                }]}
            return {
                "action": "final", "findings": [],
                "hypotheses": [{
                    "hypothesis_id": "hyp-1",
                    "claim": "The change may alter correctness behavior.",
                    "status": "unresolved",
                    "explanation": "No repository context is available in this test.",
                    "required_proof": "Inspect callers and related tests.",
                }],
            }
        if role == "security":
            return {
                "action": "final", "findings": [],
                "hypotheses": [{
                    "hypothesis_id": "hyp-1",
                    "claim": "The change may alter a security boundary.",
                    "status": "unresolved",
                    "explanation": "No repository context is available in this test.",
                    "required_proof": "Inspect callers and trust boundaries.",
                }],
            }
        if role == "critic":
            return {
                "action": "final",
                "decisions": [
                    {
                        "finding_index": index,
                        "accepted": True,
                        "introduced_by_diff": True,
                        "reproducible": True,
                        "evidence_sufficient": True,
                        "would_comment_on_real_pr": True,
                        "objections": [],
                        "confidence_adjustment": 0.0,
                    }
                    for index, _item in enumerate(task["candidates"])
                ],
            }
        raise AssertionError(role)


def enable_agentic_service(service):
    service.chat_client = FakeAgenticClient()
    service.reviewer = service._build_mode_router()
    service.harness.reviewer = service.reviewer
    return service
