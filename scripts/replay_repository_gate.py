"""Replay repository evidence and publication gates without new model calls."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
import re
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.agents.parsing import normalize_model_rule_id  # noqa: E402
from evoagent.review.merge import partition_publication  # noqa: E402
from evoagent.core.diff_parser import parse_unified_diff  # noqa: E402
from evoagent.eval.harness import load_jsonl  # noqa: E402
from evoagent.eval.agentic import ProductionEvaluationHarness  # noqa: E402
from evoagent.core.finding_policy import (  # noqa: E402
    REPOSITORY_EVIDENCE_TOOLS,
    is_deterministic_finding,
    repository_evidence_refs,
)
from evoagent.core.gates import FindingGate  # noqa: E402
from evoagent.tools.repository import RepositoryToolSuite  # noqa: E402
from evoagent.session.ledger import ExecutionLedger  # noqa: E402


def _read(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("report")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--case-id", action="append", default=[],
        help="Replay only the selected report case; repeat for multiple cases.",
    )
    parser.add_argument(
        "--repository-root", default="",
        help="Repository checkout override for a single selected case.",
    )
    parser.add_argument(
        "--policy-only", action="store_true",
        help="Reapply publication policy using recorded evidence without replaying tools.",
    )
    parser.add_argument(
        "--suppress-suggestions", action="store_true",
        help="Match the stable evaluation profile by omitting advisory suggestions.",
    )
    args = parser.parse_args()

    cases = {str(case["id"]): case for case in load_jsonl(args.dataset)}
    report_raw = _read(args.report)
    report = json.loads(report_raw.decode("utf-8"))
    harness = ProductionEvaluationHarness()
    rescored = []
    totals = harness._empty_totals()
    split_totals = {}
    promoted_count = 0
    demoted_count = 0
    replayed_tools = 0
    replay_attempts = []

    selected_ids = set(args.case_id)
    if args.repository_root and len(selected_ids) != 1:
        parser.error("--repository-root requires exactly one --case-id")
    report_results = report.get("case_results") or []
    if selected_ids:
        known_report_ids = {str(item.get("id", "")) for item in report_results}
        unknown = selected_ids.difference(known_report_ids)
        if unknown:
            parser.error(
                "report is missing selected cases: %s" % ", ".join(sorted(unknown))
            )
        report_results = [
            item for item in report_results
            if str(item.get("id", "")) in selected_ids
        ]

    for original in report_results:
        result = deepcopy(original)
        case = cases.get(str(result.get("id", "")))
        if case is None:
            parser.error("dataset is missing report case %s" % result.get("id"))
        parsed = parse_unified_diff(case["diff"])
        evidence_by_id = {}
        summary = result.get("agentic_summary") or {}
        if not args.policy_only:
            root = str(args.repository_root or case.get("repository_root") or "")
            if not root or not os.path.isdir(root):
                parser.error("case %s has no repository checkout" % case["id"])
            ledger = ExecutionLedger("gate-replay")
            suite = RepositoryToolSuite(root, case["diff"], parsed, ledger)
            registry = suite.registry(
                "gate-replay", set(REPOSITORY_EVIDENCE_TOOLS)
            )
            for call in summary.get("tool_call_log") or []:
                tool = str(call.get("tool", ""))
                if not call.get("ok") or tool not in registry.names():
                    continue
                try:
                    value = registry.invoke(tool, dict(call.get("arguments") or {}))
                except Exception:
                    continue
                if isinstance(value, dict) and value.get("evidence_id"):
                    evidence_by_id[str(value["evidence_id"])] = value
                    old_id = re.search(
                        r"['\"]evidence_id['\"]\s*:\s*['\"]([^'\"]+)",
                        str(call.get("result_preview", "")),
                    )
                    if old_id:
                        evidence_by_id[old_id.group(1)] = value
                    replayed_tools += 1

        formal = harness._restore_findings(result.get("predicted_findings") or [])
        suggestions = harness._restore_findings(result.get("suggested_findings") or [])
        for finding in formal + suggestions:
            normalized = normalize_model_rule_id(finding.to_dict())
            if normalized != finding.rule_id:
                if not finding.original_rule_id:
                    finding.original_rule_id = finding.rule_id
                finding.rule_id = normalized
            for ref in finding.evidence_refs:
                replayed = evidence_by_id.get(str(ref.get("evidence_id", "")))
                if replayed:
                    ref["tool"] = replayed.get("tool", ref.get("tool", ""))
                    ref["output"] = replayed.get("output")

        deterministic = [
            item for item in formal if is_deterministic_finding(item)
        ]
        formal_keys_before_revalidation = {
            (item.rule_id, item.source, item.path, item.line) for item in formal
        }
        revalidated, _demoted, _formal_decisions = (
            partition_publication(
                deterministic, formal, formal,
                [
                    {"finding_index": index, "publication_ready": True}
                    for index in range(len(formal))
                ],
                repository_available=True,
                publish_unverified_suggestions=False,
            )
        )
        formal = revalidated
        revalidated_keys = {
            (item.rule_id, item.source, item.path, item.line) for item in formal
        }
        policy_demoted_keys = formal_keys_before_revalidation - revalidated_keys
        demoted_count += len(policy_demoted_keys)
        formal_keys = {
            (item.rule_id, item.source, item.path, item.line) for item in formal
        }
        remaining = []
        decisions = summary.get("publication_decisions") or []
        critic = {
            int(item["finding_index"]): item
            for item in summary.get("critic_decisions") or []
            if str(item.get("finding_index", "")).isdigit()
        }
        accepted_indices = {
            int(value) for value in (summary.get("lead_final") or {}).get(
                "accepted_finding_indices", []
            )
        }
        replay_candidates = list(suggestions)
        for decision in decisions:
            try:
                decision_index = int(decision.get("finding_index", -1))
            except (TypeError, ValueError):
                continue
            if decision_index not in accepted_indices:
                continue
            matching = []
            for worker in summary.get("worker_results") or []:
                for value in worker.get("findings") or []:
                    if (
                        str(value.get("rule_id")) == str(decision.get("rule_id"))
                        and str(value.get("source")) == str(decision.get("source"))
                    ):
                        matching.append(value)
            if len(matching) != 1:
                continue
            recovered = harness._restore_findings(matching)[0]
            normalized = normalize_model_rule_id(recovered.to_dict())
            if normalized != recovered.rule_id:
                if not recovered.original_rule_id:
                    recovered.original_rule_id = recovered.rule_id
                recovered.rule_id = normalized
            key = (recovered.rule_id, recovered.source, recovered.path, recovered.line)
            if key in policy_demoted_keys:
                continue
            if key not in formal_keys and all(
                key != (item.rule_id, item.source, item.path, item.line)
                for item in replay_candidates
            ):
                replay_candidates.append(recovered)

        for finding in replay_candidates:
            decision = next((
                item for item in decisions
                if str(item.get("rule_id")) == finding.rule_id
                and str(item.get("source")) == finding.source
            ), None)
            index = int(decision.get("finding_index", -1)) if decision else -1
            objections = set(decision.get("reasons") or []) if decision else set()
            critic_ready = bool((critic.get(index) or {}).get("publication_ready"))
            evidence_only_block = bool(decision) and objections == {
                "no repository-backed tool evidence"
            }
            severity_only_block = bool(decision) and (
                "high-risk claim lacks behavioral or cross-call evidence" in objections
                and objections.issubset({
                    "high-risk claim lacks behavioral or cross-call evidence",
                    "stability profile suppresses unverified suggestions",
                })
            )
            confirmed_before_global_gate = bool(decision) and (
                decision.get("disposition") == "confirmed"
            )
            attempt = {
                "case_id": case["id"], "rule_id": finding.rule_id,
                "finding_index": index,
                "lead_selected": index in accepted_indices,
                "critic_ready": critic_ready,
                "evidence_only_block": evidence_only_block,
                "severity_only_block": severity_only_block,
                "confirmed_before_global_gate": confirmed_before_global_gate,
                "repository_evidence_count": len(repository_evidence_refs(finding)),
                "promoted": False,
            }
            if (
                index in accepted_indices
                and critic_ready
                and (
                    evidence_only_block
                    or severity_only_block
                    or confirmed_before_global_gate
                )
            ):
                published, _suggestions, _decisions = (
                    partition_publication(
                        [], [finding], [finding],
                        [{"finding_index": 0, "publication_ready": True}],
                        repository_available=True,
                    )
                )
                gated = FindingGate().apply(published, parsed)
                if gated.accepted:
                    promoted = gated.accepted[0]
                    promoted.disposition = "confirmed"
                    formal.append(promoted)
                    promoted_count += 1
                    attempt["promoted"] = True
                    replay_attempts.append(attempt)
                    continue
                attempt["gate_rejections"] = list(gated.rejected)
            replay_attempts.append(attempt)
            if not args.suppress_suggestions:
                remaining.append(finding)

        result["predicted_findings"] = [item.to_dict() for item in formal]
        result["suggested_findings"] = [item.to_dict() for item in remaining]
        summary["suggested_findings"] = [item.to_dict() for item in remaining]
        summary["suggestion_count"] = len(remaining)
        summary["accepted_findings"] = len(formal)
        result["agentic_summary"] = summary
        harness.rescore_cached_result(result, case)
        rescored.append(result)
        harness._accumulate(totals, result)
        bucket = split_totals.setdefault(case["split"], harness._empty_totals())
        harness._accumulate(bucket, result)

    report["case_results"] = rescored
    report["metrics"] = harness._metrics(totals)
    report["by_split"] = {
        split: harness._metrics(values)
        for split, values in sorted(split_totals.items())
    }
    report["status"] = "complete"
    report["gate_replay"] = {
        "model_calls": 0,
        "promoted_findings": promoted_count,
        "demoted_findings": demoted_count,
        "replayed_repository_tools": replayed_tools,
        "attempts": replay_attempts,
        "source_report": os.path.normpath(args.report).replace("\\", "/"),
        "source_report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "policy_only": bool(args.policy_only),
        "suggestions_suppressed": bool(args.suppress_suggestions),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps({
        "gate_replay": report["gate_replay"], "metrics": report["metrics"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
