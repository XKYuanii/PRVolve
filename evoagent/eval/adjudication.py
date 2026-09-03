"""Promote confirmed suggestion judgments into a versioned evaluation dataset."""
from copy import deepcopy
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from ..eval.harness import RULE_TO_CWE, dataset_fingerprint, validate_case
from ..core.finding_policy import normalize_rule_id


SUGGESTION_VERDICTS = frozenset({"required", "optional", "invalid", "duplicate"})


def promote_confirmed_judgments(
    cases: Iterable[dict], payload: dict,
) -> Tuple[List[dict], Dict[str, Any]]:
    """Return a v2 copy whose required judgments are formal expected findings."""
    if str(payload.get("status", "")).strip().lower() != "confirmed":
        raise ValueError("only confirmed suggestion judgments can be promoted")
    by_case = payload.get("cases")
    if not isinstance(by_case, dict):
        raise ValueError("confirmed judgment payload requires an object in cases")

    original = list(cases)
    revised = deepcopy(original)
    known_ids = {str(case["id"]) for case in revised}
    unknown = sorted(set(str(value) for value in by_case).difference(known_ids))
    if unknown:
        raise ValueError(
            "confirmed judgments reference unknown cases: %s" % ", ".join(unknown)
        )

    verdicts = Counter()
    added_required = []
    for case in revised:
        case["schema_version"] = 2
        for expected in case["expected_findings"]:
            expected.setdefault("should_comment", True)
        raw_judgments = by_case.get(str(case["id"]), [])
        if not isinstance(raw_judgments, list):
            raise ValueError("confirmed judgments must be arrays per case")
        judgments = []
        for index, raw in enumerate(raw_judgments):
            if not isinstance(raw, dict):
                raise ValueError("judgment %s[%d] must be an object" % (case["id"], index))
            judgment = deepcopy(raw)
            verdict = str(judgment.get("verdict", "")).strip().lower()
            if verdict not in SUGGESTION_VERDICTS:
                raise ValueError(
                    "judgment %s[%d] has invalid verdict: %s"
                    % (case["id"], index, verdict)
                )
            path = str(judgment.get("path", "")).strip()
            raw_rule_id = str(
                judgment.get("rule_id", judgment.get("cwe", ""))
            ).strip()
            if not raw_rule_id:
                raise ValueError("judgment %s[%d] requires rule_id or cwe" % (case["id"], index))
            rule_id = normalize_rule_id(raw_rule_id)
            try:
                line = int(judgment.get("line", judgment.get("start_line", 0)))
                end_line = int(judgment.get("end_line", line))
            except (TypeError, ValueError) as exc:
                raise ValueError("judgment %s[%d] has invalid lines" % (case["id"], index)) from exc
            if not path or line < 1 or end_line < line:
                raise ValueError("judgment %s[%d] requires a valid path and line range" % (case["id"], index))
            judgment.update({
                "path": path,
                "line": line,
                "end_line": end_line,
                "rule_id": rule_id,
                "verdict": verdict,
            })
            judgments.append(judgment)
            verdicts[verdict] += 1
            if verdict != "required":
                continue
            severity = str(judgment.get("severity", "")).strip().lower()
            if severity not in {"low", "medium", "high", "critical"}:
                raise ValueError(
                    "required judgment %s[%d] requires a valid severity"
                    % (case["id"], index)
                )
            cwe = str(judgment.get("cwe") or RULE_TO_CWE.get(rule_id, rule_id))
            expected = {
                "cwe": cwe,
                "end_line": end_line,
                "label_source": "confirmed-suggestion-adjudication",
                "path": path,
                "rule_id": rule_id,
                "severity": severity,
                "should_comment": True,
                "start_line": line,
            }
            identity = (path.replace("\\", "/"), line, end_line, cwe.upper())
            existing = {
                (
                    str(item["path"]).replace("\\", "/"),
                    int(item["start_line"]), int(item["end_line"]),
                    str(item["cwe"]).upper(),
                )
                for item in case["expected_findings"]
            }
            if identity not in existing:
                case["expected_findings"].append(expected)
                added_required.append({"case_id": case["id"], **expected})
        if judgments:
            case["suggestion_judgments"] = judgments
            case["label_revision"] = {
                "status": "confirmed",
                "source": "suggestion-adjudication",
                "reviewed_at": str(payload.get("reviewed_at", "")),
            }
        validate_case(case)

    summary = {
        "schema_version": 1,
        "status": "confirmed",
        "cases": len(revised),
        "judged_cases": sum(bool(value) for value in by_case.values()),
        "explicit_judgments": sum(verdicts.values()),
        "reviewed_suggestions": int(
            payload.get("reviewed_suggestions", sum(verdicts.values()))
        ),
        "verdicts": dict(sorted(verdicts.items())),
        "required_labels_added": len(added_required),
        "required_label_additions": added_required,
        "source_dataset_sha256": dataset_fingerprint(original),
        "revised_dataset_sha256": dataset_fingerprint(revised),
        "reviewer": str(payload.get("reviewer", "unknown")),
        "reviewed_at": str(payload.get("reviewed_at", "")),
    }
    return revised, summary
