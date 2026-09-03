"""Re-score a cached Agentic report against revised labels with zero model calls."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.eval.harness import dataset_fingerprint, load_jsonl  # noqa: E402
from evoagent.eval.agentic import (  # noqa: E402
    ProductionEvaluationHarness,
    validate_real_dataset,
)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("seed_report")
    parser.add_argument("--judgments", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    started = time.monotonic()

    cases = load_jsonl(args.dataset)
    case_by_id = {str(case["id"]): case for case in cases}
    seed_raw = _read_bytes(args.seed_report)
    seed = json.loads(seed_raw.decode("utf-8"))
    seed_results = list(seed.get("case_results") or [])
    if not seed_results:
        parser.error("seed report has no case_results")
    result_ids = [str(item.get("id", "")) for item in seed_results]
    if len(result_ids) != len(set(result_ids)):
        parser.error("seed report contains duplicate case ids")
    missing = sorted(set(result_ids).difference(case_by_id))
    if missing:
        parser.error("dataset is missing seed cases: %s" % ", ".join(missing))

    judgment_metadata = {"provided": False}
    if args.judgments:
        judgment_raw = _read_bytes(args.judgments)
        payload = json.loads(judgment_raw.decode("utf-8"))
        by_case = payload.get("cases") or {}
        if not isinstance(by_case, dict):
            parser.error("judgments requires an object in cases")
        unknown = sorted(set(by_case).difference(case_by_id))
        if unknown:
            parser.error("judgments contains unknown cases: %s" % ", ".join(unknown))
        for case_id, judgments in by_case.items():
            if not isinstance(judgments, list):
                parser.error("judgments must be arrays per case")
            case_by_id[case_id]["suggestion_judgments"] = deepcopy(judgments)
        judgment_metadata = {
            "provided": True,
            "status": str(payload.get("status", "unknown")),
            "reviewer": str(payload.get("reviewer", "unknown")),
            "reviewed_suggestions": int(payload.get("reviewed_suggestions", 0) or 0),
            "explicit_judgments": sum(len(value) for value in by_case.values()),
            "sha256": hashlib.sha256(judgment_raw).hexdigest(),
            "path": os.path.normpath(args.judgments).replace("\\", "/"),
        }

    selected = [case_by_id[case_id] for case_id in result_ids]
    harness = ProductionEvaluationHarness()
    rescored_results = []
    totals = harness._empty_totals()
    split_totals = {}
    for original, case in zip(seed_results, selected):
        result = harness.rescore_cached_result(deepcopy(original), case)
        rescored_results.append(result)
        harness._accumulate(totals, result)
        bucket = split_totals.setdefault(case["split"], harness._empty_totals())
        harness._accumulate(bucket, result)

    dataset_raw = _read_bytes(args.dataset)
    report = deepcopy(seed)
    report.update({
        "name": str(seed.get("name", "agentic")) + "-cached-rescore",
        "status": "complete",
        "metrics": harness._metrics(totals),
        "by_split": {
            split: harness._metrics(values)
            for split, values in sorted(split_totals.items())
        },
        "case_results": rescored_results,
        "dataset": {
            "cases": len(selected),
            "repositories": len({case["repository"] for case in selected}),
            "risk_cases": sum(bool(case["expected_findings"]) for case in selected),
            "clean_cases": sum(not case["expected_findings"] for case in selected),
            "source_sha256": hashlib.sha256(dataset_raw).hexdigest(),
            "selected_normalized_sha256": dataset_fingerprint(selected),
            "readiness": validate_real_dataset(selected),
            "suggestion_judgments": judgment_metadata,
        },
        "rescore": {
            "model_calls": 0,
            "source_report": os.path.normpath(args.seed_report).replace("\\", "/"),
            "source_report_sha256": hashlib.sha256(seed_raw).hexdigest(),
            "duration_seconds": round(time.monotonic() - started, 4),
        },
    })
    report.setdefault("configuration", {})["cached_rescore_only"] = True
    report["progress"] = {
        "completed": len(rescored_results), "total": len(selected), "failed": sum(
            not bool(item.get("execution_success")) for item in rescored_results
        ),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)
    print(json.dumps(report["metrics"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
