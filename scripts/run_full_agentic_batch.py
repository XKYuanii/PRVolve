"""Run a resumable, progress-visible batch through the full Agentic reviewer."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.config import Settings  # noqa: E402
from evoagent.eval.harness import (  # noqa: E402
    dataset_fingerprint,
    load_jsonl,
)
from evoagent.eval.agentic import (  # noqa: E402
    ProductArmReviewer,
    ProductionEvaluationHarness,
    validate_real_dataset,
)
from evoagent.llm.client import JsonChatClient  # noqa: E402


@lru_cache(maxsize=1)
def implementation_fingerprint() -> str:
    digest = hashlib.sha256()
    for directory, _folders, files in sorted(os.walk(os.path.join(ROOT, "evoagent"))):
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                digest.update(os.path.relpath(path, ROOT).replace("\\", "/").encode())
                with open(path, "rb") as handle:
                    digest.update(handle.read().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def require_compatible_report(path, expected):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        previous = json.load(handle)
    if (previous.get("configuration") != expected["configuration"]
            or previous.get("model") != expected["model"]
            or previous.get("dataset", {}).get("source_sha256")
            != expected["dataset"]["source_sha256"]):
        raise ValueError(
            "Cannot mix versions/settings/datasets in %s; use a new --output for this run." % path
        )


def select_cases(cases: list, limit: int) -> list:
    """Round-robin across split and positive/clean buckets."""
    keys = [
        (split, risk)
        for split in ("validation", "holdout", "train")
        for risk in (True, False)
    ]
    buckets = {
        key: [
            case for case in cases
            if case["split"] == key[0]
            and bool(case["expected_findings"]) is key[1]
        ]
        for key in keys
    }
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    return selected


def expected_repository_sha(case: dict) -> str:
    source = case.get("source") or {}
    if source.get("kind") == "public-security-fix-reversal":
        return str(source.get("vulnerable_base_sha") or "").strip()
    if source.get("direction") == "fix-reversal-regression":
        return str(source.get("base_sha") or "").strip()
    return str(source.get("head_sha") or source.get("fixed_head_sha") or "").strip()


def require_repository_context(cases: list) -> dict:
    """Fail the entire batch before model calls unless every case is a full checkout."""
    repositories = set()
    tracked_counts = []
    for case in cases:
        root = os.path.abspath(str(case.get("repository_root") or ""))
        if not root or not os.path.isdir(root):
            raise ValueError("case %s has no repository checkout" % case["id"])
        head = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=20, check=False,
        )
        if head.returncode != 0:
            raise ValueError("case %s repository_root is not a Git worktree" % case["id"])
        expected = expected_repository_sha(case)
        actual = head.stdout.strip().lower()
        if expected and actual != expected.lower():
            raise ValueError(
                "case %s checkout HEAD %s != expected %s"
                % (case["id"], actual, expected)
            )
        files = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", root, "ls-files"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False,
        )
        tracked = len([line for line in files.stdout.splitlines() if line.strip()])
        if files.returncode != 0 or tracked < 2:
            raise ValueError("case %s checkout is not a populated repository" % case["id"])
        repositories.add(case["repository"].lower())
        tracked_counts.append(tracked)
    return {
        "cases": len(cases),
        "repositories": len(repositories),
        "minimum_tracked_files": min(tracked_counts),
        "maximum_tracked_files": max(tracked_counts),
    }


def load_results(path: str, allowed_ids: set, include_failures: bool = False) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        item["id"]: item
        for item in report.get("case_results", [])
        if item.get("id") in allowed_ids
        and (include_failures or item.get("execution_success") is True)
    }


def apply_judgments(path: str, cases: list, field: str, label: str) -> dict:
    if not path:
        return {"provided": False, "cases": 0, "judgments": 0, "sha256": ""}
    absolute = os.path.abspath(path)
    with open(absolute, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    by_case = payload.get("cases") or {}
    if not isinstance(by_case, dict):
        raise ValueError("%s judgment file requires an object in cases" % label)
    case_ids = {case["id"] for case in cases}
    unknown = set(by_case).difference(case_ids)
    if unknown:
        raise ValueError(
            "%s judgments reference unknown cases: %s"
            % (label, ", ".join(sorted(unknown)))
        )
    count = 0
    for case in cases:
        judgments = by_case.get(case["id"], [])
        if not isinstance(judgments, list):
            raise ValueError("%s judgments must be arrays per case" % label)
        case[field] = judgments
        count += len(judgments)
    return {
        "provided": True,
        "status": str(payload.get("status", "unknown")),
        "reviewer": str(payload.get("reviewer", "unknown")),
        "cases": sum(bool(value) for value in by_case.values()),
        "judgments": count,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "path": absolute,
    }


def apply_suggestion_judgments(path: str, cases: list) -> dict:
    return apply_judgments(
        path, cases, "suggestion_judgments", "suggestion",
    )


def apply_formal_judgments(path: str, cases: list) -> dict:
    return apply_judgments(path, cases, "formal_judgments", "formal")


def build_report(
    harness, selected: list, completed: dict, config: dict,
    token_budget: int, time_budget: int, timeout: int,
    started: float, status: str, source_dataset_sha256: str,
    judgment_metadata: dict, max_revision_rounds: int = 1,
) -> dict:
    ordered = [completed[case["id"]] for case in selected if case["id"] in completed]
    totals = harness._empty_totals()
    for result in ordered:
        harness._accumulate(totals, result)
    by_split = {}
    for split in ("validation", "holdout"):
        split_totals = harness._empty_totals()
        for result in ordered:
            if result["split"] == split:
                harness._accumulate(split_totals, result)
        by_split[split] = harness._metrics(split_totals)
    role_calls = Counter()
    for result in ordered:
        role_calls.update(result.get("model_roles") or {})
    return {
        "schema_version": 1,
        "name": "full-agentic-batch",
        "status": status,
        "progress": {"completed": len(ordered), "total": len(selected)},
        "model": {
            "provider": config["provider"],
            "name": config["model"],
            "base_url": config["base_url"],
        },
        "configuration": {
            "roles": ["lead", "security", "correctness-reliability", "critic"],
            "token_budget_per_pr": token_budget,
            "time_budget_seconds_per_pr": time_budget,
            "model_request_timeout_seconds": timeout,
            "max_revision_rounds": max_revision_rounds,
            "revision_policy": "lead-opt-in",
            "scanner_policy": "independent-reviewed-baseline",
            "publication_policy": "critic-or-grounded-lead-proof-v1",
            "implementation_sha256": implementation_fingerprint(),
        },
        "dataset": {
            "cases": len(selected),
            "repositories": len({case["repository"] for case in selected}),
            "risk_cases": sum(bool(case["expected_findings"]) for case in selected),
            "clean_cases": sum(not case["expected_findings"] for case in selected),
            "source_sha256": source_dataset_sha256,
            "selected_normalized_sha256": dataset_fingerprint(selected),
            "readiness": validate_real_dataset(selected),
            "suggestion_judgments": judgment_metadata["suggestion"],
            "formal_judgments": judgment_metadata["formal"],
        },
        "metrics": harness._metrics(totals),
        "by_split": by_split,
        "execution": {"model_role_calls": dict(sorted(role_calls.items()))},
        "duration_seconds": round(time.monotonic() - started, 4),
        "case_results": ordered,
    }


def save_report(path: str, report: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument(
        "--case-id", action="append", default=[],
        help="Run only this case id; repeat for multiple targeted diagnostics.",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--token-budget", type=int, default=64000)
    parser.add_argument("--time-budget", type=int, default=120)
    parser.add_argument(
        "--max-revision-rounds", type=int, default=1,
        help=(
            "Maximum optional Lead-selected revision rounds (default 1). "
            "No revision is forced. Use 0 only for an explicit one-pass ablation."
        ),
    )
    parser.add_argument(
        "--seed-report", action="append", default=[],
        help="Reuse completed cases from this report; repeat for multiple reports.",
    )
    parser.add_argument(
        "--rerun-case-prefix", action="append", default=[],
        help=(
            "Discard cached cases whose id starts with this prefix; repeat for "
            "multiple prefixes. Useful after repository context changes."
        ),
    )
    parser.add_argument(
        "--cached-only", action="store_true",
        help="Fail instead of calling the model when any selected case is absent from cache.",
    )
    parser.add_argument(
        "--require-repository-context", action="store_true",
        help=(
            "Before any model call, require every selected case to reference a "
            "populated Git worktree at its exact expected base/head commit."
        ),
    )
    parser.add_argument(
        "--suggestion-judgments", default="",
        help="Optional required/optional/invalid/duplicate adjudication JSON.",
    )
    parser.add_argument(
        "--formal-judgments", default="",
        help="Optional adjudication JSON for unexpected formal findings.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            ROOT, "output", "agentic-evaluation", "full-agentic-batch.json"
        ),
    )
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error("--max-cases must be positive")
    if args.max_revision_rounds < 0:
        parser.error("--max-revision-rounds cannot be negative")

    settings = Settings.from_env()
    config = settings.resolved_llm()
    if not config:
        parser.error("a model provider must be configured")
    cases = load_jsonl(args.dataset)
    source_dataset_sha256 = dataset_fingerprint(cases)
    for case in cases:
        for finding in case["expected_findings"]:
            finding.setdefault("should_comment", True)
    judgment_metadata = {
        "suggestion": apply_suggestion_judgments(args.suggestion_judgments, cases),
        "formal": apply_formal_judgments(args.formal_judgments, cases),
    }
    if args.case_id:
        requested = set(args.case_id)
        known = {case["id"] for case in cases}
        unknown = requested.difference(known)
        if unknown:
            parser.error("unknown --case-id values: %s" % ", ".join(sorted(unknown)))
        selected = [case for case in cases if case["id"] in requested][:args.max_cases]
    else:
        selected = select_cases(cases, args.max_cases)
    if args.require_repository_context:
        readiness = require_repository_context(selected)
        print(
            "REPOSITORY_CONTEXT cases=%d repositories=%d tracked_files=%d..%d"
            % (
                readiness["cases"], readiness["repositories"],
                readiness["minimum_tracked_files"],
                readiness["maximum_tracked_files"],
            ),
            flush=True,
        )
    allowed_ids = {case["id"] for case in selected}
    output = os.path.abspath(args.output)
    profile = build_report(
        ProductionEvaluationHarness(), selected, {}, config, args.token_budget,
        args.time_budget, args.timeout, time.monotonic(), "running", source_dataset_sha256,
        judgment_metadata, args.max_revision_rounds,
    )
    for report_path in [output, *args.seed_report]:
        require_compatible_report(os.path.abspath(report_path), profile)
    completed = load_results(output, allowed_ids)
    for seed_report in args.seed_report:
        completed.update(load_results(
            os.path.abspath(seed_report), allowed_ids,
            include_failures=args.cached_only,
        ))
    if args.rerun_case_prefix:
        completed = {
            case_id: result for case_id, result in completed.items()
            if not any(
                case_id.startswith(prefix) for prefix in args.rerun_case_prefix
            )
        }
    missing_cached = [case["id"] for case in selected if case["id"] not in completed]
    if args.cached_only and missing_cached:
        parser.error(
            "--cached-only is missing selected cases: %s"
            % ", ".join(missing_cached)
        )

    client = JsonChatClient(
        str(config["base_url"]), str(config["api_key"]), str(config["model"]),
        provider=str(config["provider"]), timeout=args.timeout,
    )
    reviewer = ProductArmReviewer(
        "full-agentic", client, args.token_budget, args.time_budget,
        max_revision_rounds=args.max_revision_rounds,
    )
    harness = ProductionEvaluationHarness()
    for case in selected:
        if case["id"] in completed:
            harness.rescore_cached_result(completed[case["id"]], case)
    started = time.monotonic()
    save_report(output, build_report(
        harness, selected, completed, config, args.token_budget,
        args.time_budget, args.timeout, started, "running", source_dataset_sha256,
        judgment_metadata, args.max_revision_rounds,
    ))
    for index, case in enumerate(selected, 1):
        if case["id"] in completed:
            print(
                "SKIP %d/%d case=%s checkpoint=complete"
                % (index, len(selected), case["id"]),
                flush=True,
            )
            continue
        print(
            "START %d/%d case=%s split=%s risk=%s"
            % (
                index, len(selected), case["id"], case["split"],
                bool(case["expected_findings"]),
            ),
            flush=True,
        )
        case_started = time.monotonic()
        result = harness.run(reviewer, [case], "full-agentic")["case_results"][0]
        completed[case["id"]] = result
        save_report(output, build_report(
            harness, selected, completed, config, args.token_budget,
            args.time_budget, args.timeout, started, "running", source_dataset_sha256,
            judgment_metadata, args.max_revision_rounds,
        ))
        print(
            "DONE %d/%d case=%s success=%s predicted=%d suggestions=%d "
            "tp=%d fp=%d fn=%d tokens=%d elapsed=%.1fs"
            % (
                index, len(selected), case["id"], result["execution_success"],
                result["predicted"], result.get("suggestions", 0),
                result["tp"], result["fp"], result["fn"],
                result["total_tokens"], time.monotonic() - case_started,
            ),
            flush=True,
        )
        if result.get("error"):
            print("ERROR case=%s %s" % (case["id"], result["error"]), flush=True)
    save_report(output, build_report(
        harness, selected, completed, config, args.token_budget,
        args.time_budget, args.timeout, started, "complete", source_dataset_sha256,
        judgment_metadata, args.max_revision_rounds,
    ))
    print("COMPLETE %s" % output, flush=True)


if __name__ == "__main__":
    main()
