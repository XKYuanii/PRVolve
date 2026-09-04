"""Run one dataset case through the full four-role agentic reviewer."""
import argparse
import json
import os
import subprocess
import sys
import time


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.config import Settings  # noqa: E402
from evoagent.eval.harness import load_jsonl  # noqa: E402
from evoagent.eval.agentic import (  # noqa: E402
    ProductArmReviewer,
    ProductionEvaluationHarness,
)
from evoagent.llm.client import JsonChatClient  # noqa: E402


def expected_repository_sha(case: dict) -> str:
    """Return the commit representing the new-file state of the case Diff."""
    source = case.get("source") or {}
    if source.get("kind") == "public-security-fix-reversal":
        return str(source.get("vulnerable_base_sha") or "").strip()
    if source.get("direction") == "fix-reversal-regression":
        return str(source.get("base_sha") or "").strip()
    return str(
        source.get("head_sha") or source.get("fixed_head_sha") or ""
    ).strip()


def validate_repository_checkout(case: dict, repository_root: str) -> None:
    expected = expected_repository_sha(case)
    if not expected:
        return
    actual = ""
    try:
        completed = subprocess.run(
            ["git", "-C", repository_root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if completed.returncode == 0:
            actual = completed.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not actual:
        head_path = os.path.join(repository_root, ".git", "HEAD")
        try:
            with open(head_path, "r", encoding="utf-8") as handle:
                head = handle.read().strip()
            if not head.startswith("ref:"):
                actual = head
            else:
                ref = head[4:].strip().replace("/", os.sep)
                with open(
                    os.path.join(repository_root, ".git", ref),
                    "r", encoding="utf-8",
                ) as handle:
                    actual = handle.read().strip()
        except OSError:
            actual = ""
    if not actual:
        raise ValueError("--repository-root is not a readable Git checkout")
    if actual.lower() != expected.lower():
        raise ValueError(
            "--repository-root HEAD mismatch: expected %s, got %s"
            % (expected, actual)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one controlled PR through the full Agentic topology."
    )
    parser.add_argument("dataset")
    parser.add_argument("--case-id", default="")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--token-budget", type=int, default=16000)
    parser.add_argument("--time-budget", type=int, default=120)
    parser.add_argument(
        "--max-revision-rounds", type=int, default=0,
        help="Allow the existing Lead to request bounded Worker rework.",
    )
    parser.add_argument(
        "--without-repository-context", action="store_true",
        help="Remove repository_root for a controlled Diff-only comparison.",
    )
    parser.add_argument(
        "--repository-root", default="",
        help="Use this checkout as repository context for the selected case.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            ROOT, "output", "agentic-evaluation", "single-case.json"
        ),
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    config = settings.resolved_llm()
    if not config:
        parser.error("a model provider must be configured")

    cases = load_jsonl(args.dataset)
    case = next(
        (item for item in cases if item["id"] == args.case_id),
        cases[0] if cases and not args.case_id else None,
    )
    if case is None:
        parser.error("case was not found")
    if args.without_repository_context:
        case = dict(case)
        case.pop("repository_root", None)
    if args.repository_root:
        repository_root = os.path.abspath(args.repository_root)
        if not os.path.isdir(repository_root):
            parser.error("--repository-root must be an existing directory")
        try:
            validate_repository_checkout(case, repository_root)
        except ValueError as exc:
            parser.error(str(exc))
        case = dict(case)
        case["repository_root"] = repository_root
    for finding in case["expected_findings"]:
        finding.setdefault("should_comment", True)

    client = JsonChatClient(
        str(config["base_url"]),
        str(config["api_key"]),
        str(config["model"]),
        provider=str(config["provider"]),
        timeout=args.timeout,
    )
    reviewer = ProductArmReviewer(
        "full-agentic", client, args.token_budget, args.time_budget,
        max_revision_rounds=args.max_revision_rounds,
    )
    print(
        "START case=%s split=%s model=%s timeout=%ss"
        % (case["id"], case["split"], config["model"], args.timeout),
        flush=True,
    )
    started = time.monotonic()
    report = ProductionEvaluationHarness().run(
        reviewer, [case], "single-full-agentic"
    )
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    result = report["case_results"][0]
    print(
        "DONE case=%s success=%s predicted=%s suggestions=%s tp=%s fp=%s fn=%s elapsed=%.1fs"
        % (
            result["id"], result["execution_success"], result["predicted"],
            result.get("suggestions", 0),
            result["tp"], result["fp"], result["fn"],
            time.monotonic() - started,
        ),
        flush=True,
    )
    if result.get("error"):
        print("ERROR %s" % result["error"], flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
