"""Evaluate the full agentic topology against its no-critic variant."""
import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.eval.harness import load_jsonl  # noqa: E402
from evoagent.eval.agentic import (  # noqa: E402
    FairAblationSuite,
    product_reviewer_factories,
)
from evoagent.llm.client import JsonChatClient  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the complete four-role agentic topology with the same "
            "agentic workflow running without Critic."
        )
    )
    parser.add_argument("dataset", help="Human-labelled public/historical PR JSONL")
    parser.add_argument("--base-url", default=os.getenv("EVOAGENT_LLM_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("EVOAGENT_LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("EVOAGENT_LLM_MODEL", ""))
    parser.add_argument("--provider", default=os.getenv("EVOAGENT_LLM_PROVIDER", "custom"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--token-budget", type=int, default=12000)
    parser.add_argument("--time-budget", type=int, default=240)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    parser.add_argument(
        "--allow-non-production-data", action="store_true",
        help="Run for harness debugging but keep all proof/launch gates closed.",
    )
    parser.add_argument(
        "--max-cases", type=int, default=0,
        help="Limit a debug run while sampling validation/holdout in round-robin order.",
    )
    parser.add_argument(
        "--output", default=os.path.join(
            ROOT, "output", "agentic-evaluation", "evaluation.json",
        ),
    )
    args = parser.parse_args()
    if not args.base_url or not args.api_key or not args.model:
        parser.error("--base-url, --api-key and --model are required")
    if args.token_budget < 1024:
        parser.error("--token-budget must be at least 1024 for four LLM roles")
    if args.time_budget < 4:
        parser.error("--time-budget must be at least 4 seconds")
    if args.max_cases < 0:
        parser.error("--max-cases must be non-negative")

    client = JsonChatClient(
        args.base_url, args.api_key, args.model,
        provider=args.provider, timeout=args.timeout,
    )
    suite = FairAblationSuite(
        product_reviewer_factories(client, args.time_budget),
        args.model, args.token_budget,
        require_production_ready=not args.allow_non_production_data,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    cases = load_jsonl(args.dataset)
    if args.allow_non_production_data:
        # Legacy controlled fixtures predate the production should_comment
        # label. For debug-only runs, an expected finding is by definition a
        # commentable finding; production readiness remains false.
        for case in cases:
            for finding in case["expected_findings"]:
                finding.setdefault("should_comment", True)
    if args.max_cases:
        buckets = {
            split: [case for case in cases if case["split"] == split]
            for split in ("validation", "holdout", "train")
        }
        selected = []
        while len(selected) < args.max_cases and any(buckets.values()):
            for split in ("validation", "holdout", "train"):
                if buckets[split] and len(selected) < args.max_cases:
                    selected.append(buckets[split].pop(0))
        cases = selected
    report = suite.run(cases)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
