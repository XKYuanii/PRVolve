"""Evaluate versioned deterministic baselines without model calls."""
import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.eval.benchmark import ContextRuleReviewer  # noqa: E402
from evoagent.eval.harness import (  # noqa: E402
    EndToEndEvaluationHarness,
    dataset_fingerprint,
    load_jsonl,
)
from evoagent.review.reviewers import CompositeReviewer, LocalRuleReviewer, Reviewer  # noqa: E402


LEGACY_LOCAL_RULE_IDS = frozenset({
    "SEC-EVAL",
    "SEC-SUBPROCESS-SHELL",
    "SEC-HARDCODED-SECRET",
    "SEC-SQL-CONCAT",
    "REL-EMPTY-EXCEPT",
    "REL-DEBUG-PRINT",
})


class RuleFilterReviewer(Reviewer):
    name = "versioned-local-rules"

    def __init__(self, reviewer: Reviewer, rule_ids) -> None:
        self.reviewer = reviewer
        self.rule_ids = frozenset(rule_ids)

    def review(self, diff, parsed):
        return [
            finding for finding in self.reviewer.review(diff, parsed)
            if finding.rule_id in self.rule_ids
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--rules", type=int, choices=(6, 14, 18), default=18)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    local = LocalRuleReviewer()
    reviewers = [
        local if args.rules == 18
        else RuleFilterReviewer(local, LEGACY_LOCAL_RULE_IDS)
    ]
    if args.rules in {14, 18}:
        reviewers.append(ContextRuleReviewer())
    reviewer = CompositeReviewer(reviewers)
    report = EndToEndEvaluationHarness().run(
        reviewer, cases, "deterministic-%d-rule" % args.rules
    )
    report["configuration"] = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_sha256": dataset_fingerprint(cases),
        "deterministic_rules": args.rules,
        "model_calls": 0,
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
