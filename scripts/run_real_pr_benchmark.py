"""Validate a real PR dataset before wiring model-backed reviewer factories."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.eval.harness import load_jsonl  # noqa: E402
from evoagent.eval.agentic import validate_real_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Labelled public/historical PR JSONL")
    parser.add_argument("--minimum", type=int, default=300)
    args = parser.parse_args()
    result = validate_real_dataset(load_jsonl(args.dataset), args.minimum)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
