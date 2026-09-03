"""Create a versioned JSONL dataset from confirmed suggestion judgments."""
import argparse
import hashlib
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.eval.adjudication import promote_confirmed_judgments  # noqa: E402
from evoagent.eval.harness import load_jsonl  # noqa: E402


def _write_json(path: str, value) -> None:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, absolute)


def _write_jsonl(path: str, cases: list) -> None:
    absolute = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute), exist_ok=True)
    temporary = absolute + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, absolute)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("judgments")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    cases = load_jsonl(args.dataset)
    with open(args.judgments, "rb") as handle:
        raw = handle.read()
    payload = json.loads(raw.decode("utf-8"))
    revised, manifest = promote_confirmed_judgments(cases, payload)
    manifest.update({
        "source_dataset": os.path.normpath(args.dataset).replace("\\", "/"),
        "judgments": os.path.normpath(args.judgments).replace("\\", "/"),
        "judgments_sha256": hashlib.sha256(raw).hexdigest(),
        "output": os.path.normpath(args.output).replace("\\", "/"),
    })
    _write_jsonl(args.output, revised)
    _write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
