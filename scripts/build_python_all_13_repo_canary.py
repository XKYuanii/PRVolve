"""Combine the 13 Python PR families with verified container checkout paths."""
from __future__ import annotations

import copy
import json
import os
import posixpath
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.eval.harness import load_jsonl  # noqa: E402


SOURCES = [
    os.path.join(ROOT, "benchmarks", "python_interview_canary_20_v1.jsonl"),
    os.path.join(ROOT, "benchmarks", "python_three_repo_extension_v1.jsonl"),
]
OUTPUT = os.path.join(ROOT, "benchmarks", "python_all_13_repo_canary_v1.jsonl")
RUNTIME_ROOT = os.environ.get("EVOAGENT_BENCHMARK_RUNTIME_ROOT", "/app")
RUNTIME_JOIN = posixpath.join if RUNTIME_ROOT.startswith("/") else os.path.join


def relative_checkout(case: dict) -> str:
    return "output/all-13-portable-checkouts/%s" % case["id"]


def expected_repository_sha(case: dict) -> str:
    source = case.get("source") or {}
    if source.get("kind") == "public-security-fix-reversal":
        return str(source.get("vulnerable_base_sha") or "").strip()
    return str(
        source.get("head_sha") or source.get("fixed_head_sha") or ""
    ).strip()


def verify_checkout(case: dict, relative: str) -> None:
    host_root = os.path.join(ROOT, *relative.split("/"))
    if not os.path.isdir(host_root):
        raise ValueError("checkout missing for %s: %s" % (case["id"], host_root))
    completed = subprocess.run(
        ["git", "-C", host_root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=20,
        check=False,
    )
    actual = completed.stdout.strip() if completed.returncode == 0 else ""
    expected = expected_repository_sha(case)
    if not expected:
        raise ValueError("case %s has no immutable expected SHA" % case["id"])
    if actual.lower() != expected.lower():
        raise ValueError(
            "checkout HEAD mismatch for %s: expected %s, got %s"
            % (case["id"], expected, actual or "unreadable")
        )


def main() -> None:
    cases = []
    for source in SOURCES:
        cases.extend(copy.deepcopy(load_jsonl(source)))
    click_case = next(
        case for case in cases
        if case["id"] == "python-click-3299-strict-equality-default-regression"
    )
    click_aliases = click_case["expected_findings"][0]["acceptable_cwes"]
    if "CWE-754" not in click_aliases:
        click_aliases.append("CWE-754")
        click_aliases.sort()
    ids = [case["id"] for case in cases]
    if len(cases) != 26 or len(set(ids)) != 26:
        raise ValueError("expected 26 unique records, got %d" % len(set(ids)))
    if len({case["repository"] for case in cases}) != 13:
        raise ValueError("expected exactly 13 repositories")
    if sum(bool(case["expected_findings"]) for case in cases) != 13:
        raise ValueError("expected exactly 13 risk records")
    for case in cases:
        relative = relative_checkout(case)
        verify_checkout(case, relative)
        case["repository_root"] = RUNTIME_JOIN(
            RUNTIME_ROOT, *relative.split("/")
        )
    temporary = OUTPUT + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, OUTPUT)
    load_jsonl(OUTPUT)
    print("built 26 verified records across 13 repositories")


if __name__ == "__main__":
    main()
