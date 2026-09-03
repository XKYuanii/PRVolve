"""Build the 20-case Python interview canary from immutable public fixes."""
from __future__ import annotations

import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.core.diff_parser import parse_unified_diff  # noqa: E402
from evoagent.eval.harness import load_jsonl  # noqa: E402


SECURITY_PAIRS = os.path.join(ROOT, "benchmarks", "python_security_pairs_v1.jsonl")
OUTPUT = os.path.join(ROOT, "benchmarks", "python_interview_canary_20_v1.jsonl")
CACHE_ROOT = os.path.join(ROOT, "output", "real-pr-repositories")
CHECKOUT_ROOT = os.path.join(ROOT, "output", "easy-pr-checkouts")


SPECS = [
    {
        "slug": "sqlmodel-997-mutating-update-dict",
        "repository": "tiangolo/sqlmodel",
        "pull_request": 997,
        "base": "fb331e7ce7798cb0a7d9b2a3e934191ffabfb4a8",
        "head": "6cec19c8dcd867d940b773d6e3f6e2dca316cb0f",
        "base_checkout": "sqlmodel-pr-997-vulnerable-fb331e7c",
        "head_checkout": "sqlmodel-pr-997-fixed-6cec19c8",
        "paths": ["sqlmodel/main.py", "tests/test_update.py"],
        "finding_path": "sqlmodel/main.py",
        "needle": "for remaining_key in use_update:",
        "cwe": "CWE-248",
        "acceptable_cwes": ["CWE-664", "CWE-703", "CWE-670"],
        "rule_id": "RELIABILITY-MUTATE-DICT-DURING-ITERATION",
        "severity": "medium",
        "summary": "Popping from use_update while iterating it raises RuntimeError when an update-only field is present.",
    },
    {
        "slug": "pytest-13086-scandir-missing-directory",
        "repository": "pytest-dev/pytest",
        "pull_request": 13086,
        "base": "bdfc3a99bd733f385f150446caef6d5843bb6418",
        "head": "476e31a7fbb94f4f11bb292a80caec4b0e74b573",
        "base_checkout": "pytest-pr-13086-vulnerable-bdfc3a99",
        "head_checkout": "pytest-pr-13086-fixed-476e31a7",
        "paths": ["src/_pytest/pathlib.py", "testing/test_pathlib.py"],
        "finding_path": "src/_pytest/pathlib.py",
        "needle": "with os.scandir(path) as s:",
        "cwe": "CWE-248",
        "acceptable_cwes": ["CWE-703"],
        "rule_id": "RELIABILITY-UNHANDLED-FILENOTFOUND",
        "severity": "medium",
        "summary": "A directory removed during collection raises FileNotFoundError and aborts pytest instead of being treated as empty.",
    },
    {
        "slug": "requests-7205-empty-netrc-credentials",
        "repository": "psf/requests",
        "pull_request": 7205,
        "base": "1b40fdd004bfc8ba5301bcf8a6908264e9b6b877",
        "head": "77f1e8c7a9296eac1aaef7fec53dd088a4e0f65b",
        "base_checkout": "requests-pr-7205-vulnerable-1b40fdd0",
        "head_checkout": "requests-pr-7205-fixed-77f1e8c7",
        "paths": ["src/requests/utils.py", "tests/test_utils.py"],
        "finding_path": "src/requests/utils.py",
        "needle": "if _netrc:",
        "cwe": "CWE-287",
        "acceptable_cwes": [
            "CWE-20", "CWE-252", "CWE-522", "CWE-670", "CWE-697",
        ],
        "rule_id": "CORRECTNESS-EMPTY-NETRC-CREDENTIALS",
        "severity": "medium",
        "summary": "A truthy all-empty netrc tuple is accepted and converted into blank HTTP Basic credentials.",
    },
    {
        "slug": "fastapi-12935-decimal-nan-exponent",
        "repository": "fastapi/fastapi",
        "pull_request": 12935,
        "base": "6ba09082a0b8455a890a4877c8ab1e3f143be8d1",
        "head": "29006661466ac7c758f13082c3a2080fd2542bc0",
        "base_checkout": "fastapi-pr-12935-vulnerable-6ba09082",
        "head_checkout": "fastapi-pr-12935-fixed-29006661",
        "paths": ["fastapi/encoders.py", "tests/test_jsonable_encoder.py"],
        "finding_path": "fastapi/encoders.py",
        "needle": "if dec_value.as_tuple().exponent >= 0:  # type: ignore[operator]",
        "cwe": "CWE-248",
        "acceptable_cwes": ["CWE-20", "CWE-703", "CWE-670"],
        "rule_id": "RELIABILITY-DECIMAL-NAN-TYPEERROR",
        "severity": "medium",
        "summary": "Decimal NaN and Infinity use a non-integer exponent sentinel, so comparing it with zero raises TypeError.",
    },
    {
        "slug": "werkzeug-1542-unhashable-exception",
        "repository": "pallets/werkzeug",
        "pull_request": 1542,
        "base": "bdc17e4cd10bbb17449006cef385ec953a11fc36",
        "head": "0e669f6be532801267d35de23c5f5237b8406d8a",
        "base_checkout": "werkzeug-pr-1542-vulnerable-bdc17e4c",
        "head_checkout": "werkzeug-pr-1542-fixed-0e669f6b",
        "paths": ["src/werkzeug/debug/tbtools.py", "tests/test_debug.py"],
        "finding_path": "src/werkzeug/debug/tbtools.py",
        "needle": "memo.add(exc_value)",
        "cwe": "CWE-248",
        "acceptable_cwes": ["CWE-20", "CWE-703", "CWE-704", "CWE-843", "CWE-670"],
        "rule_id": "RELIABILITY-UNHASHABLE-EXCEPTION",
        "severity": "medium",
        "summary": "Storing exception objects directly in a set crashes traceback rendering for unhashable exception classes.",
    },
    {
        "slug": "rich-2305-refresh-cleanup",
        "repository": "Textualize/rich",
        "pull_request": 2305,
        "base": "aa7926c1431eebfb2ccaab9f3b63a4ac6cd8dfe6",
        "head": "cf99a8d8753fd0270998d522cb49e6e472d68677",
        "base_checkout": "rich-pr-2305-vulnerable-aa7926c1",
        "head_checkout": "rich-pr-2305-fixed-cf99a8d8",
        "paths": ["rich/live.py"],
        "finding_path": "rich/live.py",
        "needle": "self.refresh()",
        "cwe": "CWE-459",
        "acceptable_cwes": ["CWE-404", "CWE-664", "CWE-703", "CWE-772"],
        "rule_id": "RELIABILITY-REFRESH-CLEANUP",
        "severity": "medium",
        "summary": "If the initial refresh raises, redirected standard streams and live-console state are left installed.",
    },
]


def git_diff(spec: dict, old: str, new: str) -> str:
    repository = os.path.join(
        CACHE_ROOT, spec["repository"].replace("/", "__")
    )
    completed = subprocess.run(
        [
            "git", "-C", repository, "diff", "--no-ext-diff", "--unified=20",
            old, new, "--", *spec["paths"],
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.replace("\r\n", "\n")


def added_range(diff: str, path: str, needle: str, span: bool = False) -> tuple[int, int]:
    matches = [
        item.line
        for item in parse_unified_diff(diff).added_lines
        if item.path == path and item.content.strip() == needle
    ]
    if not matches or (len(matches) != 1 and not span):
        raise ValueError(
            "expected scoreable added line(s) for %s %r, found %d"
            % (path, needle, len(matches))
        )
    return min(matches), max(matches)


def pair(spec: dict) -> list[dict]:
    public_url = "https://github.com/%s/pull/%d" % (
        spec["repository"], spec["pull_request"]
    )
    risk_diff = git_diff(spec, spec["head"], spec["base"])
    clean_diff = git_diff(spec, spec["base"], spec["head"])
    start_line, end_line = added_range(
        risk_diff, spec["finding_path"], spec["needle"],
        bool(spec.get("line_span")),
    )
    common = {
        "repository": spec["repository"],
        "pull_request": spec["pull_request"],
        "repair_validation": {},
        "split": "validation",
    }
    risk = {
        **common,
        "id": "python-%s-regression" % spec["slug"],
        "diff": risk_diff,
        "expected_findings": [
            {
                "acceptable_cwes": spec["acceptable_cwes"],
                "cwe": spec["cwe"],
                "end_line": end_line,
                "path": spec["finding_path"],
                "rule_id": spec["rule_id"],
                "severity": spec["severity"],
                "should_comment": True,
                "start_line": start_line,
                "summary": spec["summary"],
            }
        ],
        "repository_root": os.path.join(CHECKOUT_ROOT, spec["base_checkout"]),
        "source": {
            "derived_from": public_url,
            "fixed_head_sha": spec["head"],
            "head_sha": spec["base"],
            "kind": "public-bug-fix-reversal",
            "label_completeness": "targeted-single-defect",
            "language": "python",
            "public_url": public_url,
            "selection_tier": "L0-obvious-correctness",
            "transformation": (
                "The public fix is reversed to replay the documented bug; "
                "this is not presented as the original PR direction."
            ),
            "vulnerable_base_sha": spec["base"],
        },
    }
    clean = {
        **common,
        "id": "python-%s-fix" % spec["slug"],
        "diff": clean_diff,
        "expected_findings": [],
        "repository_root": os.path.join(CHECKOUT_ROOT, spec["head_checkout"]),
        "source": {
            "base_sha": spec["base"],
            "head_sha": spec["head"],
            "kind": "public-bug-fix-slice",
            "label_completeness": "targeted-single-defect",
            "language": "python",
            "public_url": public_url,
            "selection_tier": "L0-clean-correctness-fix",
            "transformation": (
                "Only production and regression-test paths relevant to the "
                "documented fix are retained."
            ),
        },
    }
    return [risk, clean]


def main() -> None:
    cases = load_jsonl(SECURITY_PAIRS)
    for spec in SPECS:
        cases.extend(pair(spec))
    if len(cases) != 20:
        raise ValueError("expected exactly 20 cases, got %d" % len(cases))
    temporary = OUTPUT + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, OUTPUT)
    load_jsonl(OUTPUT)
    print("built %d cases across %d paired defect families" % (len(cases), len(cases) // 2))


if __name__ == "__main__":
    main()
