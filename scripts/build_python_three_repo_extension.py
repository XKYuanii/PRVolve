"""Build a six-case extension from three immutable public Python bug fixes."""
from __future__ import annotations

import json
import os
import posixpath
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.core.diff_parser import parse_unified_diff  # noqa: E402
from evoagent.eval.harness import load_jsonl  # noqa: E402


OUTPUT = os.path.join(ROOT, "benchmarks", "python_three_repo_extension_v1.jsonl")
CACHE_ROOT = os.path.join(ROOT, "output", "real-pr-repositories")
RUNTIME_ROOT = os.environ.get("EVOAGENT_BENCHMARK_RUNTIME_ROOT", ROOT)
RUNTIME_JOIN = posixpath.join if RUNTIME_ROOT.startswith("/") else os.path.join
CHECKOUT_ROOT = RUNTIME_JOIN(RUNTIME_ROOT, "output", "easy-pr-checkouts")


SPECS = [
    {
        "slug": "click-3299-strict-equality-default",
        "repository": "pallets/click",
        "pull_request": 3299,
        "base": "04ef3a6f473deb2499721a8d11f92a7d2c0912f2",
        "head": "d340b0c1202284a1b9d0ca892549208527d586ce",
        "base_checkout": "click-pr-3299-vulnerable-04ef3a6f",
        "head_checkout": "click-pr-3299-fixed-d340b0c1",
        "paths": ["src/click/core.py", "tests/test_options.py"],
        "findings": [
            {
                "path": "src/click/core.py",
                "needle": 'elif default_value == "":',
                "cwe": "CWE-248",
                "acceptable_cwes": ["CWE-691", "CWE-703", "CWE-755"],
                "rule_id": "RELIABILITY-STRICT-EQUALITY-DEFAULT",
                "severity": "medium",
                "summary": (
                    "Help rendering compares every default object to an empty "
                    "string, so a valid object with a strict __eq__ can raise and "
                    "crash --help."
                ),
            }
        ],
    },
    {
        "slug": "flask-6096-ipv6-server-name",
        "repository": "pallets/flask",
        "pull_request": 6096,
        "base": "514fc6b3e8402e4c646d5284e97a4f0ab50a7c4b",
        "head": "7203feabf723edae0286ae5dc64fec8ac4c91735",
        "base_checkout": "flask-pr-6096-vulnerable-514fc6b3",
        "head_checkout": "flask-pr-6096-fixed-7203feab",
        "paths": ["src/flask/app.py", "tests/test_basic.py"],
        "findings": [
            {
                "path": "src/flask/app.py",
                "needle": 'sn_host, _, sn_port = server_name.partition(":")',
                "cwe": "CWE-20",
                "acceptable_cwes": ["CWE-180", "CWE-703"],
                "rule_id": "CORRECTNESS-IPV6-SERVER-NAME-PARSING",
                "severity": "medium",
                "summary": (
                    "Splitting SERVER_NAME on its first colon treats an IPv6 "
                    "address as an empty host and a malformed port, so the "
                    "development server cannot honor a bracketed IPv6 endpoint."
                ),
            }
        ],
    },
    {
        "slug": "sphinx-10183-unhashable-annotation",
        "repository": "sphinx-doc/sphinx",
        "pull_request": 10183,
        "base": "b07ca9dbcc43f344f4aa15b970c0ce708f7b36a8",
        "head": "e1ea3bb53a7b043a70cef3356f9990b4e6bfd285",
        "base_checkout": "sphinx-pr-10183-vulnerable-b07ca9db",
        "head_checkout": "sphinx-pr-10183-fixed-e1ea3bb5",
        "paths": ["sphinx/util/typing.py"],
        "findings": [
            {
                "path": "sphinx/util/typing.py",
                "needle": "elif annotation in INVALID_BUILTIN_CLASSES:",
                "cwe": "CWE-248",
                "acceptable_cwes": ["CWE-703", "CWE-755"],
                "rule_id": "RELIABILITY-UNHASHABLE-ANNOTATION-MEMBERSHIP",
                "severity": "medium",
                "summary": (
                    "restify and stringify perform dictionary membership on an "
                    "arbitrary annotation before checking whether it is hashable, "
                    "causing the same TypeError and requiring one consolidated "
                    "review comment."
                ),
            },
        ],
    },
]


def git_diff(spec: dict, old: str, new: str) -> str:
    repository = os.path.join(CACHE_ROOT, spec["repository"].replace("/", "__"))
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


def added_line(diff: str, path: str, needle: str) -> int:
    matches = [
        item.line
        for item in parse_unified_diff(diff).added_lines
        if item.path == path and item.content.strip() == needle
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected one scoreable added line for %s %r, found %d"
            % (path, needle, len(matches))
        )
    return matches[0]


def pair(spec: dict) -> list[dict]:
    public_url = "https://github.com/%s/pull/%d" % (
        spec["repository"], spec["pull_request"]
    )
    risk_diff = git_diff(spec, spec["head"], spec["base"])
    clean_diff = git_diff(spec, spec["base"], spec["head"])
    expected = []
    for finding in spec["findings"]:
        line = added_line(risk_diff, finding["path"], finding["needle"])
        expected.append({
            "acceptable_cwes": finding["acceptable_cwes"],
            "cwe": finding["cwe"],
            "end_line": line,
            "path": finding["path"],
            "rule_id": finding["rule_id"],
            "severity": finding["severity"],
            "should_comment": True,
            "start_line": line,
            "summary": finding["summary"],
        })
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
        "expected_findings": expected,
        "repository_root": RUNTIME_JOIN(CHECKOUT_ROOT, spec["base_checkout"]),
        "source": {
            "derived_from": public_url,
            "fixed_head_sha": spec["head"],
            "head_sha": spec["base"],
            "kind": "public-bug-fix-reversal",
            "label_completeness": "targeted-review-comments",
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
        "repository_root": RUNTIME_JOIN(CHECKOUT_ROOT, spec["head_checkout"]),
        "source": {
            "base_sha": spec["base"],
            "head_sha": spec["head"],
            "kind": "public-bug-fix-slice",
            "label_completeness": "exhaustive-public-fix-slice",
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
    cases = []
    for spec in SPECS:
        cases.extend(pair(spec))
    if len(cases) != 6:
        raise ValueError("expected exactly 6 cases, got %d" % len(cases))
    temporary = OUTPUT + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    os.replace(temporary, OUTPUT)
    load_jsonl(OUTPUT)
    print("built 6 cases across exactly 3 paired public bug-fix families")


if __name__ == "__main__":
    main()
