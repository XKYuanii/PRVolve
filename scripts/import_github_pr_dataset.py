"""Import labelled public GitHub PR diffs into the Evaluation Harness JSONL format.

The manifest must contain repository, pull_request, split, expected_findings and
repair_validation. Public PR content alone is not ground truth, so unlabelled
records are intentionally rejected.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.core.diff_parser import parse_unified_diff  # noqa: E402
from evoagent.eval.harness import validate_case  # noqa: E402
from evoagent.eval.agentic import validate_real_dataset  # noqa: E402


def fetch_diff(repository, pull_request, token=""):
    url = "https://api.github.com/repos/%s/pulls/%d" % (repository, pull_request)
    headers = {
        "Accept": "application/vnd.github.v3.diff",
        "User-Agent": "evoagent-evaluation-importer",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8", errors="replace"), url
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "GitHub returned HTTP %d for %s#%d"
            % (exc.code, repository, pull_request)
        ) from exc


def fetch_metadata(repository, pull_request, token=""):
    url = "https://api.github.com/repos/%s/pulls/%d" % (repository, pull_request)
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "evoagent-evaluation-importer",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "GitHub returned HTTP %d for %s#%d metadata"
            % (exc.code, repository, pull_request)
        ) from exc


def validate_checkout(repository_root, expected_head_sha):
    root = os.path.abspath(str(repository_root))
    if not os.path.isdir(root):
        raise ValueError("repository_root is not an existing directory: %s" % root)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, timeout=30, check=False,
    )
    if result.returncode != 0:
        raise ValueError("repository_root is not a readable Git checkout: %s" % root)
    actual = result.stdout.strip().lower()
    expected = str(expected_head_sha).strip().lower()
    if actual != expected:
        raise ValueError(
            "repository_root HEAD %s does not match PR head %s" % (actual, expected)
        )
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", help="Labelled JSONL manifest")
    parser.add_argument("output", help="Evaluation JSONL output")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument(
        "--require-checkout", action="store_true",
        help="Require repository_root at the exact PR head for repository-context evaluation.",
    )
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")
    records = []
    with open(args.manifest, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            item = json.loads(raw)
            if "expected_findings" not in item:
                raise ValueError(
                    "manifest line %d has no human-reviewed expected_findings"
                    % line_number
                )
            if any("should_comment" not in finding for finding in item["expected_findings"]):
                raise ValueError(
                    "manifest line %d has a finding without human should_comment label"
                    % line_number
                )
            repository = str(item["repository"])
            pull_request = int(item["pull_request"])
            metadata = fetch_metadata(repository, pull_request, token)
            diff, _api_url = fetch_diff(repository, pull_request, token)
            parsed = parse_unified_diff(diff)
            repository_root = str(item.get("repository_root", "")).strip()
            if args.require_checkout and not repository_root:
                raise ValueError(
                    "manifest line %d requires repository_root" % line_number
                )
            if repository_root:
                repository_root = validate_checkout(
                    repository_root, metadata["head"]["sha"]
                )
            record = {
                "schema_version": 1,
                "id": item.get(
                    "id", "%s#%s" % (item["repository"], item["pull_request"])
                ),
                "repository": item["repository"],
                "pull_request": int(item["pull_request"]),
                "split": item["split"],
                "source": {
                    "kind": "public-github-pr",
                    "public_url": str(metadata.get("html_url") or (
                        "https://github.com/%s/pull/%d" % (repository, pull_request)
                    )),
                    "base_sha": str(metadata["base"]["sha"]),
                    "head_sha": str(metadata["head"]["sha"]),
                },
                "diff": diff,
                "after_files": item.get("after_files", {}),
                "expected_findings": item["expected_findings"],
                "repair_validation": item.get("repair_validation", {}),
            }
            if repository_root:
                record["repository_root"] = repository_root
            validate_case(record)
            if not parsed.added_lines:
                raise ValueError("PR %s has no added lines" % record["id"])
            records.append(record)
            if len(records) >= args.limit:
                break
    if len(records) < args.limit:
        raise ValueError(
            "manifest produced %d records; %d required" % (len(records), args.limit)
        )
    readiness = validate_real_dataset(records, args.limit)
    if not readiness["ready"]:
        raise ValueError("dataset failed production readiness gates: %s" % readiness["gates"])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print("wrote %d labelled public PRs to %s" % (len(records), args.output))


if __name__ == "__main__":
    main()
