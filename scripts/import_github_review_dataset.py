"""Build real PR cases from trusted GitHub review comments at exact commit snapshots."""
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

from evoagent.eval.harness import validate_case  # noqa: E402


TRUSTED_ASSOCIATIONS = frozenset({"MEMBER", "OWNER", "COLLABORATOR"})


def fetch_review_comment(repository: str, comment_id: int, token: str = "") -> dict:
    url = "https://api.github.com/repos/%s/pulls/comments/%d" % (
        repository, comment_id,
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "evoagent-review-benchmark-importer",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=60,
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "GitHub returned HTTP %d for review comment %s/%d"
            % (exc.code, repository, comment_id)
        ) from exc


def git_output(repository_root: str, arguments: list) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repository_root, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(
            "git %s failed in %s: %s"
            % (" ".join(arguments), repository_root, result.stderr.strip()[:1000])
        )
    return result.stdout


def _review_line(comment: dict) -> int:
    value = comment.get("original_line") or comment.get("line")
    if not value:
        raise ValueError("review comment has no scoreable line")
    return int(value)


def build_case(
    item: dict, checkout_root: str, token: str = "", cached_comments=None,
) -> dict:
    for field in (
        "id", "repository", "pull_request", "split", "base_sha",
        "snapshot_sha", "checkout", "expected_findings",
    ):
        if field not in item:
            raise ValueError("review manifest is missing %s" % field)
    repository = str(item["repository"])
    pull_request = int(item["pull_request"])
    base_sha = str(item["base_sha"]).strip().lower()
    snapshot_sha = str(item["snapshot_sha"]).strip().lower()
    repository_root = os.path.abspath(os.path.join(checkout_root, str(item["checkout"])))
    checkout_boundary = os.path.abspath(checkout_root) + os.sep
    if not repository_root.startswith(checkout_boundary):
        raise ValueError("checkout must remain below --checkout-root")
    if not os.path.isdir(repository_root):
        raise ValueError("checkout does not exist: %s" % repository_root)
    actual_head = git_output(repository_root, ["rev-parse", "HEAD"]).strip().lower()
    if actual_head != snapshot_sha:
        raise ValueError(
            "checkout HEAD %s does not match snapshot %s" % (actual_head, snapshot_sha)
        )
    git_output(repository_root, ["cat-file", "-e", base_sha + "^{commit}"])
    effective_base_sha = git_output(
        repository_root, ["merge-base", base_sha, snapshot_sha]
    ).strip().lower()
    if not effective_base_sha:
        raise ValueError(
            "base %s and snapshot %s do not have a merge base"
            % (base_sha, snapshot_sha)
        )

    evidence = {}
    for expected in item["expected_findings"]:
        comment_id = int(expected.get("review_comment_id", 0) or 0)
        if comment_id < 1:
            raise ValueError("every expected finding requires review_comment_id")
        if comment_id in evidence:
            raise ValueError("review_comment_id cannot label multiple findings")
        comment = dict((cached_comments or {}).get(comment_id) or {})
        if not comment:
            comment = fetch_review_comment(repository, comment_id, token)
        association = str(comment.get("author_association", "")).upper()
        if association not in TRUSTED_ASSOCIATIONS:
            raise ValueError(
                "review comment %d has untrusted author association %s"
                % (comment_id, association)
            )
        comment_pr = int(str(comment.get("pull_request_url", "")).rstrip("/").split("/")[-1])
        if comment_pr != pull_request:
            raise ValueError("review comment %d belongs to a different PR" % comment_id)
        comment_snapshot = str(
            comment.get("original_commit_id") or comment.get("commit_id") or ""
        ).lower()
        if comment_snapshot != snapshot_sha:
            raise ValueError(
                "review comment %d snapshot %s does not match %s"
                % (comment_id, comment_snapshot, snapshot_sha)
            )
        path = str(comment.get("path", ""))
        line = _review_line(comment)
        if str(expected.get("path", "")) != path:
            raise ValueError("review comment %d path does not match its label" % comment_id)
        if not int(expected["start_line"]) <= line <= int(expected["end_line"]):
            raise ValueError("review comment %d line does not match its label" % comment_id)
        evidence[comment_id] = {
            "author_association": association,
            "body": str(comment.get("body", ""))[:4000],
            "comment_id": comment_id,
            "html_url": str(comment.get("html_url", "")),
            "line": line,
            "path": path,
            "snapshot_sha": comment_snapshot,
        }

    diff = git_output(
        repository_root,
        [
            "diff", "--no-ext-diff", "--unified=3",
            effective_base_sha, snapshot_sha, "--",
        ],
    )
    expected_findings = []
    for raw in item["expected_findings"]:
        expected = dict(raw)
        expected["should_comment"] = bool(expected.get("should_comment", True))
        expected["label_source"] = "public-github-review-comment"
        expected_findings.append(expected)
    record = {
        "schema_version": 2,
        "id": str(item["id"]),
        "repository": repository,
        "pull_request": pull_request,
        "split": str(item["split"]),
        "source": {
            "kind": "public-github-pr",
            "label_kind": "public-github-review-comment",
            "label_completeness": "targeted-review-comments",
            "public_url": "https://github.com/%s/pull/%d" % (repository, pull_request),
            "base_sha": effective_base_sha,
            "requested_base_sha": base_sha,
            "base_was_merge_base": effective_base_sha == base_sha,
            "head_sha": snapshot_sha,
            "review_evidence": list(evidence.values()),
        },
        "repository_root": repository_root,
        "diff": diff,
        "expected_findings": expected_findings,
        "repair_validation": dict(item.get("repair_validation") or {}),
    }
    validate_case(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--checkout-root", required=True)
    parser.add_argument(
        "--reuse-evidence-from", default="",
        help="Reuse review evidence from a previously GitHub-verified dataset.",
    )
    args = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")
    cached_comments = {}
    if args.reuse_evidence_from:
        with open(args.reuse_evidence_from, "r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                prior = json.loads(raw)
                repository = str(prior.get("repository", ""))
                pull_request = int(prior.get("pull_request", 0) or 0)
                for evidence in (prior.get("source") or {}).get("review_evidence") or []:
                    comment_id = int(evidence.get("comment_id", 0) or 0)
                    if comment_id:
                        cached_comments[comment_id] = {
                            "author_association": evidence.get("author_association"),
                            "body": evidence.get("body", ""),
                            "html_url": evidence.get("html_url", ""),
                            "original_commit_id": evidence.get("snapshot_sha", ""),
                            "original_line": evidence.get("line"),
                            "path": evidence.get("path", ""),
                            "pull_request_url": (
                                "https://api.github.com/repos/%s/pulls/%d"
                                % (repository, pull_request)
                            ),
                        }
    records = []
    with open(args.manifest, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                records.append(build_case(
                    json.loads(raw), args.checkout_root, token, cached_comments,
                ))
            except Exception as exc:
                raise ValueError("manifest line %d: %s" % (line_number, exc)) from exc
    if not records:
        parser.error("manifest produced no cases")
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print("wrote %d reviewed PR snapshots to %s" % (len(records), output))


if __name__ == "__main__":
    main()
