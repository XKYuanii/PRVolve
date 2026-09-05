"""Prepare immutable full-source base/head worktrees for the 37 new PR pairs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT / "output" / "python-50-repo-canary-v1" / "selection-manifest.json"
)
DEFAULT_METADATA = (
    ROOT / "output" / "python-50-repo-canary-v1" / "pr-metadata.json"
)
DEFAULT_CACHE_ROOT = (
    ROOT / "output" / "python-50-repo-canary-v1" / "repository-cache"
)
DEFAULT_CHECKOUT_ROOT = (
    ROOT / "output" / "python-50-repo-canary-v1" / "checkouts"
)
DEFAULT_RUNTIME_DATASET = (
    ROOT / "output" / "python-50-repo-canary-v1" / "runtime-dataset.jsonl"
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def api_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "EvoAgent-public-PR-benchmark",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def request_pr(repository: str, pull_request: int) -> dict:
    url = "https://api.github.com/repos/%s/pulls/%d" % (repository, pull_request)
    last_error = ""
    for attempt in range(4):
        try:
            response = requests.get(url, headers=api_headers(), timeout=30)
            if response.status_code == 200:
                value = response.json()
                if not value.get("merged_at"):
                    raise ValueError("PR is not merged: %s" % value.get("html_url", url))
                return {
                    "repository": repository,
                    "pull_request": pull_request,
                    "public_url": str(value["html_url"]),
                    "title": str(value["title"]),
                    "base_ref": str(value["base"]["ref"]),
                    "base_sha": str(value["base"]["sha"]).lower(),
                    "head_ref": str(value["head"]["ref"]),
                    "head_sha": str(value["head"]["sha"]).lower(),
                    "merge_commit_sha": str(value.get("merge_commit_sha") or "").lower(),
                    "merged_at": str(value["merged_at"]),
                    "additions": int(value.get("additions", 0)),
                    "deletions": int(value.get("deletions", 0)),
                    "changed_files": int(value.get("changed_files", 0)),
                }
            last_error = "HTTP %d: %s" % (response.status_code, response.text[:500])
            if response.status_code not in {403, 429, 500, 502, 503, 504}:
                break
        except (requests.RequestException, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2 ** attempt)
    raise RuntimeError("GitHub PR metadata failed for %s#%d: %s" % (
        repository, pull_request, last_error,
    ))


def load_or_fetch_metadata(manifest: dict, path: Path) -> dict:
    existing = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = {
            (item["repository"], int(item["pull_request"])): item
            for item in payload.get("pull_requests", [])
        }
    pull_requests = []
    for index, selected in enumerate(manifest["selected"], 1):
        key = (selected["repository"], int(selected["pull_request"]))
        item = existing.get(key)
        if item is None:
            print("METADATA %d/%d %s#%d" % (
                index, len(manifest["selected"]), key[0], key[1],
            ), flush=True)
            item = request_pr(*key)
        pull_requests.append(item)
        atomic_json(path, {
            "schema_version": 1,
            "source": "GitHub REST pull request API",
            "pull_requests": pull_requests,
        })
    payload = {
        "schema_version": 1,
        "source": "GitHub REST pull request API",
        "pull_requests": pull_requests,
    }
    atomic_json(path, payload)
    return payload


def safe_child(root: Path, name: str) -> Path:
    child = (root / name).resolve()
    boundary = str(root.resolve()) + os.sep
    if not str(child).startswith(boundary):
        raise ValueError("path escapes configured root: %s" % child)
    return child


def run_git(arguments: list[str], cwd: Path | None = None, timeout: int = 1800) -> str:
    command = ["git", "-c", "core.longpaths=true", *arguments]
    environment = dict(os.environ)
    environment.update({
        "GIT_TERMINAL_PROMPT": "0",
        # Preserve every Git-tracked source file without downloading unrelated
        # large-file payloads; LFS pointer files remain available as context.
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    result = subprocess.run(
        command, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError("%s failed: %s" % (
            " ".join(command), (result.stderr or result.stdout).strip()[:4000],
        ))
    return result.stdout.strip()


def has_commit(cache: Path, sha: str) -> bool:
    result = subprocess.run(
        ["git", "-c", "core.longpaths=true", "cat-file", "-e", sha + "^{commit}"],
        cwd=str(cache), capture_output=True, check=False,
    )
    return result.returncode == 0


def ensure_cache(repository: str, cache: Path) -> None:
    if cache.is_dir():
        run_git(["rev-parse", "--git-dir"], cache, timeout=30)
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    run_git([
        "clone", "--filter=blob:none", "--no-checkout", "--depth=1",
        "https://github.com/%s.git" % repository, str(cache),
    ])


def ensure_commit(cache: Path, sha: str, refspec: str = "") -> None:
    if has_commit(cache, sha):
        return
    target = refspec or sha
    run_git([
        "fetch", "--filter=blob:none", "--depth=1", "--no-tags", "origin", target,
    ], cache)
    if not has_commit(cache, sha):
        # A PR ref fetch stores FETCH_HEAD without a durable local ref, but the
        # object must still resolve to the immutable SHA.
        raise RuntimeError("fetched revision does not expose expected SHA %s" % sha)


def ensure_worktree(cache: Path, checkout: Path, sha: str) -> dict:
    if checkout.is_dir():
        actual = run_git(["rev-parse", "HEAD"], checkout, timeout=30).lower()
        if actual != sha:
            raise ValueError("checkout %s is at %s, expected %s" % (
                checkout, actual, sha,
            ))
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run_git(["worktree", "add", "--detach", str(checkout), sha], cache)
    actual = run_git(["rev-parse", "HEAD"], checkout, timeout=30).lower()
    if actual != sha:
        raise ValueError("worktree HEAD mismatch: %s != %s" % (actual, sha))
    files = sum(1 for path in checkout.rglob("*") if path.is_file())
    if files < 2:
        raise ValueError("worktree is not populated: %s" % checkout)
    return {"path": str(checkout), "head_sha": actual, "files": files}


def validate_defect_line(checkout: Path, selected: dict) -> None:
    target = safe_child(checkout, selected["primary_path"])
    if not target.is_file():
        raise ValueError("primary source path is missing: %s" % target)
    content = target.read_text(encoding="utf-8", errors="replace")
    if str(selected["needle"]) not in content:
        raise ValueError("regression checkout lacks defect needle %r in %s" % (
            selected["needle"], selected["primary_path"],
        ))


def prepare_one(
    selected: dict, metadata: dict, cache_root: Path, checkout_root: Path,
) -> dict:
    repository = selected["repository"]
    pull_request = int(selected["pull_request"])
    cache = safe_child(cache_root, repository.replace("/", "__"))
    ensure_cache(repository, cache)
    ensure_commit(cache, metadata["base_sha"])
    ensure_commit(
        cache, metadata["head_sha"],
        "refs/pull/%d/head" % pull_request,
    )
    regression = safe_child(checkout_root, selected["risk_case_id"])
    clean = safe_child(checkout_root, selected["clean_case_id"])
    regression_info = ensure_worktree(cache, regression, metadata["base_sha"])
    clean_info = ensure_worktree(cache, clean, metadata["head_sha"])
    validate_defect_line(regression, selected)
    return {
        "repository": repository,
        "pull_request": pull_request,
        "base_sha": metadata["base_sha"],
        "head_sha": metadata["head_sha"],
        "regression": regression_info,
        "clean": clean_info,
        "validated_defect_line": True,
    }


def rewrite_runtime_dataset(
    dataset_path: Path, checkout_root: Path, metadata_by_key: dict,
) -> None:
    records = []
    for raw in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        case = json.loads(raw)
        pull_request = case.get("pull_request")
        key = (
            (case["repository"], int(pull_request))
            if pull_request is not None
            else None
        )
        # A larger benchmark may reuse already-prepared cases whose worktrees
        # live under an earlier checkout root. Rewrite only the PRs named by
        # this preparation manifest and preserve every reused repository_root.
        if key is not None and key in metadata_by_key:
            checkout = safe_child(checkout_root, case["id"])
            if not checkout.is_dir():
                raise ValueError("missing checkout for %s" % case["id"])
            metadata = metadata_by_key[key]
            expected = (
                metadata["base_sha"] if case["id"].endswith("-regression")
                else metadata["head_sha"]
            )
            actual = run_git(["rev-parse", "HEAD"], checkout, timeout=30).lower()
            if actual != expected:
                raise ValueError("%s HEAD %s != %s" % (case["id"], actual, expected))
            case["repository_root"] = str(checkout)
            case.setdefault("source", {}).update({
                "base_sha": metadata["base_sha"],
                "head_sha": metadata["head_sha"],
                "merged_at": metadata["merged_at"],
                "checkout_kind": "immutable-full-source-worktree",
            })
        records.append(case)
    temporary = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for case in records:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, dataset_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--checkout-root", default=str(DEFAULT_CHECKOUT_ROOT))
    parser.add_argument("--runtime-dataset", default=str(DEFAULT_RUNTIME_DATASET))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")

    manifest_path = Path(args.manifest).resolve()
    metadata_path = Path(args.metadata).resolve()
    cache_root = Path(args.cache_root).resolve()
    checkout_root = Path(args.checkout_root).resolve()
    dataset_path = Path(args.runtime_dataset).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_payload = load_or_fetch_metadata(manifest, metadata_path)
    metadata_by_key = {
        (item["repository"], int(item["pull_request"])): item
        for item in metadata_payload["pull_requests"]
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    checkout_root.mkdir(parents=True, exist_ok=True)
    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for selected in manifest["selected"]:
            key = (selected["repository"], int(selected["pull_request"]))
            future = executor.submit(
                prepare_one, selected, metadata_by_key[key], cache_root, checkout_root,
            )
            futures[future] = selected
        for future in as_completed(futures):
            selected = futures[future]
            try:
                result = future.result()
                completed.append(result)
                print("READY %d/%d %s#%d base_files=%d head_files=%d" % (
                    len(completed), len(futures), result["repository"],
                    result["pull_request"], result["regression"]["files"],
                    result["clean"]["files"],
                ), flush=True)
            except Exception as exc:
                failures.append({
                    "repository": selected["repository"],
                    "pull_request": int(selected["pull_request"]),
                    "error": str(exc),
                })
                print("FAILED %s#%d %s" % (
                    selected["repository"], int(selected["pull_request"]), exc,
                ), flush=True)

    verification = {
        "schema_version": 1,
        "status": "complete" if not failures else "failed",
        "repositories": len(completed),
        "worktrees": len(completed) * 2,
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "third_party_code_executed": False,
        "lfs_payloads_materialized": False,
        "completed": sorted(completed, key=lambda item: item["repository"].lower()),
        "failures": failures,
    }
    verification_path = metadata_path.with_name("checkout-verification.json")
    atomic_json(verification_path, verification)
    if failures:
        raise RuntimeError("%d repository checkout(s) failed; see %s" % (
            len(failures), verification_path,
        ))
    rewrite_runtime_dataset(dataset_path, checkout_root, metadata_by_key)
    print(
        "COMPLETE %d repositories, %d immutable full-source worktrees"
        % (len(completed), len(completed) * 2),
        flush=True,
    )


if __name__ == "__main__":
    main()
