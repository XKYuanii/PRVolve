"""Build the frozen 100-repository, 200-case public Python PR benchmark.

The first 50 repositories are reused byte-for-byte from the prior benchmark.
The extension is selected before model execution from a fixed, shuffled public
PR inventory.  Every extension PR has a scoreable changed line and is reviewed
against immutable full-source base/head worktrees by the preparation step.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from build_python_50_repo_canary import (  # noqa: E402
    INDEX_ERROR,
    MISSING_KEY,
    NONE_ERROR,
    WRONG_RESULT,
    WRONG_TYPE,
    public_case_pair,
    sha256_text,
    spec,
    write_jsonl,
)
from evoagent.eval.harness import load_jsonl  # noqa: E402


SEED = 20260905
EXISTING_PORTABLE = ROOT / "benchmarks" / "python_50_repo_canary_v1.jsonl"
EXISTING_RUNTIME = ROOT / "output" / "python-50-repo-canary-v1" / "runtime-dataset.jsonl"
DEFAULT_INVENTORY = ROOT / "output" / "python-50-repo-canary-v1" / "candidate-inventory-v2.json"
PORTABLE_OUTPUT = ROOT / "benchmarks" / "python_100_repo_benchmark_v1.jsonl"
OUTPUT_ROOT = ROOT / "output" / "python-100-repo-benchmark-v1"
RUNTIME_OUTPUT = OUTPUT_ROOT / "runtime-dataset.jsonl"
MANIFEST_OUTPUT = OUTPUT_ROOT / "selection-manifest.json"
METADATA_OUTPUT = OUTPUT_ROOT / "pr-metadata.json"


ASSERTION = ("CWE-754", ["CWE-248", "CWE-703"])
RESOURCE = ("CWE-404", ["CWE-459", "CWE-664", "CWE-772"])
CONFIG = ("CWE-20", ["CWE-248", "CWE-703"])


def chosen(
    repository: str,
    pull_request: int,
    slug: str,
    primary_path: str,
    needle: str,
    taxonomy: tuple[str, list[str]],
    summary: str,
    audit_tier: int,
    audit_rank: int,
    severity: str = "medium",
) -> dict:
    value = spec(
        repository, pull_request, slug, primary_path, needle, taxonomy,
        summary, severity,
    )
    value.update({"audit_tier": audit_tier, "audit_rank": audit_rank})
    return value


# Fixed before the final model run.  Rank is the deterministic position after
# sorting eligible inventory entries by repository/PR and shuffling with SEED.
SPECS = [
    chosen("wimble3/melony", 18, "melony-optional-result-backend", "melony/core/consumers.py", "assert isinstance(result_backend_saver, IAsyncResultBackendSaver)", ASSERTION, "An omitted async result backend is asserted as though it were a supplied backend.", 1, 1),
    chosen("cherrytea-dev/la_searcher_bot", 562, "la-searcher-missing-reply-markup", "src/communicate/main.py", "if isinstance(params['reply_markup'], ReplyKeyboardMarkup):", MISSING_KEY, "Messages without reply_markup are indexed directly and raise KeyError.", 1, 2),
    chosen("VirusTotal/vt-py", 213, "vt-py-datetime-utc-compat", "vt/object.py", "datetime.datetime.fromtimestamp(value, datetime.UTC)", ("CWE-248", ["CWE-703", "CWE-754"]), "Python versions without datetime.UTC raise AttributeError during timestamp conversion.", 1, 3),
    chosen("AlainSainteCroix/auto-re-agent", 11, "auto-re-agent-empty-name-tokens", "src/re_agent/backend/ghidra_bridge.py", "name = stripped.split(\"(\")[0].split()[-1] if \"(\" in stripped else target", INDEX_ERROR, "A decompiled signature beginning with a cast indexes an empty token list.", 1, 5),
    chosen("genuineknowledge/psi-agent", 137, "psi-agent-missing-file-field", "src/psi_agent/gateway/server.py", "for file_field in data.getall(\"file\"):", MISSING_KEY, "A multipart request without a file field raises instead of yielding an empty list.", 1, 6),
    chosen("frappe/press", 6862, "press-none-roles", "press/press/doctype/team/team.py", "for role in roles:", NONE_ERROR, "A missing roles collection is iterated and raises TypeError.", 1, 7),
    chosen("Ostorlab/oxo", 965, "oxo-empty-version", "src/ostorlab/cli/agent_fetcher.py", "if t_name == image and version is None:", WRONG_RESULT, "An empty version is not treated as an unspecified version, so an available image is missed.", 1, 8),
    chosen("pirlouix-dev/PDLS", 4, "pdls-modify-menu-actions", "src/main.py", "self.FocusFunction = [None, self.TypeButton0Press, self.TypeButton1Press, self.TypeButton2Press, self.SeasonButton0Press, self.SeasonButton1Press, self.SeasonButton2Press, self.SeasonButton3Press, None, self.CancelCreation, self.SaveDish]", INDEX_ERROR, "The modify menu has fewer action callbacks than focusable controls and indexes past the list.", 1, 9),
    chosen("k15z/hexchess-zero", 91, "hexchess-wrong-outcome-field", "training/trainer_loop.py", "if winner == \"white\":", WRONG_RESULT, "Gauntlet scoring reads winner even though the result contract stores outcome, so wins are not counted.", 1, 13),
    chosen("AuroraQuen/MCP", 40, "mcp-empty-argv", "home.py", "if args[0] == \"--check\":", INDEX_ERROR, "Calling the entry point without arguments indexes an empty argument list.", 1, 14),
    chosen("yt-dlp/yt-dlp", 8681, "yt-dlp-empty-facebook-entries", "yt_dlp/extractor/facebook.py", "video_info = entries[0]", INDEX_ERROR, "A Facebook attachment with no extracted entries is indexed at zero.", 1, 15),
    chosen("benevpi/conhamriver", 29, "conhamriver-string-duration", "scripts/investigate_nearby_csos.py", "\"duration_hours\": round((a.get(\"Duration\") or 0) / 60, 2) if a.get(\"Duration\") else \"\",", WRONG_TYPE, "The API Duration field can be a string and is divided as if it were numeric.", 1, 16),
    chosen("cormoran/zmk-west-commands", 7, "zmk-failed-prefix", "scripts/zmk_test.py", "or line.startswith(\"FAIL:\")", WRONG_RESULT, "Failed test output uses the FAILED prefix, so failures are not recognized.", 1, 17),
    chosen("jundot/omlx", 1747, "omlx-none-logits-processors", "omlx/scheduler.py", "logits_processors=logits_processors if logits_processors else None,", NONE_ERROR, "An empty processor list is converted to None although the downstream batch iterates it.", 1, 19),
    chosen("okorach/sonar-tools", 2401, "sonar-missing-period", "sonar/measures.py", "self.value = self.__converted_value(data.get(\"value\") or data[\"period\"].get(\"value\"))", MISSING_KEY, "Cloud measures without a period field are indexed directly.", 1, 21),
    chosen("project-bluebird/BluebirdATC", 103, "bluebird-none-route", "bluebird-dt/bluebird_dt/events/event_handler.py", "aircraft.flight_plan.route.current = list(row.route_current)", NONE_ERROR, "A nullable current route is unconditionally converted to a list.", 1, 24),
    chosen("MitchelTurner/Glasshouse", 2, "glasshouse-query-parameter-count", "src/db/transcripts.py", "cur.execute(query, (settings.lookback_days, settings.max_transcripts))", ("CWE-685", ["CWE-628", "CWE-703"]), "The fallback query has one placeholder but is always executed with two parameters.", 1, 26),
    chosen("mi-cloud-tech/vecmocon_customization", 13, "vecmocon-date-string-subtraction", "vecmocon_customization/override/quality_inspection.py", "doc.custom_delay_days = (doc.custom_submitted_date - doc.custom_due_date).days if doc.custom_due_date else 0", WRONG_TYPE, "A date object is subtracted from an unparsed date string.", 1, 27),
    chosen("Furglitch/modorganizer2-linux-installer", 1027, "modorganizer-dict-choice-index", "src/mo2-lint/step/external_resources.py", "choice = matches[0]", MISSING_KEY, "A dictionary of matches is indexed with numeric zero instead of its first key.", 1, 28),
    chosen("AcePeak/naturo", 1181, "naturo-keyerror-user-message", "naturo/cli/selector_cmd.py", "raise KeyError(f\"Selector not found: @{app_name}/{name}\")", WRONG_RESULT, "KeyError stringification adds quotes to a user-facing selector error message.", 1, 29, "low"),
    chosen("godon-dev/godon-breeders", 142, "godon-single-column-row", "engine/detection_coordinator.py", "logger.info(f\"Refreshed baseline params from DB (best value: {row[1]:.4f})\")", INDEX_ERROR, "A single-column database result is logged by indexing a nonexistent second column.", 1, 31),
    chosen("glupta/token-optimizer", 6, "token-optimizer-none-store", "skills/token-optimizer/scripts/read_cache.py", "files = store.get_all_file_entries()", NONE_ERROR, "A failed cache connection returns None and is dereferenced.", 1, 32),
    chosen("strands-labs/robots-sim", 141, "robots-sim-scalar-rgba", "strands_robots_sim/isaac/simulation.py", "except (RuntimeError, ValueError, AttributeError, TypeError) as e:", INDEX_ERROR, "A scalar RTX warm-up buffer raises IndexError outside the render recovery handler.", 1, 33),
    chosen("steveyminecraft/aws-account-audit", 15, "aws-audit-none-waf-error", "aws_account_audit/inventory.py", "errors.extend(waf_errors)", NONE_ERROR, "The optional WAF error is extended as an iterable even when it is None.", 1, 34),
    chosen("frappe/erpnext", 56661, "erpnext-none-bulk-args", "erpnext/utilities/bulk_transaction.py", "args = frappe._dict(frappe.parse_json(args))", NONE_ERROR, "A missing argument payload is converted into a mapping without a fallback.", 1, 35),
    chosen("atlassian-api/atlassian-python-api", 1641, "bitbucket-nested-admin-user", "atlassian/bitbucket/__init__.py", "repo_administrators.append(user)", WRONG_TYPE, "The API wrapper returns administrator envelopes instead of their nested user objects.", 1, 37),
    chosen("NovaSky-AI/SkyRL", 1608, "skyrl-mamba-missing-mlp", "skyrl/backends/skyrl_train/distributed/megatron/megatron_utils.py", "if hasattr(layer.mlp, \"router\"):", NONE_ERROR, "Mamba layers without an mlp attribute are dereferenced during router freezing.", 1, 38),
    chosen("vllm-ascend/vllm_ascend_dashboard", 79, "dashboard-null-email", "backend/app/schemas/__init__.py", "email: str = Field(..., max_length=100)", WRONG_TYPE, "The update schema requires an email even though partial updates may legitimately omit it.", 1, 39),
    chosen("StarGazer1995/stargazing-place-finder", 53, "stargazing-none-pollution", "src/gis_service/parsers.py", "key=lambda x: x[\"pollution_info\"].brightness,", NONE_ERROR, "Sorting dereferences nullable pollution metadata.", 1, 40),
    chosen("sgl-project/sglang", 18500, "sglang-flatten-hidden-scale", "python/sglang/srt/layers/moe/fused_moe_triton/layer.py", "hidden_states_scale=hs_scale_linear.view(torch.float8_e4m3fn).flatten(),", WRONG_RESULT, "Flattening the per-token hidden-state scale destroys the shape required by FP4 MoE autotuning.", 1, 42),
    chosen("Shiroko253/Reimubot", 4, "reimubot-missing-colon", "Reimu.py", "fortune_type = result_text.split(\"\\n\")[0].split(\":\")[1].strip()", INDEX_ERROR, "A fortune response without a colon is indexed past the split result.", 1, 43),
    chosen("n24q02m/mnemo-mcp", 879, "mnemo-none-topic-suggestion", "src/mnemo_mcp/server.py", "closest = difflib.get_close_matches(topic, list(valid_topics.keys()), n=1)", NONE_ERROR, "A missing topic is passed to difflib as though it were a string.", 1, 44),
    chosen("akunnft-ux/auto-post-soal-matematika", 3, "auto-post-history-missing-soal", "main.py", "f\"- {(h['soal'] if isinstance(h, dict) else h)[:80]}\"", MISSING_KEY, "Non-question history dictionaries do not contain the directly indexed soal key.", 1, 45),
    chosen("Bright0505/mcp-db", 8, "mcp-db-optional-asyncpg-exception", "src/database/async_connectors.py", "except (asyncio.TimeoutError, asyncpg.exceptions.QueryCanceledError):", ("CWE-248", ["CWE-703", "CWE-754"]), "When asyncpg is unavailable, evaluating its exception class raises AttributeError while handling a timeout.", 1, 47),
    chosen("candidelabs/voltaire", 57, "voltaire-arbitrum-fee-tasks", "voltaire_bundler/execution_endpoint.py", "if self.chain_id == 999 or self.chain_id == 998:  # HyperEVM", INDEX_ERROR, "Arbitrum skips the priority-fee task but is not handled as a one-task chain, so the result list is indexed past its end.", 1, 50),
    chosen("shosho-chang/nakama", 902, "nakama-wrong-calendar-event-field", "thousand_sunny/routers/bridge_weekly.py", "if entry.is_linked and entry.calendar_event_id:", WRONG_RESULT, "Deletion checks a nonexistent calendar_event_id field instead of the stored event_id.", 1, 53),
    chosen("amami-cell/susabiyu-ig-auto", 18, "susabiyu-missing-category", "susabiyu-remotion/fetch_drive_photos.py", "print(\"DL:\", dest, \"| [%s]\" % f[\"cat\"], \"caption:\", cap)", MISSING_KEY, "Approved items without a category fail during logging before publication.", 1, 54),
    chosen("kimbonnie91-hue/bobobobo", 4, "bobobobo-missing-send-id", "push_analytics_dashboard.py", "msg_sub = msg_df[[c for c in msg_cols if c in msg_df.columns]].copy()", MISSING_KEY, "An empty or unrecognized message sheet proceeds without the send_id column required by the join.", 1, 55),
    chosen("MIJUNG-HEZO/HEZO-Agent", 254, "hezo-null-html-name", "agents/build/renderer/html_renderer.py", "h3.string = svc.get(\"name\", \"\")", NONE_ERROR, "An explicitly null service name is assigned to an HTML text node and raises TypeError.", 1, 57),
    chosen("ziutus/ai_assistant_lenie", 112, "lenie-short-https-prefix", "backend/library/lenie_markdown.py", "if text[i] == \"h\" and text[i+1] == \"t\" and text[i+2] == \"t\" and text[i+3] == \"p\" and text[i+4] == \"s\" and \\", INDEX_ERROR, "The URL detector reads seven characters past the current position without checking remaining length.", 1, 58),
    chosen("AlexandreCamerini/b3agente", 1, "b3agente-empty-ttl-env", "server/app/auth.py", "_SESSION_TTL_DAYS = int(os.environ.get(\"B3_SESSION_TTL_DAYS\", \"90\"))", CONFIG, "An explicitly empty TTL environment variable is passed to int and prevents startup.", 1, 60),
    chosen("croweykid/ephemeraldaddy", 144, "ephemeraldaddy-pyside-scroll-enum", "ephemeraldaddy/gui/app.py", "self.scrollToItem(item, QAbstractItemView.PositionAtCenter)", ("CWE-248", ["CWE-703", "CWE-754"]), "PySide6 no longer exposes PositionAtCenter directly on QAbstractItemView.", 1, 61),
    chosen("NathanNeurotic/PS2-Servers", 18, "ps2-none-sys-argv", "launcher/main.py", "argv = list(getattr(sys, \"argv\", [])[1:] if argv is None else argv)", NONE_ERROR, "An explicitly None sys.argv value is sliced during startup.", 1, 62),
    chosen("TheFab21/ha-samsungtv-smart", 107, "samsungtv-missing-options", "custom_components/samsungtv_smart/media_player.py", "option = self._entry_data[DATA_OPTIONS].get(param)", MISSING_KEY, "A transient reload state omits DATA_OPTIONS and the media player indexes it directly.", 2, 3),
    chosen("hackolite/SoccerAnalytics", 10, "soccer-empty-control-history", "main.py", "team_ball_control.append(team_ball_control[-1])", INDEX_ERROR, "The first frame without an assignment copies from an empty control history.", 2, 4),
    chosen("tamzrod/Librarian", 48, "librarian-none-document-text", "ingestion/indexer.py", "text = document.get('text', '')", NONE_ERROR, "An explicitly null document text bypasses the default and is processed as a string.", 2, 5),
    chosen("sinotca529/ai-chat-tui", 1, "ai-chat-empty-stream-choices", "infrastructure/api_handler.py", "delta = chunk.choices[0].delta.content", INDEX_ERROR, "Streaming APIs can emit chunks with no choices, which are indexed at zero.", 2, 6),
    chosen("0bnoxide/AutoGIS", 63, "autogis-wrong-capability-key", "autogis/adapters/cli.py", "_guard(\"LOCAL\")", MISSING_KEY, "The import command passes a runtime name where the capability registry expects the tool name.", 2, 7),
    chosen("hackolite/CV_Studio", 759, "cv-studio-malformed-link-alias", "node_editor/node_main.py", "source_type = source.split(\":\")[2]", INDEX_ERROR, "A missing or malformed node alias is split and indexed without validating its shape.", 2, 9),
    chosen("Romeromarcov/fabrica-software", 99, "fabrica-empty-static-interrupt", "graph_project.py", "return build_project_graph().compile(", INDEX_ERROR, "A static pre-node interrupt emits an empty payload that the approval loop later indexes at zero.", 2, 15),
]


REJECTIONS = {
    "joewpb/legalclear#7": "broad-exception-fix",
    "wasserth/TotalSegmentator#567": "multiple-independent-production-fixes",
    "mikosavolainen/sondehub-alert-v2#11": "multiple-independent-behavior-changes",
    "veritasfuji-japan/veritas_os#499": "original-line-already-handles-null-client",
    "abrignoni/iLEAPP#1381": "causal-risk-line-not-present-in-reversal",
    "ArthurBernard/Fynance#201": "multiple-independent-scoreable-defects",
    "DINA-community/DDDC-Netbox-plugin#51": "fix-semantics-not-unambiguous",
    "aiidalab/aiidalab-widgets-base#759": "causal-contract-not-local-enough",
    "Ostorlab/agent_whatweb#157": "excluded-network-security-domain",
    "pumpingstationone/deepharbor#299": "multiple-independent-behavior-changes",
    "ansible/ansible-ai-connect-service#947": "causal-risk-line-not-unambiguous",
    "pylint-dev/astroid#3072": "add-only-guard-has-no-scoreable-regression-line",
    "ubclaunchpad/rocket2#521": "add-only-guard-has-no-scoreable-regression-line",
    "danniesidequestmaxxing/geo-deal-sourcing#5": "add-only-guard-has-no-scoreable-regression-line",
    "mlcommons/storage#400": "reversal-adds-no-causal-risk-line",
    "cohere-ai/cohere-python#778": "add-only-guard-has-no-scoreable-regression-line",
    "fl4p/batmon-ha#373": "add-only-guard-has-no-scoreable-regression-line",
    "ASAC-DE-bigkk/ASAC-DAG#33": "ambiguous-duplicate-causal-lines",
    "NVIDIA/TensorRT-LLM#14267": "ambiguous-duplicate-causal-lines",
}


def candidate_pool(inventory: dict, existing_repositories: set[str], tier: int) -> list[dict]:
    count = 1 if tier == 1 else 2
    values = [
        item for item in inventory["candidates"]
        if item["repository"].lower() not in existing_repositories
        and len(item["production_python_paths"]) == count
        and item["additions"] + item["deletions"] <= 120
        and not any(term in item["repository"].lower() for term in ("dummy", "bug-test", "sample"))
    ]
    values.sort(key=lambda item: (item["repository"].lower(), int(item["pull_request"])))
    random.Random(SEED).shuffle(values)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--portable-output", default=str(PORTABLE_OUTPUT))
    parser.add_argument("--runtime-output", default=str(RUNTIME_OUTPUT))
    parser.add_argument("--manifest-output", default=str(MANIFEST_OUTPUT))
    args = parser.parse_args()

    if len(SPECS) != 50 or len({item["repository"].lower() for item in SPECS}) != 50:
        raise ValueError("extension must contain 50 unique repositories")
    existing_portable = load_jsonl(str(EXISTING_PORTABLE))
    existing_runtime = load_jsonl(str(EXISTING_RUNTIME))
    existing_repositories = {case["repository"].lower() for case in existing_portable}
    if len(existing_repositories) != 50:
        raise ValueError("existing benchmark must contain 50 repositories")

    inventory_path = Path(args.inventory).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    by_key = {
        (item["repository"], int(item["pull_request"])): item
        for item in inventory["candidates"]
    }
    pools = {
        1: candidate_pool(inventory, existing_repositories, 1),
        2: candidate_pool(inventory, existing_repositories, 2),
    }
    ranks = {
        tier: {
            (item["repository"], int(item["pull_request"])): index
            for index, item in enumerate(values, 1)
        }
        for tier, values in pools.items()
    }

    added: list[dict] = []
    selected_manifest = []
    for index, item in enumerate(SPECS, 1):
        key = (item["repository"], int(item["pull_request"]))
        candidate = by_key.get(key)
        if candidate is None:
            raise ValueError("candidate inventory is missing %s#%d" % key)
        actual_rank = ranks[item["audit_tier"]].get(key)
        if actual_rank != item["audit_rank"]:
            raise ValueError("audit rank drift for %s: %s != %s" % (key, actual_rank, item["audit_rank"]))
        pair = public_case_pair(candidate, item, index)
        for case in pair:
            case["repository_root"] = "/app/output/python-100-repo-benchmark-v1/checkouts/" + case["id"]
            case["source"].update({
                "sampling_seed": SEED,
                "sampling_tier": item["audit_tier"],
                "sampling_rank": item["audit_rank"],
            })
        added.extend(pair)
        selected_manifest.append({
            **item,
            "public_url": candidate["public_url"],
            "title": candidate["title"],
            "has_python_regression_test": bool(candidate.get("test_python_paths")),
            "fix_diff_sha256": pair[0]["source"]["original_fix_diff_sha256"],
            "risk_case_id": pair[0]["id"],
            "clean_case_id": pair[1]["id"],
        })

    portable = existing_portable + added
    runtime = copy.deepcopy(existing_runtime) + copy.deepcopy(added)
    if len(portable) != 200 or len(runtime) != 200:
        raise ValueError("expected exactly 200 cases")
    if len({case["repository"].lower() for case in portable}) != 100:
        raise ValueError("expected exactly 100 repositories")
    if sum(bool(case["expected_findings"]) for case in portable) != 100:
        raise ValueError("expected 100 regression and 100 clean cases")

    portable_path = Path(args.portable_output).resolve()
    runtime_path = Path(args.runtime_output).resolve()
    write_jsonl(portable_path, portable)
    write_jsonl(runtime_path, runtime)
    pool_projection = [
        {
            "tier": tier,
            "rank": index,
            "repository": item["repository"],
            "pull_request": int(item["pull_request"]),
        }
        for tier, values in pools.items()
        for index, item in enumerate(values, 1)
    ]
    manifest = {
        "schema_version": 1,
        "name": "python-100-repo-benchmark-v1",
        "frozen_system_commit": "a0e1760",
        "cases": 200,
        "repositories": 100,
        "pull_requests": 100,
        "regressions": 100,
        "clean_fixes": 100,
        "reused_cases": 100,
        "reused_repositories": 50,
        "new_cases": 100,
        "new_repositories": 50,
        "inventory": str(inventory_path),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "portable_dataset": str(portable_path),
        "portable_dataset_sha256": hashlib.sha256(portable_path.read_bytes()).hexdigest(),
        "runtime_dataset": str(runtime_path),
        "runtime_dataset_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
        "pull_request_metadata": str(METADATA_OUTPUT),
        "selection_policy": {
            "source": "merged public GitHub pull requests",
            "language": "Python production changes",
            "sampling_seed": SEED,
            "sampling_frame": "repositories absent from the frozen 50-repository benchmark",
            "tier_1": "exactly one production Python file and at most 120 added/deleted lines",
            "tier_2": "exactly two production Python files and at most 120 added/deleted lines; retain only files causally tied to the target defect",
            "shared_exclusions": [
                "documentation, dependency, formatting, and security-domain changes",
                "repository names explicitly identifying dummy, bug-test, or sample data",
                "no unambiguous changed risk line after mechanical fix reversal",
                "multiple unrelated production behavior changes",
            ],
            "model_independence": "selection and labels frozen before the final EvoAgent run",
            "execution_policy": "each case requires an immutable full-source Git worktree at the exact PR base or head SHA",
        },
        "pool_projection_sha256": sha256_text(json.dumps(pool_projection, sort_keys=True)),
        "pool_projection": pool_projection,
        "rejections": REJECTIONS,
        "selected": selected_manifest,
    }
    manifest_path = Path(args.manifest_output).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("built 200 cases across 100 repositories (50 reused, 50 new)")


if __name__ == "__main__":
    main()
