#!/usr/bin/env python3
"""Replay imported Claude transcript days through Quaid extraction + janitor.

This is a stress/migration utility, not a scored benchmark lane. It reuses the
same runtime extraction and janitor helpers as the benchmark harness so the
results are representative of real product behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval import run_production_benchmark as rpb

_CHECKPOINT_QUAID_DIR = (
    Path(os.environ.get("BENCHMARK_PLUGIN_DIR", "")).expanduser()
    if os.environ.get("BENCHMARK_PLUGIN_DIR", "").strip()
    else (Path.home() / "quaid" / "benchmark-checkpoint" / "modules" / "quaid")
)
os.environ.setdefault("BENCHMARK_PLUGIN_DIR", str(_CHECKPOINT_QUAID_DIR))
rpb._QUAID_DIR = _CHECKPOINT_QUAID_DIR
rpb._MEMORY_GRAPH_SCRIPT = rpb._resolve_quaid_script("memory_graph.py", "datastore/memorydb/memory_graph.py")
rpb._JANITOR_SCRIPT = rpb._resolve_quaid_script("janitor.py", "core/lifecycle/janitor.py")
rpb._EXTRACT_SCRIPT = rpb._resolve_quaid_script("extract.py", "ingest/extract.py")

_TABLE_COUNT_FALLBACKS: Dict[str, tuple[str, ...]] = {
    # Vec virtual tables require the sqlite-vec module; plain sqlite3 telemetry
    # readers should fall back to the companion rowid mirror when the module
    # is unavailable. Keep this mapping schema-oriented so migration tooling can
    # reuse the same fallback logic.
    "vec_nodes": ("vec_nodes_rowids",),
    "vec_doc_chunks": ("vec_doc_chunks_rowids",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/imported-claude-dev-first2/manifest.json"),
        help="Manifest describing imported Claude day JSONL files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Fresh workspace/results directory",
    )
    parser.add_argument(
        "--backend",
        choices=["claude-code", "oauth", "api"],
        default="oauth",
        help="Runtime backend to use for extraction/janitor",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Deep extraction model",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="Limit to the first N imported days (0 = all)",
    )
    parser.add_argument(
        "--rolling",
        action="store_true",
        help="Use rolling staged extraction per imported day",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=8000,
        help="Rolling chunk token budget when --rolling is enabled",
    )
    parser.add_argument(
        "--chunk-max-lines",
        type=int,
        default=144,
        help="Rolling chunk line cap when --rolling is enabled",
    )
    parser.add_argument(
        "--extract-timeout",
        type=int,
        default=7200,
        help="Per-day extraction timeout in seconds",
    )
    parser.add_argument(
        "--skip-janitor",
        action="store_true",
        help="Skip janitor between imported days",
    )
    parser.add_argument(
        "--janitor-timeout",
        type=int,
        default=3600,
        help="Per-day janitor timeout in seconds",
    )
    parser.add_argument(
        "--repair-summary",
        action="store_true",
        help="Rewrite extract telemetry in an existing imported-Claude summary from raw rolling metrics",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    days = payload.get("days")
    if not isinstance(days, list) or not days:
        raise RuntimeError(f"Manifest has no days: {path}")
    return payload


def _ensure_fresh_results_dir(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise RuntimeError(f"Results dir already exists and is not empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _summary_path(results_dir: Path) -> Path:
    return results_dir / "logs" / "imported_claude_history_summary.json"


def _write_summary_payload(summary_path: Path, payload: Dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2))


def _rewrite_workspace_for_claude_history(workspace: Path) -> None:
    for rel in ["projects/recipe-app", "projects/portfolio-site"]:
        target = workspace / rel
        if target.exists():
            shutil.rmtree(target)
    (workspace / "projects" / "quaid").mkdir(parents=True, exist_ok=True)

    config_path = workspace / "config" / "memory.json"
    config = json.loads(config_path.read_text())
    config["users"]["defaultOwner"] = "solomon"
    config["users"]["identities"] = {
        "solomon": {
            "channels": {"cli": ["*"]},
            "speakers": ["Solomon", "The user", "User"],
            "personNodeName": "Solomon",
        }
    }
    config["projects"]["definitions"] = {
        "quaid": {
            "label": "Quaid",
            "homeDir": "projects/quaid/",
            "sourceRoots": ["projects/quaid/"],
            "autoIndex": True,
            "patterns": ["*.md"],
            "exclude": [".git/"],
            "description": "Quaid runtime and operational reference",
        }
    }
    config_path.write_text(json.dumps(config, indent=2))

    (workspace / "SOUL.md").write_text(
        "# Soul\n\n"
        "Imported Claude conversation stress workspace.\n"
    )
    (workspace / "USER.md").write_text(
        "# User Profile\n\n"
        "Imported user history for Claude dev session replay.\n"
    )
    (workspace / "ENVIRONMENT.md").write_text(
        "# Shared Environment\n\n"
        "Imported historical Claude transcript context.\n"
    )
    (workspace / "IDENTITY.md").write_text("# Identity\n\nName: Assistant\n")
    domain_rows = rpb._load_active_domains(workspace)
    root_tools = rpb._inject_domains_into_tools_md(rpb._load_quaid_tools_template(), domain_rows)
    (workspace / "TOOLS.md").write_text(root_tools.rstrip() + "\n", encoding="utf-8")
    rpb._seed_quaid_project_docs(workspace)


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
        (table_name,),
    ).fetchone()
    return bool(row)


def _direct_table_count(conn: sqlite3.Connection, table_name: str) -> int:
    return int(conn.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0])


def _safe_table_count(conn: sqlite3.Connection, table_name: str) -> int:
    if not _table_exists(conn, table_name):
        return 0
    try:
        return _direct_table_count(conn, table_name)
    except sqlite3.OperationalError as exc:
        if "no such module" not in str(exc).lower():
            raise
        for fallback_table in _TABLE_COUNT_FALLBACKS.get(table_name, ()):
            if _table_exists(conn, fallback_table):
                return _direct_table_count(conn, fallback_table)
        return 0


def _count_map(conn: sqlite3.Connection, query: str) -> Dict[str, int]:
    rows = conn.execute(query).fetchall()
    out: Dict[str, int] = {}
    for key, count in rows:
        label = str(key) if key is not None else "null"
        out[label] = int(count)
    return out


def _db_stats(workspace: Path) -> Dict[str, Any]:
    db_path = workspace / "data" / "memory.db"
    if not db_path.exists():
        return {
            "db_path": str(db_path),
            "exists": False,
        }
    conn = sqlite3.connect(str(db_path))
    try:
        total_nodes = int(conn.execute("SELECT count(*) FROM nodes").fetchone()[0])
        total_edges = int(conn.execute("SELECT count(*) FROM edges").fetchone()[0])
        try:
            by_type = _count_map(conn, "SELECT type, count(*) FROM nodes GROUP BY type")
        except sqlite3.OperationalError:
            by_type = _count_map(conn, "SELECT category, count(*) FROM nodes GROUP BY category")
        by_status = _count_map(conn, "SELECT status, count(*) FROM nodes GROUP BY status")
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        freelist_count = int(conn.execute("PRAGMA freelist_count").fetchone()[0])
        table_counts = {
            "nodes_fts": _safe_table_count(conn, "nodes_fts"),
            "vec_nodes": _safe_table_count(conn, "vec_nodes"),
            "vec_nodes_chunks": _safe_table_count(conn, "vec_nodes_chunks"),
            "vec_doc_chunks": _safe_table_count(conn, "vec_doc_chunks"),
            "embedding_cache": _safe_table_count(conn, "embedding_cache"),
            "dedup_log": _safe_table_count(conn, "dedup_log"),
            "recall_log": _safe_table_count(conn, "recall_log"),
            "session_logs": _safe_table_count(conn, "session_logs"),
            "session_log_chunks": _safe_table_count(conn, "session_log_chunks"),
            "doc_chunks": _safe_table_count(conn, "doc_chunks"),
            "project_definitions": _safe_table_count(conn, "project_definitions"),
            "janitor_runs": _safe_table_count(conn, "janitor_runs"),
        }
        return {
            "db_path": str(db_path),
            "exists": True,
            "file_size_bytes": int(db_path.stat().st_size),
            "page_count": page_count,
            "page_size": page_size,
            "freelist_count": freelist_count,
            "nodes": total_nodes,
            "edges": total_edges,
            "fact_nodes": int(by_type.get("Fact", 0)),
            "event_nodes": int(by_type.get("Event", 0)),
            "preference_nodes": int(by_type.get("Preference", 0)),
            "active_nodes": int(by_status.get("active", 0)),
            "pending_nodes": int(by_status.get("pending", 0)),
            "flagged_nodes": int(by_status.get("flagged", 0)),
            "by_type": by_type,
            "by_status": by_status,
            "tables": table_counts,
        }
    finally:
        conn.close()


def _diff_count_map(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, int]:
    keys = sorted(set(before) | set(after))
    return {
        str(key): int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
        for key in keys
    }


def _db_delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "file_size_bytes": int(after.get("file_size_bytes", 0) or 0) - int(before.get("file_size_bytes", 0) or 0),
        "page_count": int(after.get("page_count", 0) or 0) - int(before.get("page_count", 0) or 0),
        "freelist_count": int(after.get("freelist_count", 0) or 0) - int(before.get("freelist_count", 0) or 0),
        "nodes": int(after.get("nodes", 0) or 0) - int(before.get("nodes", 0) or 0),
        "edges": int(after.get("edges", 0) or 0) - int(before.get("edges", 0) or 0),
        "fact_nodes": int(after.get("fact_nodes", 0) or 0) - int(before.get("fact_nodes", 0) or 0),
        "event_nodes": int(after.get("event_nodes", 0) or 0) - int(before.get("event_nodes", 0) or 0),
        "preference_nodes": int(after.get("preference_nodes", 0) or 0) - int(before.get("preference_nodes", 0) or 0),
        "active_nodes": int(after.get("active_nodes", 0) or 0) - int(before.get("active_nodes", 0) or 0),
        "pending_nodes": int(after.get("pending_nodes", 0) or 0) - int(before.get("pending_nodes", 0) or 0),
        "flagged_nodes": int(after.get("flagged_nodes", 0) or 0) - int(before.get("flagged_nodes", 0) or 0),
        "by_type": _diff_count_map(
            before.get("by_type", {}) if isinstance(before.get("by_type"), dict) else {},
            after.get("by_type", {}) if isinstance(after.get("by_type"), dict) else {},
        ),
        "by_status": _diff_count_map(
            before.get("by_status", {}) if isinstance(before.get("by_status"), dict) else {},
            after.get("by_status", {}) if isinstance(after.get("by_status"), dict) else {},
        ),
        "tables": _diff_count_map(
            before.get("tables", {}) if isinstance(before.get("tables"), dict) else {},
            after.get("tables", {}) if isinstance(after.get("tables"), dict) else {},
        ),
    }


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_rolling_flush_metric(metric_path: Any, session_id: Optional[str]) -> Dict[str, Any]:
    if not metric_path:
        return {}
    path = Path(str(metric_path)).expanduser()
    if not path.exists():
        return {}
    latest: Dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("event") != "rolling_flush":
                    continue
                if session_id and str(row.get("session_id", "")) != session_id:
                    continue
                latest = row
    except OSError:
        return {}
    return latest


def _extract_telemetry(extract_result: Dict[str, Any], *, session_id: Optional[str] = None) -> Dict[str, Any]:
    rolling_metric_path = extract_result.get("rolling_metric_path")
    flush_metric = _load_rolling_flush_metric(rolling_metric_path, session_id=session_id)

    def _metric_int(name: str) -> int:
        if name in extract_result and extract_result.get(name) is not None:
            return int(extract_result.get(name, 0) or 0)
        return int(flush_metric.get(name, 0) or 0)

    return {
        "facts_extracted": _metric_int("facts_extracted"),
        "facts_stored": _metric_int("facts_stored"),
        "facts_skipped": _metric_int("facts_skipped"),
        "edges_created": _metric_int("edges_created"),
        "rolling_batches": _metric_int("rolling_batches"),
        "root_chunks": _metric_int("root_chunks"),
        "leaf_chunks": _metric_int("leaf_chunks"),
        "split_events": _metric_int("split_events"),
        "split_child_chunks": _metric_int("split_child_chunks"),
        "extract_wall_seconds": _float_or_none(extract_result.get("extract_wall_seconds")),
        "publish_wall_seconds": _float_or_none(extract_result.get("publish_wall_seconds")),
        "flush_wall_seconds": _float_or_none(extract_result.get("flush_wall_seconds")),
        "signal_to_publish_seconds": _float_or_none(extract_result.get("signal_to_publish_seconds")),
        "rolling_stage_wall_seconds": _float_or_none(extract_result.get("rolling_stage_wall_seconds")),
        "rolling_driver_stage_wall_seconds": _float_or_none(
            extract_result.get("rolling_driver_stage_wall_seconds")
        ),
        "rolling_driver_flush_wall_seconds": _float_or_none(
            extract_result.get("rolling_driver_flush_wall_seconds")
        ),
        "metric_path": rolling_metric_path,
        "project_logs": dict(extract_result.get("project_log_metrics", {}) or {}),
        "dedup": {
            "hash_exact_hits": _metric_int("dedup_hash_exact_hits"),
            "scanned_rows": _metric_int("dedup_scanned_rows"),
            "gray_zone_rows": _metric_int("dedup_gray_zone_rows"),
            "llm_checks": _metric_int("dedup_llm_checks"),
            "llm_same_hits": _metric_int("dedup_llm_same_hits"),
            "llm_different_hits": _metric_int("dedup_llm_different_hits"),
            "fallback_reject_hits": _metric_int("dedup_fallback_reject_hits"),
            "auto_reject_hits": _metric_int("dedup_auto_reject_hits"),
            "vec_query_count": _metric_int("dedup_vec_query_count"),
            "vec_candidates_returned": _metric_int("dedup_vec_candidates_returned"),
            "vec_candidate_limit": _metric_int("dedup_vec_candidate_limit"),
            "vec_limit_hits": _metric_int("dedup_vec_limit_hits"),
            "fts_query_count": _metric_int("dedup_fts_query_count"),
            "fts_candidates_returned": _metric_int("dedup_fts_candidates_returned"),
            "fts_candidate_limit": _metric_int("dedup_fts_candidate_limit"),
            "fts_limit_hits": _metric_int("dedup_fts_limit_hits"),
            "fallback_scan_count": _metric_int("dedup_fallback_scan_count"),
            "fallback_candidates_returned": _metric_int("dedup_fallback_candidates_returned"),
            "token_prefilter_terms": _metric_int("dedup_token_prefilter_terms"),
            "token_prefilter_skips": _metric_int("dedup_token_prefilter_skips"),
        },
        "embedding_cache": {
            "requested": _metric_int("embedding_cache_requested"),
            "unique": _metric_int("embedding_cache_unique"),
            "hits": _metric_int("embedding_cache_hits"),
            "warmed": _metric_int("embedding_cache_warmed"),
            "failed": _metric_int("embedding_cache_failed"),
            "edge_requested": _metric_int("edge_embedding_cache_requested"),
            "edge_unique": _metric_int("edge_embedding_cache_unique"),
            "edge_hits": _metric_int("edge_embedding_cache_hits"),
            "edge_warmed": _metric_int("edge_embedding_cache_warmed"),
            "edge_failed": _metric_int("edge_embedding_cache_failed"),
        },
    }


def _janitor_telemetry(janitor_stats: Dict[str, Any]) -> Dict[str, Any]:
    metrics = janitor_stats.get("metrics", {}) if isinstance(janitor_stats, dict) else {}
    task_durations = metrics.get("task_durations", {}) if isinstance(metrics, dict) else {}
    applied_changes = janitor_stats.get("applied_changes", {}) if isinstance(janitor_stats, dict) else {}
    return {
        "success": bool(janitor_stats.get("success")) if isinstance(janitor_stats, dict) else False,
        "last_run": janitor_stats.get("last_run") if isinstance(janitor_stats, dict) else None,
        "total_duration_seconds": _float_or_none(metrics.get("total_duration_seconds")),
        "task_durations": {
            str(key): float(value)
            for key, value in task_durations.items()
            if value is not None
        },
        "applied_changes": {
            str(key): value
            for key, value in applied_changes.items()
            if not isinstance(value, dict)
        },
    }


def _load_latest_janitor_stats(workspace: Path) -> Dict[str, Any]:
    path = workspace / "benchrunner" / "logs" / "janitor-stats.json"
    if not path.exists():
        path = workspace / "logs" / "janitor-stats.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _repair_summary_extract_telemetry(results_dir: Path) -> Dict[str, Any]:
    summary_path = _summary_path(results_dir)
    if not summary_path.exists():
        raise RuntimeError(f"Summary file not found: {summary_path}")
    payload = json.loads(summary_path.read_text())
    rows = payload.get("days")
    if not isinstance(rows, list):
        raise RuntimeError(f"Summary has no day rows: {summary_path}")
    repaired = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        session_id = str(row.get("session_id", "") or "")
        extract_result = row.get("extract_result", {})
        if not isinstance(extract_result, dict):
            continue
        telemetry = row.get("telemetry")
        if not isinstance(telemetry, dict):
            telemetry = {}
            row["telemetry"] = telemetry
        telemetry["extract"] = _extract_telemetry(extract_result, session_id=session_id or None)
        repaired += 1
    payload["summary_repaired_from_metrics"] = True
    _write_summary_payload(summary_path, payload)
    return {
        "summary_path": str(summary_path),
        "days_repaired": repaired,
    }


def _clear_stale_janitor_lock(workspace: Path) -> bool:
    lock_path = workspace / "benchrunner" / "data" / ".janitor.lock"
    if not lock_path.exists():
        return False
    try:
        proc = subprocess.run(
            ["ps", "-Ao", "command="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commands = proc.stdout.splitlines() if proc.returncode == 0 else []
        workspace_text = str(workspace)
        janitor_running = any("janitor.py" in cmd and workspace_text in cmd for cmd in commands)
    except Exception:
        janitor_running = False
    if janitor_running:
        return False
    # No janitor process is using this workspace; treat the lock as stale.
    lock_path.unlink(missing_ok=True)
    return True


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if args.repair_summary:
        repaired = _repair_summary_extract_telemetry(results_dir)
        print(f"Repaired summary telemetry for {repaired['days_repaired']} day(s)")
        print(f"Summary: {repaired['summary_path']}")
        return

    manifest = _load_manifest(args.manifest.expanduser().resolve())
    days: List[Dict[str, Any]] = list(manifest["days"])
    if args.max_days > 0:
        days = days[: args.max_days]
    if not days:
        raise RuntimeError("No imported Claude days selected")

    _ensure_fresh_results_dir(results_dir)

    backend = "oauth" if args.backend == "api" else args.backend
    rpb._BACKEND = backend
    if backend != "claude-code":
        credential = (
            os.environ.get("BENCHMARK_ANTHROPIC_OAUTH_TOKEN", "").strip()
            or os.environ.get("ANTHROPIC_API_KEY", "").strip()
            or (rpb._load_claude_code_oauth_token() or "").strip()
        )
        if not credential:
            raise RuntimeError(
                "No Anthropic credential found. Set BENCHMARK_ANTHROPIC_OAUTH_TOKEN/ANTHROPIC_API_KEY "
                "or ensure ~/.claude/.credentials.json contains a Claude OAuth token."
            )
        os.environ.setdefault("BENCHMARK_ANTHROPIC_OAUTH_TOKEN", credential)
        os.environ.setdefault("ANTHROPIC_API_KEY", credential)
    rpb.setup_workspace(results_dir, extraction_model=args.model)
    _rewrite_workspace_for_claude_history(results_dir)
    if args.rolling:
        rpb._set_workspace_obd_capture_limits(
            results_dir,
            chunk_tokens=args.chunk_tokens,
            chunk_max_lines=args.chunk_max_lines,
        )

    summary_rows: List[Dict[str, Any]] = []
    for idx, day in enumerate(days, start=1):
        session_id = str(day["session_id"])
        operational_day = str(day["operational_day"])
        session_file = Path(day["path"]).expanduser().resolve()
        db_before = _db_stats(results_dir)
        env = rpb._benchmark_env(results_dir, "ingest")
        env = rpb._with_quaid_now(env, operational_day)
        env["QUAID_EXTRACT_WALL_TIMEOUT"] = str(int(args.extract_timeout))

        print("=" * 60)
        print(f"Imported Claude Day {idx}/{len(days)}: {session_id} ({operational_day})")
        print(f"  Transcript: {session_file}")
        print(f"  Rolling: {'yes' if args.rolling else 'no'}")

        if args.rolling:
            extract_result = rpb._run_runtime_rolling_obd_extract(
                workspace=results_dir,
                env=env,
                session_file=session_file,
                session_id=session_id,
                chunk_tokens=int(args.chunk_tokens),
                chunk_max_lines=int(args.chunk_max_lines) if args.chunk_max_lines else None,
                timeout_seconds=int(args.extract_timeout),
            )
        else:
            extract_result = rpb._run_runtime_extract_jsonl(
                workspace=results_dir,
                env=env,
                session_file=session_file,
                owner_id="solomon",
                label=f"Imported Claude Day {idx}",
                session_id=session_id,
                timeout_seconds=int(args.extract_timeout),
            )

        janitor_stats: Dict[str, Any] = {}
        if not args.skip_janitor:
            cleared = _clear_stale_janitor_lock(results_dir)
            if cleared:
                print("  Cleared stale janitor lock before retry")
            rpb.run_janitor(results_dir, timeout_seconds=int(args.janitor_timeout))
            rpb.verify_post_janitor(results_dir)
            janitor_stats = _load_latest_janitor_stats(results_dir)
            if not janitor_stats or not bool(janitor_stats.get("success")):
                raise RuntimeError(
                    f"Janitor failed for {session_id}: {json.dumps(janitor_stats, indent=2)[:600]}"
                )

        db_after = _db_stats(results_dir)
        extract_summary = _extract_telemetry(extract_result, session_id=session_id)
        janitor_summary = _janitor_telemetry(janitor_stats)
        row = {
            "index": idx,
            "session_id": session_id,
            "operational_day": operational_day,
            "message_count": int(day.get("message_count", 0) or 0),
            "manifest_day": {
                str(key): value
                for key, value in day.items()
                if key != "path"
            },
            "extract_result": extract_result,
            "db_stats": db_after,
            "db_stats_before": db_before,
            "db_stats_after": db_after,
            "db_delta": _db_delta(db_before, db_after),
            "janitor_stats": janitor_stats,
            "telemetry": {
                "extract": extract_summary,
                "janitor": janitor_summary,
            },
        }
        summary_rows.append(row)
        print(f"  Facts stored: {extract_result.get('facts_stored')}")
        print(
            "  Extract telemetry:"
            f" stage={extract_summary.get('rolling_stage_wall_seconds')}s"
            f" flush={extract_summary.get('flush_wall_seconds')}s"
            f" signal_to_publish={extract_summary.get('signal_to_publish_seconds')}s"
        )
        print(
            "  Dedup prefilter:"
            f" fts_queries={extract_summary['dedup']['fts_query_count']}"
            f" candidates={extract_summary['dedup']['fts_candidates_returned']}"
            f" scanned={extract_summary['dedup']['scanned_rows']}"
            f" limit_hits={extract_summary['dedup']['fts_limit_hits']}"
        )
        if janitor_summary.get("success"):
            task_durations = janitor_summary.get("task_durations", {})
            print(
                "  Janitor telemetry:"
                f" total={janitor_summary.get('total_duration_seconds')}s"
                f" review={task_durations.get('review')}"
                f" rag_reindex={task_durations.get('rag_reindex')}"
            )
        print(
            "  DB delta:"
            f" nodes=+{row['db_delta']['nodes']}"
            f" facts=+{row['db_delta']['fact_nodes']}"
            f" fts=+{row['db_delta']['tables'].get('nodes_fts', 0)}"
            f" vec=+{row['db_delta']['tables'].get('vec_nodes', 0)}"
            f" db_bytes=+{row['db_delta']['file_size_bytes']}"
        )
        print(f"  DB nodes/edges: {db_after.get('nodes')} / {db_after.get('edges')}")

        _write_summary_payload(
            _summary_path(results_dir),
            {
                "schema_version": 2,
                "summary_type": "imported_claude_history_replay",
                "manifest_path": str(args.manifest.expanduser().resolve()),
                "manifest": {
                    "schema_version": manifest.get("schema_version"),
                    "source_path": manifest.get("source_path"),
                    "cutoff_hour": manifest.get("cutoff_hour"),
                    "days_exported": manifest.get("days_exported"),
                    "messages_exported": manifest.get("messages_exported"),
                },
                "results_dir": str(results_dir),
                "backend": backend,
                "model": args.model,
                "rolling": bool(args.rolling),
                "chunk_tokens": int(args.chunk_tokens),
                "chunk_max_lines": int(args.chunk_max_lines),
                "days": summary_rows,
            },
        )

    print("=" * 60)
    print(f"Completed imported Claude replay: {results_dir}")
    print(f"Summary: {results_dir / 'logs' / 'imported_claude_history_summary.json'}")


if __name__ == "__main__":
    main()
