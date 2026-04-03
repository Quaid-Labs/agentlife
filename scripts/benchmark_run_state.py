#!/usr/bin/env python3
"""Shared benchmark run-state helpers.

Single source of truth for benchmark run classification/progress used by:
- monitor_benchmarks.py
- benchmark_dashboard.py
- bench-monitor.sh (via small remote Python calls)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_RUN_NAME_TS_RE = re.compile(r"-(\d{8}-\d{6})$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _iter_visible_run_dirs(runs_dirs: Iterable[Path]) -> Iterable[Path]:
    seen_names: set[str] = set()
    for runs_dir in runs_dirs:
        if not runs_dir.exists():
            continue
        for p in runs_dir.iterdir():
            if not p.is_dir():
                continue
            if p.name in ("successful-runs", "failed-runs"):
                continue
            if p.name in seen_names:
                continue
            seen_names.add(p.name)
            yield p


def _parse_run_name_start_ts(run_name: str) -> Optional[float]:
    m = _RUN_NAME_TS_RE.search(run_name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.timestamp()


def run_start_sort_key(run_dir: Path) -> Tuple[float, str]:
    name_ts = _parse_run_name_start_ts(run_dir.name)
    if name_ts is not None:
        return (name_ts, run_dir.name)
    try:
        return (run_dir.stat().st_mtime, run_dir.name)
    except OSError:
        return (0.0, run_dir.name)


def detect_active_runs(root: Path, runs_dirs: List[Path]) -> Dict[str, int]:
    return {name: info["pid"] for name, info in detect_active_processes(root, runs_dirs).items()}


def detect_active_processes(root: Path, runs_dirs: List[Path]) -> Dict[str, Dict[str, Any]]:
    proc = run_cmd(["ps", "-eo", "pid,cmd"], cwd=root)
    active: Dict[str, Dict[str, Any]] = {}
    if proc.returncode != 0:
        return active

    patterns = [
        "run_production_benchmark.py",
        "run_locomo.py",
        "run_longmemeval.py",
    ]

    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if not any(p in cmd for p in patterns):
            continue

        m = re.search(r"--results-dir\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))", cmd)
        if m:
            rdir = m.group(1) or m.group(2) or m.group(3) or ""
            active[Path(rdir).name] = {"pid": pid, "cmd": cmd}
            continue

        for p in _iter_visible_run_dirs(runs_dirs):
            if p.name in cmd:
                active[p.name] = {"pid": pid, "cmd": cmd}
                break

    return active


def detect_kind(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "locomo" in name:
        return "locomo"
    if "longmemeval" in name:
        return "longmemeval"
    if "quaid" in name or "al-" in name or "oc-native" in name:
        return "agentlife"
    return "unknown"


def parse_score(run_dir: Path, kind: str) -> Optional[float]:
    if kind == "agentlife":
        p = run_dir / "scores.json"
        if p.exists():
            d = load_json(p)
            if d:
                try:
                    return float(d["scores"]["overall"]["accuracy"])
                except Exception:
                    return None
    if kind == "locomo":
        lp = run_dir / "locomo_results.json"
        if lp.exists():
            d = load_json(lp)
            if d:
                try:
                    return float(d["scores"]["overall"]["llm_judge"])
                except Exception:
                    pass
        pp = run_dir / "locomo_results.partial.json"
        if pp.exists():
            d = load_json(pp)
            if d:
                try:
                    return float(d["scores"]["overall"]["token_f1"])
                except Exception:
                    pass
        p = run_dir / "scores.json"
        if p.exists():
            d = load_json(p)
            if d:
                try:
                    return float(d["scores"]["overall"]["accuracy"])
                except Exception:
                    return None
    if kind == "longmemeval":
        p = run_dir / "longmemeval_results.json"
        d = load_json(p) if p.exists() else None
        if d:
            try:
                return float(d["metrics"]["overall_accuracy"])
            except Exception:
                return None
    return None


def has_completion(run_dir: Path, kind: str) -> bool:
    if kind == "agentlife":
        return (run_dir / "scores.json").exists() or (run_dir / "ingest_complete.json").exists()
    if kind == "locomo":
        return (run_dir / "locomo_results.json").exists() or (run_dir / "scores.json").exists()
    if kind == "longmemeval":
        return (run_dir / "longmemeval_results.json").exists()
    return (run_dir / "scores.json").exists()


def find_log_files(runs_dirs: List[Path], run_name: str) -> List[Path]:
    out: List[Path] = []
    for runs_dir in runs_dirs:
        sidecar = runs_dir / f"{run_name}.launch.log"
        if sidecar.exists():
            out.append(sidecar)
        run_dir = runs_dir / run_name
        for rel in ("launch.log", "launcher.log", "run.log", "resume.launch.log"):
            p = run_dir / rel
            if p.exists():
                out.append(p)
    return out


def failure_marker(log_text: str) -> bool:
    markers = [
        "Traceback (most recent call last):",
        "RuntimeError:",
        "subprocess.TimeoutExpired",
        "FATAL:",
        "ERROR:",
        "All attempts failed",
    ]
    return any(m in log_text for m in markers)


def _launch_candidates(root: Path, run_name: str) -> List[Path]:
    run = root / "runs" / str(run_name)
    return [
        root / "runs" / f"{run_name}.launch.log",
        run / "launch.log",
        run / "run.log",
    ]


def _first_launch_text(root: Path, run_name: str) -> str:
    for p in _launch_candidates(root, run_name):
        if not p.exists():
            continue
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        if txt:
            return txt
    return ""


def infer_ingest_schedule(root: Path, run_name: str) -> Optional[str]:
    run = root / "runs" / str(run_name)
    meta = load_json(root / "runs" / str(run_name) / "run_metadata.json") or {}
    for key in ("ingest_schedule", "schedule_mode"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    # Eval-only clones can carry OBD extraction artifacts while launch metadata
    # defaults to per-day; prefer persisted extraction mode when available.
    checkpoint = load_json(run / "logs" / "extraction_checkpoint.json") or {}
    mode = str(checkpoint.get("mode") or "").strip().lower()
    if mode in {"rolling-obd", "obd"}:
        return mode
    if mode == "obd-post-extract":
        return "obd"
    if mode in {"per-day", "daily", "rolling"}:
        return mode

    txt = _first_launch_text(root, run_name)
    patterns = (
        re.compile(r"^\s*Ingest schedule:\s*([A-Za-z0-9._-]+)\s*$", re.M),
        re.compile(r"^\s*ingest schedule=([A-Za-z0-9._-]+)\s*$", re.M),
        re.compile(r"--ingest-schedule\s+([A-Za-z0-9._-]+)"),
    )
    for pat in patterns:
        m = pat.search(txt)
        if m:
            return str(m.group(1)).strip().lower()
    return None


def _is_obd_schedule(schedule: Optional[str]) -> bool:
    return schedule in {"obd", "rolling-obd"}


def infer_parallel(root: Path, run_name: str) -> Optional[int]:
    meta = load_json(root / "runs" / str(run_name) / "run_metadata.json") or {}
    try:
        parallel = meta.get("parallel")
        if parallel is not None:
            return int(parallel)
    except Exception:
        pass
    patterns = [
        re.compile(r"Parallel (?:chunk )?extraction workers:\s*(\d+)"),
        re.compile(r"Parallel day extraction workers:\s*(\d+)"),
        re.compile(r"Parallel workers:\s*(\d+)"),
        re.compile(r"--parallel\s+(\d+)"),
    ]
    txt = _first_launch_text(root, run_name)
    for pat in patterns:
        m = pat.search(txt)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def infer_provider_lane(root: Path, run_name: str, active_cmd: Optional[str] = None) -> Optional[str]:
    txt = _first_launch_text(root, run_name)
    if active_cmd:
        txt = f"{txt}\n{active_cmd}"
    m = re.search(r"^\s*Backend:\s*([A-Za-z0-9._-]+)\s*$", txt, re.M)
    if m:
        return m.group(1).lower()
    low = txt.lower()
    if "claude-code" in low or "claude -p" in low:
        return "claude-code"
    if "backend=vllm" in low or "--backend vllm" in low:
        return "vllm"
    if "benchmark anthropic oauth" in low or "--backend oauth" in low:
        return "oauth"
    if "--backend api" in low:
        return "api"
    return None


def _short_model_name(model: str) -> str:
    raw = str(model or "").strip()
    if not raw:
        return ""
    low = raw.lower()
    if "haiku" in low:
        return "Haiku"
    if "sonnet" in low:
        return "Sonnet"
    if "opus" in low:
        return "Opus"
    if "gpt-4o-mini" in low:
        return "4o-mini"
    if "gpt-4o" in low:
        return "4o"
    if raw.startswith("claude-"):
        raw = raw[len("claude-") :]
    return raw


def infer_model_lane(root: Path, run_name: str, active_cmd: Optional[str] = None) -> Optional[str]:
    run = root / "runs" / str(run_name)
    meta = load_json(run / "run_metadata.json") or {}
    ingest_usage = load_json(run / "ingest_usage.json") or {}
    ingest_complete = load_json(run / "ingest_complete.json") or {}

    extraction_model = str(meta.get("model", "") or "").strip()
    eval_model = str(meta.get("eval_model", "") or "").strip()
    mode = str(meta.get("mode", "") or "").strip().lower()

    # Lineage-truth for extraction model should come from ingest artifacts, not
    # eval-only launch defaults.
    lineage_extraction_model = str(
        (ingest_usage.get("ingest") or {}).get("model")
        or ingest_complete.get("extraction_model")
        or ""
    ).strip()
    if lineage_extraction_model:
        extraction_model = lineage_extraction_model

    scores = load_json(run / "scores.json") or {}
    scores_meta = scores.get("metadata") or {}
    if not extraction_model:
        extraction_model = str(scores_meta.get("extraction_model", "") or "").strip()
    if not eval_model:
        eval_model = str(scores_meta.get("eval_model", "") or "").strip()
    if not mode:
        mode = str(scores_meta.get("mode", "") or "").strip().lower()

    txt = _first_launch_text(root, run_name)
    if not mode:
        for pat in (
            re.compile(r"^\s*Mode:\s*([A-Za-z0-9._-]+)\s*$", re.M),
            re.compile(r"--mode\s+([A-Za-z0-9._-]+)"),
        ):
            m = pat.search(txt)
            if m:
                mode = str(m.group(1)).strip().lower()
                break
    if not extraction_model:
        for pat in (
            re.compile(r"^\s*Model:\s*([A-Za-z0-9._-]+)\s*$", re.M),
            re.compile(r"Loaded \d+ sessions \(model:\s*([A-Za-z0-9._-]+)\)"),
            re.compile(r"--model\s+([A-Za-z0-9._-]+)"),
        ):
            m = pat.search(txt)
            if m:
                extraction_model = str(m.group(1)).strip()
                break
    if not eval_model:
        for pat in (
            re.compile(r"PHASE 5: EVALUATION \(([A-Za-z0-9._-]+)"),
            re.compile(r"^\s*FC answer models:\s*([A-Za-z0-9._,-]+)\s*$", re.M),
            re.compile(r"--eval-model\s+([A-Za-z0-9._-]+)"),
            re.compile(r"--fc-models\s+([A-Za-z0-9._,-]+)"),
            re.compile(r"FC Baseline \(([A-Za-z0-9._-]+)\)"),
        ):
            m = pat.search(txt)
            if m:
                eval_model = str(m.group(1)).strip().split(",")[0].strip()
                break

    if active_cmd:
        if not mode:
            m = re.search(r"--mode\s+([A-Za-z0-9._-]+)", active_cmd)
            if m:
                mode = str(m.group(1)).strip().lower()
        if not extraction_model:
            m = re.search(r"--model\s+([A-Za-z0-9._-]+)", active_cmd)
            if m:
                extraction_model = str(m.group(1)).strip()
        if not eval_model:
            m = re.search(r"--eval-model\s+([A-Za-z0-9._-]+)", active_cmd)
            if m:
                eval_model = str(m.group(1)).strip()
        if not eval_model:
            m = re.search(r"--fc-models\s+([A-Za-z0-9._,-]+)", active_cmd)
            if m:
                eval_model = str(m.group(1)).strip().split(",")[0].strip()

    # Guardrail: eval-only runs can inherit a misleading `model` default in
    # run_metadata when ingest artifacts are absent (lineage copied from an
    # eval-only run). In that case, prefer blank over mislabeling a lane.
    if mode == "eval" and not lineage_extraction_model:
        meta_model = str(meta.get("model", "") or "").strip()
        if extraction_model and meta_model and extraction_model == meta_model:
            extraction_model = ""

    short_extract = _short_model_name(extraction_model)
    short_eval = _short_model_name(eval_model)
    if mode == "fc":
        return f"FC {short_eval or short_extract}".strip()
    if short_extract and short_eval:
        return f"{short_extract}/{short_eval}"
    if short_extract:
        return short_extract
    if short_eval:
        return short_eval
    return None


def extract_final_score(root: Path, run_name: str) -> Optional[float]:
    run = root / "runs" / str(run_name)
    for fn in ("final_scores.json", "scores.json"):
        p = run / fn
        if not p.exists():
            continue
        d = load_json(p)
        if not d:
            continue
        try:
            b = d.get("blended", {}).get("blended", {}).get("pct")
            if b is not None:
                return float(b)
        except Exception:
            pass
        try:
            o = d.get("scores", {}).get("overall", {}).get("accuracy")
            if o is not None:
                return float(o)
        except Exception:
            pass
    return None


def extract_preview_score(root: Path, run_name: str) -> Optional[float]:
    run = root / "runs" / str(run_name)
    p = run / "evaluation_results.json"
    if p.exists():
        rows = load_json(p)
        if isinstance(rows, list) and rows:
            correct = sum(1 for r in rows if r.get("judge_label") == "CORRECT")
            partial = sum(1 for r in rows if r.get("judge_label") == "PARTIAL")
            wrong = sum(1 for r in rows if r.get("judge_label") == "WRONG")
            scored = correct + partial + wrong
            if scored > 0:
                return round((correct + 0.5 * partial) / scored * 100.0, 2)
    txt = _first_launch_text(root, run_name)
    for pat in (
        r"\[(\d+)/(\d+)\|q\d+\].*?\[(\d+(?:\.\d+)?)%\]",
        r"\[(\d+)/(\d+)\].*?\[(\d+(?:\.\d+)?)%\]",
    ):
        matches = re.findall(pat, txt)
        if matches:
            return float(matches[-1][2])
    return None


def infer_metric_label(root: Path, run_name: str) -> str:
    name = str(run_name or "")
    low_name = name.lower()
    is_obd = _is_obd_schedule(infer_ingest_schedule(root, run_name))
    if ("quaid-l" in low_name or "al-l" in low_name) and is_obd:
        return "AL-L Quaid OBD"
    if ("quaid-s" in low_name or "al-s" in low_name) and is_obd:
        return "AL-S Quaid OBD"
    if "oc-native" in low_name and ("als" in low_name or "al-s" in low_name):
        return "AL-S OC Native"
    if "oc-native" in low_name and ("all" in low_name or "al-l" in low_name):
        return "AL-L OC Native"
    if "locomo" in low_name:
        return "LoCoMo Quaid"
    if "longmemeval" in low_name:
        return "LongMemEval Quaid"
    if "quaid-s" in low_name or "al-s" in low_name:
        return "AL-S Quaid"
    if "quaid-l" in low_name or "al-l" in low_name:
        return "AL-L Quaid"
    if "mem0" in low_name:
        return "mem0 AL-L"
    return "Unknown"


def _safe_count_lines(path: Path) -> Optional[int]:
    try:
        with path.open("r", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def rolling_status(run: Path) -> Optional[Dict[str, Any]]:
    candidates = [
        run / "benchrunner" / "logs" / "daemon" / "rolling-extraction.jsonl",
        run / "logs" / "daemon" / "rolling-extraction.jsonl",
    ]
    metrics_path = next((p for p in candidates if p.exists()), None)
    if metrics_path is None:
        return None
    rows: List[dict] = []
    try:
        with metrics_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return None
    if not rows:
        return None
    last = rows[-1]
    stages = [r for r in rows if r.get("event") == "rolling_stage"]
    cursor = last.get("new_cursor_offset") if isinstance(last.get("new_cursor_offset"), int) else None
    total_lines = None
    session_id = str(last.get("session_id") or "").strip()
    if session_id:
        cursor_candidates = [
            run / "benchrunner" / "data" / "session-cursors" / f"{session_id}.json",
            run / "data" / "session-cursors" / f"{session_id}.json",
        ]
        cursor_meta = None
        for cp in cursor_candidates:
            if not cp.exists():
                continue
            try:
                cursor_meta = json.loads(cp.read_text())
                break
            except Exception:
                cursor_meta = None
        if cursor_meta:
            if cursor is None and isinstance(cursor_meta.get("line_offset"), int):
                cursor = cursor_meta.get("line_offset")
            transcript_path = cursor_meta.get("transcript_path")
            if transcript_path:
                total_lines = _safe_count_lines(Path(str(transcript_path)))
    avg_wall = None
    walls = [float(r.get("wall_seconds") or 0.0) for r in stages if r.get("wall_seconds") is not None]
    if walls:
        avg_wall = sum(walls) / len(walls)
    return {
        "event": last.get("event"),
        "cursor": cursor,
        "total_lines": total_lines,
        "rolling_batches": last.get("rolling_batches") if last.get("rolling_batches") is not None else len(stages),
        "staged_fact_count": last.get("staged_fact_count"),
        "avg_wall_seconds": avg_wall,
        "signal_to_publish_seconds": last.get("signal_to_publish_seconds"),
        "publish_wall_seconds": last.get("publish_wall_seconds"),
    }


def plain_obd_status(run: Path, launch_text: str) -> Optional[str]:
    if "Runtime extraction: one compaction event via ingest/extract.py" not in launch_text:
        return None
    checkpoint = load_json(run / "logs" / "extraction_checkpoint.json") or {}
    if str(checkpoint.get("mode") or "").strip() != "obd":
        return None
    progress = load_json(run / "logs" / "obd_extract_progress.json") or {}
    if progress:
        total_chunks = progress.get("total_chunks")
        current_chunk = progress.get("current_chunk")
        try:
            total_chunks = int(total_chunks)
            current_chunk = int(current_chunk)
        except Exception:
            total_chunks = None
            current_chunk = None
        if total_chunks is not None and total_chunks > 0 and current_chunk is not None:
            return f"obd extract {current_chunk}/{total_chunks}"
    trace_rows = _safe_count_lines(run / "logs" / "llm-call-trace.jsonl")
    if trace_rows and trace_rows > 0:
        return f"obd extract | calls {trace_rows}"
    if str(checkpoint.get("state") or "").strip() == "running":
        return "obd extract"
    return None


def day_plan_status(root: Path, run_name: str) -> Optional[Dict[str, Any]]:
    total_days = None
    current_day = None
    current_total = None
    current_date = None
    for lp in _launch_candidates(root, run_name):
        if not lp.exists():
            continue
        try:
            txt = lp.read_text(errors="ignore")
        except Exception:
            continue
        m = re.search(r"Grouped into (\d+) days:", txt)
        if m:
            try:
                total_days = int(m.group(1))
            except Exception:
                total_days = None
        for mm in re.finditer(r"^--- Day (\d+)/(\d+):\s*([0-9-]+)", txt, re.M):
            try:
                current_day = int(mm.group(1))
                current_total = int(mm.group(2))
            except Exception:
                current_day = None
                current_total = None
            current_date = mm.group(3)
        if total_days is not None or current_day is not None:
            break
    if total_days is None and current_total is not None:
        total_days = current_total
    if total_days is None:
        return None
    return {
        "total_days": total_days,
        "current_day": current_day,
        "current_date": current_date,
    }


def run_progress(root: Path, run_name: str) -> str:
    run = root / "runs" / str(run_name)
    launch_text = _first_launch_text(root, run_name)
    is_obd_run = _is_obd_schedule(infer_ingest_schedule(root, run_name))
    chunk_count = 0
    try:
        chunk_count = len(list((run / "extraction_cache").glob("chunk-*.json")))
    except Exception:
        chunk_count = 0
    last_chunk = None
    total_chunks = None
    prog = run / "extraction_cache" / "progress.json"
    if prog.exists():
        try:
            p = json.loads(prog.read_text())
            lc = p.get("last_completed_chunk")
            tc = p.get("total_chunks")
            if isinstance(lc, int):
                last_chunk = lc
            if isinstance(tc, int) and tc > 0:
                total_chunks = tc
        except Exception:
            pass
    if total_chunks is None:
        m = re.search(r"Extraction chunks: (\d+)", launch_text)
        if m:
            total_chunks = int(m.group(1))

    eval_done = None
    eval_total = None
    ev = run / "logs" / "eval_progress.json"
    if ev.exists():
        try:
            e = json.loads(ev.read_text())
            lcq = e.get("last_completed_query")
            if isinstance(lcq, int):
                eval_done = max(0, lcq + 1)
            else:
                eval_done = e.get("completed")
            eval_total = e.get("total_queries") if e.get("total_queries") is not None else e.get("total")
        except Exception:
            pass
    if eval_total is not None:
        tier5 = _tier5_progress(run, launch_text)
        if tier5 is not None:
            tier5_done, tier5_total = tier5
            base_done = int(eval_done or 0)
            return f"eval {base_done + tier5_done}/{int(eval_total) + tier5_total}"
        return f"eval {eval_done}/{eval_total}"

    day_plan = day_plan_status(root, run_name)
    janitor_progress = run / "logs" / "janitor_progress.json"
    if janitor_progress.exists():
        try:
            jp = json.loads(janitor_progress.read_text())
            phase = str(jp.get("phase") or "").strip()
            if phase:
                return phase.lower()
        except Exception:
            pass
    if day_plan and day_plan.get("total_days"):
        current_day = day_plan.get("current_day")
        total_days = day_plan.get("total_days")
        if current_day is not None:
            return f"extraction {current_day}/{total_days}"
        return f"extraction 0/{total_days}"

    rolling = rolling_status(run)
    if is_obd_run and rolling and rolling.get("event") == "rolling_stage":
        parts: List[str] = []
        cursor = rolling.get("cursor")
        total_lines = rolling.get("total_lines")
        batches = rolling.get("rolling_batches")
        if batches is not None:
            parts.append(f"chunk {batches}")
        if cursor is not None and total_lines is not None:
            parts.append(f"{cursor}/{total_lines}")
        elif cursor is not None:
            parts.append(f"{cursor}/?")
        else:
            parts.append("rolling")
        staged = rolling.get("staged_fact_count")
        if staged is not None:
            parts.append(f"facts {staged}")
        return " | ".join(parts)
    if is_obd_run and rolling and rolling.get("event") == "rolling_flush":
        parts = ["rolling flush"]
        signal_to_publish = rolling.get("signal_to_publish_seconds")
        publish_wall = rolling.get("publish_wall_seconds")
        if signal_to_publish is not None:
            parts.append(f"signal {float(signal_to_publish):.1f}s")
        if publish_wall is not None:
            parts.append(f"publish {float(publish_wall):.1f}s")
        return " | ".join(parts)

    matches = re.findall(r"\[(\d+)/(\d+)\|q\d+\].*?\[(\d+(?:\.\d+)?)%\]", launch_text)
    if matches:
        cur, total, pct = matches[-1]
        return f"eval {cur}/{total} ({pct}%)"
    plain_obd = plain_obd_status(run, launch_text)
    if is_obd_run and plain_obd is not None:
        return plain_obd
    janitor_seen = (run / "logs" / "janitor-task-telemetry.jsonl").exists() or (run / "logs" / "janitor-stats.json").exists()
    if janitor_seen and total_chunks is not None:
        current = 1 if last_chunk is None else max(1, min(total_chunks, last_chunk + 1))
        return f"janitor {current}/{total_chunks}"
    if last_chunk is not None and total_chunks is not None:
        done = max(0, min(total_chunks, last_chunk + 1))
        return f"extraction {done}/{total_chunks}"
    if chunk_count > 0:
        if total_chunks is not None:
            done = max(0, min(total_chunks, chunk_count))
            return f"extraction {done}/{total_chunks}"
        return f"extraction {chunk_count}/?"

    fc_step = None
    if "FULL-CONTEXT BASELINE (" in launch_text or "TIER 5 FC BASELINE (" in launch_text:
        matches = re.findall(r"\[(\d+)/(\d+)\].*?\[(\d+(?:\.\d+)?)%\]", launch_text)
        if matches:
            cur, total, pct = matches[-1]
            fc_step = f"fc {cur}/{total} ({pct}%)"
        else:
            m = re.search(r"\b(\d+) queries, (\d+) sessions\b", launch_text)
            if m:
                fc_step = f"fc 0/{m.group(1)}"
            else:
                fc_step = "fc starting"
    if fc_step is not None:
        return fc_step
    return "starting"


def _classify_run(run_dir: Path, runs_dirs: List[Path], active: Dict[str, int]) -> Dict[str, Any]:
    name = run_dir.name
    kind = detect_kind(run_dir)
    score = parse_score(run_dir, kind)
    if name in active:
        return {
            "name": name,
            "path": str(run_dir),
            "kind": kind,
            "state": "active",
            "reason": "process running",
            "score": score,
            "active_pid": active[name],
        }
    if has_completion(run_dir, kind):
        return {
            "name": name,
            "path": str(run_dir),
            "kind": kind,
            "state": "complete",
            "reason": "completion artifact present",
            "score": score,
            "active_pid": None,
        }
    logs = find_log_files(runs_dirs, name)
    log_blob = ""
    for p in logs:
        try:
            log_blob += "\n" + p.read_text(errors="ignore")[-20000:]
        except Exception:
            pass
    if log_blob and failure_marker(log_blob):
        return {
            "name": name,
            "path": str(run_dir),
            "kind": kind,
            "state": "failed",
            "reason": "failure marker in logs",
            "score": score,
            "active_pid": None,
        }
    return {
        "name": name,
        "path": str(run_dir),
        "kind": kind,
        "state": "incomplete",
        "reason": "no completion artifact and no running process",
        "score": score,
        "active_pid": None,
    }


def enrich_run(root: Path, row: Dict[str, Any], active_cmd: Optional[str] = None) -> Dict[str, Any]:
    run_name = str(row.get("name") or "")
    meta_path = root / "runs" / run_name / "run_metadata.json"
    elapsed = None
    parallel = None
    started_at = None
    completed_at = None
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text())
            parallel = m.get("parallel")
            started_at = m.get("started_at")
            completed_at = m.get("completed_at") or m.get("ended_at")
            if m.get("total_elapsed_seconds") is not None:
                elapsed = float(m["total_elapsed_seconds"])
            elif m.get("duration_sec") is not None:
                elapsed = float(m["duration_sec"])
            elif started_at and completed_at:
                started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
                ended = datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
                elapsed = max(0.0, (ended - started).total_seconds())
        except Exception:
            elapsed = None
            parallel = None
    if parallel is None:
        parallel = infer_parallel(root, run_name)
    if completed_at and row.get("state") == "incomplete":
        row["state"] = "complete"
        row["reason"] = "run_metadata.completed_at present"
    final_score = extract_final_score(root, run_name)
    row["elapsed_seconds"] = elapsed
    row["parallel"] = parallel
    row["provider_lane"] = infer_provider_lane(root, run_name, active_cmd=active_cmd)
    row["model_lane"] = infer_model_lane(root, run_name, active_cmd=active_cmd)
    row["metric_label"] = infer_metric_label(root, run_name)
    row["note"] = infer_run_note(root, run_name)
    row["started_at"] = started_at
    row["completed_at"] = completed_at
    row["final_score"] = final_score
    row["preview_score"] = None if final_score is not None else extract_preview_score(root, run_name)
    row["current_active_item"] = run_progress(root, run_name) if row.get("state") == "active" else (row.get("reason") or "")
    return row


def infer_run_note(root: Path, run_name: str) -> str:
    run = root / "runs" / str(run_name)
    note_path = run / "run_note.txt"
    if note_path.exists():
        try:
            return str(note_path.read_text(encoding="utf-8").strip())
        except Exception:
            return ""
    meta = load_json(run / "run_metadata.json") or {}
    for key in ("note", "notes", "description", "run_note"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _tier5_progress(run: Path, launch_text: str) -> Optional[Tuple[int, int]]:
    """Return (done, total) for Tier-5 progress when eval has entered EI phase."""
    tier5_file = run / "tier5_results.json"
    if tier5_file.exists():
        payload = load_json(tier5_file)
        if isinstance(payload, list):
            count = len(payload)
            return (count, count)

    if "TIER 5: EMOTIONAL INTELLIGENCE" not in launch_text:
        return None

    total = None
    m_total = re.search(r"^\s*(\d+)\s+EI queries\s*$", launch_text, re.M)
    if m_total:
        try:
            total = int(m_total.group(1))
        except Exception:
            total = None

    matches = re.findall(r"^\s*\[(\d+)/(\d+)\]\s+.*EI-\d+", launch_text, re.M)
    if matches:
        try:
            done = int(matches[-1][0])
            seen_total = int(matches[-1][1])
            return (done, seen_total)
        except Exception:
            pass

    if total is not None:
        return (0, total)
    return None


def build_status_report(root: Path, runs_dir: str = "runs", extra_runs_dirs: Iterable[str] = ()) -> Dict[str, Any]:
    root = root.resolve()
    primary_runs_dir = (root / runs_dir).resolve()
    visible_runs_dirs = [primary_runs_dir, *(Path(p).expanduser().resolve() for p in extra_runs_dirs)]
    active_processes = detect_active_processes(root, visible_runs_dirs)
    active_map = {name: int(info["pid"]) for name, info in active_processes.items()}
    run_dirs = list(_iter_visible_run_dirs(visible_runs_dirs))
    run_dirs.sort(key=run_start_sort_key, reverse=True)
    runs = [
        enrich_run(
            root,
            _classify_run(p, visible_runs_dirs, active_map),
            active_cmd=(active_processes.get(p.name) or {}).get("cmd"),
        )
        for p in run_dirs
    ]
    counts = {"active": 0, "complete": 0, "failed": 0, "incomplete": 0}
    for item in runs:
        state = str(item.get("state") or "")
        if state in counts:
            counts[state] += 1
    return {
        "timestamp": utc_now_iso(),
        "root": str(root),
        "runs_dir": str(primary_runs_dir),
        "visible_runs_dirs": [str(p) for p in visible_runs_dirs],
        "counts": counts,
        "actions": [],
        "runs": runs,
    }


def build_run_detail(root: Path, run_name: str) -> Dict[str, Any]:
    root = root.resolve()
    run = root / "runs" / run_name
    visible_runs_dirs = [root / "runs"]
    active_processes = detect_active_processes(root, visible_runs_dirs)
    active_map = {name: int(info["pid"]) for name, info in active_processes.items()}
    classified = enrich_run(
        root,
        _classify_run(run, visible_runs_dirs, active_map),
        active_cmd=(active_processes.get(run_name) or {}).get("cmd"),
    )
    out: Dict[str, Any] = {
        "name": run_name,
        "state": classified.get("state") or "unknown",
        "reason": classified.get("reason") or "",
        "final_score": None,
        "elapsed_seconds": None,
        "parallel": classified.get("parallel"),
        "started_at": classified.get("started_at"),
        "completed_at": classified.get("completed_at"),
        "provider_lane": classified.get("provider_lane"),
        "model_lane": classified.get("model_lane"),
        "metric_label": classified.get("metric_label"),
        "active_pid": classified.get("active_pid"),
        "per_type": {},
        "per_theme": {},
        "per_difficulty": {},
        "phase": None,
        "current_active_item": classified.get("current_active_item") or "",
        "chunk_count": 0,
        "day_count": None,
        "current_day": None,
        "eval_completed": None,
        "eval_total": None,
        "nodes": {},
    }
    meta = load_json(run / "run_metadata.json") or {}
    out["started_at"] = meta.get("started_at")
    out["completed_at"] = meta.get("completed_at") or meta.get("ended_at")
    out["elapsed_seconds"] = meta.get("total_elapsed_seconds") if meta.get("total_elapsed_seconds") is not None else meta.get("duration_sec")
    out["parallel"] = meta.get("parallel")
    status = str(meta.get("status", "")).lower()
    if out["completed_at"] or status == "completed":
        out["state"] = "complete"

    for fn in ("final_scores.json", "scores.json"):
        p = run / fn
        if not p.exists():
            continue
        d = load_json(p)
        if not d:
            continue
        b = d.get("blended", {}).get("blended", {}).get("pct")
        if b is not None:
            out["final_score"] = b
        else:
            out["final_score"] = d.get("scores", {}).get("overall", {}).get("accuracy")
        s = d.get("scores", {})
        out["per_type"] = s.get("per_type", {}) or {}
        out["per_theme"] = s.get("per_theme", {}) or {}
        out["per_difficulty"] = s.get("per_difficulty", {}) or {}
        break

    out["chunk_count"] = len(list((run / "extraction_cache").glob("chunk-*.json")))
    ev = load_json(run / "logs" / "eval_progress.json")
    if ev:
        lcq = ev.get("last_completed_query")
        if isinstance(lcq, int):
            out["eval_completed"] = max(0, lcq + 1)
        else:
            out["eval_completed"] = ev.get("completed")
        out["eval_total"] = ev.get("total_queries") if ev.get("total_queries") is not None else ev.get("total")
    launch_text = _first_launch_text(root, run_name)
    tier5 = _tier5_progress(run, launch_text)
    if tier5 is not None and out.get("eval_total") is not None:
        tier5_done, tier5_total = tier5
        out["eval_completed"] = int(out.get("eval_completed") or 0) + tier5_done
        out["eval_total"] = int(out.get("eval_total") or 0) + tier5_total

    db = run / "data" / "memory.db"
    if db.exists():
        try:
            conn = sqlite3.connect(str(db))
            cur = conn.cursor()
            for s in ("pending", "approved", "active", "archived", "flagged"):
                out["nodes"][s] = int(cur.execute("SELECT COUNT(*) FROM nodes WHERE status=?", (s,)).fetchone()[0])
            conn.close()
        except Exception:
            pass

    dp = day_plan_status(root, run_name)
    if dp:
        out["day_count"] = dp.get("total_days")
        out["current_day"] = dp.get("current_day")

    if out.get("final_score") is not None:
        out["phase"] = "complete"
        out["state"] = "complete"
    elif out.get("eval_total") is not None:
        out["phase"] = f"eval {out.get('eval_completed')}/{out.get('eval_total')}"
    else:
        out["phase"] = out.get("current_active_item") or run_progress(root, run_name)
        if out["state"] == "unknown":
            out["state"] = "active" if run_name in active_map else "incomplete"
    if out.get("parallel") is None:
        out["parallel"] = infer_parallel(root, run_name)
    return out
