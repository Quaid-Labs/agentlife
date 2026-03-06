#!/usr/bin/env python3
"""Cron-safe benchmark monitor + queue launcher.

Runs one pass (no internal sleep). Intended to be invoked periodically, e.g.:
  */20 * * * * /usr/bin/python3 /path/to/monitor_benchmarks.py --execute

Behavior:
1) Inspect runs/ for active/completed/failed/incomplete runs.
2) Optionally auto-remediate recoverable incomplete runs (agentlife + locomo).
3) Optionally launch next queued benchmark job when no runs are active.
4) Emit machine-readable status JSON and a compact text summary.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunStatus:
    name: str
    path: Path
    kind: str  # agentlife | locomo | longmemeval | unknown
    state: str  # active | complete | failed | incomplete
    reason: str
    score: Optional[float] = None
    active_pid: Optional[int] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=120,
    )


def load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def detect_active_runs(root: Path, runs_dir: Path) -> Dict[str, int]:
    """Return mapping run_name -> pid for known benchmark processes."""
    proc = run_cmd(["ps", "-eo", "pid,cmd"])
    active: Dict[str, int] = {}
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

        # Prefer explicit --results-dir.
        m = re.search(r"--results-dir\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))", cmd)
        if m:
            rdir = m.group(1) or m.group(2) or m.group(3) or ""
            name = Path(rdir).name
            active[name] = pid
            continue

        # Fallback: any run dir name embedded in command.
        if runs_dir.exists():
            for p in runs_dir.iterdir():
                if not p.is_dir():
                    continue
                if p.name in ("successful-runs", "failed-runs"):
                    continue
                if p.name in cmd:
                    active[p.name] = pid
                    break

    return active


def detect_kind(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "locomo" in name:
        return "locomo"
    if "longmemeval" in name:
        return "longmemeval"
    if "quaid" in name or "al-" in name:
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
        # Prefer native LoCoMo result schema.
        lp = run_dir / "locomo_results.json"
        if lp.exists():
            d = load_json(lp)
            if d:
                try:
                    return float(d["scores"]["overall"]["llm_judge"])
                except Exception:
                    pass
        # Back-compat fallback if a scores.json was produced.
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
        return (run_dir / "scores.json").exists()
    if kind == "locomo":
        return (run_dir / "locomo_results.json").exists() or (run_dir / "scores.json").exists()
    if kind == "longmemeval":
        return (run_dir / "longmemeval_results.json").exists()
    return (run_dir / "scores.json").exists()


def find_log_files(runs_dir: Path, run_name: str) -> List[Path]:
    out: List[Path] = []
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


def classify_run(run_dir: Path, runs_dir: Path, active: Dict[str, int]) -> RunStatus:
    name = run_dir.name
    kind = detect_kind(run_dir)
    score = parse_score(run_dir, kind)

    if name in active:
        return RunStatus(name, run_dir, kind, "active", "process running", score=score, active_pid=active[name])

    if has_completion(run_dir, kind):
        return RunStatus(name, run_dir, kind, "complete", "completion artifact present", score=score)

    logs = find_log_files(runs_dir, name)
    log_blob = ""
    for p in logs:
        try:
            log_blob += "\n" + p.read_text(errors="ignore")[-20000:]
        except Exception:
            pass
    if log_blob and failure_marker(log_blob):
        return RunStatus(name, run_dir, kind, "failed", "failure marker in logs", score=score)

    return RunStatus(name, run_dir, kind, "incomplete", "no completion artifact and no running process", score=score)


_RUN_NAME_TS_RE = re.compile(r"-(\d{8}-\d{6})$")


def _parse_run_name_start_ts(run_name: str) -> Optional[float]:
    """Parse run start time from canonical run name suffix YYYYMMDD-HHMMSS."""
    m = _RUN_NAME_TS_RE.search(run_name)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.timestamp()


def run_start_sort_key(run_dir: Path) -> Tuple[float, str]:
    """Return sortable key for run recency based on start time (descending)."""
    # 1) Prefer explicit timestamp encoded in run directory name.
    name_ts = _parse_run_name_start_ts(run_dir.name)
    if name_ts is not None:
        return (name_ts, run_dir.name)
    # 2) Fallback to dir mtime if name has no timestamp.
    try:
        return (run_dir.stat().st_mtime, run_dir.name)
    except OSError:
        return (0.0, run_dir.name)


def build_agentlife_resume_cmd(run_dir: Path) -> Optional[List[str]]:
    meta = load_json(run_dir / "run_metadata.json") or {}
    mode = str(meta.get("mode", "full") or "full").strip().lower()
    if mode == "per-day":
        mode = "full"
    if mode not in {"full", "ingest", "eval", "fc"}:
        mode = "full"
    model = str(meta.get("model", "claude-sonnet-4-5-20250929"))
    eval_model = str(meta.get("eval_model", model))
    judge = str(meta.get("judge", "gpt-4o-mini"))
    backend = str(meta.get("backend", "claude-code"))
    vllm_url = str(meta.get("vllm_url", "") or "").strip()
    vllm_model = str(meta.get("vllm_model", "") or "").strip()
    try:
        parallel = int(meta.get("parallel", 6) or 6)
    except (TypeError, ValueError):
        parallel = 4
    parallel = max(1, min(parallel, 64))
    tier5_raw = meta.get("tier5", True)
    if isinstance(tier5_raw, bool):
        tier5 = tier5_raw
    else:
        tier5 = str(tier5_raw).strip().lower() in {"1", "true", "yes", "on"}
    max_sessions_raw = meta.get("max_sessions")
    max_sessions: Optional[int] = None
    if max_sessions_raw is not None:
        try:
            max_sessions = int(max_sessions_raw)
        except (TypeError, ValueError):
            max_sessions = None
        if max_sessions is not None and max_sessions < 1:
            max_sessions = None
    eval_token_budget_raw = meta.get("eval_token_budget")
    eval_token_budget: Optional[int] = None
    if eval_token_budget_raw is not None:
        try:
            eval_token_budget = int(eval_token_budget_raw)
        except (TypeError, ValueError):
            eval_token_budget = None
        if eval_token_budget is not None and eval_token_budget < 0:
            eval_token_budget = 0
    filler_dir = meta.get("filler_dir")

    # Conservative default: full mode resume.
    cmd = [
        "python3",
        "agentlife/eval/run_production_benchmark.py",
        "--mode",
        mode,
        "--results-dir",
        f"runs/{run_dir.name}",
        "--backend",
        backend,
        "--model",
        model,
        "--eval-model",
        eval_model,
        "--judge",
        judge,
        "--parallel",
        str(parallel),
        "--resume-extraction",
        "--resume-eval",
    ]
    if backend == "vllm":
        if vllm_url:
            cmd.extend(["--vllm-url", vllm_url])
        if vllm_model:
            cmd.extend(["--vllm-model", vllm_model])
    if max_sessions is not None:
        cmd.extend(["--max-sessions", str(max_sessions)])
    if eval_token_budget is not None:
        cmd.extend(["--eval-token-budget", str(eval_token_budget)])
    if filler_dir:
        cmd.extend(["--filler-dir", str(filler_dir)])
    if not tier5:
        cmd.append("--no-tier5")
    return cmd


def build_locomo_resume_cmd(run_dir: Path) -> List[str]:
    # Conservative default: skip ingestion and re-run scoring/eval path.
    return [
        "python3",
        "memory-stress-test/runner/locomo/run_locomo.py",
        "--conversations",
        "all",
        "--results-dir",
        f"runs/{run_dir.name}",
        "--skip-ingest",
    ]


def launch_background(root: Path, runs_dir: Path, cmd: List[str], run_name: str) -> Tuple[bool, str]:
    log_path = runs_dir / f"{run_name}.restart.log"
    shell = " ".join(shlex.quote(c) for c in cmd)
    launch = (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(root))}; "
        f"nohup env PYTHONUNBUFFERED=1 {shell} >> {shlex.quote(str(log_path))} 2>&1 < /dev/null & echo $!"
    )
    proc = run_cmd(["bash", "-lc", launch], cwd=root)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[-400:]
    return True, proc.stdout.strip()


def read_queue(queue_path: Path) -> List[dict]:
    if not queue_path.exists():
        return []
    data = load_json(queue_path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def write_queue(queue_path: Path, items: List[dict]) -> None:
    atomic_write_text(queue_path, json.dumps(items, indent=2) + "\n")


def default_queue_item_valid(item: dict) -> bool:
    if not item.get("cmd"):
        return False
    if not item.get("results_dir"):
        return False
    return True


def maybe_launch_from_queue(
    root: Path,
    runs_dir: Path,
    queue_path: Path,
    any_active: bool,
    execute: bool,
) -> Optional[str]:
    if any_active:
        return None
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.touch(exist_ok=True)
    with queue_path.open("r+", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            data = json.loads(raw) if raw else []
            items = [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
            if not items:
                return None
            item = items[0]
            if not default_queue_item_valid(item):
                items.pop(0)
                f.seek(0)
                f.truncate()
                f.write(json.dumps(items, indent=2))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
                return "queue item invalid; dropped first item"

            cmd = item["cmd"]
            if isinstance(cmd, str):
                cmd_list = shlex.split(cmd)
            elif isinstance(cmd, list):
                cmd_list = [str(x) for x in cmd]
            else:
                items.pop(0)
                f.seek(0)
                f.truncate()
                f.write(json.dumps(items, indent=2))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
                return "queue item cmd invalid type; dropped first item"
            if not cmd_list:
                items.pop(0)
                f.seek(0)
                f.truncate()
                f.write(json.dumps(items, indent=2))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
                return "queue item cmd empty; dropped first item"

            run_name = Path(str(item["results_dir"])).name
            if execute:
                # Recheck active runs under queue lock to avoid double-launch races
                # when multiple monitor instances run concurrently.
                active_now = detect_active_runs(root, runs_dir)
                if active_now:
                    return "queue launch skipped: active run detected during lock recheck"
                ok, detail = launch_background(root, runs_dir, cmd_list, run_name)
                if not ok:
                    return f"queue launch failed: {detail}"
                items.pop(0)
                f.seek(0)
                f.truncate()
                f.write(json.dumps(items, indent=2))
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
                return f"launched queued job {run_name} pid={detail}"
            return f"would launch queued job {run_name}"
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark monitor + queue launcher")
    p.add_argument("--root", default=os.environ.get("BENCHMARK_ROOT", str(Path.home() / "agentlife-benchmark")))
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--queue", default="runs/benchmark-queue.json")
    p.add_argument("--status-out", default="runs/monitor-status.json")
    p.add_argument("--summary-out", default="runs/monitor-summary.txt")
    p.add_argument("--auto-fix", action="store_true", help="attempt safe auto-resume for incomplete runs")
    p.add_argument("--launch-queue", action="store_true", help="launch next queued job when idle")
    p.add_argument("--execute", action="store_true", help="apply actions; default is dry-run")
    p.add_argument(
        "--notify-thread-id",
        default=os.environ.get("CODEX_THREAD_ID", ""),
        help="Codex thread/session id to post a wake-up turn into (default: CODEX_THREAD_ID env)",
    )
    p.add_argument(
        "--notify-on",
        choices=["always", "action", "problem"],
        default="action",
        help="When to post thread wake-up message",
    )
    p.add_argument(
        "--notify-prefix",
        default="[benchmark-monitor]",
        help="Prefix for posted thread messages",
    )
    return p.parse_args()


def should_notify(report: dict, mode: str) -> bool:
    if mode == "always":
        return True
    actions = report.get("actions") or []
    counts = report.get("counts") or {}
    has_problem = bool(counts.get("failed", 0) or counts.get("incomplete", 0))
    if mode == "action":
        return bool(actions)
    if mode == "problem":
        return has_problem
    return False


def build_notify_message(report: dict, prefix: str) -> str:
    counts = report.get("counts") or {}
    actions = report.get("actions") or []
    runs = report.get("runs") or []
    ts = report.get("timestamp", utc_now_iso())
    lines = [
        f"{prefix} heartbeat {ts}",
        f"active={counts.get('active',0)} complete={counts.get('complete',0)} failed={counts.get('failed',0)} incomplete={counts.get('incomplete',0)}",
    ]
    problems = [r for r in runs if r.get("state") in ("failed", "incomplete")]
    actives = [r for r in runs if r.get("state") == "active"]
    for r in problems[:8]:
        lines.append(
            f"problem: {r.get('name')} state={r.get('state')} kind={r.get('kind')} score={r.get('score')} reason={r.get('reason')}"
        )
    for r in actives[:5]:
        lines.append(
            f"active: {r.get('name')} kind={r.get('kind')} pid={r.get('active_pid')} score={r.get('score')}"
        )
    if actions:
        lines.append("actions:")
        for a in actions[:8]:
            lines.append(f"- {a}")
    lines.append("Please inspect benchmark state, decide next action, and report status.")
    return "\n".join(lines)


def post_thread_turn(thread_id: str, prompt: str, cwd: Path) -> Tuple[bool, str]:
    if not thread_id.strip():
        return False, "missing thread id"
    cmd = [
        "codex",
        "exec",
        "resume",
        thread_id.strip(),
        prompt,
        "--skip-git-repo-check",
        "--json",
    ]
    proc = run_cmd(cmd, cwd=cwd)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[-500:]
    return True, "posted"


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    runs_dir = (root / args.runs_dir).resolve()
    queue_path = (root / args.queue).resolve()
    status_out = (root / args.status_out).resolve()
    summary_out = (root / args.summary_out).resolve()

    if not runs_dir.exists():
        print(f"runs dir not found: {runs_dir}", file=sys.stderr)
        return 2

    active_map = detect_active_runs(root, runs_dir)
    run_dirs = [
        p for p in runs_dir.iterdir()
        if p.is_dir() and p.name not in ("successful-runs", "failed-runs")
    ]
    run_dirs.sort(key=run_start_sort_key, reverse=True)

    statuses = [classify_run(p, runs_dir, active_map) for p in run_dirs]
    by_state: Dict[str, List[RunStatus]] = {"active": [], "complete": [], "failed": [], "incomplete": []}
    for st in statuses:
        by_state.setdefault(st.state, []).append(st)

    actions: List[str] = []

    if args.auto_fix:
        for st in by_state.get("incomplete", []):
            cmd: Optional[List[str]] = None
            if st.kind == "agentlife":
                cmd = build_agentlife_resume_cmd(st.path)
            elif st.kind == "locomo":
                cmd = build_locomo_resume_cmd(st.path)

            if not cmd:
                continue

            if args.execute:
                ok, detail = launch_background(root, runs_dir, cmd, st.name)
                if ok:
                    actions.append(f"resumed {st.name} pid={detail}")
                else:
                    actions.append(f"resume failed {st.name}: {detail}")
            else:
                actions.append(f"would resume {st.name}: {' '.join(shlex.quote(c) for c in cmd)}")

    if args.launch_queue:
        q_action = maybe_launch_from_queue(
            root,
            runs_dir,
            queue_path,
            any_active=bool(by_state.get("active")),
            execute=args.execute,
        )
        if q_action:
            actions.append(q_action)

    report = {
        "timestamp": utc_now_iso(),
        "root": str(root),
        "runs_dir": str(runs_dir),
        "counts": {k: len(v) for k, v in by_state.items()},
        "actions": actions,
        "runs": [
            {
                "name": s.name,
                "kind": s.kind,
                "state": s.state,
                "reason": s.reason,
                "score": s.score,
                "active_pid": s.active_pid,
            }
            for s in statuses
        ],
    }

    if args.notify_thread_id and should_notify(report, args.notify_on):
        msg = build_notify_message(report, args.notify_prefix)
        ok, detail = post_thread_turn(args.notify_thread_id, msg, root)
        if ok:
            report["actions"].append(f"posted wake-up turn to thread {args.notify_thread_id}")
        else:
            report["actions"].append(f"failed to post wake-up turn: {detail}")

    atomic_write_text(status_out, json.dumps(report, indent=2) + "\n")

    lines = [
        f"[{report['timestamp']}] active={report['counts'].get('active',0)} "
        f"complete={report['counts'].get('complete',0)} failed={report['counts'].get('failed',0)} "
        f"incomplete={report['counts'].get('incomplete',0)}",
    ]
    for a in actions:
        lines.append(f"  action: {a}")
    for s in statuses:
        lines.append(f"  {s.state:10s} {s.name:40s} kind={s.kind} score={s.score!s}")
    atomic_write_text(summary_out, "\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
