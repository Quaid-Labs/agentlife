#!/usr/bin/env python3
"""Simple benchmark status dashboard.

Serves a tiny web UI with live run state from Spark monitor output.

Usage:
  python3 scripts/benchmark_dashboard.py --port 8765
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import parse_qs, urlparse, quote
from typing import Any, Dict


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Status</title>
  <style>
    :root { --bg:#0b1020; --fg:#e8eeff; --muted:#9fb0da; --ok:#21c17a; --warn:#f0a53a; --bad:#ff5d6c; }
    body { margin:0; padding:20px; background:linear-gradient(140deg,#0b1020,#14213a); color:var(--fg); font:14px/1.45 "SF Mono","JetBrains Mono","Menlo",monospace; }
    h1 { margin:0 0 8px; font-size:20px; }
    .muted { color:var(--muted); }
    .cards { display:flex; gap:10px; flex-wrap:wrap; margin:12px 0 18px; }
    .card { border:1px solid #2c3f6b; border-radius:8px; padding:10px 12px; min-width:110px; background:rgba(255,255,255,0.02); }
    .n { font-size:18px; font-weight:700; }
    table { width:100%; border-collapse:collapse; background:rgba(255,255,255,0.02); }
    th,td { border-bottom:1px solid #273a63; padding:8px 6px; text-align:left; vertical-align:top; }
    th { color:#b8c7ef; }
    .active{ color:var(--ok);} .failed{ color:var(--bad);} .incomplete{ color:var(--warn);} .complete{ color:#8bd3ff; }
    code { color:#c6d5ff; }
  </style>
</head>
<body>
  <h1>Benchmark Status</h1>
  <div id="meta" class="muted">Loading…</div>
  <div class="cards">
    <div class="card"><div class="muted">Active</div><div id="c_active" class="n">-</div></div>
    <div class="card"><div class="muted">Complete</div><div id="c_complete" class="n">-</div></div>
    <div class="card"><div class="muted">Failed</div><div id="c_failed" class="n">-</div></div>
    <div class="card"><div class="muted">Incomplete</div><div id="c_incomplete" class="n">-</div></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Run #</th>
        <th>Benchmark Metric</th>
        <th>Type</th>
        <th>Status</th>
        <th>Provider Lane</th>
        <th>Threads</th>
        <th>Current Active Item</th>
        <th>ETA</th>
        <th>Final Score</th>
        <th>Deep Dive</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
<script>
function metricFromName(name) {
  const n = (name || '').toLowerCase();
  if (n.includes('oc-native') && (n.includes('als') || n.includes('al-s'))) return 'AL-S OC Native';
  if (n.includes('oc-native') && (n.includes('all') || n.includes('al-l'))) return 'AL-L OC Native';
  if (n.includes('locomo')) return 'LoCoMo Quaid';
  if (n.includes('longmemeval')) return 'LongMemEval Quaid';
  if (n.includes('quaid-s') || n.includes('al-s')) return 'AL-S Quaid';
  if (n.includes('quaid-l') || n.includes('al-l')) return 'AL-L Quaid';
  if (n.includes('mem0')) return 'mem0 AL-L';
  return 'Unknown';
}
function typeFromName(name) {
  const n = (name || '').toLowerCase();
  return n.includes('smoke') ? 'SMOKE' : 'FULL';
}
function laneFromName(name) {
  const n = (name || '').toLowerCase();
  if (n.includes('oc-native')) return 'local-vm';
  if (n.includes('vllm')) return 'vLLM';
  if (n.includes('-api') || n.includes('api-')) return 'api';
  return 'mixed';
}
function statusMap(state) {
  const m = {active:'ACTIVE', complete:'DONE', failed:'FAILED', incomplete:'INCOMPLETE'};
  return m[state] || (state || '').toUpperCase();
}
function runIdFromName(name) {
  const m = String(name || '').match(/r\\d+/i);
  return m ? m[0].toLowerCase() : (name || '');
}
function fmtDuration(sec) {
  if (sec == null || Number.isNaN(Number(sec))) return 'unknown';
  const s = Math.max(0, Math.round(Number(sec)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${ss}s`;
  return `${ss}s`;
}
function runCreatedTs(name) {
  const s = String(name || '');
  const m = s.match(/-(\\d{8})-(\\d{6})(?:$|[^0-9])/);
  if (!m) return 0;
  const ds = m[1];
  const ts = m[2];
  const iso = `${ds.slice(0,4)}-${ds.slice(4,6)}-${ds.slice(6,8)}T${ts.slice(0,2)}:${ts.slice(2,4)}:${ts.slice(4,6)}Z`;
  return Date.parse(iso) || 0;
}
async function refresh() {
  const r = await fetch(`/api/status?ts=${Date.now()}`, { cache: 'no-store' });
  const d = await r.json();
  document.getElementById('meta').textContent = `Updated: ${d.timestamp || '-'} | root: ${d.root || '-'}`;
  const c = d.counts || {};
  for (const k of ['active','complete','failed','incomplete']) {
    document.getElementById('c_'+k).textContent = c[k] ?? 0;
  }
  const runs = (d.runs || []).slice().sort((a,b)=>{
    const aActive = a.state === 'active' ? 1 : 0;
    const bActive = b.state === 'active' ? 1 : 0;
    if (aActive !== bActive) return bActive - aActive;
    const at = runCreatedTs(a.name) || Date.parse(a.started_at || a.completed_at || '') || 0;
    const bt = runCreatedTs(b.name) || Date.parse(b.started_at || b.completed_at || '') || 0;
    if (at !== bt) return bt - at;
    return String(a.name || '').localeCompare(String(b.name || ''));
  });
  const rows = document.getElementById('rows');
  rows.innerHTML = runs.map(x => `
    <tr>
      <td><code>${runIdFromName(x.name)}</code></td>
      <td>${metricFromName(x.name)}</td>
      <td>${typeFromName(x.name)}</td>
      <td class="${x.state || ''}">${statusMap(x.state)}</td>
      <td>${laneFromName(x.name)}</td>
      <td>${x.parallel ?? 'unknown'}</td>
      <td>${x.current_active_item || (x.reason || '')}</td>
      <td>${x.state === 'complete' ? fmtDuration(x.elapsed_seconds) : 'unknown'}</td>
      <td>${x.final_score != null ? `${Number(x.final_score).toFixed(2)}%` : (x.preview_score != null ? `~${Number(x.preview_score).toFixed(2)}%` : '')}</td>
      <td><a style="color:#9fd0ff" href="/run?name=${encodeURIComponent(x.name || '')}">Open</a></td>
      <td>${x.name || ''}${x.score != null ? ` | monitor=${x.score}` : ''}</td>
    </tr>
  `).join('');
}
refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""

DETAIL_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Run Deep Dive</title>
  <style>
    body { margin:0; padding:20px; background:#0b1020; color:#e8eeff; font:14px/1.45 "SF Mono","JetBrains Mono","Menlo",monospace; }
    h1,h2 { margin:0 0 10px; }
    .muted { color:#9fb0da; margin-bottom:14px; }
    a { color:#9fd0ff; }
    table { width:100%; border-collapse:collapse; margin:14px 0 22px; }
    th,td { border-bottom:1px solid #273a63; padding:7px 6px; text-align:left; }
    th { color:#b8c7ef; }
  </style>
</head>
<body>
  <h1 id="title">Run Deep Dive</h1>
  <div class="muted"><a href="/">← Back</a></div>
  <div id="meta" class="muted">Loading…</div>
  <h2>Per Type</h2><table><thead><tr><th>Type</th><th>Count</th><th>Accuracy</th></tr></thead><tbody id="per_type"></tbody></table>
  <h2>Per Theme</h2><table><thead><tr><th>Theme</th><th>Count</th><th>Accuracy</th></tr></thead><tbody id="per_theme"></tbody></table>
  <h2>Per Difficulty</h2><table><thead><tr><th>Difficulty</th><th>Count</th><th>Accuracy</th></tr></thead><tbody id="per_difficulty"></tbody></table>
<script>
function rows(id, obj) {
  const el = document.getElementById(id);
  const entries = Object.entries(obj || {}).sort((a,b)=> (b[1]?.count || 0) - (a[1]?.count || 0));
  el.innerHTML = entries.map(([k,v]) => `<tr><td>${k}</td><td>${v.count ?? ''}</td><td>${v.accuracy ?? ''}</td></tr>`).join('');
}
async function run() {
  const p = new URLSearchParams(location.search);
  const name = p.get('name') || '';
  document.getElementById('title').textContent = `Run Deep Dive: ${name}`;
  const r = await fetch('/api/run?name=' + encodeURIComponent(name));
  const d = await r.json();
  document.getElementById('meta').textContent =
    `state=${d.state || '-'} phase=${d.phase || '-'} final=${d.final_score ?? '-'} elapsed=${d.elapsed_seconds ?? '-'}s completed_at=${d.completed_at || '-'} chunks=${d.chunk_count ?? 0} eval=${d.eval_completed ?? '-'}/${d.eval_total ?? '-'} nodes=${JSON.stringify(d.nodes || {})}`;
  rows('per_type', d.per_type || {});
  rows('per_theme', d.per_theme || {});
  rows('per_difficulty', d.per_difficulty || {});
}
run();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve simple benchmark status dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--spark-host", default="spark")
    p.add_argument("--remote-root", default="/home/solomon/agentlife-benchmark")
    p.add_argument("--ttl-seconds", type=int, default=15)
    args = p.parse_args()
    if not re.fullmatch(r"/[A-Za-z0-9._/\-]+", args.remote_root or ""):
        p.error("--remote-root must be an absolute path with safe characters [A-Za-z0-9._/-]")
    return args


_resolved_root_cache: Dict[str, Any] = {"root": None, "ts": 0.0}


def resolve_remote_root(spark_host: str, preferred_root: str) -> str:
    now = time.time()
    if _resolved_root_cache["root"] and now - float(_resolved_root_cache["ts"]) < 300:
        return str(_resolved_root_cache["root"])

    candidates = []
    for c in (preferred_root, "/home/solomon/quaid/benchmark", "/home/solomon/clawd-benchmark"):
        if c and c not in candidates:
            candidates.append(c)
    cmd = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"candidates = {candidates!r}\n"
        "for c in candidates:\n"
        "    r = Path(c)\n"
        "    if not (r / 'runs').exists():\n"
        "        continue\n"
        "    if (r / 'eval' / 'run_production_benchmark.py').exists() or (r / 'agentlife' / 'eval' / 'run_production_benchmark.py').exists():\n"
        "        print(c)\n"
        "        raise SystemExit(0)\n"
        "print(candidates[0] if candidates else '')\n"
        "PY"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", spark_host, cmd],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        return preferred_root
    out = (proc.stdout or "").strip().splitlines()
    resolved = out[-1].strip() if out else preferred_root
    _resolved_root_cache["root"] = resolved
    _resolved_root_cache["ts"] = now
    return resolved


def _local_results_base() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _local_vm_eval_count(run_dir: Path) -> int | None:
    eval_path = run_dir / "oc-native" / "eval_results.json"
    if not eval_path.exists():
        return None
    try:
        payload = json.loads(eval_path.read_text())
    except Exception:
        return None
    return len(payload) if isinstance(payload, list) else None


def _local_vm_live_results_dirs() -> set[str]:
    try:
        proc = subprocess.run(
            ["pgrep", "-af", "vm_benchmark.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return set()
    if proc.returncode not in (0, 1):
        return set()
    live: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        m = re.search(r"--results-dir\s+(\S+)", line)
        if m:
            live.add(m.group(1))
    return live


def _fetch_local_vm_status() -> Dict[str, Any]:
    base = _local_results_base()
    live_results_dirs = _local_vm_live_results_dirs()
    runs: list[Dict[str, Any]] = []
    for path in sorted(base.glob("results-vm-*")):
        if not path.is_dir():
            continue
        if "oc-native" not in path.name:
            continue
        all_results = path / "all_results.json"
        name = path.name.replace("results-vm-", "", 1)
        row: Dict[str, Any] = {
            "name": name,
            "state": "active",
            "parallel": 1,
            "started_at": None,
            "completed_at": None,
            "elapsed_seconds": None,
            "final_score": None,
            "current_active_item": "starting",
            "reason": "",
            "source": "local-vm",
        }
        if all_results.exists():
            try:
                payload = json.loads(all_results.read_text())
                native = payload.get("oc-native") or {}
                overall = ((native.get("scores") or {}).get("overall") or {})
                row["final_score"] = overall.get("accuracy")
                row["elapsed_seconds"] = (
                    ((native.get("injection") or {}).get("elapsed_s") or 0)
                    + sum((item.get("eval_duration_s") or 0) for item in json.loads((path / "oc-native" / "eval_results.json").read_text()))
                    if (path / "oc-native" / "eval_results.json").exists()
                    else (native.get("injection") or {}).get("elapsed_s")
                )
                row["state"] = "complete"
                row["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(all_results.stat().st_mtime))
                row["current_active_item"] = "complete"
            except Exception:
                row["state"] = "incomplete"
                row["reason"] = "bad all_results.json"
        else:
            eval_count = _local_vm_eval_count(path)
            rel_path = str(path.relative_to(base.parent))
            is_live = rel_path in live_results_dirs
            if eval_count is not None and is_live:
                row["current_active_item"] = f"eval {eval_count}/268"
            elif (path / "oc-native" / "injection_stats.json").exists() and is_live:
                row["current_active_item"] = "eval starting"
            elif (path / "oc-native").exists() and is_live:
                row["current_active_item"] = "injecting"
            elif eval_count is not None:
                row["state"] = "incomplete"
                row["current_active_item"] = f"stopped at eval {eval_count}/268"
                row["reason"] = "local VM process not running"
            else:
                row["state"] = "incomplete"
                row["current_active_item"] = "stopped before eval"
                row["reason"] = "local VM process not running"
        runs.append(row)
    counts = {
        "active": sum(1 for r in runs if r["state"] == "active"),
        "complete": sum(1 for r in runs if r["state"] == "complete"),
        "failed": sum(1 for r in runs if r["state"] == "failed"),
        "incomplete": sum(1 for r in runs if r["state"] == "incomplete"),
    }
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(base),
        "counts": counts,
        "runs": runs,
    }


def _merge_status_reports(remote: Dict[str, Any], local: Dict[str, Any]) -> Dict[str, Any]:
    runs = list(remote.get("runs") or []) + list(local.get("runs") or [])
    counts = {"active": 0, "complete": 0, "failed": 0, "incomplete": 0}
    for item in runs:
        state = str(item.get("state") or "")
        if state in counts:
            counts[state] += 1
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": f"{remote.get('root') or '-'} | local={local.get('root') or '-'}",
        "counts": counts,
        "runs": runs,
    }


def _empty_status(root: str, error: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": root,
        "counts": {"active": 0, "complete": 0, "failed": 0, "incomplete": 0},
        "runs": [],
    }
    if error:
        payload["error"] = error
    return payload


def _fetch_local_vm_run_detail(name: str) -> Dict[str, Any]:
    path = _local_results_base() / f"results-vm-{name}"
    out: Dict[str, Any] = {
        "name": name,
        "state": "unknown",
        "final_score": None,
        "elapsed_seconds": None,
        "started_at": None,
        "completed_at": None,
        "per_type": {},
        "per_theme": {},
        "per_difficulty": {},
        "phase": "starting",
        "chunk_count": 0,
        "eval_completed": None,
        "eval_total": 268,
        "nodes": {},
    }
    eval_path = path / "oc-native" / "eval_results.json"
    all_results = path / "all_results.json"
    if eval_path.exists():
        try:
            payload = json.loads(eval_path.read_text())
            if isinstance(payload, list):
                out["eval_completed"] = len(payload)
                out["phase"] = f"eval {len(payload)}/268"
        except Exception:
            pass
    if all_results.exists():
        payload = json.loads(all_results.read_text())
        native = payload.get("oc-native") or {}
        scores = native.get("scores") or {}
        out["state"] = "complete"
        out["final_score"] = ((scores.get("overall") or {}).get("accuracy"))
        out["per_type"] = scores.get("per_type") or {}
        out["per_theme"] = scores.get("per_theme") or {}
        out["per_difficulty"] = scores.get("per_difficulty") or {}
        out["phase"] = "complete"
        out["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(all_results.stat().st_mtime))
        if eval_path.exists():
            ep = json.loads(eval_path.read_text())
            out["elapsed_seconds"] = sum((item.get("eval_duration_s") or 0) for item in ep)
    elif (path / "oc-native").exists():
        out["state"] = "active"
    return out


def fetch_remote_status(spark_host: str, remote_root: str) -> Dict[str, Any]:
    remote_root = resolve_remote_root(spark_host, remote_root)
    cmd = (
        "python3 "
        f"{shlex.quote(remote_root)}/scripts/monitor_benchmarks.py "
        f"--root {shlex.quote(remote_root)} "
        "--status-out runs/monitor-status.json "
        "--summary-out runs/monitor-summary.txt >/dev/null && "
        f"python3 - {shlex.quote(remote_root)} <<'PY'\n"
        "import json\n"
        "import re\n"
        "import sys\n"
        "from datetime import datetime\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "report = json.loads((root / 'runs' / 'monitor-status.json').read_text())\n"
        "def _infer_parallel(run_name):\n"
        "    candidates = [\n"
        "        root / 'runs' / f'{run_name}.launch.log',\n"
        "        root / 'runs' / str(run_name) / 'launch.log',\n"
        "        root / 'runs' / str(run_name) / 'run.log',\n"
        "    ]\n"
        "    pat = re.compile(r'--parallel\\s+(\\d+)')\n"
        "    for p in candidates:\n"
        "        if not p.exists():\n"
        "            continue\n"
        "        try:\n"
        "            txt = p.read_text(errors='ignore')\n"
        "        except Exception:\n"
        "            continue\n"
        "        m = pat.search(txt)\n"
        "        if m:\n"
        "            try:\n"
        "                return int(m.group(1))\n"
        "            except Exception:\n"
        "                pass\n"
        "    return None\n"
        "def _extract_final_score(run_name):\n"
        "    candidates = [\n"
        "      root / 'runs' / str(run_name) / 'final_scores.json',\n"
        "      root / 'runs' / str(run_name) / 'scores.json',\n"
        "    ]\n"
        "    for p in candidates:\n"
        "      if not p.exists():\n"
        "        continue\n"
        "      try:\n"
        "        d = json.loads(p.read_text())\n"
        "      except Exception:\n"
        "        continue\n"
        "      try:\n"
        "        b = d.get('blended',{}).get('blended',{}).get('pct')\n"
        "        if b is not None:\n"
        "          return float(b)\n"
        "      except Exception:\n"
        "        pass\n"
        "      try:\n"
        "        o = d.get('scores',{}).get('overall',{}).get('accuracy')\n"
        "        if o is not None:\n"
        "          return float(o)\n"
        "      except Exception:\n"
        "        pass\n"
        "    return None\n"
        "def _extract_preview_score(run_name):\n"
        "    p = root / 'runs' / str(run_name) / 'evaluation_results.json'\n"
        "    if p.exists():\n"
        "      try:\n"
        "        rows = json.loads(p.read_text())\n"
        "      except Exception:\n"
        "        rows = None\n"
        "      if isinstance(rows, list) and rows:\n"
        "        correct = sum(1 for r in rows if r.get('judge_label') == 'CORRECT')\n"
        "        partial = sum(1 for r in rows if r.get('judge_label') == 'PARTIAL')\n"
        "        wrong = sum(1 for r in rows if r.get('judge_label') == 'WRONG')\n"
        "        scored = correct + partial + wrong\n"
        "        if scored > 0:\n"
        "          return round((correct + 0.5 * partial) / scored * 100.0, 2)\n"
        "    launch_candidates = [\n"
        "      root / 'runs' / f'{run_name}.launch.log',\n"
        "      root / 'runs' / str(run_name) / 'launch.log',\n"
        "      root / 'runs' / str(run_name) / 'run.log',\n"
        "    ]\n"
        "    for lp in launch_candidates:\n"
        "      if not lp.exists():\n"
        "        continue\n"
        "      try:\n"
        "        txt = lp.read_text(errors='ignore')\n"
        "      except Exception:\n"
        "        continue\n"
        "      matches = re.findall(r'\\[(\\d+)/(\\d+)\\|q\\d+\\].*?\\[(\\d+(?:\\.\\d+)?)%\\]', txt)\n"
        "      if matches:\n"
        "        return float(matches[-1][2])\n"
        "    return None\n"
        "def _run_progress(run_name):\n"
        "    run = root / 'runs' / str(run_name)\n"
        "    launch_candidates = [\n"
        "        root / 'runs' / f'{run_name}.launch.log',\n"
        "        run / 'launch.log',\n"
        "        run / 'run.log',\n"
        "    ]\n"
        "    chunk_count = 0\n"
        "    try:\n"
        "        chunk_count = len(list((run / 'extraction_cache').glob('chunk-*.json')))\n"
        "    except Exception:\n"
        "        chunk_count = 0\n"
        "    last_chunk = None\n"
        "    total_chunks = None\n"
        "    prog = run / 'extraction_cache' / 'progress.json'\n"
        "    if prog.exists():\n"
        "        try:\n"
        "            p = json.loads(prog.read_text())\n"
        "            lc = p.get('last_completed_chunk')\n"
        "            tc = p.get('total_chunks')\n"
        "            if isinstance(lc, int):\n"
        "                last_chunk = lc\n"
        "            if isinstance(tc, int) and tc > 0:\n"
        "                total_chunks = tc\n"
        "        except Exception:\n"
        "            pass\n"
        "    if total_chunks is None:\n"
        "        for lp in launch_candidates:\n"
        "            if not lp.exists():\n"
        "                continue\n"
        "            try:\n"
        "                m = re.search(r'Extraction chunks: (\\d+)', lp.read_text(errors='ignore'))\n"
        "            except Exception:\n"
        "                m = None\n"
        "            if m:\n"
        "                total_chunks = int(m.group(1))\n"
        "                break\n"
        "    eval_done = None\n"
        "    eval_total = None\n"
        "    ev = run / 'logs' / 'eval_progress.json'\n"
        "    if ev.exists():\n"
        "        try:\n"
        "            e = json.loads(ev.read_text())\n"
        "            lcq = e.get('last_completed_query')\n"
        "            if isinstance(lcq, int):\n"
        "                eval_done = max(0, lcq + 1)\n"
        "            else:\n"
        "                eval_done = e.get('completed')\n"
        "            eval_total = e.get('total_queries') if e.get('total_queries') is not None else e.get('total')\n"
        "        except Exception:\n"
        "            pass\n"
        "    if eval_total is not None:\n"
        "        return f'eval {eval_done}/{eval_total}'\n"
        "    for lp in launch_candidates:\n"
        "        if not lp.exists():\n"
        "            continue\n"
        "        try:\n"
        "            txt = lp.read_text(errors='ignore')\n"
        "        except Exception:\n"
        "            continue\n"
        "        matches = re.findall(r'\\[(\\d+)/(\\d+)\\|q\\d+\\].*?\\[(\\d+(?:\\.\\d+)?)%\\]', txt)\n"
        "        if matches:\n"
        "            cur, total, pct = matches[-1]\n"
        "            return f'eval {cur}/{total} ({pct}%)'\n"
        "    janitor_progress = run / 'logs' / 'janitor_progress.json'\n"
        "    if janitor_progress.exists():\n"
        "        try:\n"
        "            jp = json.loads(janitor_progress.read_text())\n"
        "            phase = str(jp.get('phase') or '').strip()\n"
        "            if phase:\n"
        "                return phase.lower()\n"
        "        except Exception:\n"
        "            pass\n"
        "    # If janitor telemetry/stats exist, show janitor progress by current chunk.\n"
        "    janitor_seen = (run / 'logs' / 'janitor-task-telemetry.jsonl').exists() or (run / 'logs' / 'janitor-stats.json').exists()\n"
        "    if janitor_seen and total_chunks is not None:\n"
        "        if last_chunk is None:\n"
        "            current = 1\n"
        "        else:\n"
        "            current = max(1, min(total_chunks, last_chunk + 1))\n"
        "        return f'janitor {current}/{total_chunks}'\n"
        "    if last_chunk is not None and total_chunks is not None:\n"
        "        done = max(0, min(total_chunks, last_chunk + 1))\n"
        "        return f'extraction {done}/{total_chunks}'\n"
        "    if chunk_count > 0:\n"
        "        if total_chunks is not None:\n"
        "            done = max(0, min(total_chunks, chunk_count))\n"
        "            return f'extraction {done}/{total_chunks}'\n"
        "        return f'extraction {chunk_count}/?'\n"
        "    # FC baselines do not emit eval_progress.json; parse the live launch log instead.\n"
        "    fc_step = None\n"
        "    for lp in launch_candidates:\n"
        "        if not lp.exists():\n"
        "            continue\n"
        "        try:\n"
        "            txt = lp.read_text(errors='ignore')\n"
        "        except Exception:\n"
        "            continue\n"
        "        if 'FULL-CONTEXT BASELINE (' not in txt and 'TIER 5 FC BASELINE (' not in txt:\n"
        "            continue\n"
        "        matches = re.findall(r'\\[(\\d+)/(\\d+)\\].*?\\[(\\d+(?:\\.\\d+)?)%\\]', txt)\n"
        "        if matches:\n"
        "            cur, total, pct = matches[-1]\n"
        "            fc_step = f'fc {cur}/{total} ({pct}%)'\n"
        "            break\n"
        "        m = re.search(r'\\b(\\d+) queries, (\\d+) sessions\\b', txt)\n"
        "        if m:\n"
        "            fc_step = f'fc 0/{m.group(1)}'\n"
        "            break\n"
        "        fc_step = 'fc starting'\n"
        "        break\n"
        "    if fc_step is not None:\n"
        "        return fc_step\n"
        "    return 'starting'\n"
        "for r in report.get('runs', []):\n"
        "    meta_path = root / 'runs' / str(r.get('name', '')) / 'run_metadata.json'\n"
        "    elapsed = None\n"
        "    parallel = None\n"
        "    started_at = None\n"
        "    completed_at = None\n"
        "    if meta_path.exists():\n"
        "        try:\n"
        "            m = json.loads(meta_path.read_text())\n"
        "            parallel = m.get('parallel')\n"
        "            started_at = m.get('started_at')\n"
        "            completed_at = m.get('completed_at')\n"
        "            if m.get('total_elapsed_seconds') is not None:\n"
        "                elapsed = float(m['total_elapsed_seconds'])\n"
        "            elif m.get('started_at') and m.get('completed_at'):\n"
        "                started = datetime.fromisoformat(str(m['started_at']).replace('Z', '+00:00'))\n"
        "                ended = datetime.fromisoformat(str(m['completed_at']).replace('Z', '+00:00'))\n"
        "                elapsed = max(0.0, (ended - started).total_seconds())\n"
        "        except Exception:\n"
        "            elapsed = None\n"
        "            parallel = None\n"
        "    if parallel is None:\n"
        "        parallel = _infer_parallel(r.get('name', ''))\n"
        "    if completed_at and r.get('state') == 'incomplete':\n"
        "        r['state'] = 'complete'\n"
        "        r['reason'] = 'run_metadata.completed_at present'\n"
        "    r['elapsed_seconds'] = elapsed\n"
        "    r['parallel'] = parallel\n"
        "    r['started_at'] = started_at\n"
        "    r['completed_at'] = completed_at\n"
        "    r['final_score'] = _extract_final_score(r.get('name', ''))\n"
        "    r['preview_score'] = None if r['final_score'] is not None else _extract_preview_score(r.get('name', ''))\n"
        "    r['current_active_item'] = _run_progress(r.get('name', '')) if r.get('state') == 'active' else (r.get('reason') or '')\n"
        "report['root'] = str(root)\n"
        "print(json.dumps(report))\n"
        "PY"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", spark_host, cmd],
        capture_output=True,
        text=True,
        timeout=25,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ssh error").strip()[-400:])
    return json.loads(proc.stdout)


def fetch_remote_run_detail(spark_host: str, remote_root: str, run_name: str) -> Dict[str, Any]:
    remote_root = resolve_remote_root(spark_host, remote_root)
    cmd = (
        f"python3 - {shlex.quote(remote_root)} {shlex.quote(run_name)} <<'PY'\n"
        "import json\n"
        "import glob\n"
        "import sqlite3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "run_name = sys.argv[2]\n"
        "run = root / 'runs' / run_name\n"
        "out = {\n"
        "  'name': run_name,\n"
        "  'state': 'unknown',\n"
        "  'final_score': None,\n"
        "  'elapsed_seconds': None,\n"
        "  'started_at': None,\n"
        "  'completed_at': None,\n"
        "  'per_type': {},\n"
        "  'per_theme': {},\n"
        "  'per_difficulty': {},\n"
        "  'phase': None,\n"
        "  'chunk_count': 0,\n"
        "  'eval_completed': None,\n"
        "  'eval_total': None,\n"
        "  'nodes': {},\n"
        "}\n"
        "meta = run / 'run_metadata.json'\n"
        "if meta.exists():\n"
        "  try:\n"
        "    m = json.loads(meta.read_text())\n"
        "    out['started_at'] = m.get('started_at')\n"
        "    out['completed_at'] = m.get('completed_at') or m.get('ended_at')\n"
        "    out['elapsed_seconds'] = m.get('total_elapsed_seconds') if m.get('total_elapsed_seconds') is not None else m.get('duration_sec')\n"
        "    status = str(m.get('status', '')).lower()\n"
        "    out['state'] = 'complete' if (out['completed_at'] or status == 'completed') else 'incomplete'\n"
        "  except Exception:\n"
        "    pass\n"
        "for fn in ('final_scores.json','scores.json'):\n"
        "  p = run / fn\n"
        "  if not p.exists():\n"
        "    continue\n"
        "  try:\n"
        "    d = json.loads(p.read_text())\n"
        "  except Exception:\n"
        "    continue\n"
        "  b = d.get('blended',{}).get('blended',{}).get('pct')\n"
        "  if b is not None:\n"
        "    out['final_score'] = b\n"
        "  elif d.get('scores',{}).get('overall',{}).get('accuracy') is not None:\n"
        "    out['final_score'] = d.get('scores',{}).get('overall',{}).get('accuracy')\n"
        "  s = d.get('scores',{})\n"
        "  out['per_type'] = s.get('per_type', {}) or {}\n"
        "  out['per_theme'] = s.get('per_theme', {}) or {}\n"
        "  out['per_difficulty'] = s.get('per_difficulty', {}) or {}\n"
        "  break\n"
        "chunks = sorted(glob.glob(str(run / 'extraction_cache' / 'chunk-*.json')))\n"
        "out['chunk_count'] = len(chunks)\n"
        "ev = run / 'logs' / 'eval_progress.json'\n"
        "if ev.exists():\n"
        "  try:\n"
        "    e = json.loads(ev.read_text())\n"
        "    lcq = e.get('last_completed_query')\n"
        "    if isinstance(lcq, int):\n"
        "      out['eval_completed'] = max(0, lcq + 1)\n"
        "    else:\n"
        "      out['eval_completed'] = e.get('completed')\n"
        "    out['eval_total'] = e.get('total_queries') if e.get('total_queries') is not None else e.get('total')\n"
        "  except Exception:\n"
        "    pass\n"
        "db = run / 'data' / 'memory.db'\n"
        "if db.exists():\n"
        "  try:\n"
        "    conn = sqlite3.connect(str(db))\n"
        "    cur = conn.cursor()\n"
        "    for s in ('pending','approved','active','archived','flagged'):\n"
        "      out['nodes'][s] = int(cur.execute('SELECT COUNT(*) FROM nodes WHERE status=?',(s,)).fetchone()[0])\n"
        "    conn.close()\n"
        "  except Exception:\n"
        "    pass\n"
        "if out.get('final_score') is not None:\n"
        "  out['phase'] = 'complete'\n"
        "elif out.get('eval_total') is not None:\n"
        "  out['phase'] = f\"eval {out.get('eval_completed')}/{out.get('eval_total')}\"\n"
        "else:\n"
        "  last_chunk = None\n"
        "  total_chunks = None\n"
        "  prog = run / 'extraction_cache' / 'progress.json'\n"
        "  if prog.exists():\n"
        "    try:\n"
        "      p = json.loads(prog.read_text())\n"
        "      lc = p.get('last_completed_chunk')\n"
        "      tc = p.get('total_chunks')\n"
        "      if isinstance(lc, int):\n"
        "        last_chunk = lc\n"
        "      if isinstance(tc, int) and tc > 0:\n"
        "        total_chunks = tc\n"
        "    except Exception:\n"
        "      pass\n"
        "  if total_chunks is None:\n"
        "    launch_candidates = [\n"
        "      root / 'runs' / f'{run_name}.launch.log',\n"
        "      run / 'launch.log',\n"
        "      run / 'run.log',\n"
        "    ]\n"
        "    for lp in launch_candidates:\n"
        "      if not lp.exists():\n"
        "        continue\n"
        "      try:\n"
        "        m = re.search(r'Extraction chunks: (\\d+)', lp.read_text(errors='ignore'))\n"
        "      except Exception:\n"
        "        m = None\n"
        "      if m:\n"
        "        total_chunks = int(m.group(1))\n"
        "        break\n"
        "  janitor_progress = run / 'logs' / 'janitor_progress.json'\n"
        "  if janitor_progress.exists():\n"
        "    try:\n"
        "      jp = json.loads(janitor_progress.read_text())\n"
        "      phase = str(jp.get('phase') or '').strip()\n"
        "      if phase:\n"
        "        out['phase'] = phase.lower()\n"
        "      else:\n"
        "        out['phase'] = 'janitor'\n"
        "    except Exception:\n"
        "      pass\n"
        "  if out.get('phase') is None:\n"
        "    janitor_seen = (run / 'logs' / 'janitor-task-telemetry.jsonl').exists() or (run / 'logs' / 'janitor-stats.json').exists()\n"
        "    if janitor_seen and total_chunks is not None:\n"
        "      current = 1 if last_chunk is None else max(1, min(total_chunks, last_chunk + 1))\n"
        "      out['phase'] = f\"janitor {current}/{total_chunks}\"\n"
        "    elif total_chunks is not None and last_chunk is not None:\n"
        "      out['phase'] = f\"extraction {max(0, min(total_chunks, last_chunk + 1))}/{total_chunks}\"\n"
        "    elif out.get('chunk_count', 0) > 0:\n"
        "      out['phase'] = f\"extract {out.get('chunk_count')} chunks\"\n"
        "    else:\n"
        "      out['phase'] = 'starting'\n"
        "print(json.dumps(out))\n"
        "PY"
    )
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", spark_host, cmd],
        capture_output=True,
        text=True,
        timeout=25,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ssh error").strip()[-400:])
    return json.loads(proc.stdout)


def run_server(args: argparse.Namespace) -> None:
    cache: Dict[str, Any] = {"ts": 0.0, "data": None, "error": None}

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            q = parse_qs(parsed.query)
            if self.path in ("/", "/index.html"):
                raw = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/run":
                raw = DETAIL_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/api/run":
                name = (q.get("name") or [""])[0]
                if not name:
                    self._send_json(400, {"error": "missing name"})
                    return
                try:
                    if name.startswith("oc-native-"):
                        detail = _fetch_local_vm_run_detail(name)
                    else:
                        detail = fetch_remote_run_detail(args.spark_host, args.remote_root, name)
                except Exception as e:
                    self._send_json(502, {"error": str(e)})
                    return
                self._send_json(200, detail)
                return
            if path == "/api/status":
                now = time.time()
                if cache["data"] is None or now - cache["ts"] > args.ttl_seconds:
                    remote_error = None
                    try:
                        remote = fetch_remote_status(args.spark_host, args.remote_root)
                    except Exception as e:
                        remote = _empty_status(args.remote_root, error=str(e))
                        remote_error = str(e)
                    try:
                        local = _fetch_local_vm_status()
                    except Exception as e:
                        local = _empty_status(str(_local_results_base()), error=str(e))
                    cache["data"] = _merge_status_reports(remote, local)
                    if remote_error:
                        cache["data"]["warning"] = f"remote status unavailable: {remote_error}"
                    cache["ts"] = now
                    cache["error"] = None
                self._send_json(200, cache["data"] or {})
                return
            self.send_error(404, "Not found")

        def log_message(self, fmt: str, *a: Any) -> None:
            return

    srv = ThreadedHTTPServer((args.host, args.port), Handler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Spark host: {args.spark_host}")
    srv.serve_forever()


if __name__ == "__main__":
    run_server(parse_args())
