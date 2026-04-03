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
        <th>Model</th>
        <th>Score</th>
        <th>Deep Dive</th>
        <th>Notes</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
<script>
function metricFromRun(x) {
  if (x.metric_label) return x.metric_label;
  const n = (x.name || '').toLowerCase();
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
  if (n.includes('llama-cpp')) return 'llama.cpp';
  if (n.includes('vllm')) return 'vLLM';
  if (n.includes('-api') || n.includes('api-')) return 'api';
  return 'mixed';
}
function laneDisplay(x) {
  if (x.provider_lane) return x.provider_lane;
  return laneFromName(x.name);
}
function modelDisplay(x) {
  return x.model_lane || '';
}
function statusMap(state) {
  const m = {active:'ACTIVE', complete:'DONE', failed:'FAILED', incomplete:'INCOMPLETE'};
  return m[state] || (state || '').toUpperCase();
}
function runIdFromName(name) {
  const m = String(name || '').match(/r\\d+/i);
  return m ? m[0].toLowerCase() : (name || '');
}
function runNumFromName(name) {
  const m = String(name || '').match(/r(\\d+)/i);
  return m ? Number(m[1]) : -1;
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
    const ar = runNumFromName(a.name);
    const br = runNumFromName(b.name);
    if (ar !== br) return br - ar;
    const at = runCreatedTs(a.name) || Date.parse(a.started_at || a.completed_at || '') || 0;
    const bt = runCreatedTs(b.name) || Date.parse(b.started_at || b.completed_at || '') || 0;
    if (at !== bt) return bt - at;
    return String(a.name || '').localeCompare(String(b.name || ''));
  });
  const rows = document.getElementById('rows');
  rows.innerHTML = runs.map(x => `
    <tr>
      <td><code>${runIdFromName(x.name)}</code></td>
      <td>${metricFromRun(x)}</td>
      <td>${typeFromName(x.name)}</td>
      <td class="${x.state || ''}">${statusMap(x.state)}</td>
      <td>${laneDisplay(x)}</td>
      <td>${x.parallel ?? 'unknown'}</td>
      <td>${x.current_active_item || (x.reason || '')}</td>
      <td>${modelDisplay(x)}</td>
      <td>${x.final_score != null ? `${Number(x.final_score).toFixed(2)}%` : (x.preview_score != null ? `Live ${Number(x.preview_score).toFixed(2)}%` : '')}</td>
      <td><a style="color:#9fd0ff" href="/run?name=${encodeURIComponent(x.name || '')}">Open</a></td>
      <td>${x.note || ''}</td>
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
  const progressLabel = (d.day_count != null)
    ? `${d.current_day ?? 0}/${d.day_count} days`
    : `${d.chunk_count ?? 0} chunks`;
  document.getElementById('meta').textContent =
    `state=${d.state || '-'} phase=${d.phase || '-'} final=${d.final_score ?? '-'} elapsed=${d.elapsed_seconds ?? '-'}s completed_at=${d.completed_at || '-'} progress=${progressLabel} eval=${d.eval_completed ?? '-'}/${d.eval_total ?? '-'} nodes=${JSON.stringify(d.nodes || {})}`;
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
            "provider_lane": "local-vm",
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
        f"python3 {shlex.quote(remote_root)}/scripts/monitor_benchmarks.py "
        f"--root {shlex.quote(remote_root)} "
        "--status-out runs/monitor-status.json "
        "--summary-out runs/monitor-summary.txt >/dev/null && "
        f"python3 - {shlex.quote(remote_root)} <<'PY'\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "report = json.loads((root / 'runs' / 'monitor-status.json').read_text())\n"
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
        "import sys\n"
        "from pathlib import Path\n"
        "root = Path(sys.argv[1])\n"
        "run_name = sys.argv[2]\n"
        "sys.path.insert(0, str(root / 'scripts'))\n"
        "from benchmark_run_state import build_run_detail\n"
        "print(json.dumps(build_run_detail(root, run_name)))\n"
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
