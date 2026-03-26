import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_run_state as brs  # noqa: E402


def _write_lines(path: Path, count: int) -> None:
    path.write_text("".join(f"line {i}\n" for i in range(count)), encoding="utf-8")


def test_build_run_detail_matches_status_report_for_active_rolling_run(tmp_path, monkeypatch):
    root = tmp_path
    run_name = "quaid-l-r999-20260325-000000"
    run_dir = root / "runs" / run_name
    (run_dir / "logs" / "daemon").mkdir(parents=True)
    (run_dir / "data" / "session-cursors").mkdir(parents=True)

    transcript = run_dir / "synthetic.jsonl"
    _write_lines(transcript, 100)

    cursor_meta = {
        "line_offset": 40,
        "transcript_path": str(transcript),
    }
    (run_dir / "data" / "session-cursors" / "obd-session.json").write_text(json.dumps(cursor_meta), encoding="utf-8")

    rolling_rows = [
        {
            "event": "rolling_stage",
            "session_id": "obd-session",
            "rolling_batches": 3,
            "new_cursor_offset": 40,
            "staged_fact_count": 17,
            "wall_seconds": 12.3,
        }
    ]
    (run_dir / "logs" / "daemon" / "rolling-extraction.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rolling_rows),
        encoding="utf-8",
    )
    (root / "runs" / f"{run_name}.launch.log").write_text("ingest schedule=rolling-obd\n", encoding="utf-8")

    monkeypatch.setattr(brs, "detect_active_runs", lambda _root, _dirs: {run_name: 12345})

    report = brs.build_status_report(root)
    detail = brs.build_run_detail(root, run_name)

    expected = "chunk 3 | 40/100 | facts 17"
    assert report["runs"][0]["current_active_item"] == expected
    assert detail["state"] == "active"
    assert detail["current_active_item"] == expected
    assert detail["phase"] == expected
    assert detail["active_pid"] == 12345


def test_run_progress_ignores_rolling_flush_for_per_day_runs(tmp_path):
    root = tmp_path
    run_name = "quaid-s-r997-20260325-000000"
    run_dir = root / "runs" / run_name
    (run_dir / "logs" / "daemon").mkdir(parents=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    rolling_rows = [
        {
            "event": "rolling_flush",
            "signal_to_publish_seconds": 86.0,
            "publish_wall_seconds": 5.8,
        }
    ]
    (run_dir / "logs" / "daemon" / "rolling-extraction.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rolling_rows),
        encoding="utf-8",
    )
    (run_dir / "logs" / "janitor_progress.json").write_text(
        json.dumps({"phase": "janitor(9/20)"}),
        encoding="utf-8",
    )
    (root / "runs" / f"{run_name}.launch.log").write_text(
        "  Ingest schedule: per-day\n  Auto-rolling days: 2026-03-11, 2026-03-18\n",
        encoding="utf-8",
    )

    assert brs.run_progress(root, run_name) == "janitor(9/20)"


def test_infer_metric_label_does_not_mark_per_day_run_as_obd(tmp_path):
    root = tmp_path
    run_name = "quaid-s-r996-20260325-000000"
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True)
    (root / "runs" / f"{run_name}.launch.log").write_text(
        "  Ingest schedule: per-day\n  Auto-rolling days: 2026-03-11, 2026-03-18\n",
        encoding="utf-8",
    )

    assert brs.infer_metric_label(root, run_name) == "AL-S Quaid"


def test_build_run_detail_uses_shared_failed_classification(tmp_path, monkeypatch):
    root = tmp_path
    run_name = "quaid-l-r998-20260325-000000"
    run_dir = root / "runs" / run_name
    run_dir.mkdir(parents=True)
    (root / "runs" / f"{run_name}.launch.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: boom\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(brs, "detect_active_runs", lambda _root, _dirs: {})

    report = brs.build_status_report(root)
    detail = brs.build_run_detail(root, run_name)

    assert report["runs"][0]["state"] == "failed"
    assert detail["state"] == "failed"
    assert detail["reason"] == "failure marker in logs"
    assert detail["current_active_item"] == "failure marker in logs"
    assert detail["phase"] == "failure marker in logs"
