#!/usr/bin/env python3
"""Export a Claude Code transcript into benchmark day-sliced JSONL files.

The exported format matches the existing imported Claude history fixtures:
each line is a simplified JSON object with `role`, `content`, and `timestamp`.
Only human-readable `user` turns and assistant text blocks are retained.
Operational days are bucketed using a configurable cutoff hour.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Claude Code transcript JSONL to export",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for manifest + imported-claude-day-XXX.jsonl files",
    )
    parser.add_argument(
        "--cutoff-hour",
        type=int,
        default=4,
        help="Operational day boundary in UTC hours (default: 4)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="Limit export to the first N operational days (0 = all)",
    )
    return parser.parse_args()


def _parse_timestamp(raw: str) -> datetime:
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def _format_timestamp(ts: datetime) -> str:
    utc_ts = ts.astimezone(timezone.utc)
    if utc_ts.microsecond:
        return utc_ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return utc_ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _flatten_record(obj: Dict[str, object]) -> Optional[Dict[str, str]]:
    record_type = obj.get("type")
    message = obj.get("message")
    if not isinstance(message, dict):
        return None

    if record_type == "user":
        content = message.get("content")
        role = "user"
        parts: List[str] = []
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        text = "\n\n".join(parts).strip()
        if not text:
            return None
    elif record_type == "assistant":
        content = message.get("content")
        role = "assistant"
        parts: List[str] = []
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        text = "\n\n".join(parts).strip()
        if not text:
            return None
    else:
        return None

    raw_ts = obj.get("timestamp")
    if not isinstance(raw_ts, str) or not raw_ts.strip():
        return None
    timestamp = _format_timestamp(_parse_timestamp(raw_ts))
    return {"role": role, "content": text, "timestamp": timestamp}


def _iter_bucketed_rows(source_path: Path, cutoff_hour: int) -> Iterator[tuple[str, Dict[str, str]]]:
    cutoff = timedelta(hours=int(cutoff_hour))
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            flattened = _flatten_record(obj)
            if not flattened:
                continue
            ts = _parse_timestamp(flattened["timestamp"])
            operational_day = (ts - cutoff).date().isoformat()
            yield operational_day, flattened


def _group_rows(source_path: Path, cutoff_hour: int) -> OrderedDict[str, List[Dict[str, str]]]:
    grouped: OrderedDict[str, List[Dict[str, str]]] = OrderedDict()
    for operational_day, row in _iter_bucketed_rows(source_path, cutoff_hour):
        grouped.setdefault(operational_day, []).append(row)
    return grouped


def _ensure_empty_output_dir(path: Path) -> None:
    if path.exists():
        if any(path.iterdir()):
            raise RuntimeError(f"Output directory already exists and is not empty: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _day_export_metadata(rows: List[Dict[str, str]]) -> Dict[str, object]:
    role_counts = {
        "user": sum(1 for row in rows if row.get("role") == "user"),
        "assistant": sum(1 for row in rows if row.get("role") == "assistant"),
    }
    content_chars = sum(len(str(row.get("content", ""))) for row in rows)
    return {
        "message_count": len(rows),
        "role_counts": role_counts,
        "content_chars": content_chars,
        "first_timestamp": rows[0]["timestamp"] if rows else None,
        "last_timestamp": rows[-1]["timestamp"] if rows else None,
    }


def main() -> None:
    args = _parse_args()
    source_path = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not source_path.exists():
        raise RuntimeError(f"Source transcript not found: {source_path}")
    _ensure_empty_output_dir(output_dir)

    grouped = _group_rows(source_path, args.cutoff_hour)
    ordered_days = list(grouped.items())
    if args.max_days > 0:
        ordered_days = ordered_days[: int(args.max_days)]
    if not ordered_days:
        raise RuntimeError("No exportable operational days found")

    manifest_days: List[Dict[str, object]] = []
    total_messages = 0
    total_content_chars = 0
    for index, (operational_day, rows) in enumerate(ordered_days, start=1):
        session_id = f"imported-claude-day-{index:03d}"
        output_path = output_dir / f"{session_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True))
                handle.write("\n")
        day_meta = _day_export_metadata(rows)
        total_messages += int(day_meta["message_count"])
        total_content_chars += int(day_meta["content_chars"])
        manifest_days.append(
            {
                "session_id": session_id,
                "operational_day": operational_day,
                **day_meta,
                "path": str(output_path),
            }
        )

    manifest = {
        "schema_version": 2,
        "source_format": "claude-code-jsonl",
        "export_format": "imported-claude-day-jsonl",
        "message_fields": ["role", "content", "timestamp"],
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "cutoff_hour": int(args.cutoff_hour),
        "days_exported": len(manifest_days),
        "messages_exported": total_messages,
        "content_chars_exported": total_content_chars,
        "days": manifest_days,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Exported {len(manifest_days)} day(s) and {total_messages} message(s).")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
