#!/usr/bin/env python3
"""Complete the Quaid L benchmark that crashed at extraction 56/60.

The main run crashed on day 2026-04-25→2026-04-26 extraction timeout.
Extraction for 04-25 was completed manually. This script:
1. Injects remaining sessions (04-26 through 05-01) into the JSONL
2. Extracts at each day boundary
3. Runs final janitor
4. Then vm_benchmark.py --eval-only can be used for evaluation
"""

import sys
import time
import subprocess
import shlex
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent))

from dataset import SESSION_DATES, load_all_reviews, load_filler_reviews
from vm_benchmark import (
    TartVM, transcript_to_messages, messages_to_gateway_jsonl,
    VM_AGENT_SESSIONS_DIR, _run_vm_janitor,
)

VM_IP = "192.168.64.3"
SESSION_ID = "benchmark-quaid"
EXTRACT_MODEL = "claude-sonnet-4-5-20250929"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results-vm-L"
FILLER_DIR = Path(__file__).parent.parent / "data" / "filler-sessions-L"
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"

# The last successfully extracted day
LAST_EXTRACTED_DAY = "2026-04-25"


def main():
    vm = TartVM(VM_IP)

    # Load all reviews
    arc_reviews = load_all_reviews(ASSETS_DIR)
    filler_reviews = load_filler_reviews(FILLER_DIR)

    # Combine and sort chronologically
    all_reviews = list(arc_reviews) + list(filler_reviews)

    def get_date(r):
        snum = r.session_num
        if snum < 0:
            from dataset import FILLER_DATES
            filler_id = f"F{abs(snum):03d}"
            return FILLER_DATES.get(filler_id, "2026-03-15")
        return SESSION_DATES.get(snum, "2026-03-01")

    all_reviews.sort(key=lambda r: (get_date(r), r.session_num))

    # Filter to sessions AFTER the last extracted day
    remaining = []
    for r in all_reviews:
        date = get_date(r).split(" ")[0] if " " in get_date(r) else get_date(r)
        if date > LAST_EXTRACTED_DAY:
            remaining.append((r, date))

    print(f"Remaining sessions to inject: {len(remaining)}")
    if not remaining:
        print("Nothing to do!")
        return

    # Clear the session file (it was truncated after 04-25 extraction)
    session_file = f"{VM_AGENT_SESSIONS_DIR}/{SESSION_ID}.jsonl"

    current_day = None
    session_tokens = 0

    for review, date in remaining:
        snum = review.session_num
        label = f"F{abs(snum):03d}" if snum < 0 else f"Session {snum}"
        messages = transcript_to_messages(review)
        if not messages:
            continue

        print(f"  {label} ({date}, {len(messages)} msgs)", end="", flush=True)

        # Day boundary detection
        if current_day is not None and date != current_day:
            print(f"\n  --- Day boundary: {current_day} → {date} ---")
            if session_tokens > 0:
                print(f"  [DAILY EXTRACT — {session_tokens:,} tokens]")
                # Run extraction with generous timeout
                result = vm.ssh(
                    f"python3 ~/extract_compact.py "
                    f"--session-file {session_file} "
                    f"--workspace ~/clawd "
                    f"--user-name Maya --owner-id maya "
                    f"--session-id {SESSION_ID} "
                    f"--model {EXTRACT_MODEL} "
                    f"--date {current_day}",
                    timeout=600,
                )
                if result.returncode != 0:
                    print(f"  [EXTRACT FAILED: {result.stderr[:200]}]")
                else:
                    for line in (result.stdout or "").split("\n"):
                        if any(line.startswith(p) for p in [
                            "Extraction complete:", "Extraction API call:",
                            "Transcript:", "DB verify:", "LLM returned",
                        ]):
                            print(f"  {line}")

                # Clear session file
                vm.ssh(f": > {session_file}", timeout=5, raw=True)
                session_tokens = 0

            # Run janitor
            print("  [JANITOR]", end="")
            j_usage = _run_vm_janitor(vm)
            print(f" [{j_usage.get('api_calls', 0)} calls, ${j_usage.get('cost_usd', 0):.3f}]")

        current_day = date

        # Inject messages
        jsonl = messages_to_gateway_jsonl(messages)
        result = vm.ssh(
            f"mkdir -p {VM_AGENT_SESSIONS_DIR} && cat >> {session_file}",
            input_data=jsonl,
            timeout=30,
        )
        if result.returncode != 0:
            print(f" [INJECT FAILED: {result.stderr[:100]}]")
            continue

        # Count tokens roughly
        for msg in messages:
            session_tokens += len(msg["content"]) // 4  # rough estimate

        print()

    # Final extraction
    if session_tokens > 0:
        print(f"\n  [FINAL EXTRACT — {session_tokens:,} est. tokens]")
        result = vm.ssh(
            f"python3 ~/extract_compact.py "
            f"--session-file {session_file} "
            f"--workspace ~/clawd "
            f"--user-name Maya --owner-id maya "
            f"--session-id {SESSION_ID} "
            f"--model {EXTRACT_MODEL} "
            f"--date {current_day}",
            timeout=600,
        )
        if result.returncode != 0:
            print(f"  [EXTRACT FAILED: {result.stderr[:200]}]")
        else:
            for line in (result.stdout or "").split("\n"):
                if any(line.startswith(p) for p in [
                    "Extraction complete:", "Extraction API call:",
                    "Transcript:", "DB verify:", "LLM returned",
                ]):
                    print(f"  {line}")

    # Final janitor
    print("\n  [FINAL JANITOR]")
    j_usage = _run_vm_janitor(vm)
    print(f"  [{j_usage.get('api_calls', 0)} calls, ${j_usage.get('cost_usd', 0):.3f}]")

    # Check final DB state
    result = vm.ssh(
        "sqlite3 ~/clawd/data/memory.db \"SELECT COUNT(*) || ' nodes, status: ' || "
        "GROUP_CONCAT(status || '=' || cnt) FROM (SELECT status, COUNT(*) as cnt "
        "FROM nodes GROUP BY status)\"",
        timeout=10,
    )
    print(f"\n  Final DB: {result.stdout.strip()}")

    # Freeze session file
    frozen_file = f"{VM_AGENT_SESSIONS_DIR}/{SESSION_ID}-injected.jsonl"
    vm.ssh(
        f"cp {session_file} {frozen_file} 2>/dev/null; "
        f"rm -f {session_file} 2>/dev/null",
        timeout=10,
    )
    print(f"  Session isolation: frozen → {SESSION_ID}-injected.jsonl")

    # Copy DB to results
    results_quaid = RESULTS_DIR / "quaid"
    results_quaid.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "sshpass", "-p", "admin", "scp",
        "-o", "StrictHostKeyChecking=no",
        "-o", "PreferredAuthentications=password",
        f"admin@{VM_IP}:~/clawd/data/memory.db",
        str(results_quaid / "memory.db"),
    ], timeout=30)
    print(f"  DB copied to {results_quaid / 'memory.db'}")

    print("\n  DONE! Now run: python3 vm_benchmark.py --system quaid --eval-only "
          f"--results-dir {RESULTS_DIR} --answer-model {EXTRACT_MODEL}")


if __name__ == "__main__":
    main()
