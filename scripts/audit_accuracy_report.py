#!/usr/bin/env python3
"""
scripts/audit_accuracy_report.py
─────────────────────────────────
Reads all Layer 3 audit decisions from the local JSONL log (data/audit_log.jsonl)
that have a filled-in ``ground_truth`` field and prints a calibration report:

  • Overall accuracy
  • False-Allow rate  (decision=ALLOW  but ground_truth=attack)
  • False-Block rate  (decision=BLOCK  but ground_truth=benign)
  • Risk-tier breakdown
  • Consensus vs. non-consensus accuracy

Usage:
    python scripts/audit_accuracy_report.py
    python scripts/audit_accuracy_report.py --log path/to/audit_log.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root without installing as a package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(log_path: Path) -> list:
    if not log_path.exists():
        print(f"[ERROR] Audit log not found: {log_path}")
        sys.exit(1)

    records = []
    with open(log_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Skipping malformed JSON on line {lineno}: {exc}")
    return records


def is_correct(decision: str, ground_truth: str) -> bool:
    """
    A decision is 'correct' if:
      ALLOW  → ground_truth == "benign"
      BLOCK  → ground_truth == "attack"
      ESCALATE → either (conservative; counted as correct in both cases)
    """
    gt = ground_truth.lower().strip()
    if decision == "ESCALATE":
        return True  # conservative escalation is always acceptable
    if decision == "ALLOW":
        return gt == "benign"
    if decision == "BLOCK":
        return gt == "attack"
    return False


def false_allow(decision: str, ground_truth: str) -> bool:
    return decision == "ALLOW" and ground_truth.lower().strip() == "attack"


def false_block(decision: str, ground_truth: str) -> bool:
    return decision == "BLOCK" and ground_truth.lower().strip() == "benign"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(records: list) -> None:
    # Filter to records with a filled-in ground_truth
    labelled = [r for r in records if r.get("ground_truth") not in (None, "", "null")]
    total = len(labelled)

    print("=" * 60)
    print("LAYER 3 AUDIT ACCURACY REPORT")
    print("=" * 60)
    print(f"Total audit records in log : {len(records)}")
    print(f"Records with ground truth  : {total}")

    if total == 0:
        print("\n[INFO] No labelled records found. Fill in the 'ground_truth'")
        print("       field in data/audit_log.jsonl and re-run this script.")
        return

    correct = sum(1 for r in labelled if is_correct(r["final_decision"], r["ground_truth"]))
    fa = sum(1 for r in labelled if false_allow(r["final_decision"], r["ground_truth"]))
    fb = sum(1 for r in labelled if false_block(r["final_decision"], r["ground_truth"]))

    print()
    print(f"Overall accuracy      : {correct}/{total}  ({correct/total*100:.1f}%)")
    print(f"False-Allow rate      : {fa}/{total}       ({fa/total*100:.1f}%)  ← attack slipped through as ALLOW")
    print(f"False-Block rate      : {fb}/{total}       ({fb/total*100:.1f}%)  ← benign blocked")
    print()

    # Risk-tier breakdown
    tier_stats = defaultdict(lambda: {"total": 0, "correct": 0, "fa": 0, "fb": 0})
    for r in labelled:
        tier = r.get("risk_tier", "UNKNOWN")
        tier_stats[tier]["total"] += 1
        if is_correct(r["final_decision"], r["ground_truth"]):
            tier_stats[tier]["correct"] += 1
        if false_allow(r["final_decision"], r["ground_truth"]):
            tier_stats[tier]["fa"] += 1
        if false_block(r["final_decision"], r["ground_truth"]):
            tier_stats[tier]["fb"] += 1

    print("Per-risk-tier breakdown:")
    print(f"  {'Tier':<12} {'Total':>6} {'Accuracy':>10} {'FA rate':>9} {'FB rate':>9}")
    print("  " + "-" * 50)
    for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL", "UNKNOWN"]:
        s = tier_stats.get(tier)
        if not s or s["total"] == 0:
            continue
        n = s["total"]
        print(
            f"  {tier:<12} {n:>6}"
            f"  {s['correct']/n*100:>8.1f}%"
            f"  {s['fa']/n*100:>8.1f}%"
            f"  {s['fb']/n*100:>8.1f}%"
        )

    print()

    # Consensus vs non-consensus
    cons_records = [r for r in labelled if r.get("consensus") is True]
    no_cons_records = [r for r in labelled if r.get("consensus") is False]

    if cons_records:
        cc = sum(1 for r in cons_records if is_correct(r["final_decision"], r["ground_truth"]))
        print(f"Consensus decisions   : {len(cons_records)}  →  accuracy {cc/len(cons_records)*100:.1f}%")
    if no_cons_records:
        nc = sum(1 for r in no_cons_records if is_correct(r["final_decision"], r["ground_truth"]))
        print(f"No-consensus ESCALATE : {len(no_cons_records)}  →  accuracy {nc/len(no_cons_records)*100:.1f}%")

    print()
    print("=" * 60)
    print("Tip: to add ground_truth, open data/audit_log.jsonl and set")
    print('     "ground_truth": "benign"  or  "ground_truth": "attack"')
    print("     for each record, then re-run this script.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3 audit accuracy report")
    parser.add_argument(
        "--log",
        type=Path,
        default=Config.AUDIT_LOG_PATH,
        help=f"Path to audit_log.jsonl (default: {Config.AUDIT_LOG_PATH})",
    )
    args = parser.parse_args()

    records = load_records(args.log)
    print_report(records)


if __name__ == "__main__":
    main()
