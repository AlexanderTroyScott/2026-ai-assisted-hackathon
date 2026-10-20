#!/usr/bin/env python
"""Validate, submit and log a Kaggle submission for the IAA hackathon.

Usage
-----
  python scripts/kaggle_submit.py --input outputs/submissions/x.csv --validate-only
  python scripts/kaggle_submit.py --input outputs/submissions/x.csv \
      --tag exp-014 --message "D-014 experience features" --local-d2 0.118

Nothing is uploaded unless every validation check passes AND --submit is given.
Submissions are irreversible and capped at 10/day, so the default is to validate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw"
LOG_MD = ROOT / "docs" / "experiment_log.md"
LOG_CSV = ROOT / "outputs" / "experiment_log.csv"
COMP = "iaa-ai-assisted-hackathon-2026-part-1"
EXPECTED_ROWS = 626_639
FLOOR = 1e-3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(path: Path) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    try:
        df = pd.read_csv(path, dtype={"row_id": str})
    except Exception as exc:  # noqa: BLE001
        return [("CSV readable", False, str(exc))]

    checks.append(("columns", list(df.columns) == ["row_id", "prediction"], f"{list(df.columns)}"))
    checks.append(("row count", len(df) == EXPECTED_ROWS, f"{len(df)} expected {EXPECTED_ROWS}"))
    checks.append(("no duplicate row_id", df["row_id"].is_unique, f"{int(df['row_id'].duplicated().sum())} dups"))
    checks.append(("no NaN", not df.isna().any().any(), f"{int(df.isna().sum().sum())} NaN"))
    checks.append(("strictly positive", bool((df["prediction"] >= FLOOR).all()),
                   f"min={df['prediction'].min()} floor={FLOOR}"))
    checks.append(("numeric prediction", pd.api.types.is_numeric_dtype(df["prediction"]), str(df["prediction"].dtype)))

    sample = pd.read_parquet(DATA / "sample_submission.parquet")
    checks.append(("row_id set == sample", set(df["row_id"]) == set(sample["row_id"]),
                   f"{len(set(df['row_id']) ^ set(sample['row_id']))} differing ids"))

    # prefix must match source test file (M / B / C)
    prefixes = {"M": "motor", "B": "buildings", "C": "contents"}
    bad = 0
    for pref, line in prefixes.items():
        src = pd.read_parquet(DATA / f"test_{line}.parquet", columns=["row_id"])
        ids = set(src["row_id"])
        got = set(df.loc[df["row_id"].str.startswith(pref), "row_id"])
        bad += len(got ^ ids)
    checks.append(("prefix/LOB consistency", bad == 0, f"{bad} mismatched ids"))
    return checks


def log_submission(tag: str, message: str, local_d2: str, digest: str, lb_score: str, lb_rank: str, notes: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    row = [ts, tag, message, local_d2, lb_score, lb_rank, digest[:12], notes]
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="") as fh:
        w = csv.writer(fh)
        if write_header:
            w.writerow(["timestamp", "tag", "message", "local_d2", "lb_score", "lb_rank", "sha256", "notes"])
        w.writerow(row)

    if LOG_MD.exists():
        text = LOG_MD.read_text()
        marker = "\n## Submissions\n"
        line = (f"| {ts} | {tag} | {message} | {local_d2} | {lb_score} | {lb_rank} | "
                f"{digest[:12]} | {notes} |\n")
        if marker in text:
            text = text.replace(marker, marker + "\n" + line, 1)
            LOG_MD.write_text(text)


def fetch_leaderboard() -> tuple[str, str]:
    """Return (our best score, our best rank) if visible, else ('-','-')."""
    try:
        out = subprocess.run(
            ["kaggle", "competitions", "leaderboard", "-c", COMP, "--show"],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout
    except Exception:  # noqa: BLE001
        return ("-", "-")
    rows = [ln.split() for ln in out.splitlines() if ln.strip() and ln.split()[-1].replace(".", "").isdigit()]
    if not rows:
        return ("-", "-")
    best = max(rows, key=lambda r: float(r[-1]))
    return (best[-1], "posted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--message", default="")
    ap.add_argument("--local-d2", default="n/a")
    ap.add_argument("--notes", default="")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found")
        return 1

    checks = validate(args.input)
    width = max(len(c[0]) for c in checks)
    print("\nValidation gate")
    ok = True
    for name, passed, detail in checks:
        ok &= passed
        print(f"  {name:<{width}}  {'PASS' if passed else 'FAIL'}   {detail}")
    digest = sha256(args.input)
    print(f"  {'sha256':<{width}}  {digest}")

    if not ok:
        print("\nREFUSING TO SUBMIT — validation failed. Fix the generating code, not the CSV.")
        return 1

    if args.validate_only or not args.submit:
        print("\nValidation passed. Not uploading (use --submit to upload).")
        return 0

    print(f"\nUploading {args.input.name} to {COMP} with message: {args.message!r}")
    proc = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", COMP,
         "-f", str(args.input), "-m", args.message or args.tag],
        capture_output=True, text=True, timeout=180, check=False,
    )
    print(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0:
        print("Upload failed — nothing logged.")
        return proc.returncode

    score, rank = fetch_leaderboard()
    log_submission(args.tag, args.message, args.local_d2, digest, score, rank, args.notes)
    print(f"\nLogged to docs/experiment_log.md  (leaderboard: score={score} rank={rank})")
    print("Note: the leaderboard may take a few minutes to reflect a new submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
