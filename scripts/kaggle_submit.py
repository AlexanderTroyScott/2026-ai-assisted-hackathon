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
import io
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

    # D-052: exposure-scaling sanity. The training truth has a FLAT implied rate
    # (cost/exposure) across exposure bands: 437/430/417/389, max/min = 1.12.
    # A submission whose implied rate rises steeply with exposure is
    # double-counting exposure and will collapse on the leaderboard even when
    # local D2 looks excellent — that is exactly what exp-024 did (0.1365 local,
    # 0.06019 LB). This check would have caught it before spending a slot.
    try:
        exp_rows = []
        for ln, pre in (("motor", "M"), ("buildings", "B"), ("contents", "C")):
            t = pd.read_parquet(DATA / f"test_{ln}.parquet",
                                columns=["row_id", "period_start", "period_end"])
            ps = pd.to_datetime(t["period_start"])
            pe = pd.to_datetime(t["period_end"])
            t = t.assign(exposure=((pe - ps).dt.days + 1) / 365.0)
            exp_rows.append(t[["row_id", "exposure"]])
        exp_df = pd.concat(exp_rows, ignore_index=True)
        d = df.merge(exp_df, on="row_id", how="left")
        d["implied_rate"] = d["prediction"] / d["exposure"].clip(lower=1e-3)
        bands = pd.cut(d["exposure"], [0, 0.3, 0.6, 0.9, 1.01])
        g = d.groupby(bands, observed=True)["implied_rate"].mean()
        ratio = float(g.max() / g.min()) if g.min() > 0 else float("inf")
        flat = ratio <= 1.5
        checks.append(("exposure scaling flat", flat,
                       f"implied rate max/min={ratio:.2f} (truth 1.12; >1.5 => "
                       f"exposure likely double-counted)"))
    except Exception as exc:  # noqa: BLE001
        # D-055: this gate must FAIL CLOSED. It was previously fail-open, so a
        # broken exposure merge let a double-counted submission through as a
        # pass - the exact failure mode the gate exists to catch (D-052).
        checks.append(("exposure scaling flat", False,
                       f"check FAILED TO RUN: {exc} - refusing rather than skipping"))
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
        lines = text.splitlines(keepends=True)
        hdr = next((i for i, ln in enumerate(lines) if ln.startswith("| # | timestamp |")), None)
        if hdr is not None:
            # next row number = highest existing | n | in the table
            n = 0
            i = hdr + 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                first = lines[i].strip().lstrip("|").split("|")[0].strip()
                if first.isdigit():
                    n = max(n, int(first))
                i += 1
            row10 = (f"| {n + 1} | {ts} | {tag} | {message} | {notes} | {local_d2} | "
                     f"{lb_score} | — | clean | `{digest[:12]}` |\n")
            lines.insert(i, row10)
            LOG_MD.write_text("".join(lines))
        else:
            marker = "\n## Submissions\n"
            line = (f"| {ts} | {tag} | {message} | {local_d2} | {lb_score} | {lb_rank} | "
                    f"{digest[:12]} | {notes} |\n")
            if marker in text:
                LOG_MD.write_text(text.replace(marker, marker + "\n" + line, 1))


def fetch_our_score() -> tuple[str, str]:
    """Return (score of our latest submission, 'posted') from our own submission
    list. Never take the board maximum — that is another team's score."""
    try:
        out = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", COMP, "--csv"],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout
    except Exception:  # noqa: BLE001
        return ("-", "-")
    rows = list(csv.DictReader(io.StringIO(out)))
    if not rows:
        return ("-", "-")
    rows.sort(key=lambda r: r.get("date", ""), reverse=True)
    score = rows[0].get("publicScore") or "-"
    return (score, "posted")


def check_quota() -> int:
    """Print the daily submission quota; return the number of slots left today.

    User-directed hold rule: if the quota has reset to 10 (fresh window), do not
    submit without explicit human confirmation.
    """
    try:
        out = subprocess.run(
            [sys.executable, "-c",
             "from kaggle.api.kaggle_api_extended import KaggleApi;"
             "a=KaggleApi(); a.authenticate();"
             "l=a.competition_get_submission_limits(%r);"
             "print(l.num_today, l.num_allowed_now)" % COMP],
            capture_output=True, text=True, timeout=60, check=False,
        ).stdout.split()
        # The `kaggle` package prints a version warning to stdout on import, so
        # filter to the numeric tokens (the two limit values come last).
        nums = [t for t in out if t.lstrip("-").isdigit()]
        num_today, allowed = int(nums[0]), int(nums[1])
        print(f"  quota           used today {num_today}/10, remaining {allowed}")
        return allowed
    except Exception as exc:  # noqa: BLE001
        print(f"  quota           UNAVAILABLE ({exc}) — refuse to submit blind")
        return -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--tag", default="untagged")
    ap.add_argument("--message", default="")
    ap.add_argument("--local-d2", default="n/a")
    ap.add_argument("--notes", default="")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--ack-quota-reset", action="store_true",
                    help="required to submit when the daily quota has reset to 10")
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
    allowed = check_quota()

    if not ok:
        print("\nREFUSING TO SUBMIT — validation failed. Fix the generating code, not the CSV.")
        return 1

    if args.validate_only or not args.submit:
        print("\nValidation passed. Not uploading (use --submit to upload).")
        return 0

    if allowed == 0:
        print("\nREFUSING TO SUBMIT — daily quota exhausted.")
        return 1
    if allowed < 0:
        print("\nREFUSING TO SUBMIT — quota could not be verified.")
        return 1
    if allowed == 10 and not args.ack_quota_reset:
        print("\nHOLD — the daily quota has reset to 10 (fresh window).")
        print("User-directed policy: do not submit without explicit human confirmation.")
        print("Re-run with --ack-quota-reset if a human has approved spending a fresh window.")
        return 1

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

    score, rank = fetch_our_score()
    log_submission(args.tag, args.message, args.local_d2, digest, score, rank, args.notes)
    print(f"\nLogged to docs/experiment_log.md  (leaderboard: score={score} rank={rank})")
    print("Note: the leaderboard may take a few minutes to reflect a new submission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
