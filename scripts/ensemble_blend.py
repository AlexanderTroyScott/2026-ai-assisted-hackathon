#!/usr/bin/env python
"""Average several submission CSVs into one (equal weights).

The D-025 population-shift finding says near-ties flip between local CV and the
leaderboard; averaging seeds reduces the variance that drives those flips.

Usage:
  python scripts/ensemble_blend.py --out outputs/submissions/exp-009.csv \
      outputs/submissions/exp-007-s0.csv outputs/submissions/exp-007-s1.csv \
      outputs/submissions/exp-007-s2.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLOOR = 1e-3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    frames = [pd.read_csv(p, dtype={"row_id": str}) for p in args.inputs]
    ref = frames[0]
    for p, f in zip(args.inputs, frames):
        if list(f.columns) != ["row_id", "prediction"]:
            raise SystemExit(f"{p}: bad columns {list(f.columns)}")
        if not (f["row_id"] == ref["row_id"]).all():
            raise SystemExit(f"{p}: row_id order/content differs from {args.inputs[0]}")

    pred = sum(f["prediction"].to_numpy() for f in frames) / len(frames)
    out = pd.DataFrame({"row_id": ref["row_id"], "prediction": pred.clip(min=FLOOR)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  from {len(frames)} files  "
          f"mean={out.prediction.mean():.2f}  min={out.prediction.min():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
