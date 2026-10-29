#!/usr/bin/env python
"""Scale a submission CSV by a constant level factor (D-085 level probes).

One primary change per probe: the global scalar K applied to every prediction.
Used for the anchoring-convention experiment — the level scalar locates the
expected 2025 total that maximises board D², which is the empirical answer to
the D-009 anchoring ambiguity.

Usage
-----
  python scripts/scale_submission.py --input outputs/submissions/exp-046-xgb-lgb-blend.csv \
      --scale 0.955 --out outputs/submissions/exp-047-scale0955.csv
  python scripts/scale_submission.py --input outputs/submissions/blend.csv \
      --target-total 206079305 --out outputs/submissions/exp-054-calibrated.csv

`--target-total` implements the D-085 adopted convention: scale so the
predicted total equals the AY2024 developed actual ultimate total (a
data-derived quantity, `developed/` not a fitted constant).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 1e-3


def scale_frame(df: pd.DataFrame, k: float) -> pd.DataFrame:
    if k <= 0:
        raise ValueError(f"scale must be positive, got {k}")
    out = df.copy()
    out["prediction"] = np.maximum(out["prediction"].astype(float) * k, FLOOR)
    return out


def calibrate_to_total(df: pd.DataFrame, target_total: float) -> tuple[pd.DataFrame, float]:
    """D-085 adopted convention: scale so sum(prediction) == target_total.

    Returns (scaled frame, implied scale factor). The floor is applied after
    scaling; on real submissions no row is near the floor so the sum matches
    the target to machine precision.
    """
    if target_total <= 0:
        raise ValueError(f"target_total must be positive, got {target_total}")
    total = float(df["prediction"].astype(float).sum())
    if total <= 0:
        raise ValueError("input predictions sum to a non-positive total")
    k = target_total / total
    return scale_frame(df, k), k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--scale", type=float, default=None,
                    help="constant multiplier applied to every prediction")
    ap.add_argument("--target-total", type=float, default=None,
                    help="D-085: scale so sum(prediction) equals this total "
                         "(e.g. the AY2024 developed actual ultimate total)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if (args.scale is None) == (args.target_total is None):
        ap.error("give exactly one of --scale or --target-total")

    df = pd.read_csv(args.input, dtype={"row_id": str})
    if list(df.columns) != ["row_id", "prediction"]:
        raise SystemExit(f"FATAL: expected columns ['row_id', 'prediction'], got {list(df.columns)}")
    if args.target_total is not None:
        out, k = calibrate_to_total(df, args.target_total)
    else:
        k = args.scale
        out = scale_frame(df, k)
    if not np.isfinite(out["prediction"]).all():
        raise SystemExit("FATAL: non-finite predictions after scaling")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  K={k:.6f}  rows={len(out):,}  "
          f"sum={out['prediction'].sum():.6e}  min={out['prediction'].min():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
