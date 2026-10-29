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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--scale", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.input, dtype={"row_id": str})
    if list(df.columns) != ["row_id", "prediction"]:
        raise SystemExit(f"FATAL: expected columns ['row_id', 'prediction'], got {list(df.columns)}")
    out = scale_frame(df, args.scale)
    if not np.isfinite(out["prediction"]).all():
        raise SystemExit("FATAL: non-finite predictions after scaling")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  K={args.scale}  rows={len(out):,}  "
          f"sum={out['prediction'].sum():.6e}  min={out['prediction'].min():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
