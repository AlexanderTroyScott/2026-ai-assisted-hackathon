#!/usr/bin/env python
"""Average several submission CSVs into one (equal or explicit weights).

The D-025 population-shift finding says near-ties flip between local CV and the
leaderboard; averaging seeds or model families reduces variance (D-029, D-064).

Usage:
  python scripts/ensemble_blend.py --out outputs/submissions/exp-009.csv \
      outputs/submissions/exp-007-s0.csv outputs/submissions/exp-007-s1.csv \
      outputs/submissions/exp-007-s2.csv
  python scripts/ensemble_blend.py --weights 0.40,0.42,0.18 \
      --out outputs/submissions/exp-036-xgb-lgb-blend.csv \
      outputs/submissions/exp-035-xgb.csv outputs/submissions/exp-033.csv \
      outputs/submissions/exp-032.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FLOOR = 1e-3


def blend_frames(frames: list[pd.DataFrame], weights: list[float] | None = None) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one frame required")
    ref = frames[0]
    for i, f in enumerate(frames):
        if list(f.columns) != ["row_id", "prediction"]:
            raise ValueError(f"frame {i}: bad columns {list(f.columns)}")
        if not (f["row_id"] == ref["row_id"]).all():
            raise ValueError(f"frame {i}: row_id order/content differs from frame 0")

    if weights is None:
        w = np.full(len(frames), 1.0 / len(frames), dtype=float)
    else:
        if len(weights) != len(frames):
            raise ValueError(f"expected {len(frames)} weights, got {len(weights)}: {weights}")
        w = np.asarray(weights, dtype=float)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"weights must be non-negative with positive sum, got {weights}")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got sum={w.sum():.6f} ({weights})")
        w = w / w.sum()

    pred = sum(wi * f["prediction"].to_numpy(float) for wi, f in zip(w, frames))
    return pd.DataFrame({"row_id": ref["row_id"], "prediction": np.maximum(pred, FLOOR)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weights", default=None,
                    help="comma-separated blend weights summing to 1.0 (default: equal 1/N)")
    args = ap.parse_args()

    frames = [pd.read_csv(p, dtype={"row_id": str}) for p in args.inputs]
    weights = [float(x) for x in args.weights.split(",")] if args.weights else None
    try:
        out = blend_frames(frames, weights=weights)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  from {len(frames)} files  "
          f"mean={out.prediction.mean():.2f}  min={out.prediction.min():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
