#!/usr/bin/env python
"""C-5 / R-04: model diagnostics — normalised Gini gap, Lorenz, lift charts.

R-04 identifies the **train-versus-holdout normalised-Gini gap** as the overfit
measure to watch, rather than the level of the Gini itself:

    "A model generalizes well if its normalized Gini in a testing or holdout
     sample is reasonably close to its normalized Gini in the training sample."

This was the diagnostic gap flagged in `docs/actuarial_report.md` §6. It costs
no submission slot and is the one tool we had not been using while local D²
mis-ranked candidates twice.

Usage:
  python scripts/run_diagnostics.py
  python scripts/run_diagnostics.py --features base,premium,experience --fold 2023
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics import (  # noqa: E402
    calibration,
    gini_gap,
    lift_chart,
    lift_summary,
    lorenz_curve,
    normalized_gini,
    weighted_gini,
    data_gini,
)

MET = ROOT / "outputs" / "metrics"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="base,premium,experience,weather,flags,relativity,address,customer")
    ap.add_argument("--fold", default="2024", choices=["2024", "2023"])
    ap.add_argument("--tag", default="diag-001")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    import lightgbm as lgb
    from scripts.run_experiment import FOLDS, CAT_FEATURES, align, feature_cols, prepare

    holdout_year, train_max_year = FOLDS[args.fold]
    train, _ = prepare(args.features, development="lob", weather_lag=False)
    tr = train[train.acc_year <= train_max_year]
    va = train[train.acc_year == holdout_year]
    print(f"fold {args.fold}: train AY <= {train_max_year} ({len(tr):,}) "
          f"-> holdout AY {holdout_year} ({len(va):,})")

    cols = feature_cols(args.features)
    X_tr, X_va, use, cats = align(tr, va, cols, CAT_FEATURES)

    # D-053 tuned configuration, which is what our presented model uses
    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=3000, learning_rate=0.02, num_leaves=16,
        min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=3.0, verbose=-1, random_state=args.seed,
    )
    model.fit(X_tr, tr["y_ult"], eval_set=[(X_va, va["y_ult"])], eval_metric="tweedie",
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
              categorical_feature=cats)

    p_tr = np.maximum(model.predict(X_tr), 1e-3)
    p_va = np.maximum(model.predict(X_va), 1e-3)
    y_tr = tr["y_ult"].to_numpy()
    y_va = va["y_ult"].to_numpy()
    w_tr = tr["exposure"].to_numpy()
    w_va = va["exposure"].to_numpy()

    print("\n=== R-04 normalised Gini (model Gini / data Gini) ===")
    g_tr = weighted_gini(y_tr, w_tr, p_tr)
    g_va = weighted_gini(y_va, w_va, p_va)
    dg_tr, dg_va = data_gini(y_tr, w_tr), data_gini(y_va, w_va)
    ng_tr, ng_va = g_tr / dg_tr, g_va / dg_va
    print(f"  train   model Gini {g_tr:.4f}  data Gini {dg_tr:.4f}  normalized {ng_tr:.4f}")
    print(f"  holdout model Gini {g_va:.4f}  data Gini {dg_va:.4f}  normalized {ng_va:.4f}")
    print(f"  ** GAP (train - holdout) = {ng_tr - ng_va:+.4f} **")
    print()
    print("  R-04 worked example for scale: random forest gap 0.0106 beat")
    print("  linear regression gap 0.0201. Smaller gap = better generalisation.")

    gap = gini_gap((y_tr, w_tr, p_tr), (y_va, w_va, p_va))

    print(f"\n=== R-04 lift chart, holdout, {args.bins} equal-exposure bins ===")
    ch = lift_chart(y_va, w_va, p_va, bins=args.bins)
    s = lift_summary(ch)
    print(ch[["bin", "n", "exposure", "actual", "predicted", "actual_norm", "predicted_norm"]]
          .to_string(index=False, float_format=lambda x: f"{x:12.2f}"))
    print(f"\n  lift (last bin actual_norm - first) = {s['lift_distance']:+.3f}")
    print(f"  lift ratio (last / first)          = {s['lift_ratio']:.3f}")
    print(f"  calibration MAE (|actual-pred|)    = {s['calibration_mae']:.2f}")
    print("  R-04: read for (a) monotonicity of actuals — small reversals fine;")
    print("  (b) lift; (c) how closely predicted tracks actual.")

    cal_tr = calibration(y_tr, w_tr, p_tr)
    cal_va = calibration(y_va, w_va, p_va)
    print(f"\n=== Calibration (the gap Gini/lift cannot see) ===")
    print(f"  {'':10s} {'ratio':>9s} {'slope':>9s} {'in-the-large':>14s}")
    for nm, c in (("train", cal_tr), ("holdout", cal_va)):
        print(f"  {nm:10s} {c['calibration_ratio']:9.4f} {c['calibration_slope']:9.4f} "
              f"{c['calibration_in_large']:14.2f}")
    print("  ratio 1.0 = level calibrated; slope 1.0 = no over/under-prediction at the top.")
    print("  D-054: exp-025 outranked exp-022 on Gini and lift yet lost 0.011 on the")
    print("  board, because these are rank metrics and blind to level.")

    lc = lorenz_curve(y_va, w_va, p_va)
    # Gini = area between the Lorenz curve and the diagonal
    area = np.trapezoid(lc["cum_loss"], lc["cum_exposure"])
    print(f"\n=== R-04 Lorenz curve (holdout, sorted ascending by prediction) ===")
    print(f"  area under curve {area:.4f}  => Gini = 1 - 2*area = {1 - 2*area:.4f}")
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        i = int(frac * (len(lc) - 1))
        print(f"    lowest {int(frac*100):3d}% of exposure by prediction carries "
              f"{100*lc['cum_loss'].iloc[i]:5.1f}% of loss")

    MET.mkdir(parents=True, exist_ok=True)
    (MET / f"{args.tag}.json").write_text(json.dumps({
        "tag": args.tag, "fold": args.fold, "features": args.features,
        "seed": args.seed,
        "normalized_gini_train": ng_tr, "normalized_gini_holdout": ng_va,
        "gap": gap["gap"],
        "lift_distance": s["lift_distance"], "lift_ratio": s["lift_ratio"],
        "calibration_mae": s["calibration_mae"],
        **{f"train_{k}": v for k, v in cal_tr.items()},
        **{f"holdout_{k}": v for k, v in cal_va.items()},
    }, indent=2))
    ch.to_csv(MET / f"{args.tag}_lift.csv", index=False)
    print(f"\nwrote {MET / (args.tag + '.json')} and {MET / (args.tag + '_lift.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
