#!/usr/bin/env python
"""C-2: Tweedie GLM benchmark against the LGBM stack.

Usage:
  python scripts/run_glm.py --features base,premium,address,customer
  python scripts/run_glm.py --save-pred glm-001

Gate from docs/references.md: the GLM must land within ~0.01 local D2 of LGBM
to be worth using as the second leg of the C-3 form ensemble. The number is
reported either way — a defensible relativities table is worth having even if
the GLM is never blended.

Reports honest (train-fitted) D2 and, for the write-up, exp(coefficient)
relativities against the dropped base level of each categorical.
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
from sklearn.metrics import d2_tweedie_score  # noqa: E402

from src.dataset import LINES  # noqa: E402
from src.glm import fit_glm, predict_cost, prepare_design, relativities  # noqa: E402

OUT = ROOT / "outputs" / "submissions"
MET = ROOT / "outputs" / "metrics"

# An interpretable rating-plan spec: the factors a pricing actuary would put in
# a GLM, deliberately narrower than the LGBM feature soup so the relativities
# read cleanly.
#
# `occupation` is deliberately EXCLUDED (measured 2026-10-21). It costs 21
# dummy parameters and adds 0.0001 of D2 to the tree model (honest 0.1331 ->
# 0.1330 on removal), while being collinear with driver_age / years_licensed /
# ncb_years. Decomposition: 64% of its dispersion is Student+Retired, i.e. age
# in disguise (median driver age 22 and 77); the residual 1.24x spread is an
# age-by-occupation interaction concentrated in drivers under 35. In a GLM it
# would destabilise every coefficient it touches and wreck the relativities
# table. `driver_age` carries the signal.
NUM_SPEC = [
    "driver_age", "years_licensed", "ncb_years", "vehicle_value_group",
    "vehicle_age", "engine_size_cc", "annual_mileage", "sum_insured",
    "year_built", "distance_to_river_km", "elevation_m", "bedrooms",
    "household_size", "years_at_address", "renewal_count",
    "missed_payments_12m", "written_premium", "premium_per_exposure",
    "voluntary_excess", "compulsory_excess", "discount_applied",
]
CAT_SPEC = [
    "line", "cover_type", "fuel_type", "body_type", "transmission",
    "vehicle_use", "overnight_parking", "channel", "payment_method",
    "region", "urban_density", "flood_zone", "construction_type",
    "property_type", "roof_type", "energy_rating", "soil_type",
    "occupancy", "tenure", "security_level", "marital_status",
    "credit_band", "new_business", "telematics_opt_in", "named_young_driver",
]

DENYLIST = {"renewal_decision", "customer_service_contacts", "telematics_score",
            "claim_id", "paid_to_date", "incurred_to_date", "recoveries",
            "status", "report_date", "loss_date"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="glm-001")
    ap.add_argument("--alpha", type=float, default=1e-4)
    ap.add_argument("--fold", default="2024", choices=["2024", "2023"])
    ap.add_argument("--save-pred", action="store_true")
    args = ap.parse_args()

    # never let a deny-listed field reach the design
    bad = DENYLIST & (set(NUM_SPEC) | set(CAT_SPEC))
    if bad:
        raise SystemExit(f"FATAL: deny-listed fields in the GLM spec: {bad}")

    from scripts.run_experiment import FOLDS, prepare
    holdout_year, train_max_year = FOLDS[args.fold]
    train, test = prepare("base,premium,address,customer",
                          development="lob", weather_lag=False)
    tr = train[train["acc_year"] <= train_max_year]
    va = train[train["acc_year"] == holdout_year]
    print(f"fold {args.fold}: train AY <= {train_max_year} ({len(tr):,}) "
          f"-> holdout AY {holdout_year} ({len(va):,})")

    X_tr, cols = prepare_design(tr, NUM_SPEC, CAT_SPEC)
    X_va, _ = prepare_design(va, NUM_SPEC, CAT_SPEC, impute_from=tr)
    print(f"design: {X_tr.shape[1]} terms ({len(NUM_SPEC)} numeric, "
          f"{len(CAT_SPEC)} categorical)")

    exp_tr = tr["exposure"].astype(float).clip(lower=1e-3)
    exp_va = va["exposure"].astype(float).clip(lower=1e-3)
    rate_tr = tr["y_ult"].astype(float) / exp_tr
    rate_va = va["y_ult"].astype(float) / exp_va

    print(f"\nfitting Tweedie GLM  power=1.5  link=log  alpha={args.alpha} ...")
    model = fit_glm(X_tr, rate_tr, exp_tr, power=1.5, alpha=args.alpha)
    print(f"  converged in {model.n_iter_} iterations")

    p_tr = predict_cost(model, X_tr, exp_tr)
    p_va = predict_cost(model, X_va, exp_va)
    y_tr = tr["y_ult"].to_numpy(float)
    yv = va["y_ult"].to_numpy(float)

    def d2(y, p):
        return d2_tweedie_score(y, np.maximum(p, 1e-3), power=1.5)

    print(f"\nGLM  train D2 = {d2(y_tr, p_tr):+.4f}   holdout D2 = {d2(yv, p_va):+.4f}")
    print(f"     sum pred {p_va.sum()/1e6:.2f}M  sum actual {yv.sum()/1e6:.2f}M  "
          f"ratio {p_va.sum()/yv.sum():.4f}")

    for ln in LINES:
        m = va["line"].to_numpy() == ln
        if m.sum():
            print(f"     {ln:10s} D2 = {d2(yv[m], p_va[m]):+.4f}")

    rel = relativities(model, cols, NUM_SPEC)
    MET.mkdir(parents=True, exist_ok=True)
    rel.to_csv(MET / f"{args.tag}_relativities.csv", index=False)

    cats = rel[rel.kind == "categorical"]
    print(f"\nmost expensive categorical levels (relativity vs base level):")
    for _, r in cats.head(12).iterrows():
        print(f"  {r.term:42s} {r.relativity:7.3f}")
    print(f"\ncheapest:")
    for _, r in cats.tail(6).iterrows():
        print(f"  {r.term:42s} {r.relativity:7.3f}")

    nums = rel[rel.kind == "numeric"]
    print(f"\nnumeric terms (coefficient, i.e. log-rate per unit):")
    for _, r in nums.reindex(nums.coef.abs().sort_values(ascending=False).index).head(10).iterrows():
        print(f"  {r.term:28s} coef {r.coef:+.5f}")

    if args.save_pred:
        te = test.copy()
        X_te, _ = prepare_design(te, NUM_SPEC, CAT_SPEC, impute_from=tr)
        exp_te = te["exposure"].astype(float).clip(lower=1e-3)
        p_te = predict_cost(model, X_te, exp_te)
        sub = pd.DataFrame({"row_id": te["row_id"], "prediction": p_te})
        assert len(sub) == 626_639, len(sub)
        OUT.mkdir(parents=True, exist_ok=True)
        sub.to_csv(OUT / f"{args.tag}.csv", index=False)
        print(f"\nwrote {OUT / (args.tag + '.csv')}  mean={sub.prediction.mean():.2f}")

    (MET / f"{args.tag}.json").write_text(json.dumps({
        "name": args.tag, "kind": "glm_tweedie_1.5", "fold": args.fold,
        "alpha": args.alpha, "n_terms": int(X_tr.shape[1]),
        "d2_holdout": float(d2(yv, p_va)), "d2_train": float(d2(y_tr, p_tr)),
        "sum_ratio": float(p_va.sum() / yv.sum()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
