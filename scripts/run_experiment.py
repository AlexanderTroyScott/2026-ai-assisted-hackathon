#!/usr/bin/env python
"""Run one experiment: build frame, time-split, train, score, predict.

Usage:
  python scripts/run_experiment.py --name exp-002 --features base,premium,experience
  python scripts/run_experiment.py --name exp-003 --features base,premium,experience,weather

Always prints local D² for every model AND for the two baselines so the
leaderboard movement is attributable to a specific change.
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

from src.dataset import build_frame, LINES, FORWARD_FACTOR  # noqa: E402
from sklearn.metrics import d2_tweedie_score  # noqa: E402

OUT = ROOT / "outputs" / "submissions"
MET = ROOT / "outputs" / "metrics"

# D-011/D-016 deny-list: never features.
# customer_service_contacts is outcome leakage (D-016).
# telematics_score has an ambiguous observation window (D-017).
# renewal_decision is training-only outcome information (D-011).
DENYLIST = ["renewal_decision", "customer_service_contacts", "telematics_score",
            "claim_id", "paid_to_date", "incurred_to_date", "recoveries", "status",
            "report_date", "loss_date"]

BASE_FEATURES = [
    "exposure", "voluntary_excess", "compulsory_excess", "missed_payments_12m",
    "renewal_count", "days_quote_to_inception", "discount_applied", "written_premium",
    "new_business",
    "policyholder_age", "sum_insured", "driver_age", "years_licensed", "ncb_years",
    "vehicle_value_group", "vehicle_age", "engine_size_cc", "annual_mileage",
    "telematics_opt_in",
]
CAT_FEATURES = [
    "line", "channel", "payment_method", "cover_type", "fuel_type", "body_type",
    "transmission", "overnight_parking", "vehicle_use", "construction_type",
    "occupancy", "tenure", "security_level", "urban_density", "soil_type",
    "property_type", "roof_type", "energy_rating", "region", "flood_zone",
]
EXPERIENCE = [
    "cust_claims_1y", "cust_cost_1y", "cust_claims_2y", "cust_cost_2y",
    "cust_claims_4y", "cust_cost_4y", "cust_claims_all", "cust_cost_all",
    "pol_claims_all", "pol_cost_all", "days_since_claim",
]
WEATHER = [
    "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
    "wildfire_index", "rainfall_mm", "mean_temperature_c",
]
FLAGS = ["driver_assist", "named_young_driver", "accidental_damage_cover", "mortgage",
         "basement", "trees_within_5m", "solar_panels", "works_from_home", "lodgers",
         "bicycle_cover", "smoke_alarms"]


def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_parquet(ROOT / "data_raw" / "weather.parquet")
    w["year"] = w["year"].astype(int)
    df = df.copy()
    df["weather_year"] = np.where(df["is_test"], 2025, df["acc_year"])
    df = df.merge(w, left_on=["address_id", "weather_year"], right_on=["address_id", "year"], how="left")
    return df


def feature_cols(groups: str) -> list[str]:
    cols = list(BASE_FEATURES)
    if "experience" in groups:
        cols += EXPERIENCE
    if "weather" in groups:
        cols += WEATHER
    if "flags" in groups:
        cols += FLAGS
    bad = [c for c in cols if c in DENYLIST]
    if bad:
        raise SystemExit(f"FATAL: deny-listed features requested: {bad}")
    return cols


def prepare(groups: str):
    df = build_frame("train")
    if "weather" in groups:
        df = add_weather(df)
    te = build_frame("test")
    if "weather" in groups:
        te = add_weather(te)
    return df, te


def align(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], cats: list[str]):
    train = train.copy()
    test = test.copy()
    cats = [c for c in cats if c in train.columns]
    for c in cats:
        train[c] = train[c].astype(str).fillna("other")
        test[c] = test[c].astype(str).fillna("other")
        levels = sorted(set(train[c]) | set(test[c]))
        train[c] = pd.Categorical(train[c], categories=levels)
        test[c] = pd.Categorical(test[c], categories=levels)
    use = [c for c in cols + cats if c in train.columns]
    num = [c for c in use if c not in cats]
    for c in num:
        if train[c].dtype == object or train[c].dtype == bool:
            train[c] = pd.to_numeric(train[c], errors="coerce")
            test[c] = pd.to_numeric(test[c], errors="coerce")
        train[c] = train[c].astype(np.float64)
        test[c] = test[c].astype(np.float64)
    train[num] = train[num].fillna(-1)
    test[num] = test[num].fillna(-1)
    return train[use], test[use], use, cats


def lob_recalibrate(pred: np.ndarray, lines: pd.Series, actual: np.ndarray | None = None,
                    ref_lines: pd.Series | None = None, ref_pred: np.ndarray | None = None,
                    ref_actual: np.ndarray | None = None) -> np.ndarray:
    """Scale each LOB so its predicted total matches its expected total.

    D-009/D-005: without this the model systematically underpredicts the LOBs
    with the largest development factors (measured: buildings needs x1.15).
    When `actual` is given the scales are fit on that sample; otherwise they are
    carried over from a reference sample.
    """
    out = pred.copy()
    if actual is not None:
        for ln in sorted(lines.unique()):
            m = (lines == ln).to_numpy()
            if m.sum() and out[m].sum() > 0:
                out[m] *= actual[m].sum() / out[m].sum()
    else:
        for ln in sorted(lines.unique()):
            m_ref = (ref_lines == ln).to_numpy()
            m = (lines == ln).to_numpy()
            if m.sum() and m_ref.sum() and out[m].sum() > 0:
                scale = ref_actual[m_ref].sum() / ref_pred[m_ref].sum()
                out[m] *= scale
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="base,premium")
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    groups = set(args.features.split(","))

    print(f"=== {args.name}  features={args.features} ===")
    train, test = prepare(args.features)
    print(f"train rows {len(train):,}  test rows {len(test):,}")

    # ---- baselines on the same holdout (developed truth) ----
    y_true = train["y_ult"].to_numpy()
    tr = train[train["acc_year"] <= 2023]
    va = train[train["acc_year"] == 2024]
    yv = va["y_ult"].to_numpy()

    lob_rate = tr.groupby("line").apply(
        lambda g: g["y_ult"].sum() / g["exposure"].sum(), include_groups=False
    )
    pred_const = va["exposure"].to_numpy() * va["line"].map(lob_rate).to_numpy()
    lob_lr = tr.groupby("line").apply(
        lambda g: g["y_ult"].sum() / g["written_premium"].sum(), include_groups=False
    )
    pred_prem = va["written_premium"].to_numpy() * va["line"].map(lob_lr).to_numpy()

    def d2(p):
        return d2_tweedie_score(yv, np.maximum(p, 1e-3), power=1.5)

    print(f"baseline constant_rate   D2 = {d2(pred_const):+.4f}")
    print(f"baseline premium_lr      D2 = {d2(pred_prem):+.4f}")

    # ---- model ----
    import lightgbm as lgb

    cols = feature_cols(args.features)
    X_tr, X_va, use, cats = align(tr, va, cols, CAT_FEATURES)
    X_all, X_te, _, _ = align(train, test, cols, CAT_FEATURES)

    params = dict(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=2500, learning_rate=0.04, num_leaves=64,
        min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=3.0, verbose=-1, random_state=0,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, tr["y_ult"], eval_set=[(X_va, va["y_ult"])],
        eval_metric="tweedie",
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
        categorical_feature=cats,
    )
    p_va = np.maximum(model.predict(X_va), 1e-3)
    print(f"model (raw)            D2 = {d2(p_va):+.4f}   (n_iter={model.best_iteration_})")

    # D-009: recalibrate each LOB so predicted totals match expected totals.
    p_va = lob_recalibrate(p_va, va["line"], actual=yv)
    print(f"model (LOB-recalibrated) D2 = {d2(p_va):+.4f}")

    for ln in LINES:
        m = va["line"].to_numpy() == ln
        if m.sum():
            print(f"   {ln:10s} D2 = {d2_tweedie_score(yv[m], p_va[m], power=1.5):+.4f}")

    imp = pd.Series(model.feature_importances_, index=use).sort_values(ascending=False)
    print("top features:", ", ".join(f"{k}={v}" for k, v in imp.head(12).items()))

    # ---- score the test set ----
    p_te = np.maximum(model.predict(X_te), 1e-3)
    # carry the holdout LOB scales over to the test set
    p_te = lob_recalibrate(p_te, test["line"], ref_lines=va["line"],
                           ref_pred=np.maximum(model.predict(X_va), 1e-3), ref_actual=yv)
    fwd = test["line"].map(FORWARD_FACTOR).to_numpy()
    # forward factor is applied because labels are on an accident-year basis
    # and the model has learned the 2021-2024 average level (D-008).
    p_te = p_te * fwd
    sub = pd.DataFrame({"row_id": test["row_id"], "prediction": np.maximum(p_te, 1e-3)})
    assert len(sub) == 626_639, len(sub)
    OUT.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{args.name}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}  mean={sub.prediction.mean():.2f}  p50={sub.prediction.median():.2f}")

    (MET / f"{args.name}.json").write_text(json.dumps({
        "name": args.name, "features": args.features,
        "d2_constant": float(d2(pred_const)), "d2_premium_lr": float(d2(pred_prem)),
        "d2_model": float(d2(p_va)), "n_iter": int(model.best_iteration_),
        "top_features": imp.head(20).to_dict(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
