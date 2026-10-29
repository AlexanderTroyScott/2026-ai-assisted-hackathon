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

from src.dataset import build_frame, LINES, forward_factors  # noqa: E402
from src.features import historical_relativities  # noqa: E402
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
    "new_business", "prior_claims_5y",
    "earned_premium", "age_band", "ncb_band", "vvg_log",
    "policyholder_age", "sum_insured", "driver_age", "years_licensed", "ncb_years",
    "vehicle_value_group", "vehicle_age", "engine_size_cc", "annual_mileage",
    "telematics_opt_in",
]
CAT_FEATURES = [
    "line", "channel", "payment_method", "cover_type", "fuel_type", "body_type",
    "transmission", "overnight_parking", "vehicle_use", "construction_type",
    "occupancy", "tenure", "security_level", "urban_density", "soil_type",
    "property_type", "roof_type", "energy_rating", "region", "flood_zone",
    "marital_status", "credit_band", "marketing_segment",
]
# `occupation` is deliberately EXCLUDED (D-056, verified 2026-10-22 on both
# folds). 64% of its dispersion is Student+Retired, i.e. driver_age in disguise
# (median ages 22 and 77 vs 44-45 for all other levels). Measured impact on
# accuracy is nil (mean +0.0003, folds disagree in direction), but calibration
# improves on BOTH folds (ratio 1.143->1.120 / 1.185->1.155; slope
# 1.189->1.158 / 1.224->1.149) - and D-054 established that our leaderboard
# failures are calibration failures, not ranking failures. It also carries two
# artefacts (Manager = one 11.9M claim; Student = age) and costs 21 dummy
# parameters in the GLM. Already excluded there (D-046).
# Address/customer rating attributes (D-030): joined from the dimension tables,
# absent from every model before exp-010. DGP findings: flood_zone 1->5 RR 190x,
# elevation_m ~330x, distance_to_river_km 36x for flood; region/urban_density in
# the premium formula.
ADDRESS_NUMERIC = [
    "distance_to_river_km", "elevation_m", "year_built", "bedrooms",
]
CUSTOMER_NUMERIC = [
    "household_size", "licence_points", "years_at_address", "customer_since_year",
]
EXPERIENCE = [
    "cust_claims_1y", "cust_cost_1y", "cust_claims_2y", "cust_cost_2y",
    "cust_claims_4y", "cust_cost_4y", "cust_claims_all", "cust_cost_all",
    "pol_claims_all", "pol_cost_all", "days_since_claim",
]
# D-031: strictly-lagged denied-claim history (loss < ps). Denied claims carry
# zero cost (D-004) but the refusal event is claim-occurrence information.
DENIED = [
    "cust_denied_all", "pol_denied_all", "days_since_denied",
]
WEATHER = [
    "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
    "wildfire_index", "rainfall_mm", "mean_temperature_c",
]
FLAGS = ["driver_assist", "named_young_driver", "accidental_damage_cover", "mortgage",
         "basement", "trees_within_5m", "solar_panels", "works_from_home", "lodgers",
         "bicycle_cover", "smoke_alarms"]


def add_weather(df: pd.DataFrame, lag: bool = False) -> pd.DataFrame:
    w = pd.read_parquet(ROOT / "developed" / "weather.parquet")
    w["year"] = w["year"].astype(int)
    df = df.copy()
    if lag:
        # sensitivity for the D-012 leakage question: use the *previous* year's
        # climate indices instead of the same year's (test rows use 2024).
        df["weather_year"] = np.where(df["is_test"], 2024,
                                      np.maximum(df["acc_year"] - 1, w["year"].min()))
    else:
        df["weather_year"] = np.where(df["is_test"], 2025, df["acc_year"])
    df = df.merge(w, left_on=["address_id", "weather_year"], right_on=["address_id", "year"], how="left")
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """D-021/D-024: expose the structure of the decoded premium formula so the
    model can learn deviations rather than re-deriving the formula from scratch.

    written_premium = base x theta_age x theta_ncb x theta_vvg x theta_cover
                      x theta_region x theta_year x exp(-1.04 x discount) x noise
    with sharp jumps at driver age 25 and 70 and 3 flat NCB bands.
    """
    df = df.copy()
    # D-051: written_premium is ANNUALISED, so dividing by exposure gives an
    # artefact (up to 916,241 on a 1-day row). earned_premium is built in
    # load_policies as written_premium x exposure.
    age = df["driver_age"].fillna(df["policyholder_age"]).fillna(50)
    df["age_band"] = pd.cut(age, [0, 24, 69, 200], labels=[0, 1, 2]).astype(float)
    df["ncb_band"] = pd.cut(df["ncb_years"].fillna(0), [-1, 0, 4, 200], labels=[0, 1, 2]).astype(float)
    df["vvg_log"] = np.log1p(df["vehicle_value_group"].fillna(0))
    si = pd.to_numeric(df.get("sum_insured", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["si_artefact"] = (si > 1e7).astype(float)
    df["sum_insured_clean"] = np.where(si.notna(), np.minimum(si, 1e7), np.nan)
    return df


def feature_cols(groups: str) -> list[str]:
    cols = list(BASE_FEATURES)
    if "si_clean" in groups:
        cols = [c for c in cols if c != "sum_insured"] + ["sum_insured_clean", "si_artefact"]
    if "address" in groups:
        cols += ADDRESS_NUMERIC + ["listed_building"]
    if "customer" in groups:
        cols += CUSTOMER_NUMERIC
    if "experience" in groups:
        cols += EXPERIENCE
    if "denied" in groups:
        cols += DENIED
    if "weather" in groups:
        cols += WEATHER
    if "flags" in groups:
        cols += FLAGS
    if "relativity" in groups:
        cols += [f"{c}_hist_{k}" for c in CAT_FEATURES for k in ("rate", "freq")]
    bad = [c for c in cols if c in DENYLIST]
    if bad:
        raise SystemExit(f"FATAL: deny-listed features requested: {bad}")
    return cols


def prepare(groups: str, development: str = "lob", weather_lag: bool = False,
            legacy_features: bool = False):
    df = add_derived(build_frame("train", development=development,
                                 legacy_features=legacy_features))
    if "weather" in groups:
        df = add_weather(df, lag=weather_lag)
    te = add_derived(build_frame("test", legacy_features=legacy_features))
    if "weather" in groups:
        te = add_weather(te, lag=weather_lag)
    if "relativity" in groups:
        df, te = historical_relativities(
            df, te, CAT_FEATURES, prior="legacy" if legacy_features else "strict")
    return df, te


def align(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], cats: list[str]):
    train = train.copy()
    test = test.copy()
    missing = [c for c in cols + cats if c not in train.columns or c not in test.columns]
    if missing:
        raise SystemExit(
            f"FATAL: requested features absent from the frame: {missing}. "
            "Fix the frame builder (build_frame / add_derived / add_weather), "
            "not the feature list silently."
        )
    cats = [c for c in cats if c in train.columns]
    for c in cats:
        train[c] = train[c].astype(str).fillna("other")
        test[c] = test[c].astype(str).fillna("other")
        # D-084: the category space is defined by the training sample; test-only
        # levels map to "other" instead of leaking the test category space into
        # the fit (the old union(train, test) was transductive). Inert on this
        # book (no train-unseen test levels exist in developed/).
        train_levels = set(train[c])
        levels = sorted(train_levels | {"other"})
        test[c] = pd.Categorical(test[c].where(test[c].isin(train_levels), "other"),
                                 categories=levels)
        train[c] = pd.Categorical(train[c], categories=levels)
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
    # D-011/D-037: assert on the matrix actually returned, not on the request
    # list or on source text. Anything deny-listed that reaches `use` is outcome
    # leakage regardless of how it got here (a feature group, a derived name,
    # a silent join).
    leaked = [c for c in use if c in DENYLIST]
    if leaked:
        raise SystemExit(f"FATAL: deny-listed features in the model matrix: {leaked}")
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


def fit_freq_sev(tr, va, cols, cats, seed=0):
    """Frequency/severity split (D-023).

    Frequency  : Poisson on claim counts, exposure-weighted
    Severity   : Gamma on cost conditional on a claim occurring
    Combined   : E[cost] = E[count] x E[cost per claim]

    This is the actuarially standard way to handle a low-frequency, heavy-tailed
    book, and it targets the tail directly (see docs/synthetic_dgp_findings.md 6b:
    top 1% of bodily_injury claims carry 18.3% of BI cost).
    """
    import lightgbm as lgb

    has = (tr["n_claims"] > 0) & (tr["y_ult"] > 0)
    Xf_tr, Xf_va, use, cats2 = align(tr, va, cols, cats)
    fmodel = lgb.LGBMRegressor(
        objective="poisson", n_estimators=2000, learning_rate=0.05, num_leaves=64,
        min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=3.0, verbose=-1, random_state=seed,
    )
    fmodel.fit(Xf_tr, tr["n_claims"], sample_weight=tr["exposure"],
               eval_set=[(Xf_va, va["n_claims"])], eval_metric="poisson",
               callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
               categorical_feature=cats2)

    Xs_tr, Xs_va, _, _ = align(tr[has], va, cols, cats)
    smodel = lgb.LGBMRegressor(
        objective="gamma", n_estimators=2000, learning_rate=0.05, num_leaves=64,
        min_data_in_leaf=150, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=5.0, verbose=-1, random_state=seed,
    )
    smodel.fit(Xs_tr, tr.loc[has, "y_ult"],
               callbacks=[lgb.log_evaluation(0)], categorical_feature=cats2)

    # fmodel already predicts the expected CLAIM COUNT for the row (exposure is a
    # feature and the loss is exposure-weighted), so do NOT multiply by exposure
    # again — that was a double count and cost ~0.06 of D2.
    p_freq = np.maximum(fmodel.predict(Xf_va), 1e-9)
    p_sev = np.maximum(smodel.predict(Xs_va), 1e-3)
    return p_freq * p_sev, fmodel, smodel, use, cats2


# Rolling-origin folds (D-006). Each is (holdout year, training max year).
# The 2024 fold is the primary one -- it is closest to the 2025 scoring
# population (D-018). The 2023 fold is a stability check, not a second tuning
# opportunity (D-025: local near-ties do not rank on the leaderboard).
FOLDS = {"2024": (2024, 2023), "2023": (2023, 2022)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--features", default="base,premium")
    ap.add_argument("--mode", default="tweedie", choices=["tweedie", "freq_sev", "per_lob", "per_lob_xgb"])
    ap.add_argument("--development", default="lob", choices=["lob", "peril", "bf", "split", "ht_motor"])
    ap.add_argument("--norm-rel", action="store_true",
                    help="D-065: normalize _hist_rate and _hist_freq within (line, acc_year) to mean 1.0")
    ap.add_argument("--trend", default="central", choices=["premium", "low", "central", "high"],
                    help="2025 level scenario (D-009, still `proposed`)")
    ap.add_argument("--fold", default="2024", choices=sorted(FOLDS),
                    help="rolling-origin holdout year (D-006)")
    ap.add_argument("--target", default="cost", choices=["rate", "cost"],
                    help="D-063: 'rate' fits y_ult/exposure with exposure weights and drops "
                         "earned_premium; 'cost' fits total row cost.")
    ap.add_argument("--num-leaves", type=int, default=None,
                    help="override LGBM num_leaves (default: 64 for tweedie, 6 for per_lob)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override LGBM learning_rate (default: 0.04)")
    ap.add_argument("--l2", type=float, default=None,
                    help="override LGBM lambda_l2 (default: 3.0 for tweedie, 5.0 for per_lob)")
    ap.add_argument("--recency-lambda", type=float, default=0.0,
                    help="exp(lambda*(AY-train_max)) sample weights; >0 up-weights "
                         "recent accident years (D-041). 0 = uniform.")
    ap.add_argument("--train-max", type=int, default=None,
                    help="override the fold's training max AY (diagnostic: measures "
                         "the value of including the most recent training year)")
    ap.add_argument("--final-fit", action="store_true",
                    help="refit on ALL training years incl. the holdout before "
                         "scoring test (D-045). Uses the CV best_iteration as a "
                         "fixed tree count; local D2 is then unavailable.")
    ap.add_argument("--weather-lag", action="store_true")
    ap.add_argument("--legacy-features", action="store_true",
                    help="D-084: pre-audit feature construction (full-sample relativity "
                         "prior + (customer_id, loss, cost) claim dedup). Only for "
                         "byte-exact reproduction of pre-D-084 runs such as exp-036.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-val-pred", action="store_true")
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    if args.final_fit and args.mode == "freq_sev":
        ap.error("--final-fit is not implemented for --mode freq_sev; "
                 "refusing instead of silently ignoring it (D-084)")
    groups = set(args.features.split(","))
    holdout_year, train_max_year = FOLDS[args.fold]
    if args.train_max is not None:
        train_max_year = args.train_max

    print(f"=== {args.name}  features={args.features}  development={args.development}"
          f"  trend={args.trend}  fold={args.fold} ===")
    train, test = prepare(args.features, development=args.development,
                          weather_lag=args.weather_lag,
                          legacy_features=args.legacy_features)
    print(f"train rows {len(train):,}  test rows {len(test):,}")

    # ---- baselines on the same holdout (developed truth) ----
    y_true = train["y_ult"].to_numpy()
    tr = train[train["acc_year"] <= train_max_year]
    va = train[train["acc_year"] == holdout_year]
    yv = va["y_ult"].to_numpy()
    print(f"fold {args.fold}: train AY <= {train_max_year} ({len(tr):,} rows) "
          f"-> holdout AY {holdout_year} ({len(va):,} rows)")

    lob_rate = tr.groupby("line").apply(
        lambda g: g["y_ult"].sum() / g["exposure"].sum(), include_groups=False
    )
    pred_const = va["exposure"].to_numpy() * va["line"].map(lob_rate).to_numpy()
    lob_lr = tr.groupby("line").apply(
        lambda g: g["y_ult"].sum() / g["earned_premium"].sum(), include_groups=False
    )
    # D-051: written_premium is annualised; the benchmark must use earned
    # premium (rate x exposure) or it is unscaled for partial-year rows.
    pred_prem = va["earned_premium"].to_numpy() * va["line"].map(lob_lr).to_numpy()

    def d2(p):
        return d2_tweedie_score(yv, np.maximum(p, 1e-3), power=1.5)

    print(f"baseline constant_rate   D2 = {d2(pred_const):+.4f}")
    print(f"baseline premium_lr      D2 = {d2(pred_prem):+.4f}")

    # ---- model ----
    import lightgbm as lgb

    cols = feature_cols(args.features)
    if args.target == "rate" and "earned_premium" in cols:
        # D-063: written_premium is already an annualised rate; earned_premium is
        # written_premium x exposure, which is a total-cost feature, not a rate feature.
        cols = [c for c in cols if c != "earned_premium"]
    X_all, X_te, use, cats = align(train, test, cols, CAT_FEATURES)
    if args.norm_rel:
        # D-084: same normalization basis on both sides — within (line, acc_year)
        # on each frame (test = AY 2025). Group means are target-free.
        hist_cols = [c for c in use if "_hist_rate" in c or "_hist_freq" in c]
        for c in hist_cols:
            grp_tr = X_all[c].groupby([train["line"], train["acc_year"]]).transform("mean")
            X_all[c] = X_all[c] / grp_tr.replace(0, 1.0)
            grp_te = X_te[c].groupby([test["line"], test["acc_year"]]).transform("mean")
            X_te[c] = X_te[c] / grp_te.replace(0, 1.0)
    X_tr = X_all.loc[tr.index]
    X_va = X_all.loc[va.index]
    default_leaves = 6 if args.mode == "per_lob" else 64
    default_l2 = 5.0 if args.mode == "per_lob" else 3.0
    params = dict(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=2500,
        learning_rate=args.lr if args.lr is not None else 0.04,
        num_leaves=args.num_leaves if args.num_leaves is not None else default_leaves,
        min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=args.l2 if args.l2 is not None else default_l2,
        verbose=-1, random_state=args.seed,
    )
    rec_w = np.exp(args.recency_lambda * (tr["acc_year"].to_numpy() - train_max_year))
    rec_w = rec_w / rec_w.mean()
    if args.recency_lambda:
        print(f"recency lambda={args.recency_lambda}: sample-weight range "
              f"{rec_w.min():.3f}..{rec_w.max():.3f}")

    exp_tr_arr = tr["exposure"].to_numpy(float)
    exp_va_arr = va["exposure"].to_numpy(float)
    exp_all_arr = train["exposure"].to_numpy(float)
    exp_te_arr = test["exposure"].to_numpy(float)
    to_cost = (lambda r, e: np.maximum(r * e, 1e-3)) if args.target == "rate" else (lambda r, e: r)

    def _fit_target_weights(y_arr: np.ndarray, exp_arr: np.ndarray, rw: np.ndarray):
        if args.target == "rate":
            w = np.clip(exp_arr, 1e-3, None)
            return y_arr / w, rw * w, [w]
        return y_arr, rw, None

    if args.mode == "freq_sev":
        p_raw, fmodel, smodel, use, cats = fit_freq_sev(tr, va, cols, CAT_FEATURES, seed=args.seed)
        model = smodel
        best_iter = int(fmodel.best_iteration_)
        imp = pd.Series(fmodel.feature_importances_, index=use).sort_values(ascending=False)
        p_tr_raw = np.maximum(fmodel.predict(X_tr), 1e-9) * np.maximum(smodel.predict(X_tr), 1e-3)
    elif args.mode == "per_lob":
        lob_models = {}
        lob_active_cols = {}
        lob_best_iters = {}
        p_raw = np.zeros(len(va), dtype=float)
        p_tr_raw = np.zeros(len(tr), dtype=float)
        imp_accum = pd.Series(0.0, index=use)
        for ln in LINES:
            m_tr_ln = (tr["line"] == ln).to_numpy()
            m_va_ln = (va["line"] == ln).to_numpy()
            sub_tr = X_tr.loc[m_tr_ln]
            c_ln = [c for c in use if sub_tr[c].nunique(dropna=False) > 1]
            cat_ln = [c for c in cats if c in c_ln]
            lob_active_cols[ln] = (c_ln, cat_ln)
            m_ln = lgb.LGBMRegressor(**params)
            y_fit_ln, w_fit_ln, _ = _fit_target_weights(
                tr.loc[m_tr_ln, "y_ult"].to_numpy(float), exp_tr_arr[m_tr_ln], rec_w[m_tr_ln]
            )
            y_eval_ln, _, w_eval_ln = _fit_target_weights(
                va.loc[m_va_ln, "y_ult"].to_numpy(float), exp_va_arr[m_va_ln], np.ones(m_va_ln.sum())
            )
            m_ln.fit(
                X_tr.loc[m_tr_ln, c_ln], y_fit_ln,
                sample_weight=w_fit_ln,
                eval_set=[(X_va.loc[m_va_ln, c_ln], y_eval_ln)],
                eval_sample_weight=w_eval_ln,
                eval_metric="tweedie",
                callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
                categorical_feature=cat_ln,
            )
            lob_models[ln] = m_ln
            lob_best_iters[ln] = max(int(m_ln.best_iteration_), 50)
            p_raw[m_va_ln] = to_cost(np.maximum(m_ln.predict(X_va.loc[m_va_ln, c_ln]), 1e-3), exp_va_arr[m_va_ln])
            p_tr_raw[m_tr_ln] = to_cost(np.maximum(m_ln.predict(X_tr.loc[m_tr_ln, c_ln]), 1e-3), exp_tr_arr[m_tr_ln])
            imp_accum[c_ln] += m_ln.feature_importances_
        best_iter = int(np.mean(list(lob_best_iters.values())))
        imp = imp_accum.sort_values(ascending=False)
        print(f"per_lob best_iters={lob_best_iters}")
    elif args.mode == "per_lob_xgb":
        import xgboost as xgb

        lob_models = {}
        lob_active_cols = {}
        lob_best_iters = {}
        p_raw = np.zeros(len(va), dtype=float)
        p_tr_raw = np.zeros(len(tr), dtype=float)
        imp_accum = pd.Series(0.0, index=use)
        xgb_params = dict(
            objective="reg:tweedie", tweedie_variance_power=1.5,
            n_estimators=2500,
            learning_rate=args.lr if args.lr is not None else 0.05,
            max_depth=3, min_child_weight=200,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=args.l2 if args.l2 is not None else 5.0,
            random_state=args.seed,
            tree_method="hist", enable_categorical=True, n_jobs=-1,
        )
        for ln in LINES:
            m_tr_ln = (tr["line"] == ln).to_numpy()
            m_va_ln = (va["line"] == ln).to_numpy()
            sub_tr = X_tr.loc[m_tr_ln]
            c_ln = [c for c in use if sub_tr[c].nunique(dropna=False) > 1]
            cat_ln = [c for c in cats if c in c_ln]
            lob_active_cols[ln] = (c_ln, cat_ln)
            m_ln = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=60, eval_metric="tweedie-nloglik@1.5")
            y_fit_ln, w_fit_ln, _ = _fit_target_weights(
                tr.loc[m_tr_ln, "y_ult"].to_numpy(float), exp_tr_arr[m_tr_ln], rec_w[m_tr_ln]
            )
            y_eval_ln, _, w_eval_ln = _fit_target_weights(
                va.loc[m_va_ln, "y_ult"].to_numpy(float), exp_va_arr[m_va_ln], np.ones(m_va_ln.sum())
            )
            m_ln.fit(
                X_tr.loc[m_tr_ln, c_ln], y_fit_ln,
                sample_weight=w_fit_ln,
                eval_set=[(X_va.loc[m_va_ln, c_ln], y_eval_ln)],
                sample_weight_eval_set=w_eval_ln,
                verbose=False,
            )
            lob_models[ln] = m_ln
            lob_best_iters[ln] = max(int(m_ln.best_iteration) + 1, 50)
            p_raw[m_va_ln] = to_cost(np.maximum(m_ln.predict(X_va.loc[m_va_ln, c_ln]), 1e-3), exp_va_arr[m_va_ln])
            p_tr_raw[m_tr_ln] = to_cost(np.maximum(m_ln.predict(X_tr.loc[m_tr_ln, c_ln]), 1e-3), exp_tr_arr[m_tr_ln])
            imp_accum[c_ln] += m_ln.feature_importances_
        best_iter = int(np.mean(list(lob_best_iters.values())))
        imp = imp_accum.sort_values(ascending=False)
        print(f"per_lob_xgb best_iters={lob_best_iters}")
    else:
        model = lgb.LGBMRegressor(**params)
        y_fit, w_fit, _ = _fit_target_weights(tr["y_ult"].to_numpy(float), exp_tr_arr, rec_w)
        y_eval, _, w_eval = _fit_target_weights(va["y_ult"].to_numpy(float), exp_va_arr, np.ones(len(va)))
        model.fit(
            X_tr, y_fit, eval_set=[(X_va, y_eval)],
            eval_sample_weight=w_eval,
            eval_metric="tweedie",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=cats,
            sample_weight=w_fit,
        )
        p_raw = to_cost(np.maximum(model.predict(X_va), 1e-3), exp_va_arr)
        p_tr_raw = to_cost(np.maximum(model.predict(X_tr), 1e-3), exp_tr_arr)
        best_iter = int(model.best_iteration_)
        imp = pd.Series(model.feature_importances_, index=use).sort_values(ascending=False)
    print(f"model ({args.mode}, target={args.target}, raw)  D2 = {d2(p_raw):+.4f}   (n_iter={best_iter})")

    p_va_oracle = lob_recalibrate(p_raw, va["line"], actual=yv)
    p_va_oof = lob_recalibrate(p_raw, va["line"], ref_lines=tr["line"],
                               ref_pred=p_tr_raw, ref_actual=tr["y_ult"].to_numpy())

    print(f"model (recal, TRAIN-fitted = honest) D2 = {d2(p_va_oof):+.4f}")
    print(f"model (recal, holdout-fitted = oracle) D2 = {d2(p_va_oracle):+.4f}"
          f"   [in-sample; +{d2(p_va_oracle) - d2(p_raw):.4f} over raw]")
    d2_oracle_inflation = float(d2(p_va_oracle) - d2(p_raw))

    for ln in LINES:
        m = va["line"].to_numpy() == ln
        if m.sum():
            print(f"   {ln:10s} D2 = {d2_tweedie_score(yv[m], p_va_oof[m], power=1.5):+.4f}")

    print("top features:", ", ".join(f"{k}={v}" for k, v in imp.head(12).items()))

    # ---- D-045 / D-063: final fit on ALL training years ----
    rec_w_all = np.exp(args.recency_lambda * (train["acc_year"].to_numpy() - train["acc_year"].max()))
    rec_w_all = rec_w_all / rec_w_all.mean()

    if args.mode == "freq_sev":
        p_te = (np.maximum(fmodel.predict(X_te), 1e-9)
                * np.maximum(smodel.predict(X_te), 1e-3))
        p_ref = (np.maximum(fmodel.predict(X_va), 1e-9)
                 * np.maximum(smodel.predict(X_va), 1e-3))
        p_te = lob_recalibrate(p_te, test["line"], ref_lines=va["line"],
                               ref_pred=p_ref, ref_actual=yv)
    elif args.mode in ("per_lob", "per_lob_xgb"):
        p_te = np.zeros(len(test), dtype=float)
        if args.final_fit:
            print(f"\nD-045/D-063 {args.mode} final refit on ALL {len(train):,} training rows "
                  f"(AY 2021-2024) with fixed n_estimators={lob_best_iters}")
            p_all = np.zeros(len(train), dtype=float)
            for ln in LINES:
                m_all_ln = (train["line"] == ln).to_numpy()
                m_te_ln = (test["line"] == ln).to_numpy()
                c_ln, cat_ln = lob_active_cols[ln]
                y_all_ln, w_all_ln, _ = _fit_target_weights(
                    train.loc[m_all_ln, "y_ult"].to_numpy(float), exp_all_arr[m_all_ln], rec_w_all[m_all_ln]
                )
                if args.mode == "per_lob_xgb":
                    import xgboost as xgb
                    f_ln = xgb.XGBRegressor(**dict(xgb_params, n_estimators=lob_best_iters[ln]))
                    f_ln.fit(X_all.loc[m_all_ln, c_ln], y_all_ln, sample_weight=w_all_ln, verbose=False)
                else:
                    f_params = dict(params, n_estimators=lob_best_iters[ln])
                    f_ln = lgb.LGBMRegressor(**f_params)
                    f_ln.fit(X_all.loc[m_all_ln, c_ln], y_all_ln, sample_weight=w_all_ln, categorical_feature=cat_ln)
                p_all[m_all_ln] = to_cost(np.maximum(f_ln.predict(X_all.loc[m_all_ln, c_ln]), 1e-3), exp_all_arr[m_all_ln])
                p_te[m_te_ln] = to_cost(np.maximum(f_ln.predict(X_te.loc[m_te_ln, c_ln]), 1e-3), exp_te_arr[m_te_ln])
            p_te = lob_recalibrate(p_te, test["line"], ref_lines=train["line"],
                                   ref_pred=p_all, ref_actual=train["y_ult"].to_numpy())
        else:
            for ln in LINES:
                m_te_ln = (test["line"] == ln).to_numpy()
                c_ln, _ = lob_active_cols[ln]
                p_te[m_te_ln] = to_cost(np.maximum(lob_models[ln].predict(X_te.loc[m_te_ln, c_ln]), 1e-3), exp_te_arr[m_te_ln])
            p_te = lob_recalibrate(p_te, test["line"], ref_lines=va["line"],
                                   ref_pred=p_raw, ref_actual=yv)
    else:
        final_model = model
        if args.final_fit:
            n_fix = max(int(best_iter), 50)
            print(f"\nD-045/D-063 final refit on ALL {len(train):,} training rows "
                  f"(AY 2021-2024) with fixed n_estimators={n_fix}")
            final_params = dict(params, n_estimators=n_fix)
            final_model = lgb.LGBMRegressor(**final_params)
            y_all_fit, w_all_fit, _ = _fit_target_weights(
                train["y_ult"].to_numpy(float), exp_all_arr, rec_w_all
            )
            final_model.fit(X_all, y_all_fit, sample_weight=w_all_fit, categorical_feature=cats)
        p_te = to_cost(np.maximum(final_model.predict(X_te), 1e-3), exp_te_arr)
        p_ref = to_cost(np.maximum(final_model.predict(X_va), 1e-3), exp_va_arr)
        if args.final_fit:
            p_all = to_cost(np.maximum(final_model.predict(X_all), 1e-3), exp_all_arr)
            p_te = lob_recalibrate(p_te, test["line"], ref_lines=train["line"],
                                   ref_pred=p_all, ref_actual=train["y_ult"].to_numpy())
        else:
            p_te = lob_recalibrate(p_te, test["line"], ref_lines=va["line"],
                                   ref_pred=p_ref, ref_actual=yv)
    fwd_tab = forward_factors(args.trend)
    fwd = test["line"].map(fwd_tab).to_numpy()
    # Forward factor is applied because labels are on an accident-year basis and
    # the model has learned the 2021-2024 average level (D-008). The factor
    # itself is a scenario, not a measured fact: D-009 blends the claims-index
    # CAGR with the premium anchor and is still `proposed` (needs sign-off).
    p_te = p_te * fwd
    sub = pd.DataFrame({"row_id": test["row_id"], "prediction": np.maximum(p_te, 1e-3)})
    assert len(sub) == 626_639, len(sub)
    OUT.mkdir(parents=True, exist_ok=True)
    MET.mkdir(parents=True, exist_ok=True)
    if args.save_val_pred:
        # store the honest (train-fitted) predictions so downstream analysis
        # does not silently inherit the in-sample oracle (D-040)
        pd.DataFrame({"row_id": va["row_id"].to_numpy(), "y_true": yv,
                      "pred": p_va_oof, "pred_oracle": p_va_oracle}).to_parquet(
            MET / f"{args.name}_valpred.parquet", index=False)
    path = OUT / f"{args.name}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}  mean={sub.prediction.mean():.2f}  p50={sub.prediction.median():.2f}")

    (MET / f"{args.name}.json").write_text(json.dumps({
        "name": args.name, "features": args.features, "mode": args.mode, "seed": args.seed,
        "development": args.development, "trend": args.trend, "fold": args.fold,
        "target": args.target, "final_fit": bool(args.final_fit),
        "norm_rel": bool(args.norm_rel), "weather_lag": bool(args.weather_lag),
        "legacy_features": bool(args.legacy_features),
        "num_leaves": args.num_leaves, "lr": args.lr, "l2": args.l2,
        "holdout_year": holdout_year, "train_max_year": train_max_year,
        "forward_factor": fwd_tab, "recency_lambda": args.recency_lambda,
        "d2_constant": float(d2(pred_const)), "d2_premium_lr": float(d2(pred_prem)),
        "d2_model": float(d2(p_va_oof)),
        "d2_model_oracle": float(d2(p_va_oracle)),
        "d2_oracle_inflation": d2_oracle_inflation,
        "d2_model_raw": float(d2(p_raw)), "n_iter": best_iter,
        "top_features": imp.head(20).to_dict(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
