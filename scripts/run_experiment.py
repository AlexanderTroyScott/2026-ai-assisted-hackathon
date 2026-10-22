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
    w = pd.read_parquet(ROOT / "data_raw" / "weather.parquet")
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
    return df


def feature_cols(groups: str) -> list[str]:
    cols = list(BASE_FEATURES)
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


def prepare(groups: str, development: str = "lob", weather_lag: bool = False):
    df = add_derived(build_frame("train", development=development))
    if "weather" in groups:
        df = add_weather(df, lag=weather_lag)
    te = add_derived(build_frame("test"))
    if "weather" in groups:
        te = add_weather(te, lag=weather_lag)
    if "relativity" in groups:
        df, te = historical_relativities(df, te, CAT_FEATURES)
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
    ap.add_argument("--mode", default="tweedie", choices=["tweedie", "freq_sev"])
    ap.add_argument("--development", default="lob", choices=["lob", "peril"])
    ap.add_argument("--trend", default="central", choices=["premium", "low", "central", "high"],
                    help="2025 level scenario (D-009, still `proposed`)")
    ap.add_argument("--fold", default="2024", choices=sorted(FOLDS),
                    help="rolling-origin holdout year (D-006)")
    ap.add_argument("--target", default="cost", choices=["rate", "cost"],
                    help="D-052: 'cost' (default) fits total row cost. 'rate' is "
                         "EXPERIMENTAL and regressed on the leaderboard (exp-024, "
                         "0.06019) by double-counting exposure via earned_premium; "
                         "kept for diagnostics only")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-val-pred", action="store_true")
    ap.add_argument("--no-submit", action="store_true")
    args = ap.parse_args()
    groups = set(args.features.split(","))
    holdout_year, train_max_year = FOLDS[args.fold]
    if args.train_max is not None:
        train_max_year = args.train_max

    print(f"=== {args.name}  features={args.features}  development={args.development}"
          f"  trend={args.trend}  fold={args.fold} ===")
    train, test = prepare(args.features, development=args.development, weather_lag=args.weather_lag)
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
    X_tr, X_va, use, cats = align(tr, va, cols, CAT_FEATURES)
    params = dict(
        objective="tweedie", tweedie_variance_power=1.5,
        # D-053: tuned 2026-10-22 by a 3-round sweep gated on out-of-time
        # stability (both the 2024 and 2023 folds must agree). Fewer leaves
        # is consistently better: 64 -> 31 -> 16 improves, 128 -> 256 collapses.
        n_estimators=3000, learning_rate=0.02, num_leaves=16,
        min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=3.0, verbose=-1, random_state=args.seed,
    )
    # D-041: recency weighting. The 2025 scoring population is the newest year,
    # and the DGP findings document non-stationary relativities (ncb=0 freq rel
    # 1.60 -> 2.53; age 17-24 2.03 -> 1.74). Uniform training weights let 2021
    # pull the fit as hard as 2023. Note the D-033 label bug was accidentally
    # doing this job by inflating 2023 labels, which is why removing it cost
    # leaderboard score -- this replaces that accident with an explicit knob.
    rec_w = np.exp(args.recency_lambda * (tr["acc_year"].to_numpy() - train_max_year))
    rec_w = rec_w / rec_w.mean()
    if args.recency_lambda:
        print(f"recency lambda={args.recency_lambda}: sample-weight range "
              f"{rec_w.min():.3f}..{rec_w.max():.3f}")

    if args.mode == "freq_sev":
        p_raw, fmodel, smodel, use, cats = fit_freq_sev(tr, va, cols, CAT_FEATURES, seed=args.seed)
        model = smodel
        best_iter = int(fmodel.best_iteration_)
        imp = pd.Series(fmodel.feature_importances_, index=use).sort_values(ascending=False)
    else:
        model = lgb.LGBMRegressor(**params)
        # D-051: written_premium is an annualised rate, so total row cost is
        # `rate x exposure` — a PRODUCT. Trees are additive and learn products
        # badly, and this handicaps the 35% of rows with exposure != 1. The
        # fix is to fit the annual RATE with exposure as the weight, then
        # multiply back outside the model (the formulation src/glm.py already
        # uses). `--target cost` retains the old behaviour for A/B only.
        exp_tr = tr["exposure"].to_numpy(float)
        exp_va = va["exposure"].to_numpy(float)
        if args.target == "rate":
            y_fit = (tr["y_ult"].to_numpy() / np.clip(exp_tr, 1e-3, None))
            y_eval = (va["y_ult"].to_numpy() / np.clip(exp_va, 1e-3, None))
            w_fit = rec_w * np.clip(exp_tr, 1e-3, None)
        else:
            y_fit = tr["y_ult"].to_numpy()
            y_eval = va["y_ult"].to_numpy()
            w_fit = rec_w
        model.fit(
            X_tr, y_fit, eval_set=[(X_va, y_eval)],
            eval_metric="tweedie",
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            categorical_feature=cats,
            sample_weight=w_fit,
        )
        def _predict(m, X, expv):
            r = np.maximum(m.predict(X), 1e-3)
            return np.maximum(r * expv, 1e-3) if args.target == "rate" else r
        p_raw = _predict(model, X_va, exp_va)
        best_iter = int(model.best_iteration_)
        imp = pd.Series(model.feature_importances_, index=use).sort_values(ascending=False)
    print(f"model ({args.mode}, target={args.target}, raw)  D2 = {d2(p_raw):+.4f}   (n_iter={best_iter})")

    # LOB recalibration (D-009/D-005) scales each LOB so predicted totals match
    # expected totals. WHERE those scales are fitted determines what the local
    # number means (D-040):
    #
    #   * fitted on the holdout and scored on the holdout => in-sample ORACLE.
    #     Measured worth +0.0020..+0.0029 of local D2, i.e. the size of the
    #     whole D-025 tie band, and structurally blind to LOB-level miss. This
    #     is NOT a generalisation estimate and must not be used to compare runs
    #     on anything level-shaped.
    #   * fitted on the TRAIN side and applied to the holdout => out-of-sample,
    #     the honest local metric.
    #
    # The submission still uses the holdout-fitted scale carried to the test
    # set, which is correct: 2024 developed truth is the right level anchor for
    # 2025 rows. So the oracle number is the right thing for the submission path
    # and the wrong thing for local model comparison.
    p_va_oracle = lob_recalibrate(p_raw, va["line"], actual=yv)
    exp_tr_arr = tr["exposure"].to_numpy(float)
    exp_all_arr = train["exposure"].to_numpy(float)
    exp_te_arr = test["exposure"].to_numpy(float)
    exp_va_arr = va["exposure"].to_numpy(float)
    to_cost = (lambda r, e: np.maximum(r * e, 1e-3)) if args.target == "rate" else (lambda r, e: r)
    p_tr_raw = (np.maximum(fmodel.predict(X_tr), 1e-9) * np.maximum(smodel.predict(X_tr), 1e-3)
                if args.mode == "freq_sev"
                else to_cost(np.maximum(model.predict(X_tr), 1e-3), exp_tr_arr))
    # Train-fitted scales applied to RAW holdout predictions. Passing
    # p_va_oracle here would apply two scales at once.
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

    # ---- D-045: final fit on ALL training years ----
    # The holdout is only for model selection. Once the tree count is chosen,
    # leaving the holdout out of the fit throws away the most recent 26% of the
    # book (2024: 626,190 rows), which is also the year closest to the 2025
    # scoring population. Standard practice is to refit on everything before
    # deploying. Local D2 above is still the honest out-of-sample number for
    # this config family; it is NOT re-measurable after refitting.
    final_model = model
    if args.final_fit and args.mode != "freq_sev":
        n_fix = max(int(best_iter), 50)
        print(f"\nD-045 final refit on ALL {len(train):,} training rows "
              f"(AY 2021-2024) with fixed n_estimators={n_fix}")
        X_all, X_te, _, _ = align(train, test, cols, CAT_FEATURES)
        final_model = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5,
            n_estimators=n_fix, learning_rate=0.04, num_leaves=64,
            min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=3.0, verbose=-1, random_state=args.seed,
        )
        final_model.fit(X_all, train["y_ult"], categorical_feature=cats)

    # ---- score the test set ----
    X_all, X_te, _, _ = align(train, test, cols, CAT_FEATURES)
    if args.mode == "freq_sev":
        p_te = (np.maximum(fmodel.predict(X_te), 1e-9)
                * np.maximum(smodel.predict(X_te), 1e-3))
        p_ref = (np.maximum(fmodel.predict(X_va), 1e-9)
                 * np.maximum(smodel.predict(X_va), 1e-3))
    else:
        p_te = to_cost(np.maximum(final_model.predict(X_te), 1e-3), exp_te_arr)
        p_ref = to_cost(np.maximum(final_model.predict(X_va), 1e-3), exp_va_arr)
    # LOB scales. After a final refit the anchor should also be all-data, so the
    # level is set on the full training book rather than the 2024 slice alone.
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
