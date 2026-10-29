#!/usr/bin/env python
"""Run Part 2 (Emerging Market) transfer learning + credibility-shrinkage offset adaptation
and generate the 20-policy office premium build-up.

Usage:
  python scripts/run_emerging.py --name emerging-exp-001
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emerging import (  # noqa: E402
    DEV_DIR,
    DEV_LDF,
    EM_DIR,
    EM_LDF,
    LINES,
    build_market_frame,
    build_office_premiums,
    estimate_forward_index,
)

OUT_SUB = ROOT / "outputs" / "submissions"
OUT_MET = ROOT / "outputs" / "metrics"
EXPECTED_EM_ROWS = 209_063
FLOOR = 1e-3

SHARED_NUM = [
    "exposure", "voluntary_excess", "compulsory_excess", "missed_payments_12m",
    "renewal_count", "days_quote_to_inception", "discount_applied", "new_business",
    "prior_claims_5y", "age_num", "sum_insured", "sum_insured_clean", "si_artefact",
    "years_licensed", "ncb_years", "ncb_band", "vehicle_value_group", "vvg_log",
    "vehicle_age", "engine_size_cc", "seats", "days_unoccupied", "high_value_items",
    "distance_to_river_km", "elevation_m", "year_built", "bedrooms",
    "household_size", "licence_points", "years_at_address",
    "pol_claims_all", "pol_cost_all", "pol_claims_1y", "pol_cost_1y", "days_since_claim",
    "hail_days", "wind_exposure_index", "freezing_days", "dry_summer_index",
    "wildfire_index", "rainfall_mm", "mean_temperature_c",
    "driver_assist", "named_young_driver", "accidental_damage_cover", "mortgage",
    "basement", "trees_within_5m", "solar_panels", "works_from_home", "lodgers",
    "bicycle_cover", "smoke_alarms", "listed_building",
]
SHARED_CAT = [
    "age_band_5", "channel", "payment_method", "cover_type", "fuel_type", "body_type",
    "transmission", "overnight_parking", "vehicle_use", "construction_type",
    "occupancy", "tenure", "security_level", "urban_density", "soil_type",
    "property_type", "roof_type", "energy_rating", "flood_zone",
    "marital_status", "marketing_segment", "occ_clean", "vehicle_colour",
]


def blend_lob_costs(p_off, p_prior, lines, w_off):
    """Per-LOB offset/prior blend (D-067 axis, D-086 arm C).

    `w_off[ln]` is the offset weight for LOB `ln`; the prior leg carries
    (1 - w_off[ln]). Both inputs must be fold/level-calibrated costs on the
    same rows. With w_off = 1.0 everywhere this returns `p_off` unchanged
    (the record behaviour).
    """
    out = np.array(p_off, dtype=float).copy()
    p_prior = np.asarray(p_prior, dtype=float)
    for ln, w in w_off.items():
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"w_off[{ln}]={w} outside [0, 1]")
        m = np.asarray(lines) == ln
        out[m] = w * out[m] + (1.0 - w) * p_prior[m]
    return out

ADAPT_CAT = ["region", "gender", "age_band_5", "channel", "cover_type", "flood_zone"]
ADAPT_NUM = [
    "voluntary_excess", "new_business", "wp_rel", "is_self_employed",
    "pol_claims_all", "prior_claims_5y", "wildfire_index", "hail_days",
]


def prep_matrices(dfs: list[pd.DataFrame], num_cols: list[str], cat_cols: list[str]):
    num_use = [c for c in num_cols if all(c in d.columns for d in dfs)]
    cat_use = [c for c in cat_cols if all(c in d.columns for d in dfs)]
    outs = [d[num_use + cat_use].copy() for d in dfs]
    for c in cat_use:
        for o in outs:
            o[c] = o[c].astype(str).fillna("other")
        levels = sorted(set().union(*(set(o[c]) for o in outs)))
        for o in outs:
            o[c] = pd.Categorical(o[c], categories=levels)
    for c in num_use:
        for o in outs:
            o[c] = pd.to_numeric(o[c], errors="coerce").astype(np.float64).fillna(-1.0)
    return outs, num_use + cat_use, cat_use


def fit_developed_priors(dev_tr: pd.DataFrame, em_tr: pd.DataFrame, em_te: pd.DataFrame):
    (X_dev, X_em_tr, X_em_te), all_shared, cats_shared = prep_matrices(
        [dev_tr, em_tr, em_te], SHARED_NUM + ["wp_rel"], SHARED_CAT
    )
    em_tr = em_tr.copy()
    em_te = em_te.copy()
    em_tr["dev_prior_rate"] = 0.0
    em_te["dev_prior_rate"] = 0.0
    em_tr["dev_prior_rate_nowp"] = 0.0
    em_te["dev_prior_rate_nowp"] = 0.0

    for ln in LINES:
        m_dev = (dev_tr["line"] == ln).to_numpy()
        m_etr = (em_tr["line"] == ln).to_numpy()
        m_ete = (em_te["line"] == ln).to_numpy()

        sub_dev = X_dev.loc[m_dev]
        c_ln = [c for c in all_shared if sub_dev[c].nunique(dropna=False) > 1]
        cat_ln = [c for c in cats_shared if c in c_ln]
        w_dev = np.clip(dev_tr.loc[m_dev, "exposure"].to_numpy(float), 1e-3, None)
        y_dev = dev_tr.loc[m_dev, "y_ult"].to_numpy(float) / w_dev

        m1 = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5,
            n_estimators=400, learning_rate=0.04, num_leaves=8,
            min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, random_state=0,
        )
        m1.fit(X_dev.loc[m_dev, c_ln], y_dev, sample_weight=w_dev, categorical_feature=cat_ln)
        em_tr.loc[m_etr, "dev_prior_rate"] = np.maximum(m1.predict(X_em_tr.loc[m_etr, c_ln]), 1e-3)
        em_te.loc[m_ete, "dev_prior_rate"] = np.maximum(m1.predict(X_em_te.loc[m_ete, c_ln]), 1e-3)

        c_ln_nowp = [c for c in c_ln if c != "wp_rel"]
        m2 = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5,
            n_estimators=400, learning_rate=0.04, num_leaves=8,
            min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, random_state=0,
        )
        m2.fit(X_dev.loc[m_dev, c_ln_nowp], y_dev, sample_weight=w_dev, categorical_feature=cat_ln)
        em_tr.loc[m_etr, "dev_prior_rate_nowp"] = np.maximum(m2.predict(X_em_tr.loc[m_etr, c_ln_nowp]), 1e-3)
        em_te.loc[m_ete, "dev_prior_rate_nowp"] = np.maximum(m2.predict(X_em_te.loc[m_ete, c_ln_nowp]), 1e-3)

    em_tr["prior_rate_blend"] = 0.6 * em_tr["dev_prior_rate"] + 0.4 * em_tr["dev_prior_rate_nowp"]
    em_te["prior_rate_blend"] = 0.6 * em_te["dev_prior_rate"] + 0.4 * em_te["dev_prior_rate_nowp"]
    return em_tr, em_te


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="emerging-exp-001")
    ap.add_argument("--use-cache", action="store_true")
    ap.add_argument("--num-leaves", type=int, default=3,
                    help="offset layer num_leaves (D-067 incumbent 3; D-075 combo_top2 = 4)")
    ap.add_argument("--min-leaf", type=int, default=600,
                    help="offset layer min_data_in_leaf (D-067 incumbent 600; D-075 combo_top2 = 1200)")
    ap.add_argument("--w-off", default="motor=1.0,buildings=1.0,contents=1.0",
                    help="per-LOB offset weight in the offset/prior blend (D-086 arm C). "
                         "Default all-1.0 = record behaviour (offset-only, D-067). "
                         "e.g. 'motor=1.0,buildings=1.0,contents=0.5'")
    args = ap.parse_args()

    print(f"=== Part 2 Emerging Market Pipeline: {args.name} ===")
    cache_tr = Path("/tmp/opencode/part2/em_tr_with_prior.parquet")
    cache_te = Path("/tmp/opencode/part2/em_te_with_prior.parquet")
    if args.use_cache and cache_tr.exists() and cache_te.exists():
        em_tr = pd.read_parquet(cache_tr)
        em_te = pd.read_parquet(cache_te)
        em_tr["prior_rate_blend"] = 0.6 * em_tr["dev_prior_rate"] + 0.4 * em_tr["dev_prior_rate_nowp"]
        em_te["prior_rate_blend"] = 0.6 * em_te["dev_prior_rate"] + 0.4 * em_te["dev_prior_rate_nowp"]
        print(f"Loaded cached frames: train={len(em_tr):,}, test={len(em_te):,}")
    else:
        dev_tr = build_market_frame(DEV_DIR, "train", DEV_LDF)
        em_tr = build_market_frame(EM_DIR, "train", EM_LDF)
        em_te = build_market_frame(EM_DIR, "test")
        em_tr, em_te = fit_developed_priors(dev_tr, em_tr, em_te)

    (X_ad_tr, X_ad_te), cols_ad, cats_ad = prep_matrices([em_tr, em_te], ADAPT_NUM, ADAPT_CAT)
    (X_full_tr, _), cols_full, cats_full = prep_matrices(
        [em_tr, em_te], SHARED_NUM + ["written_premium", "wp_rel", "is_self_employed"], SHARED_CAT + ["region", "gender"]
    )

    gkf = GroupKFold(n_splits=5)
    groups = em_tr["customer_id"].to_numpy()
    y_all = em_tr["y_ult"].to_numpy(float)
    exp_all = em_tr["exposure"].to_numpy(float)
    exp_te = em_te["exposure"].to_numpy(float)
    lines_all = em_tr["line"].to_numpy()

    p_const = np.zeros(len(em_tr))
    p_prem = np.zeros(len(em_tr))
    p_dev_zero = np.zeros(len(em_tr))
    p_em_only = np.zeros(len(em_tr))
    p_offset = np.zeros(len(em_tr))

    lob_n_est = {"motor": 35, "buildings": 30, "contents": 20}

    for tr_idx, va_idx in gkf.split(em_tr, y_all, groups):
        for ln in LINES:
            m_tr = tr_idx[lines_all[tr_idx] == ln]
            m_va = va_idx[lines_all[va_idx] == ln]

            rate_ln = y_all[m_tr].sum() / exp_all[m_tr].sum()
            lr_ln = y_all[m_tr].sum() / em_tr["earned_premium"].to_numpy(float)[m_tr].sum()
            p_const[m_va] = exp_all[m_va] * rate_ln
            p_prem[m_va] = em_tr["earned_premium"].to_numpy(float)[m_va] * lr_ln

            pr_tr = em_tr["prior_rate_blend"].to_numpy(float)[m_tr]
            pr_va = em_tr["prior_rate_blend"].to_numpy(float)[m_va]
            scale_pr = y_all[m_tr].sum() / (pr_tr * exp_all[m_tr]).sum()
            pr_tr_cal = pr_tr * scale_pr
            pr_va_cal = pr_va * scale_pr
            p_dev_zero[m_va] = pr_va_cal * exp_all[m_va]

            # Emerging-only baseline tree
            sub_f = X_full_tr.iloc[m_tr]
            c_f = [c for c in cols_full if sub_f[c].nunique(dropna=False) > 1]
            w_tr = np.clip(exp_all[m_tr], 1e-3, None)
            m_em = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=150, learning_rate=0.03, num_leaves=6,
                min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=10.0, verbose=-1, random_state=0,
            )
            m_em.fit(X_full_tr.iloc[m_tr][c_f], y_all[m_tr] / w_tr, sample_weight=w_tr,
                     categorical_feature=[c for c in cats_full if c in c_f])
            ptr_e = np.maximum(m_em.predict(X_full_tr.iloc[m_tr][c_f]) * exp_all[m_tr], 1e-3)
            pva_e = np.maximum(m_em.predict(X_full_tr.iloc[m_va][c_f]) * exp_all[m_va], 1e-3)
            p_em_only[m_va] = pva_e * (y_all[m_tr].sum() / ptr_e.sum())

            # Credibility-shrinkage offset adaptation on top of dev_prior
            sub_a = X_ad_tr.iloc[m_tr]
            c_a = [c for c in cols_ad if sub_a[c].nunique(dropna=False) > 1]
            cat_a = [c for c in cats_ad if c in c_a]
            m_off = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=lob_n_est[ln], learning_rate=0.02, num_leaves=args.num_leaves,
                min_data_in_leaf=args.min_leaf, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=100.0, verbose=-1, random_state=0,
            )
            m_off.fit(
                X_ad_tr.iloc[m_tr][c_a], y_all[m_tr] / w_tr,
                sample_weight=w_tr, init_score=np.log(np.clip(pr_tr_cal, 1e-3, None)),
                categorical_feature=cat_a,
            )
            rate_tr_hat = pr_tr_cal * np.exp(m_off.predict(X_ad_tr.iloc[m_tr][c_a], raw_score=True))
            rate_va_hat = pr_va_cal * np.exp(m_off.predict(X_ad_tr.iloc[m_va][c_a], raw_score=True))
            cost_tr_hat = rate_tr_hat * exp_all[m_tr]
            cost_va_hat = rate_va_hat * exp_all[m_va]
            p_offset[m_va] = cost_va_hat * (y_all[m_tr].sum() / cost_tr_hat.sum())

    def d2_all(p):
        return float(d2_tweedie_score(y_all, np.maximum(p, FLOOR), power=1.5))

    # D-067: the offset layer stands alone (blend grid rises monotonically to w_off=1.0)
    p_final_oof = p_offset
    p_blend_legacy = 0.75 * p_offset + 0.25 * p_dev_zero
    print(f"5-Fold CV  constant_rate           D2 = {d2_all(p_const):+.5f}")
    print(f"5-Fold CV  premium_lr              D2 = {d2_all(p_prem):+.5f}")
    print(f"5-Fold CV  emerging_only_lgbm      D2 = {d2_all(p_em_only):+.5f}")
    print(f"5-Fold CV  developed_prior_zero    D2 = {d2_all(p_dev_zero):+.5f}")
    print(f"5-Fold CV  offset_adapted_lgbm     D2 = {d2_all(p_offset):+.5f}")
    print(f"5-Fold CV  legacy_blend_075_025    D2 = {d2_all(p_blend_legacy):+.5f}")
    print(f"5-Fold CV  final_offset_only       D2 = {d2_all(p_final_oof):+.5f}")
    per_lob_d2 = {}
    for ln in LINES:
        m = lines_all == ln
        per_lob_d2[ln] = float(d2_tweedie_score(y_all[m], np.maximum(p_final_oof[m], FLOOR), power=1.5))
        print(f"   {ln:10s} D2 = {per_lob_d2[ln]:+.5f}")

    # ---- Full refit on all 2023-2024 emerging rows -> predict 2025 test ----
    w_off = {}
    for kv in args.w_off.split(","):
        k, v = kv.split("=")
        w_off[k.strip()] = float(v)
    if set(w_off) != set(LINES):
        raise SystemExit(f"FATAL: --w-off must cover exactly {tuple(LINES)}, got {sorted(w_off)}")
    p_te_pred = np.zeros(len(em_te))
    p_te_prior = np.zeros(len(em_te))
    for ln in LINES:
        m_tr = (em_tr["line"] == ln).to_numpy()
        m_te = (em_te["line"] == ln).to_numpy()
        pr_tr = em_tr["prior_rate_blend"].to_numpy(float)[m_tr]
        pr_te = em_te["prior_rate_blend"].to_numpy(float)[m_te]
        scale_pr = y_all[m_tr].sum() / (pr_tr * exp_all[m_tr]).sum()
        pr_tr_cal = pr_tr * scale_pr
        pr_te_cal = pr_te * scale_pr

        sub_a = X_ad_tr.loc[m_tr]
        c_a = [c for c in cols_ad if sub_a[c].nunique(dropna=False) > 1]
        cat_a = [c for c in cats_ad if c in c_a]
        w_tr = np.clip(exp_all[m_tr], 1e-3, None)

        preds_off = np.zeros(m_te.sum())
        for seed in [0, 1, 2]:
            m_off = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=lob_n_est[ln], learning_rate=0.02, num_leaves=args.num_leaves,
                min_data_in_leaf=args.min_leaf, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=100.0, verbose=-1, random_state=seed,
            )
            m_off.fit(
                X_ad_tr.loc[m_tr, c_a], y_all[m_tr] / w_tr,
                sample_weight=w_tr, init_score=np.log(np.clip(pr_tr_cal, 1e-3, None)),
                categorical_feature=cat_a,
            )
            rate_tr_hat = pr_tr_cal * np.exp(m_off.predict(X_ad_tr.loc[m_tr, c_a], raw_score=True))
            rate_te_hat = pr_te_cal * np.exp(m_off.predict(X_ad_te.loc[m_te, c_a], raw_score=True))
            cost_tr_hat = rate_tr_hat * exp_all[m_tr]
            cost_te_hat = rate_te_hat * exp_te[m_te]
            preds_off += cost_te_hat * (y_all[m_tr].sum() / cost_tr_hat.sum()) / 3.0

        p_te_pred[m_te] = preds_off  # D-067 offset-only
        p_te_prior[m_te] = pr_te_cal * exp_te[m_te]

    if any(w < 1.0 for w in w_off.values()):
        # D-086 arm C: per-LOB offset/prior blend at level-calibrated costs
        p_te_pred = blend_lob_costs(p_te_pred, p_te_prior,
                                    em_te["line"].to_numpy(), w_off)
        print(f"\nper-LOB offset/prior blend (D-086 arm C): {w_off}")

    # ---- D-069: 2025 forward index applied at scoring (D-008) ----
    fwd, fwd_stats = estimate_forward_index(em_tr, em_te)
    em_te = em_te.copy()
    em_te["cohort"] = np.where(em_te["new_business"].fillna(0).astype(float) > 0.5, "nb", "ren")
    f_map = fwd.set_index(["line", "cohort"])["f_final"]
    f_rows = f_map.loc[list(zip(em_te["line"], em_te["cohort"]))].to_numpy(float)
    p_te_pred = p_te_pred * f_rows
    print("\n2025 forward index (D-069):")
    print(fwd[["line", "cohort", "f_data", "f_reconciled", "f_final"]].round(4).to_string(index=False))

    sample = pd.read_parquet(EM_DIR / "sample_submission.parquet")
    sub = pd.DataFrame({"row_id": em_te["row_id"], "prediction": np.maximum(p_te_pred, FLOOR)})
    # Order row_ids to match sample_submission exactly
    sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
    assert len(sub) == EXPECTED_EM_ROWS
    assert not sub["prediction"].isna().any()

    OUT_SUB.mkdir(parents=True, exist_ok=True)
    OUT_MET.mkdir(parents=True, exist_ok=True)
    sub_path = OUT_SUB / f"{args.name}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\nwrote {sub_path}  rows={len(sub):,}  mean={sub.prediction.mean():.2f}  p50={sub.prediction.median():.2f}")

    # ---- Build Office Premiums for the 20 reference_policies ----
    ref_pols = pd.read_parquet(EM_DIR / "reference_policies.parquet")
    expenses = pd.read_parquet(EM_DIR / "expenses.parquet")
    office_df = build_office_premiums(ref_pols, sub, expenses, market="emerging")
    office_path = OUT_MET / "emerging_reference_office_premiums.csv"
    office_df.to_csv(office_path, index=False)
    print(f"wrote {office_path} ({len(office_df)} reference policies)")

    (OUT_MET / f"{args.name}.json").write_text(json.dumps({
        "name": args.name,
        "market": "emerging",
        "rows_train": int(len(em_tr)),
        "rows_test": int(len(em_te)),
        "d2_constant": d2_all(p_const),
        "d2_premium_lr": d2_all(p_prem),
        "d2_emerging_only": d2_all(p_em_only),
        "d2_developed_prior_zero": d2_all(p_dev_zero),
        "d2_offset_adapted": d2_all(p_offset),
        "d2_credibility_blend_final": d2_all(p_final_oof),
        "d2_legacy_blend_075_025": d2_all(p_blend_legacy),
        "per_lob_d2": per_lob_d2,
        "forward_index": fwd.to_dict(orient="records"),
        "forward_index_stats": fwd_stats,
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
