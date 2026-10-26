#!/usr/bin/env python
"""Part 2 trial round 3 — adaptation feature-set expansion + prior recency.

Configs (one primary change each, same offset hyper-params as the winner):
  A: extended adaptation features (full risk factor set, lv=3/min_leaf=600/l2=100)
  B: extended features, higher capacity (lv=4/min_leaf=400/l2=50)
  C: winner feature set, prior recalibrated on 2024 only (recency-weighted init)

Winner threshold: beat offset-only baseline OOF 0.14045 by > 0.0003.
Emits outputs/submissions/emerging-exp-00N.csv for the best qualifying config.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_emerging as base  # noqa: E402

from src.emerging import EM_DIR, LINES  # noqa: E402

CACHE = Path("/tmp/opencode/part2")
FLOOR = 1e-3
LOB_N_EST = {"motor": 35, "buildings": 30, "contents": 20}
BASELINE = 0.14045

EXT_NUM = base.ADAPT_NUM + [
    "sum_insured_clean", "high_value_items", "vehicle_value_group", "vehicle_age",
    "ncb_years", "renewal_count", "days_quote_to_inception", "discount_applied",
    "compulsory_excess", "distance_to_river_km", "elevation_m", "year_built",
    "bedrooms", "freezing_days", "dry_summer_index", "rainfall_mm",
    "wind_exposure_index", "pol_cost_all", "pol_claims_1y", "days_since_claim",
    "household_size", "licence_points", "years_at_address",
]
EXT_CAT = base.ADAPT_CAT + [
    "urban_density", "construction_type", "occupancy", "tenure", "security_level",
    "property_type", "roof_type", "energy_rating", "soil_type", "payment_method",
    "marketing_segment",
]


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def oof_offset(X_ad_tr, cols, cats, y, exp_tr, lines, groups, pr, hparams, mult=1.0):
    out = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(y, y, groups):
        for ln in LINES:
            m_tr = tri[lines[tri] == ln]
            m_va = vai[lines[vai] == ln]
            w = np.clip(exp_tr[m_tr], 1e-3, None)
            scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            pr_cal = pr[m_tr] * scale
            c_a = [c for c in cols if X_ad_tr.iloc[m_tr][c].nunique(dropna=False) > 1]
            m = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=int(LOB_N_EST[ln] * mult),
                categorical_feature=[c for c in cats if c in c_a],
                verbose=-1, random_state=0, **hparams,
            )
            m.fit(
                X_ad_tr.iloc[m_tr][c_a], y[m_tr] / w, sample_weight=w,
                init_score=np.log(np.clip(pr_cal, 1e-3, None)),
            )
            rate_tr = pr_cal * np.exp(m.predict(X_ad_tr.iloc[m_tr][c_a], raw_score=True))
            rate_va = pr[m_va] * scale * np.exp(m.predict(X_ad_tr.iloc[m_va][c_a], raw_score=True))
            out[m_va] = rate_va * exp_tr[m_va] * (y[m_tr].sum() / (rate_tr * exp_tr[m_tr]).sum())
    return out


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    em_te = pd.read_parquet(CACHE / "em_te_with_prior.parquet")
    for d in (em_tr, em_te):
        d["prior_rate_blend"] = 0.6 * d["dev_prior_rate"] + 0.4 * d["dev_prior_rate_nowp"]

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = em_tr["prior_rate_blend"].to_numpy(float)

    (X_ext_tr, X_ext_te), cols_ext, cats_ext = base.prep_matrices(
        [em_tr, em_te], EXT_NUM, EXT_CAT
    )
    (X_win_tr, X_win_te), cols_win, cats_win = base.prep_matrices(
        [em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT
    )

    hp_a = dict(learning_rate=0.02, num_leaves=3, min_data_in_leaf=600,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=100.0)
    hp_b = dict(learning_rate=0.02, num_leaves=4, min_data_in_leaf=400,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=50.0)

    configs = {
        "A_ext_features": (X_ext_tr, cols_ext, cats_ext, pr, hp_a),
        "B_ext_highcap": (X_ext_tr, cols_ext, cats_ext, pr, hp_b),
    }

    # C: prior recalibrated on 2024 only
    pr_24 = pr.copy()
    for ln in LINES:
        m24 = (lines == ln) & (em_tr["acc_year"].to_numpy() == 2024)
        m = lines == ln
        scale24 = y[m24].sum() / (pr[m24] * exp_tr[m24]).sum()
        pr_24[m] = pr[m] * scale24
    configs["C_prior_2024"] = (X_win_tr, cols_win, cats_win, pr_24, hp_a)

    scores = {}
    oof_best = None
    best_name, best_score = "", -9.0
    for name, (X, cols, cats, pr_cfg, hp) in configs.items():
        off = oof_offset(X, cols, cats, y, exp_tr, lines, groups, pr_cfg, hp)
        s = d2(y, off)
        scores[name] = s
        per = {ln: round(d2(y[lines == ln], off[lines == ln]), 5) for ln in LINES}
        print(f"{name:18s} OOF D2 = {s:+.5f}  {per}", flush=True)
        if s > best_score:
            best_name, best_score, oof_best = name, s, off

    print(f"best: {best_name} {best_score:+.5f} (baseline {BASELINE:+.5f})", flush=True)

    if best_score > BASELINE + 0.0003:
        # full 3-seed test refit for the winning config
        name = best_name
        X_win, cols, cats, pr_cfg, hp = configs[name]
        (X_tr_u, X_te_u) = (X_ext_tr, X_ext_te) if name.startswith(("A", "B")) else (X_win_tr, X_win_te)
        exp_te = em_te["exposure"].to_numpy(float)
        pr_te = em_te["prior_rate_blend"].to_numpy(float)
        if name == "C_prior_2024":
            for ln in LINES:
                m24 = (lines == ln) & (em_tr["acc_year"].to_numpy() == 2024)
                m = (em_te["line"] == ln).to_numpy()
                scale24 = y[m24].sum() / (pr[m24] * exp_tr[m24]).sum()
                pr_te[m] = em_te["prior_rate_blend"].to_numpy(float)[m] * scale24

        pred = np.zeros(len(em_te))
        for ln in LINES:
            m_tr = (em_tr["line"] == ln).to_numpy()
            m_te = (em_te["line"] == ln).to_numpy()
            w = np.clip(exp_tr[m_tr], 1e-3, None)
            scale = y[m_tr].sum() / (pr_cfg[m_tr] * exp_tr[m_tr]).sum()
            pr_cal = pr_cfg[m_tr] * scale
            c_a = [c for c in cols if X_tr_u.loc[m_tr][c].nunique(dropna=False) > 1]
            acc = np.zeros(m_te.sum())
            for seed in (0, 1, 2):
                m = lgb.LGBMRegressor(
                    objective="tweedie", tweedie_variance_power=1.5,
                    n_estimators=LOB_N_EST[ln],
                    categorical_feature=[c for c in cats if c in c_a],
                    verbose=-1, random_state=seed, **hp,
                )
                m.fit(X_tr_u.loc[m_tr, c_a], y[m_tr] / w, sample_weight=w,
                      init_score=np.log(np.clip(pr_cal, 1e-3, None)))
                rate_tr = pr_cal * np.exp(m.predict(X_tr_u.loc[m_tr, c_a], raw_score=True))
                rate_te = pr_te[m_te] * np.exp(m.predict(X_te_u.loc[m_te, c_a], raw_score=True))
                acc += rate_te * exp_te[m_te] * (y[m_tr].sum() / (rate_tr * exp_tr[m_tr]).sum()) / 3.0
            pred[m_te] = acc

        sample = pd.read_parquet(EM_DIR / "sample_submission.parquet")
        sub = pd.DataFrame({"row_id": em_te["row_id"], "prediction": np.maximum(pred, FLOOR)})
        sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
        assert len(sub) == 209_063 and not sub["prediction"].isna().any()
        out = ROOT / "outputs" / "submissions" / "emerging-exp-006.csv"
        sub.to_csv(out, index=False)
        print(f"wrote {out.name} ({name}) mean={sub.prediction.mean():.2f}", flush=True)
    else:
        print("no config qualifies — no CSV written", flush=True)

    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
