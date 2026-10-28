#!/usr/bin/env python
"""Part 2 ensemble-on-record — D-070 blend promoted onto the record config.

Primary change vs `emerging-exp-025`: the offset layer's prediction becomes a
0.7 / 0.3 blend of the record offset LGBM (num_leaves=4, min_data_in_leaf=1200)
with the D-070 offset XGBoost component (tweedie p=1.5, depth=3,
base_margin=log prior).  Everything else is byte-identical to the record
pipeline (`run_emerging.py --use-cache --num-leaves 4 --min-leaf 1200`): same
cached frames, same per-fold calibration, same 3-seed refit, same
PROBE_FORWARD scoring via estimate_forward_index.

Pre-registered gate (human-approved 2026-10-28 "ensemble draw #1 today,
validation-gated"): paired-seed (0/1/2) 5-fold OOF D2 of the blend minus the
LGBM-only arm at identical seeds must average >= +0.0002.  The blend weight is
FIXED at D-070's 0.7/0.3 and is not re-selected here (avoid selection bias).
Below the gate no CSV is emitted and the submission slot is held.

Usage:
  python scripts/run_emerging_ensemble2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_emerging as base  # noqa: E402

from src.emerging import EM_DIR, LINES, estimate_forward_index  # noqa: E402

CACHE = Path("/tmp/opencode/part2")
OUT_SUB = ROOT / "outputs" / "submissions"
OUT_MET = ROOT / "outputs" / "metrics"
FLOOR = 1e-3
LOB_N_EST = {"motor": 35, "buildings": 30, "contents": 20}
W_LGB, W_XGB = 0.7, 0.3
SEEDS = (0, 1, 2)
GATE = 0.0002
NUM_LEAVES = 4          # D-075 combo_top2 (record config)
MIN_LEAF = 1200         # D-075 combo_top2 (record config)
NAME = "emerging-exp-030"
RECORD_OOF_SEED0 = 0.14074  # record LGBM-only OOF at seed 0 (exp-025 contract)


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def fit_lgbm_offset(X, cols, cats, y, w, pr_cal, ln, seed):
    """Record-config offset LGBM — byte-identical hyperparameters to
    run_emerging.py's m_off at --num-leaves 4 --min-leaf 1200."""
    c = [c for c in cols if X[c].nunique(dropna=False) > 1]
    m = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=LOB_N_EST[ln], learning_rate=0.02, num_leaves=NUM_LEAVES,
        min_data_in_leaf=MIN_LEAF, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=100.0, verbose=-1, random_state=seed,
    )
    m.fit(X[c], y / w, sample_weight=w, init_score=np.log(np.clip(pr_cal, 1e-3, None)),
          categorical_feature=[x for x in cats if x in c])
    return lambda Z: np.exp(m.predict(Z[c], raw_score=True))


def _encode_num(df, cols, cats):
    out = df[cols].copy()
    for c in cats:
        out[c] = (out[c].cat.codes.astype(float) if hasattr(out[c], "cat")
                  else pd.Categorical(out[c]).codes.astype(float))
    return out


def fit_xgb_offset(X, cols, cats, y, w, pr_cal, ln, seed):
    """D-070 offset XGBoost component, unchanged.

    `base_margin` must be passed at BOTH fit and predict (D-070 mechanics
    note: predict() without it is not trees-only).
    """
    m = xgb.XGBRegressor(
        objective="reg:tweedie", tweedie_variance_power=1.5,
        n_estimators=LOB_N_EST[ln], learning_rate=0.05, max_depth=3,
        min_child_weight=300.0, reg_lambda=100.0, subsample=0.8,
        colsample_bytree=0.8, random_state=seed, verbosity=0,
    )
    m.fit(_encode_num(X, cols, cats), y / w, sample_weight=w,
          base_margin=np.log(np.clip(pr_cal, 1e-3, None)))
    return lambda Z, pr_z: m.predict(
        _encode_num(Z, cols, cats), base_margin=np.log(np.clip(pr_z, 1e-3, None)))


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    em_te = pd.read_parquet(CACHE / "em_te_with_prior.parquet")
    for d in (em_tr, em_te):
        d["prior_rate_blend"] = 0.6 * d["dev_prior_rate"] + 0.4 * d["dev_prior_rate_nowp"]

    (X_ad_tr, X_ad_te), cols_ad, cats_ad = base.prep_matrices(
        [em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT)

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    exp_te = em_te["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = em_tr["prior_rate_blend"].to_numpy(float)
    pr_te = em_te["prior_rate_blend"].to_numpy(float)

    # ---- paired-seed OOF: blend vs LGBM-only at identical seeds/folds ----
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(em_tr, y, groups))
    oof_lgb = {s: np.zeros(len(em_tr)) for s in SEEDS}
    oof_xgb = {s: np.zeros(len(em_tr)) for s in SEEDS}

    for seed in SEEDS:
        for tri, vai in folds:
            for ln in LINES:
                m_tr, m_va = tri[lines[tri] == ln], vai[lines[vai] == ln]
                w = np.clip(exp_tr[m_tr], 1e-3, None)
                scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
                pr_cal = pr[m_tr] * scale

                f_l = fit_lgbm_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad,
                                      y[m_tr], w, pr_cal, ln, seed)
                r_tr = pr_cal * f_l(X_ad_tr.iloc[m_tr])
                r_va = pr[m_va] * scale * f_l(X_ad_tr.iloc[m_va])
                oof_lgb[seed][m_va] = r_va * exp_tr[m_va] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())

                f_x = fit_xgb_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad,
                                     y[m_tr], w, pr_cal, ln, seed)
                r_tr = f_x(X_ad_tr.iloc[m_tr], pr_cal)
                r_va = f_x(X_ad_tr.iloc[m_va], pr[m_va] * scale)
                oof_xgb[seed][m_va] = r_va * exp_tr[m_va] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())

    print("paired-seed OOF (5-fold GroupKFold(customer_id)):", flush=True)
    deltas = []
    for seed in SEEDS:
        d_lgb = d2(y, oof_lgb[seed])
        d_blend = d2(y, W_LGB * oof_lgb[seed] + W_XGB * oof_xgb[seed])
        deltas.append(d_blend - d_lgb)
        note = ""
        if seed == 0:
            drift = abs(d_lgb - RECORD_OOF_SEED0)
            note = f"  (record-contract check: |{d_lgb:.5f} - {RECORD_OOF_SEED0}| = {drift:.6f})"
        print(f"  seed {seed}: lgbm {d_lgb:+.5f}  blend {d_blend:+.5f}  delta {d_blend - d_lgb:+.5f}{note}", flush=True)
    mean_delta = float(np.mean(deltas))
    print(f"mean paired delta = {mean_delta:+.5f}  gate >= +{GATE:.4f} -> "
          f"{'PASS' if mean_delta >= GATE else 'FAIL'}", flush=True)

    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / "emerging-ensemble2.json").write_text(json.dumps({
        "name": NAME,
        "primary_change": "offset layer = 0.7/0.3 blend record LGBM + D-070 XGB (weights fixed a priori)",
        "weights": {"lgbm": W_LGB, "xgb": W_XGB},
        "seeds": list(SEEDS),
        "oof_lgbm": {str(s): d2(y, oof_lgb[s]) for s in SEEDS},
        "oof_blend": {str(s): d2(y, W_LGB * oof_lgb[s] + W_XGB * oof_xgb[s]) for s in SEEDS},
        "paired_deltas": deltas,
        "mean_paired_delta": mean_delta,
        "gate": GATE,
        "pass": bool(mean_delta >= GATE),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    if mean_delta < GATE:
        print("gate FAIL — no CSV written, slot held", flush=True)
        return 0

    # ---- full refit (3 seeds per component), blend, PROBE_FORWARD scoring ----
    pred_lgb = np.zeros(len(em_te))
    pred_xgb = np.zeros(len(em_te))
    for ln in LINES:
        m_tr = (em_tr["line"] == ln).to_numpy()
        m_te = (em_te["line"] == ln).to_numpy()
        w = np.clip(exp_tr[m_tr], 1e-3, None)
        scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
        pr_cal = pr[m_tr] * scale

        acc_l = np.zeros(m_te.sum())
        acc_x = np.zeros(m_te.sum())
        for seed in SEEDS:
            f_l = fit_lgbm_offset(X_ad_tr.loc[m_tr], cols_ad, cats_ad, y[m_tr], w, pr_cal, ln, seed)
            r_tr = pr_cal * f_l(X_ad_tr.loc[m_tr])
            r_te = pr_te[m_te] * scale * f_l(X_ad_te.loc[m_te])
            acc_l += r_te * exp_te[m_te] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum()) / len(SEEDS)

            f_x = fit_xgb_offset(X_ad_tr.loc[m_tr], cols_ad, cats_ad, y[m_tr], w, pr_cal, ln, seed)
            r_tr = f_x(X_ad_tr.loc[m_tr], pr_cal)
            r_te = f_x(X_ad_te.loc[m_te], pr_te[m_te] * scale)
            acc_x += r_te * exp_te[m_te] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum()) / len(SEEDS)

        pred_lgb[m_te] = acc_l
        pred_xgb[m_te] = acc_x

    p_te_pred = W_LGB * pred_lgb + W_XGB * pred_xgb

    fwd, fwd_stats = estimate_forward_index(em_tr, em_te)
    em_te_c = em_te.copy()
    em_te_c["cohort"] = np.where(em_te_c["new_business"].fillna(0).astype(float) > 0.5, "nb", "ren")
    f_map = fwd.set_index(["line", "cohort"])["f_final"]
    p_te_pred = p_te_pred * f_map.loc[list(zip(em_te_c["line"], em_te_c["cohort"]))].to_numpy(float)

    sample = pd.read_parquet(EM_DIR / "sample_submission.parquet")
    sub = pd.DataFrame({"row_id": em_te["row_id"], "prediction": np.maximum(p_te_pred, FLOOR)})
    sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
    assert len(sub) == 209_063 and not sub["prediction"].isna().any()

    OUT_SUB.mkdir(parents=True, exist_ok=True)
    out = OUT_SUB / f"{NAME}.csv"
    sub.to_csv(out, index=False)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"\nwrote {out.name}  rows={len(sub):,}  mean={sub.prediction.mean():.2f}  "
          f"total={sub.prediction.sum():,.0f}  sha256={sha[:12]}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
