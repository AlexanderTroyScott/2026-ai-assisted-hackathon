#!/usr/bin/env python
"""Part 2 transfer-prior upgrade — "exp-036 becomes the prior" (candidate D-082).

Primary change vs the record prior (D-066): the developed-market prior models
are brought up to the Part 1 state of the art (D-056/D-064):

  * family blend 42% Per-LOB LGBM + 40% Per-LOB XGBoost + 18% Pooled LGBM
    (exp-036 weights, FIXED a priori — not re-selected here),
  * `occ_clean` dropped (D-056: occupation is driver age in disguise),
  * the two-leg structure is kept: wp_rel-informed (60%) + no-wp (40%).

Everything downstream of the prior is byte-identical to the record pipeline:
same cached frames, same offset layer (record config lv=4/min_leaf=1200),
same fold calibration, same 3-seed refit, same PROBE_FORWARD scoring.

Family hyperparameters are the Part 1 documented specs (D-063/D-064) on a
fixed 400-tree budget matched to the record prior (the prior trains on ALL
developed rows — no eval set exists at this location, so no early stopping):
  per-LOB LGBM   lv=6,  lr=0.05, l2=5.0, min_leaf=200   (exp-033)
  pooled LGBM    lv=16, lr=0.04, l2=3.0, min_leaf=200   (exp-032)
  per-LOB XGB    depth=3, lr=0.05, min_child_weight=200,
                 reg_lambda=5.0                          (exp-035, D-064)
Pooled leg gets `line` as a categorical (pooled model needs LOB identity;
per-LOB legs drop it automatically as constant). XGBoost consumes categorical
codes as numeric, as in the D-070 port (`_encode_num`).

Pre-registered gate (plan agreed with the human 2026-10-28; validation before
any slot): paired-seed (0/1/2) 5-fold OOF D2 — variant-prior arm minus
record-prior arm at identical seeds and folds — must average >= +0.0002, and
the record arm must reproduce the record OOF 0.14074 at seed 0 (contract
check). PASS -> emit a submission CSV and request ONE Part 2 slot from the
human (explicit per-submission confirmation still required). FAIL -> no CSV,
this axis closes, the slot is held.

Usage:
  python scripts/run_emerging_prior.py
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
from run_emerging_ensemble2 import _encode_num, fit_lgbm_offset  # noqa: E402

from src.emerging import DEV_DIR, DEV_LDF, EM_DIR, LINES, estimate_forward_index  # noqa: E402

CACHE = Path("/tmp/opencode/part2")
OUT_SUB = ROOT / "outputs" / "submissions"
OUT_MET = ROOT / "outputs" / "metrics"
FLOOR = 1e-3
SEEDS = (0, 1, 2)
GATE = 0.0002
NAME = "emerging-prior-exp36"
RECORD_OOF_SEED0 = 0.14074   # record LGBM-only OOF at seed 0 (exp-025 contract)
CONTRACT_TOL = 1e-5          # 0.14074 is the 5-dp rounding of the record OOF

# exp-036 blend weights (D-064) and the record two-leg blend (D-066)
W_LGBM_LOB, W_XGB_LOB, W_LGBM_POOL = 0.42, 0.40, 0.18
W_WP, W_NOWP = 0.6, 0.4
N_EST = 400                  # boosting budget matched to the record prior
VAR_CAT = [c for c in base.SHARED_CAT if c != "occ_clean"]  # D-056


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def fit_variant_prior(dev_tr: pd.DataFrame, em_tr: pd.DataFrame, em_te: pd.DataFrame):
    """exp-036-style family prior on the shared feature set, occ_clean dropped.

    Mirrors base.fit_developed_priors structurally (two legs, per-line active
    features, rate target y_ult/exposure with exposure weights) but replaces
    the single LGBM per leg with the exp-036 family blend. Writes
    dev_prior_rate_v / dev_prior_rate_nowp_v onto copies of em_tr/em_te.
    """
    (X_dev, X_em_tr, X_em_te), all_shared, cats_shared = base.prep_matrices(
        [dev_tr, em_tr, em_te], base.SHARED_NUM + ["wp_rel"], VAR_CAT + ["line"]
    )
    em_tr = em_tr.copy()
    em_te = em_te.copy()
    w_dev = np.clip(dev_tr["exposure"].to_numpy(float), 1e-3, None)
    y_dev = dev_tr["y_ult"].to_numpy(float) / w_dev
    dev_line = dev_tr["line"].to_numpy()
    em_tr_line = em_tr["line"].to_numpy()
    em_te_line = em_te["line"].to_numpy()

    legs = {
        "_v": list(all_shared),                     # wp_rel-informed leg
        "_nowp_v": [c for c in all_shared if c != "wp_rel"],
    }

    for suf, cols_leg in legs.items():
        # ---- pooled LGBM (exp-032 spec), all lines, `line` as categorical ----
        c_p = [c for c in cols_leg if X_dev[c].nunique(dropna=False) > 1]
        cat_p = [c for c in cats_shared if c in c_p]
        mp = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5,
            n_estimators=N_EST, learning_rate=0.04, num_leaves=16,
            min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=3.0, verbose=-1, random_state=0,
        )
        mp.fit(X_dev[c_p], y_dev, sample_weight=w_dev, categorical_feature=cat_p)
        pool_tr = np.maximum(mp.predict(X_em_tr[c_p]), FLOOR)
        pool_te = np.maximum(mp.predict(X_em_te[c_p]), FLOOR)
        print(f"  leg{suf}: pooled LGBM done", flush=True)

        for ln in LINES:
            m_dev = dev_line == ln
            m_etr = em_tr_line == ln
            m_ete = em_te_line == ln

            sub_dev = X_dev.loc[m_dev]
            c_ln = [c for c in cols_leg if sub_dev[c].nunique(dropna=False) > 1]
            cat_ln = [c for c in cats_shared if c in c_ln]
            Xd = X_dev.loc[m_dev, c_ln]
            y_ln, w_ln = y_dev[m_dev], w_dev[m_dev]

            # ---- Per-LOB LGBM (exp-033 spec) ----
            ml = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=N_EST, learning_rate=0.05, num_leaves=6,
                min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=5.0, verbose=-1, random_state=0,
            )
            ml.fit(Xd, y_ln, sample_weight=w_ln, categorical_feature=cat_ln)
            lg_tr = np.maximum(ml.predict(X_em_tr.loc[m_etr, c_ln]), FLOOR)
            lg_te = np.maximum(ml.predict(X_em_te.loc[m_ete, c_ln]), FLOOR)

            # ---- Per-LOB XGBoost (exp-035 spec, D-064) ----
            mx = xgb.XGBRegressor(
                objective="reg:tweedie", tweedie_variance_power=1.5,
                n_estimators=N_EST, learning_rate=0.05, max_depth=3,
                min_child_weight=200.0, reg_lambda=5.0, subsample=0.8,
                colsample_bytree=0.8, random_state=0, verbosity=0,
            )
            mx.fit(_encode_num(Xd, c_ln, cat_ln), y_ln, sample_weight=w_ln)
            xg_tr = np.maximum(
                mx.predict(_encode_num(X_em_tr.loc[m_etr, c_ln], c_ln, cat_ln)), FLOOR)
            xg_te = np.maximum(
                mx.predict(_encode_num(X_em_te.loc[m_ete, c_ln], c_ln, cat_ln)), FLOOR)

            em_tr.loc[m_etr, f"dev_prior_rate{suf}"] = (
                W_LGBM_LOB * lg_tr + W_XGB_LOB * xg_tr + W_LGBM_POOL * pool_tr[m_etr])
            em_te.loc[m_ete, f"dev_prior_rate{suf}"] = (
                W_LGBM_LOB * lg_te + W_XGB_LOB * xg_te + W_LGBM_POOL * pool_te[m_ete])
            print(f"  leg{suf}: {ln} LGBM+XGB done", flush=True)

    return em_tr, em_te


def prior_only_oof(pr, y, exp_tr, lines, folds):
    """Fold-recalibrated prior alone (no offset layer) — ladder row 4."""
    p = np.zeros(len(y))
    for tri, vai in folds:
        for ln in LINES:
            m_tr, m_va = tri[lines[tri] == ln], vai[lines[vai] == ln]
            scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            p[m_va] = pr[m_va] * scale * exp_tr[m_va]
    return d2(y, p)


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    em_te = pd.read_parquet(CACHE / "em_te_with_prior.parquet")
    for d in (em_tr, em_te):
        d["prior_rate_blend"] = 0.6 * d["dev_prior_rate"] + 0.4 * d["dev_prior_rate_nowp"]

    print("building developed frame...", flush=True)
    dev_tr = base.build_market_frame(DEV_DIR, "train", DEV_LDF)
    print("fitting variant prior (exp-036 family blend)...", flush=True)
    em_tr, em_te = fit_variant_prior(dev_tr, em_tr, em_te)
    em_tr["prior_rate_blend_v"] = W_WP * em_tr["dev_prior_rate_v"] + W_NOWP * em_tr["dev_prior_rate_nowp_v"]
    em_te["prior_rate_blend_v"] = W_WP * em_te["dev_prior_rate_v"] + W_NOWP * em_te["dev_prior_rate_nowp_v"]

    (X_ad_tr, X_ad_te), cols_ad, cats_ad = base.prep_matrices(
        [em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT)

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    exp_te = em_te["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr_rec = em_tr["prior_rate_blend"].to_numpy(float)
    pr_var = em_tr["prior_rate_blend_v"].to_numpy(float)
    pr_rec_te = em_te["prior_rate_blend"].to_numpy(float)
    pr_var_te = em_te["prior_rate_blend_v"].to_numpy(float)

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(em_tr, y, groups))

    print("prior-alone OOF (fold-recalibrated, no offset layer):", flush=True)
    d2_rec_zero = prior_only_oof(pr_rec, y, exp_tr, lines, folds)
    d2_var_zero = prior_only_oof(pr_var, y, exp_tr, lines, folds)
    print(f"  record prior alone  D2 = {d2_rec_zero:+.5f}   (ladder row: +0.1386)")
    print(f"  variant prior alone D2 = {d2_var_zero:+.5f}", flush=True)

    # ---- paired-seed OOF: variant prior vs record prior, identical folds/seeds ----
    oof_rec = {s: np.zeros(len(em_tr)) for s in SEEDS}
    oof_var = {s: np.zeros(len(em_tr)) for s in SEEDS}

    for seed in SEEDS:
        for tri, vai in folds:
            for ln in LINES:
                m_tr, m_va = tri[lines[tri] == ln], vai[lines[vai] == ln]
                w = np.clip(exp_tr[m_tr], 1e-3, None)
                for pr, oof in ((pr_rec, oof_rec), (pr_var, oof_var)):
                    scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
                    pr_cal = pr[m_tr] * scale
                    f = fit_lgbm_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad,
                                        y[m_tr], w, pr_cal, ln, seed)
                    r_tr = pr_cal * f(X_ad_tr.iloc[m_tr])
                    r_va = pr[m_va] * scale * f(X_ad_tr.iloc[m_va])
                    oof[seed][m_va] = r_va * exp_tr[m_va] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())
        print(f"  seed {seed} OOF done ({time.time() - t0:.0f}s)", flush=True)

    print("paired-seed OOF (5-fold GroupKFold(customer_id)):", flush=True)
    deltas = []
    for seed in SEEDS:
        d_r, d_v = d2(y, oof_rec[seed]), d2(y, oof_var[seed])
        deltas.append(d_v - d_r)
        note = ""
        if seed == 0:
            drift = abs(d_r - RECORD_OOF_SEED0)
            note = f"  (record-contract check: |{d_r:.5f} - {RECORD_OOF_SEED0}| = {drift:.2e})"
            assert drift < CONTRACT_TOL, f"record arm drift {drift} >= {CONTRACT_TOL} — comparison untrustworthy, aborting"
        print(f"  seed {seed}: record {d_r:+.5f}  variant {d_v:+.5f}  delta {d_v - d_r:+.5f}{note}", flush=True)
    mean_delta = float(np.mean(deltas))
    print(f"mean paired delta = {mean_delta:+.5f}  gate >= +{GATE:.4f} -> "
          f"{'PASS' if mean_delta >= GATE else 'FAIL'}", flush=True)

    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / f"{NAME}.json").write_text(json.dumps({
        "name": NAME,
        "primary_change": "transfer prior upgraded to Part 1 state of the art: "
                          "42% per-LOB LGBM (exp-033) + 40% per-LOB XGB (exp-035) "
                          "+ 18% pooled LGBM (exp-032), occ_clean dropped (D-056), "
                          "0.6/0.4 wp legs kept (record: LGBM-only, occ_clean in)",
        "weights": {"lgbm_lob": W_LGBM_LOB, "xgb_lob": W_XGB_LOB, "lgbm_pool": W_LGBM_POOL,
                    "wp": W_WP, "nowp": W_NOWP},
        "n_estimators": N_EST,
        "seeds": list(SEEDS),
        "d2_prior_only_record": d2_rec_zero,
        "d2_prior_only_variant": d2_var_zero,
        "oof_record": {str(s): d2(y, oof_rec[s]) for s in SEEDS},
        "oof_variant": {str(s): d2(y, oof_var[s]) for s in SEEDS},
        "paired_deltas": deltas,
        "mean_paired_delta": mean_delta,
        "gate": GATE,
        "pass": bool(mean_delta >= GATE),
        "record_contract_drift_seed0": abs(d2(y, oof_rec[0]) - RECORD_OOF_SEED0),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    if mean_delta < GATE:
        print("gate FAIL — no CSV written, slot held, axis closes", flush=True)
        return 0

    # ---- gate PASS: full refit (3 seeds), variant prior, PROBE_FORWARD scoring ----
    p_te_pred = np.zeros(len(em_te))
    for ln in LINES:
        m_tr = (em_tr["line"] == ln).to_numpy()
        m_te = (em_te["line"] == ln).to_numpy()
        w = np.clip(exp_tr[m_tr], 1e-3, None)
        scale = y[m_tr].sum() / (pr_var[m_tr] * exp_tr[m_tr]).sum()
        pr_cal = pr_var[m_tr] * scale

        acc = np.zeros(m_te.sum())
        for seed in SEEDS:
            f = fit_lgbm_offset(X_ad_tr.loc[m_tr], cols_ad, cats_ad, y[m_tr], w, pr_cal, ln, seed)
            r_tr = pr_cal * f(X_ad_tr.loc[m_tr])
            r_te = pr_var_te[m_te] * scale * f(X_ad_te.loc[m_te])
            acc += r_te * exp_te[m_te] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum()) / len(SEEDS)
        p_te_pred[m_te] = acc

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
    print(f"\ngate PASS — wrote {out.name}  rows={len(sub):,}  mean={sub.prediction.mean():.2f}  "
          f"total={sub.prediction.sum():,.0f}  sha256={sha[:12]}", flush=True)
    print("CSV NOT submitted — request one Part 2 slot from the human first "
          "(explicit per-submission confirmation, kaggle_submit.py serially).", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
