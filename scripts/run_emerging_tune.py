#!/usr/bin/env python
"""Part 2 hyperparameter residual sweep (D-075 candidate).

D-067 swept blend weights, capacity (n_est x0.5/x2), per-LOB tree counts,
extended adaptation feature sets and prior blends - all negatives at the
incumbent (lv=3, min_leaf=600, l2=100, lr=0.02, 35/30/20 trees, ff=0.8).
The shrinkage trio (num_leaves, min_data_in_leaf, lambda_l2), the learning-rate
trajectory, feature/bagging fractions and OOF seed sensitivity were never
individually swept.  This script closes that question.

Design (pre-registered):
  * one-at-a-time around the incumbent (attribution over interaction);
  * same 5-Fold GroupKFold(customer_id), same offset machinery and fold-level
    recalibration as run_emerging.py's D-067 layer (byte-identical for the
    incumbent config, which must reproduce OOF 0.14045 or the run aborts);
  * config 'seed_1' re-runs the incumbent at random_state=1 to measure local
    OOF seed sigma (the scoring refit already averages 3 seeds; CV does not).

Adoption gate (D-025): a config is a candidate only if total OOF D2 >
0.14045 + 0.0003.  With ~10 configs scanned, the maximum is subject to
winner's curse: any winner must be re-measured at a second seed before any
submission decision.  Nothing here touches Kaggle quota.

Usage:
  python scripts/run_emerging_tune.py
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(ROOT))

import run_emerging as base  # noqa: E402

CACHE = Path("/tmp/opencode/part2")
OUT_MET = ROOT / "outputs" / "metrics"
FLOOR = 1e-3
INCUMBENT_D2 = 0.14045
GATE = 0.0003
LOB_N_EST = {"motor": 35, "buildings": 30, "contents": 20}

# name -> (num_leaves, min_data_in_leaf, lambda_l2, lr, n_est_mult, ff, bag, seed)
CONFIGS = {
    "incumbent":   (3, 600, 100.0, 0.02, 1.0, 0.8, 0.8, 0),
    "min_leaf_300": (3, 300, 100.0, 0.02, 1.0, 0.8, 0.8, 0),
    "min_leaf_1200": (3, 1200, 100.0, 0.02, 1.0, 0.8, 0.8, 0),
    "l2_30":       (3, 600, 30.0, 0.02, 1.0, 0.8, 0.8, 0),
    "l2_300":      (3, 600, 300.0, 0.02, 1.0, 0.8, 0.8, 0),
    "lv_2":        (2, 600, 100.0, 0.02, 1.0, 0.8, 0.8, 0),
    "lv_4":        (4, 600, 100.0, 0.02, 1.0, 0.8, 0.8, 0),
    "lr_001_x2n":  (3, 600, 100.0, 0.01, 2.0, 0.8, 0.8, 0),
    "lr_004_halfn": (3, 600, 100.0, 0.04, 0.5, 0.8, 0.8, 0),
    "ff_10":       (3, 600, 100.0, 0.02, 1.0, 1.0, 1.0, 0),
    "combo_top2":  (4, 1200, 100.0, 0.02, 1.0, 0.8, 0.8, 0),
    "combo_top2_s1": (4, 1200, 100.0, 0.02, 1.0, 0.8, 0.8, 1),
    "combo_top2_s2": (4, 1200, 100.0, 0.02, 1.0, 0.8, 0.8, 2),
    "incumbent_s2": (3, 600, 100.0, 0.02, 1.0, 0.8, 0.8, 2),
    "combo_all3":  (4, 1200, 100.0, 0.02, 1.0, 1.0, 1.0, 0),
    "seed_1":      (3, 600, 100.0, 0.02, 1.0, 0.8, 0.8, 1),
}


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet").reset_index(drop=True)
    em_te = pd.read_parquet(CACHE / "em_te_with_prior.parquet").reset_index(drop=True)
    for d in (em_tr, em_te):
        d["prior_rate_blend"] = 0.6 * d["dev_prior_rate"] + 0.4 * d["dev_prior_rate_nowp"]

    (X_tr, _), cols, cats = base.prep_matrices([em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT)
    y_all = em_tr["y_ult"].to_numpy(float)
    exp_all = em_tr["exposure"].to_numpy(float)
    pr_all = em_tr["prior_rate_blend"].to_numpy(float)
    lines_all = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(em_tr, y_all, groups))

    results = {}
    for name, (lv, min_leaf, l2, lr, mult, ff, bag, seed) in CONFIGS.items():
        p_oof = np.zeros(len(em_tr))
        for tr_idx, va_idx in folds:
            for ln in LOB_N_EST:
                m_tr = tr_idx[lines_all[tr_idx] == ln]
                m_va = va_idx[lines_all[va_idx] == ln]
                pr_tr, pr_va = pr_all[m_tr], pr_all[m_va]
                scale_pr = y_all[m_tr].sum() / (pr_tr * exp_all[m_tr]).sum()
                pr_tr_cal, pr_va_cal = pr_tr * scale_pr, pr_va * scale_pr

                sub_a = X_tr.iloc[m_tr]
                c_a = [c for c in cols if sub_a[c].nunique(dropna=False) > 1]
                cat_a = [c for c in cats if c in c_a]
                w_tr = np.clip(exp_all[m_tr], 1e-3, None)

                m_off = lgb.LGBMRegressor(
                    objective="tweedie", tweedie_variance_power=1.5,
                    n_estimators=int(round(LOB_N_EST[ln] * mult)), learning_rate=lr,
                    num_leaves=lv, min_data_in_leaf=min_leaf,
                    feature_fraction=ff, bagging_fraction=bag,
                    bagging_freq=1, lambda_l2=l2, verbose=-1, random_state=seed,
                )
                m_off.fit(
                    X_tr.iloc[m_tr][c_a], y_all[m_tr] / w_tr,
                    sample_weight=w_tr, init_score=np.log(np.clip(pr_tr_cal, 1e-3, None)),
                    categorical_feature=cat_a,
                )
                rate_tr_hat = pr_tr_cal * np.exp(m_off.predict(X_tr.iloc[m_tr][c_a], raw_score=True))
                rate_va_hat = pr_va_cal * np.exp(m_off.predict(X_tr.iloc[m_va][c_a], raw_score=True))
                cost_tr_hat = rate_tr_hat * exp_all[m_tr]
                cost_va_hat = rate_va_hat * exp_all[m_va]
                p_oof[m_va] = cost_va_hat * (y_all[m_tr].sum() / cost_tr_hat.sum())

        per_lob = {ln: d2(y_all[lines_all == ln], p_oof[lines_all == ln]) for ln in LOB_N_EST}
        total = d2(y_all, p_oof)
        results[name] = {"total_d2": total, "per_lob_d2": per_lob,
                         "config": {"lv": lv, "min_leaf": min_leaf, "l2": l2, "lr": lr,
                                    "n_est_mult": mult, "ff": ff, "bag": bag, "seed": seed}}
        print(f"{name:16s} total D2 = {total:+.5f}  "
              f"({', '.join(f'{k} {v:+.4f}' for k, v in per_lob.items())})  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    if abs(results["incumbent"]["total_d2"] - INCUMBENT_D2) > 5e-5:
        print(f"ABORT: incumbent reproduces {results['incumbent']['total_d2']:.5f}, "
              f"expected {INCUMBENT_D2:.5f} - machinery mismatch, results discarded", flush=True)
        return 1

    seed_sigma = abs(results["seed_1"]["total_d2"] - results["incumbent"]["total_d2"])
    # paired replication of the winner across seeds (winner's-curse protection)
    paired = {}
    for s, ck, ik in ((0, "combo_top2", "incumbent"), (1, "combo_top2_s1", "seed_1"),
                      (2, "combo_top2_s2", "incumbent_s2")):
        paired[s] = results[ck]["total_d2"] - results[ik]["total_d2"]
    best = max((k for k in results if results[k]["config"]["seed"] == 0),
               key=lambda k: results[k]["total_d2"])
    gate_pass = results[best]["total_d2"] > INCUMBENT_D2 + GATE

    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / "emerging-tune.json").write_text(json.dumps({
        "incumbent_d2": INCUMBENT_D2,
        "results": results,
        "seed_sigma_local": seed_sigma,
        "combo_top2_paired_deltas": paired,
        "best_config": best,
        "best_d2": results[best]["total_d2"],
        "gate_pass": gate_pass,
        "note": ("winner's curse: best of 15 configs; combo_top2 sits at the +0.0003 gate "
                 "boundary - paired seed deltas decide adoption"),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    print(f"\nlocal OOF seed sigma (incumbent s0 vs s1): {seed_sigma:.5f}")
    print("combo_top2 paired deltas vs incumbent, per seed:",
          {s: round(v, 5) for s, v in paired.items()})
    print(f"best config: {best} at {results[best]['total_d2']:+.5f} "
          f"(incumbent {INCUMBENT_D2:+.5f}, delta {results[best]['total_d2'] - INCUMBENT_D2:+.5f})")
    print("GATE PASSED - second-seed confirmation required before any submission"
          if gate_pass else
          "gate not passed - incumbent retained, no candidate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
