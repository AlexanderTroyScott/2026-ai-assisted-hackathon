#!/usr/bin/env python
"""Part 2 trial round 4 — per-LOB tree counts and prior composition.

All configs keep the winning 14-feature adaptation set and lv=3/min_leaf=600/l2=100.
  D: motor 50 trees (weakest LOB gets more capacity)
  E: property-heavy counts (buildings 45, contents 30)
  F: pure written-premium prior as init (w_wp=1.0 instead of 0.6/0.4)
  G: motor-heavy extreme (motor 70)

Winner threshold: beat offset-only baseline OOF 0.14045 by > 0.0003.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts"))

import run_emerging as base  # noqa: E402
from run_emerging_trials3 import d2  # noqa: E402

from src.emerging import EM_DIR, LINES  # noqa: E402

CACHE = Path("/tmp/opencode/part2")
FLOOR = 1e-3
BASELINE = 0.14045
HP = dict(learning_rate=0.02, num_leaves=3, min_data_in_leaf=600,
          feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=100.0)


def oof_offset(X, cols, cats, y, exp_tr, lines, groups, pr, n_est):
    out = np.zeros(len(y))
    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(y, y, groups):
        for ln in LINES:
            m_tr = tri[lines[tri] == ln]
            m_va = vai[lines[vai] == ln]
            w = np.clip(exp_tr[m_tr], 1e-3, None)
            scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            pr_cal = pr[m_tr] * scale
            c_a = [c for c in cols if X.iloc[m_tr][c].nunique(dropna=False) > 1]
            m = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=n_est[ln], categorical_feature=[c for c in cats if c in c_a],
                verbose=-1, random_state=0, **HP,
            )
            m.fit(X.iloc[m_tr][c_a], y[m_tr] / w, sample_weight=w,
                  init_score=np.log(np.clip(pr_cal, 1e-3, None)))
            rate_tr = pr_cal * np.exp(m.predict(X.iloc[m_tr][c_a], raw_score=True))
            rate_va = pr[m_va] * scale * np.exp(m.predict(X.iloc[m_va][c_a], raw_score=True))
            out[m_va] = rate_va * exp_tr[m_va] * (y[m_tr].sum() / (rate_tr * exp_tr[m_tr]).sum())
    return out


def main() -> int:
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    em_tr["prior_rate_blend"] = 0.6 * em_tr["dev_prior_rate"] + 0.4 * em_tr["dev_prior_rate_nowp"]

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = em_tr["prior_rate_blend"].to_numpy(float)
    pr_wp = em_tr["dev_prior_rate"].to_numpy(float)

    (X, _), cols, cats = base.prep_matrices([em_tr, em_tr], base.ADAPT_NUM, base.ADAPT_CAT)

    configs = {
        "D_motor50": ({"motor": 50, "buildings": 30, "contents": 20}, pr),
        "E_prop_heavy": ({"motor": 35, "buildings": 45, "contents": 30}, pr),
        "F_wp_prior": ({"motor": 35, "buildings": 30, "contents": 20}, pr_wp),
        "G_motor70": ({"motor": 70, "buildings": 30, "contents": 20}, pr),
    }

    best_name, best_score = "", -9.0
    for name, (n_est, pr_cfg) in configs.items():
        off = oof_offset(X, cols, cats, y, exp_tr, lines, groups, pr_cfg, n_est)
        s = d2(y, off)
        per = {ln: round(d2(y[lines == ln], off[lines == ln]), 5) for ln in LINES}
        print(f"{name:14s} OOF D2 = {s:+.5f}  {per}", flush=True)
        if s > best_score:
            best_name, best_score = name, s

    print(f"best: {best_name} {best_score:+.5f} (baseline {BASELINE:+.5f}) "
          f"-> {'QUALIFIES' if best_score > BASELINE + 0.0003 else 'no submission'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
