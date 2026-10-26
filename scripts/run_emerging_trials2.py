#!/usr/bin/env python
"""Part 2 trial round 2 — offset capacity sweep + exact per-LOB blend score.

Rebuilds the round-1 OOF components (same seeds/configs), caches them to
/tmp/opencode/part2/oof_components.npz for fast re-blending, reports the exact
global OOF D2 of the round-1 per-LOB blend variant, and sweeps offset capacity
(n_estimators x0.5 / x1 / x2). Emits a candidate CSV only if it beats the
round-1 winner (offset-only, OOF 0.14045) by > 0.0003.
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


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def oof_offset(em_tr, X_ad_tr, cols_ad, cats_ad, y, exp_tr, lines, groups, pr, mult):
    out = np.zeros(len(em_tr))
    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(em_tr, y, groups):
        for ln in LINES:
            m_tr = tri[lines[tri] == ln]
            m_va = vai[lines[vai] == ln]
            w = np.clip(exp_tr[m_tr], 1e-3, None)
            scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            pr_cal = pr[m_tr] * scale
            c_a = [c for c in cols_ad if X_ad_tr.iloc[m_tr][c].nunique(dropna=False) > 1]
            m = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=int(LOB_N_EST[ln] * mult), learning_rate=0.02, num_leaves=3,
                min_data_in_leaf=600, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=100.0, verbose=-1, random_state=0,
            )
            m.fit(
                X_ad_tr.iloc[m_tr][c_a], y[m_tr] / w, sample_weight=w,
                init_score=np.log(np.clip(pr_cal, 1e-3, None)),
                categorical_feature=[c for c in cats_ad if c in c_a],
            )
            rate_tr = pr_cal * np.exp(m.predict(X_ad_tr.iloc[m_tr][c_a], raw_score=True))
            rate_va = pr[m_va] * scale * np.exp(m.predict(X_ad_tr.iloc[m_va][c_a], raw_score=True))
            out[m_va] = rate_va * exp_tr[m_va] * (y[m_tr].sum() / (rate_tr * exp_tr[m_tr]).sum())
    return out


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    for d in (em_tr,):
        d["prior_rate_blend"] = 0.6 * d["dev_prior_rate"] + 0.4 * d["dev_prior_rate_nowp"]

    (X_ad_tr, _), cols_ad, cats_ad = base.prep_matrices([em_tr, em_tr], base.ADAPT_NUM, base.ADAPT_CAT)

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = em_tr["prior_rate_blend"].to_numpy(float)

    # dev_zero OOF (cheap, no fits)
    dev_zero = np.zeros(len(em_tr))
    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(em_tr, y, groups):
        for ln in LINES:
            m_tr = tri[lines[tri] == ln]
            m_va = vai[lines[vai] == ln]
            scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            dev_zero[m_va] = pr[m_va] * scale * exp_tr[m_va]

    results = {}
    components = {"dev_zero": dev_zero}
    for mult in (0.5, 1.0, 2.0):
        off = oof_offset(em_tr, X_ad_tr, cols_ad, cats_ad, y, exp_tr, lines, groups, pr, mult)
        components[f"offset_x{mult}"] = off
        results[f"offset_x{mult}"] = d2(y, off)
        print(f"offset n_est x{mult}: OOF D2 = {results[f'offset_x{mult}']:+.5f}", flush=True)

    # exact global OOF D2 of round-1 per-LOB blend (motor 1.0, buildings 1.0, contents 0.5)
    off1 = components["offset_x1.0"]
    p = np.zeros(len(em_tr))
    for ln, w in (("motor", 1.0), ("buildings", 1.0), ("contents", 0.5)):
        m = lines == ln
        p[m] = w * off1[m] + (1 - w) * dev_zero[m]
    results["per_lob_blend_global"] = d2(y, p)
    print(f"round-1 per-LOB blend: global OOF D2 = {results['per_lob_blend_global']:+.5f}", flush=True)

    np.savez_compressed(
        CACHE / "oof_components.npz",
        dev_zero=dev_zero, offset_x1=off1,
        **{k: v for k, v in components.items() if k.startswith("offset_x") and k != "offset_x1.0"},
        y=y, lines=lines,
    )
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
