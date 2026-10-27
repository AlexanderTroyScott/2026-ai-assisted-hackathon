#!/usr/bin/env python
"""Part 2 ensemble trial — multi-family adaptation blend (D-064 lesson port).

Components in OOF space (same folds/seeds discipline as run_emerging_trials):
  A offset_lgbm  : current ultra-shallow offset LGBM (D-067 winner)
  B offset_xgb   : XGBoost offset adaptation (tweedie p=1.5, depth=3,
                   base_margin = log(prior)) — the D-064 family diversity
  C pooled_lgbm  : emerging-only pooled LGBM (lv=16 family per D-064's third leg)
  D dev_zero     : calibrated developed prior alone

Blend weights are searched on OOF D^2; a candidate CSV (with the D-069 forward
index applied at scoring) is emitted only if the best blend beats the
offset-only baseline (0.14045) by more than 0.0003 (D-025 noise threshold).

Usage:
  python scripts/run_emerging_ensemble.py
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
BASELINE = 0.14045


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def fit_lgbm_offset(X, cols, cats, y, w, pr_cal, ln, seed=0):
    c = [c for c in cols if X[c].nunique(dropna=False) > 1]
    m = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=LOB_N_EST[ln], learning_rate=0.02, num_leaves=3,
        min_data_in_leaf=600, feature_fraction=0.8, bagging_fraction=0.8,
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


def fit_xgb_offset(X, cols, cats, y, w, pr_cal, ln, seed=0):
    """XGBoost offset adaptation.

    `base_margin` must be passed at BOTH fit and predict (verified empirically:
    predict() without it is not trees-only — its mean sat between the
    correction-only and full-rate levels).  The returned callable takes the
    per-row prior rate and returns the RATE directly.
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

    (X_ad_tr, X_ad_te), cols_ad, cats_ad = base.prep_matrices([em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT)
    (X_full_tr, X_full_te), cols_full, cats_full = base.prep_matrices(
        [em_tr, em_te], base.SHARED_NUM + ["written_premium", "wp_rel", "is_self_employed"],
        base.SHARED_CAT + ["region", "gender"],
    )

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    exp_te = em_te["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = em_tr["prior_rate_blend"].to_numpy(float)
    pr_te = em_te["prior_rate_blend"].to_numpy(float)

    comps = ["offset_lgbm", "offset_xgb", "pooled_lgbm", "dev_zero"]
    oof = {k: np.zeros(len(em_tr)) for k in comps}

    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(em_tr, y, groups):
        for ln in LINES:
            m_tr, m_va = tri[lines[tri] == ln], vai[lines[vai] == ln]
            w = np.clip(exp_tr[m_tr], 1e-3, None)
            scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            pr_cal = pr[m_tr] * scale
            oof["dev_zero"][m_va] = pr[m_va] * scale * exp_tr[m_va]

            f_lgbm = fit_lgbm_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad, y[m_tr], w, pr_cal, ln)
            r_va = pr[m_va] * scale * f_lgbm(X_ad_tr.iloc[m_va])
            r_tr = pr_cal * f_lgbm(X_ad_tr.iloc[m_tr])
            oof["offset_lgbm"][m_va] = r_va * exp_tr[m_va] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())

            f_xgb = fit_xgb_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad, y[m_tr], w, pr_cal, ln)
            r_va = f_xgb(X_ad_tr.iloc[m_va], pr[m_va] * scale)
            r_tr = f_xgb(X_ad_tr.iloc[m_tr], pr_cal)
            oof["offset_xgb"][m_va] = r_va * exp_tr[m_va] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())

        # pooled emerging-only LGBM (D-064 third family: lv=16 pooled rate model)
        w = np.clip(exp_tr[tri], 1e-3, None)
        c_f = [c for c in cols_full if X_full_tr.iloc[tri][c].nunique(dropna=False) > 1]
        m = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5,
            n_estimators=250, learning_rate=0.04, num_leaves=16,
            min_data_in_leaf=150, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, random_state=0,
        )
        m.fit(X_full_tr.iloc[tri][c_f], y[tri] / w, sample_weight=w,
              categorical_feature=[x for x in cats_full if x in c_f])
        ptr = np.maximum(m.predict(X_full_tr.iloc[tri][c_f]) * exp_tr[tri], 1e-3)
        pva = np.maximum(m.predict(X_full_tr.iloc[vai][c_f]) * exp_tr[vai], 1e-3)
        oof["pooled_lgbm"][vai] = pva * (y[tri].sum() / ptr.sum())

    print("OOF components:", flush=True)
    for k in comps:
        print(f"  {k:12s} D2 = {d2(y, oof[k]):+.5f}", flush=True)

    # ---- blend grid over the simplex (coarse, then refine) ----
    best = (0.0, 0.0, 0.0, -9.0)
    grid = {}
    for w_x in np.round(np.arange(0.0, 0.61, 0.1), 2):
        for w_p in (0.0, 0.1, 0.2):
            for w_d in (0.0, 0.1):
                w_a = round(1 - w_x - w_p - w_d, 2)
                if w_a < 0:
                    continue
                p = (w_a * oof["offset_lgbm"] + w_x * oof["offset_xgb"]
                     + w_p * oof["pooled_lgbm"] + w_d * oof["dev_zero"])
                s = d2(y, p)
                grid[f"{w_a}/{w_x}/{w_p}/{w_d}"] = s
                if s > best[3]:
                    best = (w_a, w_x, w_p, s)
    print(f"best blend lgbm/xgb/pool/dev = {best[0]}/{best[1]}/{best[2]}  OOF D2 = {best[3]:+.5f} "
          f"(baseline {BASELINE:+.5f}, gain {best[3]-BASELINE:+.5f})", flush=True)

    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / "emerging-ensemble.json").write_text(json.dumps({
        "oof_components": {k: d2(y, oof[k]) for k in comps},
        "best_blend": {"lgbm": best[0], "xgb": best[1], "pool": best[2], "d2": best[3]},
        "grid_top": dict(sorted(grid.items(), key=lambda kv: -kv[1])[:10]),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    qualifies = best[3] > BASELINE + 0.0003
    if not qualifies:
        print("no blend qualifies (<= +0.0003) — no CSV written", flush=True)
        return 0

    # ---- full-refit test components (3 seeds for the offset families) ----
    test_comp = {k: np.zeros(len(em_te)) for k in comps}
    for ln in LINES:
        m_tr = (em_tr["line"] == ln).to_numpy()
        m_te = (em_te["line"] == ln).to_numpy()
        w = np.clip(exp_tr[m_tr], 1e-3, None)
        scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
        pr_cal = pr[m_tr] * scale
        test_comp["dev_zero"][m_te] = pr_te[m_te] * scale * exp_te[m_te]

        acc = np.zeros(m_te.sum())
        for seed in (0, 1, 2):
            f = fit_lgbm_offset(X_ad_tr.loc[m_tr], cols_ad, cats_ad, y[m_tr], w, pr_cal, ln, seed)
            r_tr = pr_cal * f(X_ad_tr.loc[m_tr])
            r_te = pr_te[m_te] * scale * f(X_ad_te.loc[m_te])
            acc += r_te * exp_te[m_te] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum()) / 3.0
        test_comp["offset_lgbm"][m_te] = acc

        acc = np.zeros(m_te.sum())
        for seed in (0, 1, 2):
            f = fit_xgb_offset(X_ad_tr.loc[m_tr], cols_ad, cats_ad, y[m_tr], w, pr_cal, ln, seed)
            r_tr = f(X_ad_tr.loc[m_tr], pr_cal)
            r_te = f(X_ad_te.loc[m_te], pr_te[m_te] * scale)
            acc += r_te * exp_te[m_te] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum()) / 3.0
        test_comp["offset_xgb"][m_te] = acc

    w = np.clip(exp_tr, 1e-3, None)
    c_f = [c for c in cols_full if X_full_tr[c].nunique(dropna=False) > 1]
    m = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=250, learning_rate=0.04, num_leaves=16,
        min_data_in_leaf=150, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=5.0, verbose=-1, random_state=0,
    )
    m.fit(X_full_tr[c_f], y / w, sample_weight=w, categorical_feature=[x for x in cats_full if x in c_f])
    ptr = np.maximum(m.predict(X_full_tr[c_f]) * exp_tr, 1e-3)
    test_comp["pooled_lgbm"] = np.maximum(m.predict(X_full_te[c_f]) * exp_te, 1e-3) * (y.sum() / ptr.sum())

    w_a, w_x, w_p = best[0], best[1], best[2]
    w_d = round(1 - w_a - w_x - w_p, 2)
    pred = (w_a * test_comp["offset_lgbm"] + w_x * test_comp["offset_xgb"]
            + w_p * test_comp["pooled_lgbm"] + w_d * test_comp["dev_zero"])

    fwd, _ = estimate_forward_index(em_tr, em_te)
    em_te_c = em_te.copy()
    em_te_c["cohort"] = np.where(em_te_c["new_business"].fillna(0).astype(float) > 0.5, "nb", "ren")
    f_map = fwd.set_index(["line", "cohort"])["f_final"]
    pred = pred * f_map.loc[list(zip(em_te_c["line"], em_te_c["cohort"]))].to_numpy(float)

    sample = pd.read_parquet(EM_DIR / "sample_submission.parquet")
    sub = pd.DataFrame({"row_id": em_te["row_id"], "prediction": np.maximum(pred, FLOOR)})
    sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
    assert len(sub) == 209_063 and not sub["prediction"].isna().any()
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    out = OUT_SUB / "emerging-exp-020.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name} (blend {w_a}/{w_x}/{w_p}/{w_d}) mean={sub.prediction.mean():.2f}", flush=True)
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
