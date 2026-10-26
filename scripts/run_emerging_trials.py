#!/usr/bin/env python
"""Part 2 time-boxed trial sprint — evaluate blend variants offline.

One round of 5-Fold fits produces out-of-fold component predictions (offset
adaptation, calibrated developed prior, emerging-only LGBM); one round of full
refits produces the matching test predictions. Every blend variant is then just
a linear combination evaluated for free on the OOF components, so a whole grid
of submission candidates costs one training round.

Writes candidate CSVs to outputs/submissions/emerging-exp-0NN.csv and the
evaluation grid to outputs/metrics/emerging-trials.json.

Usage:
  python scripts/run_emerging_trials.py
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

import run_emerging as base  # noqa: E402

from src.emerging import EM_DIR, LINES  # noqa: E402

OUT_SUB = ROOT / "outputs" / "submissions"
OUT_MET = ROOT / "outputs" / "metrics"
CACHE = Path("/tmp/opencode/part2")
FLOOR = 1e-3
LOB_N_EST = {"motor": 35, "buildings": 30, "contents": 20}


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    em_te = pd.read_parquet(CACHE / "em_te_with_prior.parquet")
    for d in (em_tr, em_te):
        d["prior_rate_blend"] = 0.6 * d["dev_prior_rate"] + 0.4 * d["dev_prior_rate_nowp"]

    (X_ad_tr, X_ad_te), cols_ad, cats_ad = base.prep_matrices(
        [em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT
    )
    (X_full_tr, X_full_te), cols_full, cats_full = base.prep_matrices(
        [em_tr, em_te],
        base.SHARED_NUM + ["written_premium", "wp_rel", "is_self_employed"],
        base.SHARED_CAT + ["region", "gender"],
    )

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    exp_te = em_te["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = em_tr["prior_rate_blend"].to_numpy(float)
    pr_te = em_te["prior_rate_blend"].to_numpy(float)

    # ---------------- OOF components (one 5-Fold round) ----------------
    oof = {k: np.zeros(len(em_tr)) for k in ("offset", "dev_zero", "em_only")}
    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(em_tr, y, groups):
        for ln in LINES:
            m_tr = tri[lines[tri] == ln]
            m_va = vai[lines[vai] == ln]
            w = np.clip(exp_tr[m_tr], 1e-3, None)

            scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            pr_cal = pr[m_tr] * scale
            oof["dev_zero"][m_va] = pr[m_va] * scale * exp_tr[m_va]

            c_a = [c for c in cols_ad if X_ad_tr.iloc[m_tr][c].nunique(dropna=False) > 1]
            m_off = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=LOB_N_EST[ln], learning_rate=0.02, num_leaves=3,
                min_data_in_leaf=600, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=100.0, verbose=-1, random_state=0,
            )
            m_off.fit(
                X_ad_tr.iloc[m_tr][c_a], y[m_tr] / w, sample_weight=w,
                init_score=np.log(np.clip(pr_cal, 1e-3, None)),
                categorical_feature=[c for c in cats_ad if c in c_a],
            )
            rate_tr = pr_cal * np.exp(m_off.predict(X_ad_tr.iloc[m_tr][c_a], raw_score=True))
            rate_va = pr[m_va] * scale * np.exp(m_off.predict(X_ad_tr.iloc[m_va][c_a], raw_score=True))
            cost_tr = rate_tr * exp_tr[m_tr]
            oof["offset"][m_va] = rate_va * exp_tr[m_va] * (y[m_tr].sum() / cost_tr.sum())

            c_f = [c for c in cols_full if X_full_tr.iloc[m_tr][c].nunique(dropna=False) > 1]
            m_em = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=150, learning_rate=0.03, num_leaves=6,
                min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=10.0, verbose=-1, random_state=0,
            )
            m_em.fit(
                X_full_tr.iloc[m_tr][c_f], y[m_tr] / w, sample_weight=w,
                categorical_feature=[c for c in cats_full if c in c_f],
            )
            ptr = np.maximum(m_em.predict(X_full_tr.iloc[m_tr][c_f]) * exp_tr[m_tr], 1e-3)
            pva = np.maximum(m_em.predict(X_full_tr.iloc[m_va][c_f]) * exp_tr[m_va], 1e-3)
            oof["em_only"][m_va] = pva * (y[m_tr].sum() / ptr.sum())

    print(f"OOF components built in {time.time()-t0:.0f}s", flush=True)
    for k, v in oof.items():
        print(f"  OOF {k:9s} D2 = {d2(y, v):+.5f}", flush=True)

    # ---------------- offline blend grid on OOF ----------------
    grid = {}
    best_global = (0.0, -9.0)
    for w_off in np.round(np.arange(0.5, 1.001, 0.05), 2):
        p = w_off * oof["offset"] + (1 - w_off) * oof["dev_zero"]
        grid[f"blend_{w_off:.2f}"] = d2(y, p)
        if grid[f"blend_{w_off:.2f}"] > best_global[1]:
            best_global = (float(w_off), grid[f"blend_{w_off:.2f}"])

    best_lob = {}
    for ln in LINES:
        m = lines == ln
        bw, bs = 0.0, -9.0
        for w_off in np.round(np.arange(0.5, 1.001, 0.05), 2):
            p = w_off * oof["offset"][m] + (1 - w_off) * oof["dev_zero"][m]
            s = d2(y[m], p)
            if s > bs:
                bw, bs = float(w_off), s
        best_lob[ln] = (bw, bs)

    best3 = (0.0, 0.0, -9.0)
    for a in (0.6, 0.7, 0.8, 0.9):
        for b in (0.0, 0.05, 0.1, 0.15):
            p = a * oof["offset"] + b * oof["em_only"] + (1 - a - b) * oof["dev_zero"]
            s = d2(y, p)
            if s > best3[2]:
                best3 = (a, b, s)

    print("blend grid:", json.dumps({k: round(v, 5) for k, v in grid.items()}), flush=True)
    print(f"best global w_off={best_global[0]:.2f} D2={best_global[1]:+.5f}", flush=True)
    print("best per-LOB w_off:", {k: (v[0], round(v[1], 5)) for k, v in best_lob.items()}, flush=True)
    print(f"best 3-way a={best3[0]:.2f} b={best3[1]:.2f} D2={best3[2]:+.5f}", flush=True)

    # ---------------- full-refit test components (one round) ----------------
    test_comp = {k: np.zeros(len(em_te)) for k in ("offset", "dev_zero", "em_only")}
    for ln in LINES:
        m_tr = (em_tr["line"] == ln).to_numpy()
        m_te = (em_te["line"] == ln).to_numpy()
        w = np.clip(exp_tr[m_tr], 1e-3, None)
        scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
        test_comp["dev_zero"][m_te] = pr_te[m_te] * scale * exp_te[m_te]

        c_a = [c for c in cols_ad if X_ad_tr.loc[m_tr][c].nunique(dropna=False) > 1]
        preds = np.zeros(m_te.sum())
        for seed in (0, 1, 2):
            m_off = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=1.5,
                n_estimators=LOB_N_EST[ln], learning_rate=0.02, num_leaves=3,
                min_data_in_leaf=600, feature_fraction=0.8, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=100.0, verbose=-1, random_state=seed,
            )
            m_off.fit(
                X_ad_tr.loc[m_tr, c_a], y[m_tr] / w, sample_weight=w,
                init_score=np.log(np.clip(pr[m_tr] * scale, 1e-3, None)),
                categorical_feature=[c for c in cats_ad if c in c_a],
            )
            rate_tr = pr[m_tr] * scale * np.exp(m_off.predict(X_ad_tr.loc[m_tr, c_a], raw_score=True))
            rate_te = pr_te[m_te] * scale * np.exp(m_off.predict(X_ad_te.loc[m_te, c_a], raw_score=True))
            cost_tr = rate_tr * exp_tr[m_tr]
            preds += rate_te * exp_te[m_te] * (y[m_tr].sum() / cost_tr.sum()) / 3.0
        test_comp["offset"][m_te] = preds

        c_f = [c for c in cols_full if X_full_tr.loc[m_tr][c].nunique(dropna=False) > 1]
        m_em = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.5,
            n_estimators=150, learning_rate=0.03, num_leaves=6,
            min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=10.0, verbose=-1, random_state=0,
        )
        m_em.fit(
            X_full_tr.loc[m_tr, c_f], y[m_tr] / w, sample_weight=w,
            categorical_feature=[c for c in cats_full if c in c_f],
        )
        ptr = np.maximum(m_em.predict(X_full_tr.loc[m_tr, c_f]) * exp_tr[m_tr], 1e-3)
        pte = np.maximum(m_em.predict(X_full_te.loc[m_te, c_f]) * exp_te[m_te], 1e-3)
        test_comp["em_only"][m_te] = pte * (y[m_tr].sum() / ptr.sum())

    # ---------------- emit candidate CSVs ----------------
    sample = pd.read_parquet(EM_DIR / "sample_submission.parquet")

    def write_csv(tag: str, pred: np.ndarray, desc: str) -> None:
        sub = pd.DataFrame({"row_id": em_te["row_id"], "prediction": np.maximum(pred, FLOOR)})
        sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
        assert len(sub) == 209_063 and not sub["prediction"].isna().any(), tag
        sub.to_csv(OUT_SUB / f"{tag}.csv", index=False)
        print(f"wrote {tag}.csv  mean={sub.prediction.mean():.2f}  {desc}", flush=True)

    candidates = {}
    candidates["offset-only"] = test_comp["offset"]
    candidates[f"blend-{best_global[0]:.2f}"] = (
        best_global[0] * test_comp["offset"] + (1 - best_global[0]) * test_comp["dev_zero"]
    )
    p_lob = np.zeros(len(em_te))
    for ln in LINES:
        m = (em_te["line"] == ln).to_numpy()
        w = best_lob[ln][0]
        p_lob[m] = w * test_comp["offset"][m] + (1 - w) * test_comp["dev_zero"][m]
    candidates["per-lob-blend"] = p_lob
    candidates["three-way"] = (
        best3[0] * test_comp["offset"] + best3[1] * test_comp["em_only"]
        + (1 - best3[0] - best3[1]) * test_comp["dev_zero"]
    )

    oof_scores = {
        "offset-only": d2(y, oof["offset"]),
        f"blend-{best_global[0]:.2f}": best_global[1],
        "per-lob-blend": d2(y, np.concatenate([
            best_lob[ln][0] * oof["offset"][lines == ln] + (1 - best_lob[ln][0]) * oof["dev_zero"][lines == ln]
            for ln in LINES
        ])) if False else float(np.mean([best_lob[ln][1] for ln in LINES])),
        "three-way": best3[2],
    }

    OUT_SUB.mkdir(parents=True, exist_ok=True)
    OUT_MET.mkdir(parents=True, exist_ok=True)
    for i, (tag, pred) in enumerate(candidates.items(), start=2):
        write_csv(f"emerging-exp-{i:03d}", pred, f"OOF D2 ~ {oof_scores.get(tag, float('nan')):.5f}")

    (OUT_MET / "emerging-trials.json").write_text(json.dumps({
        "oof_components": {k: d2(y, v) for k, v in oof.items()},
        "blend_grid": grid,
        "best_global_w_off": best_global,
        "best_per_lob_w_off": {k: v for k, v in best_lob.items()},
        "best_three_way": best3,
        "candidates": {k: oof_scores.get(k) for k in candidates},
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))
    print(f"done in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
