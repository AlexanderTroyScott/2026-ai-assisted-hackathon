#!/usr/bin/env python
"""Part 2 motor per-peril trial — decompose motor cost by peril and model each
peril separately (D-074 candidate).

Motivation: motor is the weakest LOB (OOF D2 0.071) and its 5 perils have
very different shapes — bodily_injury is 6% of claims / 35% of cost / 58% open,
glass_hail is 19% of claims / 7% of cost and growing, theft is rare and severe.
A single Tweedie on the sum cannot separate their feature drivers.

Design (pre-registered, one config):
  * labels: per-peril developed cost, same EM_LDF convention as the pipeline
    (sum over perils == y_ult, asserted).
  * model per peril: ultra-shallow offset Tweedie LGBM (D-067 machinery),
    config adapted to sparse cells: n_est=25, lr=0.02, lv=3, min_leaf=200,
    l2=100.  Offset = developed prior calibrated to the peril's train-fold
    total.  Fold-level recalibration is train-fitted (honest).
  * buildings/contents: the standard D-067 offset layer, unchanged.
  * same 5-Fold GroupKFold(customer_id) as run_emerging.py; the baseline
    motor single-model OOF is computed in the same loop for a like-for-like
    comparison.

Gate: a candidate CSV (with the D-069 forward index at scoring) is written
only if total OOF D2 beats the offset-only baseline (0.14045) by > 0.0003
(D-025 noise threshold).  Submission still requires explicit human
confirmation.

Usage:
  python scripts/run_emerging_peril.py
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

from src.emerging import EM_DIR, estimate_forward_index  # noqa: E402

CACHE = Path("/tmp/opencode/part2")
OUT_SUB = ROOT / "outputs" / "submissions"
OUT_MET = ROOT / "outputs" / "metrics"
FLOOR = 1e-3
BASELINE = 0.14045
LOB_N_EST = {"motor": 35, "buildings": 30, "contents": 20}


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def fit_offset(X, cols, cats, y, w, pr_cal, n_est, min_leaf):
    """Ultra-shallow offset Tweedie LGBM; returns rate predictor (D-067 style)."""
    c = [cc for cc in cols if X[cc].nunique(dropna=False) > 1]
    m = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=n_est, learning_rate=0.02, num_leaves=3,
        min_data_in_leaf=min_leaf, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=1, lambda_l2=100.0, verbose=-1, random_state=0,
    )
    m.fit(X[c], y / w, sample_weight=w, init_score=np.log(np.clip(pr_cal, 1e-3, None)),
          categorical_feature=[x for x in cats if x in c])
    return lambda Z: np.exp(m.predict(Z[c], raw_score=True))


def peril_targets(em_tr):
    """Per-peril developed row costs for motor.  Sum over perils == y_ult."""
    claims = pd.read_parquet(EM_DIR / "claims.parquet")
    dedup_keys = ["policy_id", "line", "peril", "loss_date", "report_date", "status",
                  "paid_to_date", "incurred_to_date", "recoveries"]
    claims = claims.drop_duplicates(subset=[k for k in dedup_keys if k in claims.columns], keep="first")
    claims = claims[claims["status"] != "repudiated"].copy()
    claims["loss"] = pd.to_datetime(claims["loss_date"])
    claims["cost"] = np.maximum(0.0, claims["incurred_to_date"] - claims["recoveries"])

    m = em_tr[em_tr["line"] == "motor"][["row_id", "policy_id", "ps", "pe", "acc_year", "y_ult"]]
    m_t = m.merge(claims[claims["line"] == "motor"], on="policy_id", how="inner")
    m_t = m_t[(m_t["loss"] >= m_t["ps"]) & (m_t["loss"] <= m_t["pe"])]
    dup = m_t.groupby("claim_id").size()
    assert (dup <= 1).all(), "claim matched more than one row"

    piv = m_t.groupby(["row_id", "peril"])["cost"].sum().unstack(fill_value=0.0)
    piv = piv.reindex(m["row_id"]).fillna(0.0)
    # Develop each peril pro-rata to the row's own y_ult: the LDF is per
    # (LOB, acc_year), so it applies uniformly across a row's perils and
    # allocating the developed total by in-period cost share is exactly the
    # per-peril development.  This also absorbs a rounding provenance gap —
    # the cached y_ult was built with motor 2024 = 1.2518*1.0301 (1.289479)
    # while EM_LDF now stores round(..., 4) = 1.2895 (~1.6bp) — and keeps
    # sum(per-peril) == y_ult exact.  The assert below still guards against
    # attribution misses (y_ult > 0 with no attributed claim cost).
    y_ult = m["y_ult"].to_numpy(float)
    raw = piv.to_numpy(float)
    tot = raw.sum(axis=1, keepdims=True)
    share = np.divide(raw, tot, out=np.zeros_like(raw), where=tot > 0)
    y_p = share * y_ult[:, None]
    assert np.allclose(y_p.sum(axis=1), y_ult), "per-peril labels do not sum to y_ult"
    return piv.columns.tolist(), pd.DataFrame(y_p, index=m["row_id"], columns=piv.columns), m.index.to_numpy()


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet").reset_index(drop=True)
    em_te = pd.read_parquet(CACHE / "em_te_with_prior.parquet").reset_index(drop=True)
    for d in (em_tr, em_te):
        d["prior_rate_blend"] = 0.6 * d["dev_prior_rate"] + 0.4 * d["dev_prior_rate_nowp"]

    (X_tr, X_te), cols, cats = base.prep_matrices([em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT)
    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    exp_te = em_te["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = em_tr["prior_rate_blend"].to_numpy(float)
    pr_te = em_te["prior_rate_blend"].to_numpy(float)

    perils, y_p_frame, m_idx = peril_targets(em_tr)
    y_p = y_p_frame.to_numpy(float)
    print(f"motor perils: {perils}", flush=True)
    print("peril developed cost shares:",
          {p: round(float(y_p[:, i].sum() / y_p.sum()), 3) for i, p in enumerate(perils)}, flush=True)

    n_tr = len(em_tr)
    oof_base = np.zeros(n_tr)      # standard single-model motor baseline
    oof_peril = np.zeros(n_tr)     # per-peril sum
    oof_full = np.zeros(n_tr)      # full 3-LOB: baseline for non-motor + per-peril motor

    gkf = GroupKFold(n_splits=5)
    for tri, vai in gkf.split(em_tr, y, groups):
        m_tr = tri[lines[tri] == "motor"]
        m_va = vai[lines[vai] == "motor"]
        w = np.clip(exp_tr[m_tr], 1e-3, None)
        pr_cal = pr[m_tr] * (y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum())

        # ---- baseline: single total-cost offset model (D-067) ----
        f = fit_offset(X_tr.iloc[m_tr], cols, cats, y[m_tr], w, pr_cal, 35, 600)
        r_tr = pr_cal * f(X_tr.iloc[m_tr])
        r_va = pr[m_va] * (y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()) * f(X_tr.iloc[m_va])
        oof_base[m_va] = r_va * exp_tr[m_va] * (y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())

        # ---- per-peril offset models ----
        acc = np.zeros(len(m_va))
        pos = {idx: j for j, idx in enumerate(m_idx)}  # row position in y_p
        tr_p = np.array([pos[i] for i in m_tr])
        va_p = np.array([pos[i] for i in m_va])
        for j, _p in enumerate(perils):
            ypp = y_p[tr_p, j]
            scale_p = ypp.sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
            pr_cal_p = pr[m_tr] * scale_p
            fp = fit_offset(X_tr.iloc[m_tr], cols, cats, ypp, w, pr_cal_p, 25, 200)
            r_trp = pr_cal_p * fp(X_tr.iloc[m_tr])
            r_vap = pr[m_va] * scale_p * fp(X_tr.iloc[m_va])
            cal = ypp.sum() / max((r_trp * exp_tr[m_tr]).sum(), 1e-9)
            acc += r_vap * exp_tr[m_va] * cal
        oof_peril[m_va] = acc

        # ---- full-LOB: standard for buildings/contents ----
        for ln in ("buildings", "contents"):
            b_tr = tri[lines[tri] == ln]
            b_va = vai[lines[vai] == ln]
            wb = np.clip(exp_tr[b_tr], 1e-3, None)
            scale = y[b_tr].sum() / (pr[b_tr] * exp_tr[b_tr]).sum()
            pr_cal_b = pr[b_tr] * scale
            fb = fit_offset(X_tr.iloc[b_tr], cols, cats, y[b_tr], wb, pr_cal_b,
                            LOB_N_EST[ln], 600)
            r_trb = pr_cal_b * fb(X_tr.iloc[b_tr])
            r_vab = pr[b_va] * scale * fb(X_tr.iloc[b_va])
            oof_full[b_va] = r_vab * exp_tr[b_va] * (y[b_tr].sum() / (r_trb * exp_tr[b_tr]).sum())
        oof_full[m_va] = acc

        print(f"fold done  motor base={d2(y[m_va], oof_base[m_va]):+.4f} "
              f"peril={d2(y[m_va], oof_peril[m_va]):+.4f}", flush=True)

    print(f"\n5-Fold CV  motor baseline   D2 = {d2(y[lines == 'motor'], oof_base[lines == 'motor']):+.5f}")
    print(f"5-Fold CV  motor per-peril  D2 = {d2(y[lines == 'motor'], oof_peril[lines == 'motor']):+.5f}")
    print(f"5-Fold CV  total baseline   D2 = {d2(y, np.where(lines == 'motor', oof_base, oof_full)):+.5f}")
    print(f"5-Fold CV  total per-peril  D2 = {d2(y, oof_full):+.5f}")

    total_base = d2(y, np.where(lines == "motor", oof_base, oof_full))
    total_peril = d2(y, oof_full)
    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / "emerging-peril.json").write_text(json.dumps({
        "perils": perils,
        "peril_cost_share": {p: float(y_p[:, i].sum() / y_p.sum()) for i, p in enumerate(perils)},
        "motor_d2_baseline": d2(y[lines == "motor"], oof_base[lines == "motor"]),
        "motor_d2_peril": d2(y[lines == "motor"], oof_peril[lines == "motor"]),
        "total_d2_baseline": total_base,
        "total_d2_peril": total_peril,
        "gate": total_peril > BASELINE + 0.0003,
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    if not (total_peril > BASELINE + 0.0003):
        print("no OOF gain above +0.0003 — no CSV written", flush=True)
        return 0

    # ---- full refit -> test predictions (per-peril motor, standard rest) ----
    p_te = np.zeros(len(em_te))
    m_tr_all = (em_tr["line"] == "motor").to_numpy()
    m_te_all = (em_te["line"] == "motor").to_numpy()
    w = np.clip(exp_tr[m_tr_all], 1e-3, None)
    acc = np.zeros(int(m_te_all.sum()))
    pos = {idx: j for j, idx in enumerate(m_idx)}
    tr_p = np.array([pos[i] for i in np.where(m_tr_all)[0]])
    for j, _p in enumerate(perils):
        ypp = y_p[tr_p, j]
        scale_p = ypp.sum() / (pr[m_tr_all] * exp_tr[m_tr_all]).sum()
        pr_cal_p = pr[m_tr_all] * scale_p
        fp = fit_offset(X_tr.loc[m_tr_all], cols, cats, ypp, w, pr_cal_p, 25, 200)
        r_trp = pr_cal_p * fp(X_tr.loc[m_tr_all])
        r_tep = pr_te[m_te_all] * scale_p * fp(X_te.loc[m_te_all])
        acc += r_tep * exp_te[m_te_all] * (ypp.sum() / (r_trp * exp_tr[m_tr_all]).sum())
    p_te[m_te_all] = acc

    for ln in ("buildings", "contents"):
        b_tr = (em_tr["line"] == ln).to_numpy()
        b_te = (em_te["line"] == ln).to_numpy()
        wb = np.clip(exp_tr[b_tr], 1e-3, None)
        scale = y[b_tr].sum() / (pr[b_tr] * exp_tr[b_tr]).sum()
        pr_cal_b = pr[b_tr] * scale
        fb = fit_offset(X_tr.loc[b_tr], cols, cats, y[b_tr], wb, pr_cal_b,
                        LOB_N_EST[ln], 600)
        r_trb = pr_cal_b * fb(X_tr.loc[b_tr])
        r_teb = pr_te[b_te] * scale * fb(X_te.loc[b_te])
        p_te[b_te] = r_teb * exp_te[b_te] * (y[b_tr].sum() / (r_trb * exp_tr[b_tr]).sum())

    fwd, _ = estimate_forward_index(em_tr, em_te)
    em_te_c = em_te.copy()
    em_te_c["cohort"] = np.where(em_te_c["new_business"].fillna(0).astype(float) > 0.5, "nb", "ren")
    f_map = fwd.set_index(["line", "cohort"])["f_final"]
    p_te = p_te * f_map.loc[list(zip(em_te_c["line"], em_te_c["cohort"]))].to_numpy(float)

    sample = pd.read_parquet(EM_DIR / "sample_submission.parquet")
    sub = pd.DataFrame({"row_id": em_te["row_id"], "prediction": np.maximum(p_te, FLOOR)})
    sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
    assert len(sub) == 209_063 and not sub["prediction"].isna().any()
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    out = OUT_SUB / "emerging-exp-023.csv"
    sub.to_csv(out, index=False)
    print(f"wrote {out.name} rows={len(sub)} mean={sub.prediction.mean():.2f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
