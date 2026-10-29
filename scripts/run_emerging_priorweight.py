#!/usr/bin/env python
"""Part 2 mean-quality candidate: prior leg blend weight (D-086 axis).

Primary change vs the record (D-066): the weight `w_wp` in

    prior_rate_blend = w_wp * dev_prior_rate + (1 - w_wp) * dev_prior_rate_nowp

is grid-searched instead of fixed at 0.6. D-067 only compared the endpoints
w_wp = 0.6 (record) and w_wp = 1.0 (pure wp leg, OOF 0.14007 — 0.00038 worse),
so the response curve interior was never located: the peak is bracketed but
not settled. Everything downstream is byte-identical to the record pipeline:
record cached prior legs, record offset config (lv=4 / min_leaf=1200), same
fold calibration, same 5-fold GroupKFold(customer_id).

Pre-registered gate (written before results, 2026-10-29):
  * arms: w_wp in {0.4, 0.5, 0.6 (record), 0.7, 0.8};
  * paired-seed (0/1/2) OOF at identical folds/seeds across arms;
  * contract check: record arm reproduces 0.14074 at seed 0 (tol 1e-5);
  * a grid arm passes only if mean paired delta vs the record arm >= +0.0002
    AND all 3 seeds positive (stricter than the single-candidate D-083 gate —
    winner's-curse control for the 4-way grid);
  * PASS -> build the submission via `run_emerging.py --wp-weight <w>` (record
    pipeline otherwise, PROBE_FORWARD scoring untouched) and request ONE Part 2
    slot; FAIL -> no CSV, axis closes. This script stops before any CSV.

Usage:
  python scripts/run_emerging_priorweight.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_emerging as base  # noqa: E402
from run_emerging_ensemble2 import fit_lgbm_offset  # noqa: E402

CACHE = Path("/tmp/opencode/part2")
OUT_SUB = ROOT / "outputs" / "submissions"
OUT_MET = ROOT / "outputs" / "metrics"
FLOOR = 1e-3
SEEDS = (0, 1, 2)
GATE = 0.0002
NAME = "emerging-priorweight"
RECORD_OOF_SEED0 = 0.14074
CONTRACT_TOL = 1e-5
W_GRID = (0.4, 0.5, 0.6, 0.7, 0.8)
W_RECORD = 0.6


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    em_te = pd.read_parquet(CACHE / "em_te_with_prior.parquet")

    (X_ad_tr, X_ad_te), cols_ad, cats_ad = base.prep_matrices(
        [em_tr, em_te], base.ADAPT_NUM, base.ADAPT_CAT)

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    leg_wp = em_tr["dev_prior_rate"].to_numpy(float)
    leg_nowp = em_tr["dev_prior_rate_nowp"].to_numpy(float)

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(em_tr, y, groups))

    priors = {w: w * leg_wp + (1.0 - w) * leg_nowp for w in W_GRID}
    oof = {w: {s: np.zeros(len(em_tr)) for s in SEEDS} for w in W_GRID}

    for seed in SEEDS:
        for tri, vai in folds:
            for ln in base.LINES:
                m_tr, m_va = tri[lines[tri] == ln], vai[lines[vai] == ln]
                w = np.clip(exp_tr[m_tr], 1e-3, None)
                for wgt in W_GRID:
                    pr = priors[wgt]
                    scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
                    pr_cal = pr[m_tr] * scale
                    f = fit_lgbm_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad,
                                        y[m_tr], w, pr_cal, ln, seed)
                    r_tr = pr_cal * f(X_ad_tr.iloc[m_tr])
                    r_va = pr[m_va] * scale * f(X_ad_tr.iloc[m_va])
                    oof[wgt][seed][m_va] = r_va * exp_tr[m_va] * (
                        y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    print("paired-seed OOF (5-fold GroupKFold(customer_id)):", flush=True)
    results = {}
    contract_ok = True
    for wgt in W_GRID:
        ds = {s: d2(y, oof[wgt][s]) for s in SEEDS}
        results[wgt] = ds
        note = ""
        if wgt == W_RECORD:
            drift = abs(ds[0] - RECORD_OOF_SEED0)
            note = f"  (contract: |{ds[0]:.5f} - {RECORD_OOF_SEED0}| = {drift:.2e})"
            contract_ok = drift < CONTRACT_TOL
        print(f"  w_wp={wgt}: " + "  ".join(f"seed {s} {ds[s]:+.5f}" for s in SEEDS) + note,
              flush=True)

    rec = results[W_RECORD]
    verdicts = {}
    for wgt in W_GRID:
        if wgt == W_RECORD:
            continue
        deltas = [results[wgt][s] - rec[s] for s in SEEDS]
        mean_delta = float(np.mean(deltas))
        passed = mean_delta >= GATE and all(d > 0 for d in deltas)
        verdicts[wgt] = {"deltas": deltas, "mean_paired_delta": mean_delta,
                         "all_seeds_positive": all(d > 0 for d in deltas),
                         "pass": bool(passed)}
        print(f"  w_wp={wgt}: deltas {[f'{d:+.5f}' for d in deltas]} "
              f"mean {mean_delta:+.5f} -> {'PASS' if passed else 'FAIL'}", flush=True)

    if not contract_ok:
        print("CONTRACT CHECK FAILED — comparison untrustworthy, aborting without CSV", flush=True)

    best_w = max(verdicts, key=lambda w: verdicts[w]["mean_paired_delta"]) if verdicts else None
    passed_any = contract_ok and any(v["pass"] for v in verdicts.values())

    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / f"{NAME}.json").write_text(json.dumps({
        "name": NAME,
        "primary_change": "prior leg blend weight w_wp grid {0.4,0.5,0.6,0.7,0.8} "
                          "(record 0.6; D-067 tested only 0.6 vs 1.0)",
        "w_grid": list(W_GRID),
        "seeds": list(SEEDS),
        "oof_by_weight": {str(w): {str(s): results[w][s] for s in SEEDS} for w in W_GRID},
        "verdicts": {str(w): v for w, v in verdicts.items()},
        "gate": GATE,
        "gate_rule": "mean paired delta >= +0.0002 AND 3/3 seeds positive "
                     "(winner's-curse control, 4-way grid)",
        "contract_ok": bool(contract_ok),
        "record_contract_drift_seed0": abs(rec[0] - RECORD_OOF_SEED0),
        "best_weight": best_w,
        "pass": bool(passed_any),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    if not passed_any:
        print("gate FAIL — no CSV written, axis closes", flush=True)
        return 0

    print(f"GATE PASS at w_wp={best_w} — building submission CSV", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
