#!/usr/bin/env python
"""Part 2 mean-quality candidate: per-LOB offset/prior blend (D-086, arm C).

Primary change vs the record (D-067): the record submits offset-only
(w_off = 1.0 per LOB — "the offset layer stands alone"). D-067's OOF grid at
the OLD config (lv=3/min_leaf=600) found a per-LOB blend optimum of motor 1.0,
buildings 1.0, contents 0.5 (global OOF 0.14050 vs 0.14045 — +0.00005,
single-seed, and exp-004's board test did not rank). Never gated at the
record config (D-075 lv=4/min_leaf=1200), whose heavier shrinkage could make
the contents prior blend useful again.

Arms: record (1.0, 1.0, 1.0) vs D-067-fixed (1.0, 1.0, 0.5) — the variant
weights are FIXED a priori from D-067, not re-selected here.

Pre-registered gate (written before results, 2026-10-29):
  * paired-seed (0/1/2) 5-fold GroupKFold(customer_id) OOF at record config,
    identical folds/seeds across arms; contract check = 0.14074 at seed 0
    (tol 1e-5) on the record arm;
  * variant passes only if mean paired delta >= +0.0002 AND 3/3 seeds positive
    (winner's-curse control, 2 arms);
  * PASS -> build the submission and request ONE Part 2 slot; FAIL -> no CSV,
    axis closes. This script stops before any CSV.

Usage:
  python scripts/run_emerging_wblend.py
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
OUT_MET = ROOT / "outputs" / "metrics"
FLOOR = 1e-3
SEEDS = (0, 1, 2)
GATE = 0.0002
NAME = "emerging-wblend"
RECORD_OOF_SEED0 = 0.14074
CONTRACT_TOL = 1e-5
W_OFF_RECORD = {"motor": 1.0, "buildings": 1.0, "contents": 1.0}
W_OFF_D067 = {"motor": 1.0, "buildings": 1.0, "contents": 0.5}


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def main() -> int:
    t0 = time.time()
    em_tr = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")
    (X_ad_tr, _), cols_ad, cats_ad = base.prep_matrices(
        [em_tr, pd.read_parquet(CACHE / "em_te_with_prior.parquet")],
        base.ADAPT_NUM, base.ADAPT_CAT)

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    pr = (0.6 * em_tr["dev_prior_rate"].to_numpy(float)
          + 0.4 * em_tr["dev_prior_rate_nowp"].to_numpy(float))

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(em_tr, y, groups))

    # OOF pieces: offset-only and fold-calibrated prior-alone, per seed
    oof_off = {s: np.zeros(len(y)) for s in SEEDS}
    oof_prior = {s: np.zeros(len(y)) for s in SEEDS}
    for seed in SEEDS:
        for tri, vai in folds:
            for ln in base.LINES:
                m_tr, m_va = tri[lines[tri] == ln], vai[lines[vai] == ln]
                w = np.clip(exp_tr[m_tr], 1e-3, None)
                scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
                pr_cal = pr[m_tr] * scale
                oof_prior[seed][m_va] = pr[m_va] * scale * exp_tr[m_va]
                f = fit_lgbm_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad,
                                    y[m_tr], w, pr_cal, ln, seed)
                r_tr = pr_cal * f(X_ad_tr.iloc[m_tr])
                r_va = pr[m_va] * scale * f(X_ad_tr.iloc[m_va])
                oof_off[seed][m_va] = r_va * exp_tr[m_va] * (
                    y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())
        print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    def blend_oof(w_off):
        out = {}
        for s in SEEDS:
            p = oof_off[s].copy()
            for ln in base.LINES:
                m = lines == ln
                p[m] = w_off[ln] * oof_off[s][m] + (1.0 - w_off[ln]) * oof_prior[s][m]
            out[s] = p
        return out

    arms = {"record": blend_oof(W_OFF_RECORD), "d067_contents_shrink": blend_oof(W_OFF_D067)}
    results = {n: {s: d2(y, arms[n][s]) for s in SEEDS} for n in arms}
    for n in results:
        note = ""
        if n == "record":
            drift = abs(results[n][0] - RECORD_OOF_SEED0)
            note = f"  (contract: |{results[n][0]:.5f} - {RECORD_OOF_SEED0}| = {drift:.2e})"
        print(f"{n}: " + "  ".join(f"seed {s} {results[n][s]:+.5f}" for s in SEEDS) + note,
              flush=True)

    rec = results["record"]
    drift = abs(rec[0] - RECORD_OOF_SEED0)
    contract_ok = drift < CONTRACT_TOL
    deltas = [results["d067_contents_shrink"][s] - rec[s] for s in SEEDS]
    mean_delta = float(np.mean(deltas))
    passed = contract_ok and mean_delta >= GATE and all(d > 0 for d in deltas)
    print(f"d067_contents_shrink: deltas {[f'{d:+.5f}' for d in deltas]} "
          f"mean {mean_delta:+.5f} -> {'PASS' if passed else 'FAIL'}", flush=True)

    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / f"{NAME}.json").write_text(json.dumps({
        "name": NAME,
        "primary_change": "per-LOB offset/prior blend w_off (1.0,1.0,0.5) from "
                          "D-067's old-config grid vs record offset-only",
        "w_off_variant": W_OFF_D067,
        "seeds": list(SEEDS),
        "oof": {n: {str(s): results[n][s] for s in SEEDS} for n in results},
        "deltas": deltas,
        "mean_paired_delta": mean_delta,
        "gate": GATE,
        "gate_rule": "mean paired delta >= +0.0002 AND 3/3 seeds positive (2 arms)",
        "contract_ok": bool(contract_ok),
        "record_contract_drift_seed0": drift,
        "pass": bool(passed),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    if not passed:
        print("gate FAIL — no CSV written, axis closes", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
