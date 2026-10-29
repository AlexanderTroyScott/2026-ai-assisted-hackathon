#!/usr/bin/env python
"""Part 2 mean-quality candidate: wp_rel cross-year normalization (audit idea).

Primary change vs the record (D-066/D-068): `wp_rel` is redefined from
within-(line, year) normalization to a FIXED per-line base shared across years
and splits:

    record:  wp_rel = written_premium / mean(line, acc_year)   # year drift erased
    variant: wp_rel = written_premium / base(line)             # base = exposure-
             weighted mean written_premium per line over TRAIN rows, applied
             unchanged to train AND test frames of the same market

Rationale (D-068's "express the trend properly"): the record normalization
strips the one feature that could carry the 2024->2025 premium level shift
(motor mean wp 608 -> 679, +11.6%), so the model cannot express the trend and
the board-tuned PROBE_FORWARD multipliers carry it instead. With a fixed base
the Stage-1 prior (developed 2021-24: premium ~+5%/yr vs claims ~+5.6%/yr)
and the Stage-2 offset can learn a level-sensitive wp_rel slope.

Known counter-evidence (measured 2026-10-29 before this run, kept honest):
in emerging train the cross-year drift does NOT track the rate trend (motor
rate +10.7% 23->24 while premium flat +0.01%; buildings rate -36.8% vs
premium +6.4%; contents rate -2.2% vs premium +5.7%). Any gain must therefore
come from the developed-prior slope, not from emerging's own years.

Arms (both compared to the record at identical folds/seeds):
  * A (full): wp_rel redefined for the prior AND the offset;
  * B (offset-only): cached record priors, redefined wp_rel in the offset only.

Pre-registered gate (written before results, 2026-10-29):
  * paired-seed (0/1/2) 5-fold GroupKFold(customer_id) OOF at record offset
    config (lv=4 / min_leaf=1200), identical folds/seeds across arms;
  * contract check: record arm reproduces 0.14074 at seed 0 (tol 1e-5);
  * an arm passes only if mean paired delta vs record >= +0.0002 AND 3/3 seeds
    positive (winner's-curse control, 2 arms);
  * PASS -> build a LEVEL-PRESERVING submission (per-(line,cohort) cell totals
    held at emerging-exp-025's, isolating the shape change) and request ONE
    Part 2 slot; FAIL -> no CSV, axis closes. This script stops before any CSV.

Usage:
  python scripts/run_emerging_wprel.py
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
NAME = "emerging-wprel"
RECORD_OOF_SEED0 = 0.14074
CONTRACT_TOL = 1e-5


def d2(y, p):
    return float(d2_tweedie_score(y, np.maximum(p, FLOOR), power=1.5))


def fixed_base_wp_rel(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Re-key wp_rel to a fixed per-line base derived from TRAIN rows only.

    frames: market -> {'train': df, 'test': df}. The base for each market is
    the exposure-weighted mean written_premium per line over its train rows;
    the same constant is applied to that market's train AND test rows.
    """
    out = {}
    for market, splits in frames.items():
        tr = splits["train"]
        base_wm = (tr.assign(wpx=tr["written_premium"] * tr["exposure"])
                    .groupby("line")["wpx"].sum()
                    / tr.groupby("line")["exposure"].sum())
        out[market] = {}
        for split, df in splits.items():
            d = df.copy()
            denom = d["line"].map(base_wm).to_numpy(float)
            d["wp_rel"] = d["written_premium"].to_numpy(float) / denom
            out[market][split] = d
    return out


def record_wp_rel(df: pd.DataFrame) -> pd.Series:
    """Record definition, reproduced byte-identically from build_market_frame:
    wp_rel = written_premium / exposure-weighted mean written_premium per
    (line, acc_year)."""
    wm = df.groupby(["line", "acc_year"]).apply(
        lambda g: (g["written_premium"] * g["exposure"]).sum() / g["exposure"].sum(),
        include_groups=False,
    )
    denom = pd.MultiIndex.from_frame(df[["line", "acc_year"]]).map(wm).to_numpy(float)
    return df["written_premium"] / denom


def oof_arm(prior_tr, X_ad_tr, cols_ad, cats_ad, y, exp_tr, lines, groups, folds):
    """Paired-seed record-config OOF for one arm. Returns {seed: oof_pred}."""
    oof = {s: np.zeros(len(y)) for s in SEEDS}
    pr = prior_tr
    for seed in SEEDS:
        for tri, vai in folds:
            for ln in base.LINES:
                m_tr, m_va = tri[lines[tri] == ln], vai[lines[vai] == ln]
                w = np.clip(exp_tr[m_tr], 1e-3, None)
                scale = y[m_tr].sum() / (pr[m_tr] * exp_tr[m_tr]).sum()
                pr_cal = pr[m_tr] * scale
                f = fit_lgbm_offset(X_ad_tr.iloc[m_tr], cols_ad, cats_ad,
                                    y[m_tr], w, pr_cal, ln, seed)
                r_tr = pr_cal * f(X_ad_tr.iloc[m_tr])
                r_va = pr[m_va] * scale * f(X_ad_tr.iloc[m_va])
                oof[seed][m_va] = r_va * exp_tr[m_va] * (
                    y[m_tr].sum() / (r_tr * exp_tr[m_tr]).sum())
    return oof


def main() -> int:
    t0 = time.time()

    dev_tr = pd.read_parquet(CACHE / "dev_tr.parquet")
    em_tr = pd.read_parquet(CACHE / "em_tr.parquet")
    em_te = pd.read_parquet(CACHE / "em_te.parquet")
    em_tr_prior = pd.read_parquet(CACHE / "em_tr_with_prior.parquet")

    # sanity: raw and with-prior caches must be the same rows in the same order
    assert (em_tr["row_id"].to_numpy() == em_tr_prior["row_id"].to_numpy()).all()
    assert (em_tr["y_ult"].to_numpy() == em_tr_prior["y_ult"].to_numpy()).all()
    # and the record wp_rel definition reproduces the cached column exactly
    rec_wp = record_wp_rel(em_tr)
    assert np.allclose(rec_wp.to_numpy(float), em_tr_prior["wp_rel"].to_numpy(float),
                       rtol=1e-12, atol=1e-12), "record wp_rel reproduction failed"

    # variant frames: fixed-base wp_rel (train-derived base per market/line)
    rebuilt = fixed_base_wp_rel({
        "dev": {"train": dev_tr},
        "em": {"train": em_tr, "test": em_te},
    })
    dev_tr_v = rebuilt["dev"]["train"]
    em_tr_v = rebuilt["em"]["train"]
    em_te_v = rebuilt["em"]["test"]

    y = em_tr["y_ult"].to_numpy(float)
    exp_tr = em_tr["exposure"].to_numpy(float)
    lines = em_tr["line"].to_numpy()
    groups = em_tr["customer_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(em_tr, y, groups))

    # ---- Arm A (full): rebuilt priors on rebuilt frames --------------------
    print("Arm A: fitting developed priors on fixed-base wp_rel frames...", flush=True)
    em_tr_a, em_te_a = base.fit_developed_priors(dev_tr_v, em_tr_v, em_te_v)
    em_tr_a = em_tr_a.copy()
    em_tr_a["prior_rate_blend"] = (0.6 * em_tr_a["dev_prior_rate"]
                                   + 0.4 * em_tr_a["dev_prior_rate_nowp"])
    (X_ad_a, _), cols_ad, cats_ad = base.prep_matrices(
        [em_tr_a, em_te_a], base.ADAPT_NUM, base.ADAPT_CAT)
    print(f"  Arm A priors done ({time.time() - t0:.0f}s)", flush=True)

    # ---- Arm B (offset-only): cached record priors, rebuilt wp_rel ---------
    em_tr_b = em_tr_v.copy()
    for c in ("dev_prior_rate", "dev_prior_rate_nowp"):
        em_tr_b[c] = em_tr_prior[c].to_numpy(float)
    em_tr_b["prior_rate_blend"] = (0.6 * em_tr_b["dev_prior_rate"]
                                   + 0.4 * em_tr_b["dev_prior_rate_nowp"])
    (X_ad_b, _), _, _ = base.prep_matrices([em_tr_b, em_te_v], base.ADAPT_NUM, base.ADAPT_CAT)

    # ---- Record arm: cached frames as-is (record wp_rel + cached priors) ---
    em_tr_r = em_tr_prior.copy()
    em_tr_r["prior_rate_blend"] = (0.6 * em_tr_r["dev_prior_rate"]
                                   + 0.4 * em_tr_r["dev_prior_rate_nowp"])
    em_te_r = pd.read_parquet(CACHE / "em_te_with_prior.parquet")
    (X_ad_r, _), _, _ = base.prep_matrices([em_tr_r, em_te_r], base.ADAPT_NUM, base.ADAPT_CAT)
    pr_rec = em_tr_r["prior_rate_blend"].to_numpy(float)

    arms = {
        "record": (pr_rec, X_ad_r),
        "A_full": (em_tr_a["prior_rate_blend"].to_numpy(float), X_ad_a),
        "B_offset_only": (em_tr_b["prior_rate_blend"].to_numpy(float), X_ad_b),
    }

    results = {}
    for name, (pr, X) in arms.items():
        t_arm = time.time()
        oof = oof_arm(pr, X, cols_ad, cats_ad, y, exp_tr, lines, groups, folds)
        results[name] = {s: d2(y, oof[s]) for s in SEEDS}
        print(f"{name}: " + "  ".join(f"seed {s} {results[name][s]:+.5f}" for s in SEEDS)
              + f"  ({time.time() - t_arm:.0f}s)", flush=True)

    rec = results["record"]
    drift = abs(rec[0] - RECORD_OOF_SEED0)
    contract_ok = drift < CONTRACT_TOL
    print(f"contract check: |{rec[0]:.5f} - {RECORD_OOF_SEED0}| = {drift:.2e} -> "
          f"{'ok' if contract_ok else 'FAILED'}", flush=True)

    verdicts = {}
    for name in ("A_full", "B_offset_only"):
        deltas = [results[name][s] - rec[s] for s in SEEDS]
        mean_delta = float(np.mean(deltas))
        passed = contract_ok and mean_delta >= GATE and all(d > 0 for d in deltas)
        verdicts[name] = {"deltas": deltas, "mean_paired_delta": mean_delta,
                          "all_seeds_positive": all(d > 0 for d in deltas),
                          "pass": bool(passed)}
        print(f"{name}: deltas {[f'{d:+.5f}' for d in deltas]} mean {mean_delta:+.5f} "
              f"-> {'PASS' if passed else 'FAIL'}", flush=True)

    passed_any = any(v["pass"] for v in verdicts.values())
    OUT_MET.mkdir(parents=True, exist_ok=True)
    (OUT_MET / f"{NAME}.json").write_text(json.dumps({
        "name": NAME,
        "primary_change": "wp_rel re-keyed from within-(line,year) to a fixed "
                          "per-line base (train-derived, shared train/test) — "
                          "cross-year premium level becomes visible to prior+offset",
        "arms": ["A_full (prior+offset)", "B_offset_only"],
        "seeds": list(SEEDS),
        "oof": {n: {str(s): results[n][s] for s in SEEDS} for n in results},
        "verdicts": verdicts,
        "gate": GATE,
        "gate_rule": "mean paired delta >= +0.0002 AND 3/3 seeds positive (2 arms)",
        "contract_ok": bool(contract_ok),
        "record_contract_drift_seed0": drift,
        "pass": bool(passed_any),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, default=float))

    if not passed_any:
        print("gate FAIL — no CSV written, axis closes", flush=True)
    else:
        print("GATE PASS — build the level-preserving submission next "
              "(cell totals held at emerging-exp-025's)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
