"""Feature engineering for the IAA hackathon.

Every function here is leakage-safe under AGENTS.md rule 6: anything derived
from claims for a row uses only source rows whose claims all have loss dates
before that row's `period_start`. For historical relativities the strict rule is
`source.pe < row.ps` — since every claim attributed to a source row has
`loss <= source.pe`, all contributing loss dates precede the target row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Exposure units of credibility for smoothing a level's historical rate toward
# the line's global rate (Bühlmann-style pseudo-exposure).
PRIOR_EXPOSURE = 200.0
MIN_LEVEL_EXPOSURE = 50.0


def historical_relativities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    y: str = "y_ult",
    expo: str = "exposure",
    freq: str = "n_claims",
    prior: str = "strict",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `<col>_hist_rate` / `<col>_hist_freq` historical relativities.

    For each training row, level statistics accumulate over source rows with
    `pe < ps` (strictly lagged). For each test row (ps >= 2025-01-01) the full
    training history (pe <= 2024-12-30) qualifies. Rates are smoothed toward a
    credibility anchor with `PRIOR_EXPOSURE` units of credibility; levels with
    little history fall back to the anchor.

    D-084: the credibility anchor itself must be strictly lagged too, or the
    holdout's own outcomes leak into every thin-cell feature through the prior
    level (the D-065 mild-leak finding). `prior` selects the anchor:

    - "strict" (default): for every row the anchor is the line rate over source
      rows with `pe < ps`. Rows with no antecedent experience at all fall back
      to the **initial prior** — the full-period line rate, labelled as such.
      On this book those rows are all first-year (AY2021-01-01) inception rows;
      holdout and test rows always have antecedent history and never draw on
      the initial prior.
    - "legacy": the pre-D-084 anchor, the full-period line rate for every row.
      Retained only for byte-exact reproduction of pre-D-084 runs.
    """
    if prior not in ("strict", "legacy"):
        raise ValueError(f"prior must be 'strict' or 'legacy', got {prior!r}")
    tr = train.copy()
    te = test.copy()
    for frame, name, req in (
        (tr, "train", {"row_id", "line", "ps", "pe", y, expo, freq}),
        (te, "test", {"row_id", "line", "ps"}),
    ):
        missing = req - set(frame.columns)
        if missing:
            raise ValueError(f"{name} frame missing {sorted(missing)}")
    cols = [c for c in cols if c in tr.columns and c in te.columns]
    if not cols:
        return tr, te

    for c in cols:
        tr[c] = tr[c].astype(str).fillna("other")
        te[c] = te[c].astype(str).fillna("other")

    gen = [f"{c}_hist_{k}" for c in cols for k in ("rate", "freq")]
    new_tr = {g: np.full(len(tr), np.nan) for g in gen}
    new_te = {g: np.full(len(te), np.nan) for g in gen}

    # Credibility anchor per row (D-084). NaN outside a line's rows.
    prior_lr_tr = np.full(len(tr), np.nan)
    prior_lf_tr = np.full(len(tr), np.nan)
    prior_lr_te = np.full(len(te), np.nan)
    prior_lf_te = np.full(len(te), np.nan)

    for line in sorted(tr["line"].unique()):
        tr_m = (tr["line"] == line).to_numpy()
        te_m = (te["line"] == line).to_numpy()
        if not tr_m.any():
            continue
        # Initial prior (experience-period start): the full-period line rate.
        lr_full = tr.loc[tr_m, y].sum() / tr.loc[tr_m, expo].sum()
        lf_full = tr.loc[tr_m, freq].sum() / tr.loc[tr_m, expo].sum()
        if prior == "legacy":
            prior_lr_tr[tr_m] = lr_full
            prior_lf_tr[tr_m] = lf_full
            prior_lr_te[te_m] = lr_full
            prior_lf_te[te_m] = lf_full
        else:
            # Strictly lagged anchor: cumulative line history over source rows
            # with pe < ps (same boundary rule as the level statistics).
            order = np.argsort(tr.loc[tr_m, "pe"].to_numpy(), kind="stable")
            cy_l = np.cumsum(tr.loc[tr_m, y].to_numpy(float)[order])
            ce_l = np.cumsum(tr.loc[tr_m, expo].to_numpy(float)[order])
            cf_l = np.cumsum(tr.loc[tr_m, freq].to_numpy(float)[order])
            pe_l = tr.loc[tr_m, "pe"].to_numpy()[order]
            for out_lr, out_lf, frame, mask in (
                (prior_lr_tr, prior_lf_tr, tr, tr_m),
                (prior_lr_te, prior_lf_te, te, te_m),
            ):
                if not mask.any():
                    continue
                q = frame.loc[mask, "ps"].to_numpy()
                i = np.clip(np.searchsorted(pe_l, q, side="left"), 0, len(pe_l))
                has = i > 0
                idx = np.maximum(i - 1, 0)
                pce = np.where(has, ce_l[idx], 0.0)
                pcy = np.where(has, cy_l[idx], 0.0)
                pcf = np.where(has, cf_l[idx], 0.0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rate_i = np.where(has, pcy / pce, lr_full)
                    freq_i = np.where(has, pcf / pce, lf_full)
                # no antecedent experience at all -> initial prior
                rate_i = np.where(np.isfinite(rate_i), rate_i, lr_full)
                freq_i = np.where(np.isfinite(freq_i), freq_i, lf_full)
                out_lr[mask] = rate_i
                out_lf[mask] = freq_i

        for c in cols:
            tr_key = tr[c].to_numpy()
            te_key = te[c].to_numpy()
            levels = pd.unique(tr_key[tr_m])
            for level in levels:
                src_m = tr_m & (tr_key == level)
                gpe = tr.loc[src_m, "pe"].to_numpy()
                order = np.argsort(gpe, kind="stable")
                gpe = gpe[order]
                cy = np.cumsum(tr.loc[src_m, y].to_numpy(float)[order])
                ce = np.cumsum(tr.loc[src_m, expo].to_numpy(float)[order])
                cf = np.cumsum(tr.loc[src_m, freq].to_numpy(float)[order])

                for out, mask, frame, plr_all, plf_all in (
                    (new_tr, tr_m & (tr_key == level), tr, prior_lr_tr, prior_lf_tr),
                    (new_te, te_m & (te_key == level), te, prior_lr_te, prior_lf_te),
                ):
                    if not mask.any():
                        continue
                    q = frame.loc[mask, "ps"].to_numpy()
                    i = np.clip(np.searchsorted(gpe, q, side="left"), 0, len(gpe))
                    has = i > 0
                    idx = np.maximum(i - 1, 0)
                    cy_q = np.where(has, cy[idx], 0.0)
                    ce_q = np.where(has, ce[idx], 0.0)
                    cf_q = np.where(has, cf[idx], 0.0)
                    plr = plr_all[mask]
                    plf = plf_all[mask]
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = (cy_q + PRIOR_EXPOSURE * plr) / (ce_q + PRIOR_EXPOSURE)
                        f = (cf_q + PRIOR_EXPOSURE * plf) / (ce_q + PRIOR_EXPOSURE)
                    small = ce_q < MIN_LEVEL_EXPOSURE
                    r[small] = plr[small]
                    f[small] = plf[small]
                    out[f"{c}_hist_rate"][mask] = r
                    out[f"{c}_hist_freq"][mask] = f

            for out, mask, plr_all, plf_all in (
                (new_tr, tr_m, prior_lr_tr, prior_lf_tr),
                (new_te, te_m, prior_lr_te, prior_lf_te),
            ):
                for g, plr in ((f"{c}_hist_rate", plr_all), (f"{c}_hist_freq", plf_all)):
                    arr = out[g]
                    sel = mask & np.isnan(arr)
                    arr[sel] = plr[sel]

    for g in gen:
        tr[g] = new_tr[g]
        te[g] = new_te[g]
    return tr, te
