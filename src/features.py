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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add `<col>_hist_rate` / `<col>_hist_freq` historical relativities.

    For each training row, level statistics accumulate over source rows with
    `pe < ps` (strictly lagged). For each test row (ps >= 2025-01-01) the full
    training history (pe <= 2024-12-30) qualifies. Rates are smoothed toward the
    line's global rate with `PRIOR_EXPOSURE` units of credibility; levels with
    little history fall back to the line global rate.
    """
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

    for line in sorted(tr["line"].unique()):
        tr_m = (tr["line"] == line).to_numpy()
        te_m = (te["line"] == line).to_numpy()
        if not tr_m.any():
            continue
        lr = tr.loc[tr_m, y].sum() / tr.loc[tr_m, expo].sum()
        lf = tr.loc[tr_m, freq].sum() / tr.loc[tr_m, expo].sum()

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

                for out, mask, frame in (
                    (new_tr, tr_m & (tr_key == level), tr),
                    (new_te, te_m & (te_key == level), te),
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
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = (cy_q + PRIOR_EXPOSURE * lr) / (ce_q + PRIOR_EXPOSURE)
                        f = (cf_q + PRIOR_EXPOSURE * lf) / (ce_q + PRIOR_EXPOSURE)
                    small = ce_q < MIN_LEVEL_EXPOSURE
                    r[small] = lr
                    f[small] = lf
                    out[f"{c}_hist_rate"][mask] = r
                    out[f"{c}_hist_freq"][mask] = f

            for out, mask in ((new_tr, tr_m), (new_te, te_m)):
                for g, fallback in ((f"{c}_hist_rate", lr), (f"{c}_hist_freq", lf)):
                    arr = out[g]
                    arr[mask & np.isnan(arr)] = fallback

    for g in gen:
        tr[g] = new_tr[g]
        te[g] = new_te[g]
    return tr, te
