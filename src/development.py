"""Development factors for label development (D-005, open question 3).

The accepted LOB x accident-year factors (docs/data_findings.md "IBNR /
development", frozen in `src/dataset.py`) are the **level anchor**. This module
redistributes that level within each (LOB, accident year) across perils in
proportion to their own age-to-ultimate factors, so long-tail perils
(bodily_injury, subsidence) are developed more than short-tail ones.

Level preservation is exact: for every (LOB, AY) the cost-weighted mean of the
per-peril factors equals the accepted LOB factor for that AY.

Per-peril factors are frozen from `scripts/build_development.py` (incurred
triangle, volume-weighted chain ladder, net of recoveries).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXTRACT_YEAR = 2024

# age k -> ultimate, k = EXTRACT_YEAR - accident_year (k >= len => fully mature)
#
# D-035: re-derived 2026-10-21 from `scripts/build_development.py` after fixing
# its triangle. The previous table was built with immature accident years left
# in the volume-weighted average, which dragged every factor toward 1.0 and
# under-developed long-tail perils most (motor BI 1.535 -> 1.708, subsidence
# 1.306 -> 1.464). Level stays anchored to the accepted D-005 LOB factors via
# `claim_factors` normalisation; only the per-peril split moved.
PERIL_LDF = {
    "motor": {
        "bodily_injury": {0: 1.7079, 1: 1.0693, 2: 1.0081, 3: 1.0},
        "third_party_prop": {0: 1.0749, 1: 1.0012, 2: 1.0, 3: 1.0},
        "own_damage": {0: 1.0388, 1: 1.001, 2: 1.0, 3: 1.0},
        "theft": {0: 1.0205, 1: 1.0012, 2: 1.0, 3: 1.0},
        "glass_hail": {0: 1.0367, 1: 1.0031, 2: 1.0, 3: 1.0},
    },
    "buildings": {
        "flood": {0: 1.0445, 1: 1.0024, 2: 1.0, 3: 1.0},
        "escape_of_water": {0: 1.0727, 1: 1.0027, 2: 1.0, 3: 1.0},
        "fire": {0: 1.0352, 1: 1.0, 2: 1.0, 3: 1.0},
        "subsidence": {0: 1.464, 1: 1.0076, 2: 1.0, 3: 1.0},
        "storm": {0: 1.0668, 1: 1.0027, 2: 1.0, 3: 1.0},
        "accidental_damage": {0: 1.0281, 1: 1.0001, 2: 1.0, 3: 1.0},
    },
    "contents": {
        "theft": {0: 1.0192, 1: 1.0017, 2: 1.0, 3: 1.0},
        "flood": {0: 1.0504, 1: 1.0, 2: 1.0, 3: 1.0},
        "escape_of_water": {0: 1.0722, 1: 1.0039, 2: 1.0, 3: 1.0},
        "accidental_damage": {0: 1.0381, 1: 1.0017, 2: 1.0, 3: 1.0},
    },
}


def peril_age_factor(line: str, peril: str, acc_year: int) -> float:
    tab = PERIL_LDF.get(line, {}).get(peril)
    if not tab:
        return 1.0
    age = EXTRACT_YEAR - acc_year
    return float(tab.get(age, 1.0))


def claim_factors(merged: pd.DataFrame, lob_ldf: dict) -> np.ndarray:
    """Per-claim development multipliers.

    `merged` needs columns: line, acc_year, peril, cost. `lob_ldf` is the
    accepted factor table {line: {acc_year: factor}} from `src.dataset.LDF`.

    factor(claim) = LOB_factor(line, AY) x raw(peril) / weighted_avg(raw | line, AY)
    so the cost-weighted mean factor within each (line, AY) equals the accepted
    LOB factor exactly.
    """
    line = merged["line"].to_numpy()
    ay = merged["acc_year"].to_numpy()
    peril = merged["peril"].to_numpy()
    cost = merged["cost"].to_numpy(float)

    raw = np.array([peril_age_factor(ln, pr, int(y)) for ln, pr, y in zip(line, peril, ay)])
    lob = np.ones(len(merged), dtype=float)
    for ln in lob_ldf:
        for y, f in lob_ldf[ln].items():
            lob[(line == ln) & (ay == y)] = f

    out = np.ones(len(merged), dtype=float)
    frame = pd.DataFrame({"line": line, "ay": ay, "raw": raw, "cost": cost})
    wavg = (
        frame.assign(wr=frame["raw"] * frame["cost"])
        .groupby(["line", "ay"])
        .agg(wr=("wr", "sum"), c=("cost", "sum"))
    )
    wavg["avg"] = np.where(wavg["c"] > 0, wavg["wr"] / wavg["c"].replace(0, np.nan), 1.0)
    key = list(zip(line, ay))
    avg = np.array([wavg["avg"].get(k, 1.0) for k in key], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = lob * raw / avg
    out[~np.isfinite(out)] = 1.0
    return out
