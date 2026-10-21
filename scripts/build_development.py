#!/usr/bin/env python
"""Estimate age-to-ultimate development factors, LOB-level and per-peril.

Replicates the documented incurred-triangle method (docs/data_findings.md
"IBNR / development") at peril granularity so long-tail perils (bodily_injury,
subsidence) develop differently from short-tail ones (D-005, open question 3).

Method
------
Cumulative triangle C(AY, age) = sum of (incurred - recoveries) over claims with
accident year AY reported by end of year AY + age. Volume-weighted chain-ladder:

    f(age) = sum_AY C(AY, age+1) / sum_AY C(AY, age)     over AYs reaching BOTH

Age-0 -> ultimate for an accident year at age `start` is the product of
f(start .. max). Matches the documented convention (docs/data_findings.md
section 5): the last observed factor is near 1 and is itself the tail -- there
is no extra tail multiplier.

Valuation diagonal (D-035)
--------------------------
At extract year E, accident year AY has only reached age `E - AY`. Cells past
that diagonal are **unreached**, not zero: no 2024 claim has been reported at
age 1 yet. Reading them as zero increments and cumsumming manufactures a value
for the future, and because volume-weighting sums over AYs, every immature AY
then drags f(age) toward 1.0 and understates development. `build_cum` blanks
those cells; `triangle_factors` drops any AY that has not reached BOTH ages.

Usage:  python scripts/build_development.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data_raw"

EXTRACT_YEAR = 2024
ACCIDENT_YEARS = (2021, 2022, 2023, 2024)

# Guardrail targets from docs/data_findings.md section 5. These are the
# "age-0 -> ultimate" column and the "0->1 / 1->2 / 2->3" column respectively.
# Comparing the wrong pair is what let the D-033 table bug survive review.
DOCUMENTED_AGE_TO_AGE = {
    "motor": {0: 1.223, 1: 1.027, 2: 1.003},
    "buildings": {0: 1.089, 1: 1.003},
    "contents": {0: 1.040, 1: 1.004},
}
DOCUMENTED_AGE0_TO_ULT = {"motor": 1.26, "buildings": 1.09, "contents": 1.04}


def load_claims() -> pd.DataFrame:
    c = pd.read_parquet(DATA / "claims.parquet")
    c = c[c["status"] != "repudiated"].copy()
    c["loss"] = pd.to_datetime(c["loss_date"])
    c["rep"] = pd.to_datetime(c["report_date"])
    c["cost"] = np.maximum(0.0, c["incurred_to_date"] - c["recoveries"])
    c["ay"] = c["loss"].dt.year
    c["age"] = c["rep"].dt.year - c["loss"].dt.year
    # The documented triangle covers origin years 2021-2024. 2020 rows are
    # pre-book orphans (data_findings.md section 2) and are far too thin to
    # anchor a factor -- contents/flood has 2 such claims and they swing
    # f(1->2) to ~10x.
    return c[c["ay"].isin(ACCIDENT_YEARS)].copy()


def build_cum(c: pd.DataFrame) -> pd.DataFrame:
    """Cumulative reported triangle indexed by AY, columns = development age.

    Cells past the valuation diagonal are NaN (unreached), never zero. See the
    module docstring for why that distinction matters (D-035).
    """
    g = c.groupby(["ay", "age"])["cost"].sum().unstack()
    g = g[sorted(g.columns)]
    inc = g.fillna(0.0)                       # reached, but no report: true zero
    cum = inc.cumsum(axis=1)
    for age in cum.columns:                   # blank the unreached diagonal
        cum.loc[(cum.index + age) > EXTRACT_YEAR, age] = np.nan
    return cum.sort_index()


def triangle_factors(cum: pd.DataFrame) -> dict[int, float]:
    """Volume-weighted age-to-age factors, using only AYs that reached both ages."""
    ages = list(cum.columns)
    fac: dict[int, float] = {}
    for i in range(len(ages) - 1):
        a, b = ages[i], ages[i + 1]
        both = cum[[a, b]].dropna()           # D-035: immature AYs excluded
        if len(both) == 0 or both[a].sum() == 0:
            continue
        fac[a] = both[b].sum() / both[a].sum()
    return fac


def age0_to_ultimate(fac: dict[int, float], start_age: int) -> float:
    """Product of observed factors from `start_age` on.

    Matches the documented convention (docs/data_findings.md): the last observed
    factor is near 1 and serves as the tail — no extra tail multiplier.
    """
    if not fac:
        return 1.0
    prod = 1.0
    for a in sorted(fac):
        if a >= start_age:
            prod *= fac[a]
    return prod


def full_table(c: pd.DataFrame) -> pd.DataFrame:
    """Per line x peril: age->ultimate factors for each start age 0..3."""
    rows = []
    for (line, peril), gg in c.groupby(["line", "peril"]):
        cum = build_cum(gg)
        fac = triangle_factors(cum)
        rows.append({
            "line": line, "peril": peril, "n": len(gg), "cost_M": gg["cost"].sum() / 1e6,
            "late": float((gg["rep"] - gg["loss"]).dt.days.gt(365).mean()),
            **{f"age{k}": age0_to_ultimate(fac, k) for k in range(4)},
        })
    return pd.DataFrame(rows).sort_values(["line", "cost_M"], ascending=[True, False])


def main() -> int:
    c = load_claims()
    print(f"claims: {len(c):,}  net cost {c['cost'].sum() / 1e6:,.1f}M")

    print("\n== LOB-level check against the documented triangle ==")
    ok = True
    for line in ["motor", "buildings", "contents"]:
        cum = build_cum(c[c["line"] == line])
        fac = triangle_factors(cum)
        a0 = age0_to_ultimate(fac, 0)
        fac_s = {a: round(f, 3) for a, f in sorted(fac.items())}
        # Compare like with like: age-to-age against age-to-age (0->1, 1->2, 2->3)
        # and age-0->ultimate against the documented "to ultimate" column.
        for a, doc in DOCUMENTED_AGE_TO_AGE[line].items():
            got = fac.get(a)
            if got is None or abs(got - doc) > 0.005:
                ok = False
                print(f"  MISMATCH {line} f({a}->{a+1}): got {got} documented {doc}")
        ult_doc = DOCUMENTED_AGE0_TO_ULT[line]
        if abs(a0 - ult_doc) > 0.005:
            ok = False
            print(f"  MISMATCH {line} age0->ult: got {a0:.3f} documented {ult_doc:.3f}")
        print(f"{line:10s} f(age) {fac_s}  age0->ult = {a0:.3f} "
              f"(documented {ult_doc:.3f})  age1->ult = {age0_to_ultimate(fac, 1):.3f}"
              f"{'  OK' if abs(a0 - ult_doc) <= 0.005 else '  <-- MISMATCH'}")
    if not ok:
        print("\nFATAL: derivation does not reproduce docs/data_findings.md section 5.")
        return 1

    print("\n== per-peril age -> ultimate (within LOB) ==")
    table = full_table(c)
    for line in ["motor", "buildings", "contents"]:
        print(f"\n-- {line} --")
        for _, r in table[table["line"] == line].iterrows():
            print(f"{r['peril']:20s} n={r['n']:7,} cost={r['cost_M']:7.1f}M "
                  f"late>365={r['late']:.3f}  "
                  + "  ".join(f"age{k}->ult={r[f'age{k}']:.3f}" for k in range(4)))

    print("\n== frozen dict (PERIL_LDF) ==")
    frozen = {}
    for line in ["motor", "buildings", "contents"]:
        frozen[line] = {}
        for _, r in table[table["line"] == line].iterrows():
            frozen[line][r["peril"]] = {k: round(float(r[f"age{k}"]), 4) for k in range(4)}
    print(frozen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
