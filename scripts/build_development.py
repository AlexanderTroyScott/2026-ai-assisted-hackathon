#!/usr/bin/env python
"""Estimate age-to-ultimate development factors, LOB-level and per-peril.

Replicates the documented incurred-triangle method (docs/data_findings.md
"IBNR / development") at peril granularity so long-tail perils (bodily_injury,
subsidence) develop differently from short-tail ones (D-005, open question 3).

Method
------
Cumulative triangle C(AY, age) = sum of (incurred - recoveries) over claims with
accident year AY reported by end of year AY + age. Volume-weighted chain-ladder:

    f(age) = sum_AY C(AY, age+1) / sum_AY C(AY, age)     over AYs in both

Age-0 -> ultimate for accident year AY (age = extract_year - AY) is the product
of f(age .. max) times a tail factor (mean of the last two factors, floored 1.0).

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

# Frozen in src/dataset.py — the script must reproduce these (guardrail).
DOCUMENTED_LOB_AGE0 = {"motor": 1.223, "buildings": 1.089, "contents": 1.040}


def load_claims() -> pd.DataFrame:
    c = pd.read_parquet(DATA / "claims.parquet")
    c = c[c["status"] != "repudiated"].copy()
    c["loss"] = pd.to_datetime(c["loss_date"])
    c["rep"] = pd.to_datetime(c["report_date"])
    c["cost"] = np.maximum(0.0, c["incurred_to_date"] - c["recoveries"])
    c["ay"] = c["loss"].dt.year
    c["age"] = c["rep"].dt.year - c["loss"].dt.year
    return c


def build_cum(c: pd.DataFrame) -> pd.DataFrame:
    """Cumulative reported triangle indexed by AY, columns = development age."""
    g = c.groupby(["ay", "age"])["cost"].sum().unstack(fill_value=0.0)
    g = g[sorted(g.columns)]
    return g.cumsum(axis=1).sort_index()


def triangle_factors(cum: pd.DataFrame) -> dict[int, float]:
    ages = list(cum.columns)
    fac: dict[int, float] = {}
    for i in range(len(ages) - 1):
        a, b = ages[i], ages[i + 1]
        both = cum[(cum[a] > 0) & (cum[b] > 0)]
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

    print("\n== LOB-level check against documented age-0 factors ==")
    for line in ["motor", "buildings", "contents"]:
        cum = build_cum(c[c["line"] == line])
        fac = triangle_factors(cum)
        a0 = age0_to_ultimate(fac, 0)
        a0_prev = age0_to_ultimate(fac, 1)
        fac_s = {a: round(f, 3) for a, f in sorted(fac.items())}
        print(f"{line:10s} f(age) {fac_s}  age0->ult = {a0:.3f} "
              f"(documented {DOCUMENTED_LOB_AGE0[line]:.3f})  age1->ult = {a0_prev:.3f}")

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
