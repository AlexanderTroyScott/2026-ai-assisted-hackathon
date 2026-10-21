"""Guardrails for label development (D-005 / open question 3).

The per-peril development factors must preserve the accepted LOB level exactly
within each (LOB, accident year) and only redistribute development across perils.

The first block pins the LOB LDF table itself to the documented incurred
triangle (docs/data_findings.md section 5). This is the test that was missing
when a duplicated copy of the table over-developed 2023 buildings/contents
labels by ~8.9% / ~4.0%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import AGE_TO_AGE, LDF, age_to_ultimate  # noqa: E402
from src.development import claim_factors, peril_age_factor  # noqa: E402


# --- D-005: the LOB table must match the documented triangle -------------
# docs/data_findings.md section 5, volume-weighted age-to-age factors:
#   motor 0->1 = 1.223, 1->2 = 1.027, 2->3 = 1.003   -> age-0 to ultimate ~1.26
#   buildings 0->1 = 1.089, 1->2 = 1.003             -> age-0 to ultimate ~1.09
#   contents  0->1 = 1.040, 1->2 = 1.004             -> age-0 to ultimate ~1.04
DOCUMENTED_AGE_TO_AGE = {
    "motor": {0: 1.223, 1: 1.027, 2: 1.003},
    "buildings": {0: 1.089, 1: 1.003},
    "contents": {0: 1.040, 1: 1.004},
}
DOCUMENTED_AGE0_TO_ULT = {"motor": 1.26, "buildings": 1.09, "contents": 1.04}


def test_age_to_age_factors_match_documented_triangle():
    assert AGE_TO_AGE == DOCUMENTED_AGE_TO_AGE


def test_age0_to_ultimate_matches_documented_column():
    for ln, expected in DOCUMENTED_AGE0_TO_ULT.items():
        assert LDF[ln][2024] == pytest.approx(expected, abs=5e-3), (
            f"{ln} age-0 -> ultimate drifted from the documented {expected}"
        )


def test_mature_years_need_no_development():
    """An accident year at or past the last tabulated age is fully developed.

    Getting this wrong over-develops mature labels: the old table applied the
    full age-0 factor to 2023 (age 1), inflating buildings labels by 8.9%.
    """
    # 2023 sits at age 1 at the 2024-12-31 valuation, so it needs only the
    # 1->2 (and 2->3, if any) factors -- NOT the 0->1 factor.
    assert LDF["buildings"][2023] == pytest.approx(1.003, rel=1e-9)
    assert LDF["contents"][2023] == pytest.approx(1.004, rel=1e-9)
    assert LDF["motor"][2023] == pytest.approx(1.027 * 1.003, rel=1e-9)

    # 2021 sits at age 3, past every tabulated age for all three lines.
    for ln in ("motor", "buildings", "contents"):
        assert LDF[ln][2021] == 1.0, f"{ln} 2021 is mature and must not be developed"


def test_younger_accident_years_develop_more():
    """The less mature the accident year, the more development it still needs.

    Walking 2021 -> 2024 the factor must never decrease; a dip would mean a
    mis-keyed age (which is how 2023 came to carry the age-0 factor).
    """
    for ln, tab in LDF.items():
        years = sorted(tab)  # 2021 (most mature) .. 2024 (least mature)
        vals = [tab[y] for y in years]
        assert vals == sorted(vals), f"{ln} LDF not monotone in maturity: {tab}"


def test_age_to_ultimate_beyond_last_age_is_flat():
    assert age_to_ultimate({0: 1.223, 1: 1.027, 2: 1.003}, 3) == 1.0
    assert age_to_ultimate({0: 1.223, 1: 1.027, 2: 1.003}, 0) == pytest.approx(
        1.223 * 1.027 * 1.003, rel=1e-12
    )


def _claims(rows):
    return pd.DataFrame(rows)


def test_level_preserved_within_lob_ay():
    merged = _claims([
        # motor 2024: BI-heavy and OD-heavy claims
        dict(line="motor", acc_year=2024, peril="bodily_injury", cost=90_000.0),
        dict(line="motor", acc_year=2024, peril="own_damage", cost=10_000.0),
        dict(line="motor", acc_year=2024, peril="own_damage", cost=20_000.0),
    ])
    f = claim_factors(merged, LDF)
    wavg = np.average(f, weights=merged["cost"].to_numpy(float))
    assert wavg == pytest.approx(LDF["motor"][2024], rel=1e-9)


def test_long_tail_peril_develops_more():
    merged = _claims([
        dict(line="motor", acc_year=2024, peril="bodily_injury", cost=1.0),
        dict(line="motor", acc_year=2024, peril="own_damage", cost=1.0),
    ])
    f = claim_factors(merged, LDF)
    assert f[0] > f[1]
    assert f[0] > LDF["motor"][2024] > f[1]
    merged = _claims([
        dict(line="buildings", acc_year=2024, peril="subsidence", cost=1.0),
        dict(line="buildings", acc_year=2024, peril="accidental_damage", cost=1.0),
    ])
    f = claim_factors(merged, LDF)
    assert f[0] > LDF["buildings"][2024] > f[1]


def test_mature_year_factors_are_flat():
    merged = _claims([
        dict(line="motor", acc_year=2021, peril="bodily_injury", cost=1.0),
        dict(line="motor", acc_year=2021, peril="own_damage", cost=1.0),
    ])
    f = claim_factors(merged, LDF)
    assert f == pytest.approx(np.array([LDF["motor"][2021]] * 2), rel=1e-9)


def test_unknown_peril_falls_back_to_lob_factor():
    merged = _claims([
        dict(line="motor", acc_year=2024, peril="bodily_injury", cost=50.0),
        dict(line="motor", acc_year=2024, peril="meteor_strike", cost=50.0),
    ])
    f = claim_factors(merged, LDF)
    wavg = np.average(f, weights=merged["cost"].to_numpy(float))
    assert wavg == pytest.approx(LDF["motor"][2024], rel=1e-9)


def test_peril_age_factor_monotone_in_tail():
    assert peril_age_factor("motor", "bodily_injury", 2024) > peril_age_factor(
        "motor", "glass_hail", 2024
    )
    assert peril_age_factor("motor", "bodily_injury", 2021) == 1.0


# --- D-035: the triangle derivation itself must reproduce the documented
# factors. This is the guardrail whose absence let a cumsum bug under-develop
# every peril while the level check passed.

def _triangle_fixture():
    """The documented motor triangle (data_findings.md section 5), incremental.

    Values are M CU exactly as published. Cells past the valuation diagonal
    (age > 2024 - AY) are unreached and must stay NaN.
    """
    import numpy as _np
    return pd.DataFrame(
        {0: [81.5, 92.2, 100.1, 90.7],
         1: [21.8, 21.5, 17.3, _np.nan],
         2: [2.36, 3.19, _np.nan, _np.nan],
         3: [0.37, _np.nan, _np.nan, _np.nan]},
        index=pd.Index([2021, 2022, 2023, 2024], name="ay"),
    )


def test_volume_weighted_factors_exclude_immature_years():
    """Immature AYs must not enter the volume-weighted average (D-035).

    2024 has reached age 0 only and 2023 age 1 only. Including them drags
    f(0->1) toward 1.0 — that was the bug.
    """
    from src.development import EXTRACT_YEAR

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_development import age0_to_ultimate, triangle_factors

    cum = _triangle_fixture().cumsum(axis=1)
    fac = triangle_factors(cum)

    # Only 2021-2023 reach age 1: f(0->1) = (103.3+113.7+117.4)/(81.5+92.2+100.1)
    assert fac[0] == pytest.approx(1.2213, abs=5e-3)
    # Only 2021-2022 reach age 2
    assert fac[1] == pytest.approx(1.0256, abs=5e-3)
    assert fac[2] == pytest.approx(1.0035, abs=5e-3)
    assert age0_to_ultimate(fac, 0) == pytest.approx(1.257, abs=5e-3)
    assert EXTRACT_YEAR - 2024 == 0, "2024 is at age 0 and must not reach age 1"


def test_fill_zero_cumsum_is_the_bug_dilutes_factors():
    """Regression pin for the exact D-035 defect.

    The old build_cum did unstack(fill_value=0) then cumsum, which writes the
    2024 age-0 amount across every later age. Volume-weighting then averages
    that phantom in and drags f(0->1) from 1.221 down to ~1.166.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_development import triangle_factors

    inc = _triangle_fixture()
    naive_cum = inc.fillna(0.0).cumsum(axis=1)   # <- the bug
    naive_fac = triangle_factors(naive_cum)
    correct_fac = triangle_factors(inc.cumsum(axis=1))

    assert naive_fac[0] < correct_fac[0] - 0.02, (
        f"fill-zero+cumsum should dilute f(0->1): {naive_fac[0]:.4f} vs {correct_fac[0]:.4f}"
    )
    assert naive_fac[0] == pytest.approx(1.1663, abs=5e-3)


def test_peril_factors_are_long_tail_ordered():
    """D-028: long-tail perils must develop more than short-tail ones, and the
    ordering must survive the D-035 re-freeze."""
    from src.development import PERIL_LDF

    assert PERIL_LDF["motor"]["bodily_injury"][0] > PERIL_LDF["motor"]["glass_hail"][0]
    assert PERIL_LDF["buildings"]["subsidence"][0] > PERIL_LDF["buildings"]["accidental_damage"][0]
    # long-tail factors should be well above the LOB average, not compressed
    assert PERIL_LDF["motor"]["bodily_injury"][0] > LDF["motor"][2024]
    assert PERIL_LDF["buildings"]["subsidence"][0] > LDF["buildings"][2024]
    # mature ages are flat
    for ln in PERIL_LDF:
        for pr, tab in PERIL_LDF[ln].items():
            assert tab[3] == 1.0, f"{ln}/{pr} age 3 should be fully developed"
