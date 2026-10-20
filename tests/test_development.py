"""Guardrails for label development (D-005 / open question 3).

The per-peril development factors must preserve the accepted LOB level exactly
within each (LOB, accident year) and only redistribute development across perils.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import LDF  # noqa: E402
from src.development import claim_factors, peril_age_factor  # noqa: E402


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
