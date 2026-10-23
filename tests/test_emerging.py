"""Guardrails for Part 2 (Emerging Market) adaptation, schema canonicalisation,
and office premium build-up (D-066).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.emerging import (  # noqa: E402
    EM_DIR,
    EM_LDF,
    build_office_premiums,
    canonicalise_construction,
    clean_occupation,
)

EXPECTED_EM_ROWS = 209_063


def test_emerging_sample_submission_and_test_coverage():
    sample = pd.read_parquet(EM_DIR / "sample_submission.parquet")
    assert len(sample) == EXPECTED_EM_ROWS
    assert list(sample.columns) == ["row_id", "prediction"]
    assert sample["row_id"].is_unique

    ids = []
    for ln, pref in [("motor", "M"), ("buildings", "B"), ("contents", "C")]:
        t = pd.read_parquet(EM_DIR / f"test_{ln}.parquet", columns=["row_id"])
        assert t["row_id"].str.startswith(pref).all()
        ids.append(t["row_id"])
    all_ids = pd.concat(ids, ignore_index=True)
    assert len(all_ids) == EXPECTED_EM_ROWS
    assert set(all_ids) == set(sample["row_id"])


def test_clean_occupation_maps_free_text_to_20_canonical_levels():
    cust_em = pd.read_parquet(EM_DIR / "customers.parquet", columns=["occupation"])
    cust_dev = pd.read_parquet(ROOT / "developed" / "customers.parquet", columns=["occupation"])
    dev_levels = set(cust_dev["occupation"].dropna().str.strip().str.lower().unique())
    clean, is_se = clean_occupation(cust_em["occupation"])
    em_levels = set(clean.unique()) - {"other"}
    assert em_levels == dev_levels, f"unmapped occupation levels: {em_levels ^ dev_levels}"
    assert set(is_se.unique()) <= {0.0, 1.0}
    assert is_se.sum() > 5000


def test_canonicalise_construction_maps_local_codes():
    b_em = pd.read_parquet(EM_DIR / "policies_buildings.parquet", columns=["construction_type"])
    mapped = canonicalise_construction(b_em["construction_type"])
    assert set(mapped.unique()) == {"brick", "stone", "timber", "non_standard"}


def test_emerging_ldfs_monotone():
    for ln, tab in EM_LDF.items():
        assert tab[2024] > tab[2023] > 1.0


def test_office_premium_formula_matches_expenses_table():
    ref = pd.read_parquet(EM_DIR / "reference_policies.parquet")
    exp = pd.read_parquet(EM_DIR / "expenses.parquet")
    preds = pd.DataFrame({"row_id": ref["row_id"], "prediction": np.full(len(ref), 200.0)})
    out = build_office_premiums(ref, preds, exp, market="emerging")
    assert len(out) == 20
    assert not out["office_premium_annual"].isna().any()
    # Check broker motor row: denom = 1 - (0.15 + 0.20 + 0.08 + 0.08) = 0.49, cat = 0
    m_brk = out[(out["line"] == "motor") & (out["channel"] == "broker")].iloc[0]
    expected_m = (m_brk["risk_premium_annual"] + 85.0) / 0.49
    assert m_brk["office_premium_annual"] == pytest.approx(expected_m, rel=1e-9)
    # Check broker buildings row: cat = 0.12 * risk_premium_annual, denom = 0.49
    b_brk = out[(out["line"] == "buildings") & (out["channel"] == "broker")].iloc[0]
    expected_b = (b_brk["risk_premium_annual"] * 1.12 + 85.0) / 0.49
    assert b_brk["office_premium_annual"] == pytest.approx(expected_b, rel=1e-9)
