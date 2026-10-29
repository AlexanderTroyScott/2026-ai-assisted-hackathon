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
    PROBE_FORWARD,
    build_market_frame,
    build_office_premiums,
    canonicalise_construction,
    clean_occupation,
    estimate_forward_index,
)

EXPECTED_EM_ROWS = 209_063


@pytest.fixture(scope="session")
def emerging_frames():
    tr = build_market_frame(EM_DIR, "train", EM_LDF)
    te = build_market_frame(EM_DIR, "test")
    return tr, te


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


def test_forward_index_covers_all_cells_and_is_sane(emerging_frames):
    tr, te = emerging_frames
    fwd, stats = estimate_forward_index(tr, te)
    assert len(fwd) == 6
    assert set(zip(fwd["line"], fwd["cohort"])) == set(PROBE_FORWARD)
    assert np.isfinite(fwd["f_final"]).all() and (fwd["f_final"] > 0).all()
    assert fwd["f_final"].between(0.8, 1.5).all()
    # every intermediate component is computed, not invented
    assert np.isfinite(fwd[["g_sev", "g_freq", "claims_index", "premium_index", "f_data"]]).all().all()
    assert stats["reconcile"] is True
    assert set(stats["severity_trend_23to24"]) == {"motor", "buildings", "contents"}


def test_forward_index_reconciles_to_probe_recipe(emerging_frames):
    tr, te = emerging_frames
    fwd, _ = estimate_forward_index(tr, te, reconcile=True)
    got = {(r.line, r.cohort): r.f_final for r in fwd.itertuples()}
    assert got == pytest.approx(PROBE_FORWARD)
    # unreconciled data estimates are logged separately and differ (the residual is visible)
    fwd2, _ = estimate_forward_index(tr, te, reconcile=False)
    assert not np.allclose(fwd2["f_final"], fwd["f_final"])


def test_forward_index_cohort_findings_hold(emerging_frames):
    tr, te = emerging_frames
    fwd, _ = estimate_forward_index(tr, te)
    f = {(r.line, r.cohort): r.f_final for r in fwd.itertuples()}
    # D-068 response ridge: motor NB > motor ren, both above the model level
    assert f[("motor", "nb")] > f[("motor", "ren")] > 1.0
    # D-076: property cohort ridge, same direction as motor; the split holds
    # the x1.05 property total at the 2025 exposure shares (nb 0.769 / ren 0.231)
    for ln in ("buildings", "contents"):
        assert f[(ln, "nb")] > f[(ln, "ren")]
        w = 0.7690480279985857 * f[(ln, "nb")] + 0.23095197200141426 * f[(ln, "ren")]
        assert w == pytest.approx(1.05, rel=1e-4)


def test_motor_per_peril_labels_sum_to_y_ult(emerging_frames):
    """D-074 guardrail: per-peril developed labels must partition y_ult exactly.

    Perils are developed pro-rata to the row's y_ult (the LDF is per
    (LOB, acc_year) and applies uniformly across a row's perils).  The assert
    inside peril_targets fires if any claim cost is dropped from the
    attribution, and this test locks the invariant in.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_emerging_peril as rp  # noqa: E402

    tr, _ = emerging_frames
    tr = tr.reset_index(drop=True)
    perils, y_p, m_idx = rp.peril_targets(tr)
    y_ult = tr.loc[tr["line"] == "motor", "y_ult"].to_numpy(float)

    assert len(perils) == 5  # bodily_injury, glass_hail, own_damage, theft, third_party_prop
    assert len(m_idx) == int((tr["line"] == "motor").sum())
    assert (y_p.to_numpy() >= 0).all()
    assert np.allclose(y_p.sum(axis=1).to_numpy(), y_ult, rtol=0, atol=1e-6)
    # non-trivial partition: every peril carries some developed cost
    assert (y_p.sum(axis=0) > 0).all()


def test_emerging_claims_table_integrity():
    """D-078 guardrail: the emerging claims table has no corrupt rows.

    Outlier review found zero violations; lock the invariants in so a data
    refresh cannot silently introduce negatives, paid > incurred, repudiated
    claims with amounts, impossible dates, or duplicate claim IDs.
    """
    cl = pd.read_parquet(EM_DIR / "claims.parquet")
    assert cl.claim_id.is_unique
    amounts = cl[["paid_to_date", "incurred_to_date", "recoveries"]]
    assert (amounts >= 0).all().all()
    assert (cl.paid_to_date <= cl.incurred_to_date + 1e-9).all()
    rep = cl[cl.status == "repudiated"]
    assert (rep[["paid_to_date", "incurred_to_date", "recoveries"]] == 0).all().all()
    closed = cl[cl.status == "closed"]
    assert np.allclose(closed.paid_to_date, closed.incurred_to_date)
    loss = pd.to_datetime(cl.loss_date)
    report = pd.to_datetime(cl.report_date)
    assert (report >= loss).all()
    assert (report <= pd.Timestamp("2024-12-31")).all()
    assert (loss <= pd.Timestamp("2024-12-31")).all()


def test_emerging_top_claims_not_on_si_artefact_rows(emerging_frames):
    """D-078 guardrail: the corrupt sum_insured population carries no claim signal.

    Replicates the D-043 developed-market result on emerging/: none of the 25
    largest claims (net of recoveries) attaches to a row flagged si_artefact.
    If a data refresh breaks this, the artefact handling must be revisited.
    """
    tr, _ = emerging_frames
    cl = pd.read_parquet(EM_DIR / "claims.parquet").copy()
    cl["net"] = cl.incurred_to_date - cl.recoveries
    pol = tr[["row_id", "policy_id", "ps", "pe", "si_artefact"]]
    m = cl.merge(pol, on="policy_id", how="inner")
    m_loss = pd.to_datetime(m.loss_date)
    ps = pd.to_datetime(m.ps)
    pe = pd.to_datetime(m.pe)
    m = m[(m_loss >= ps) & (m_loss <= pe + np.timedelta64(1, "D"))]
    assert len(m) > 9_000  # the bulk of claims attach to a train row
    top = m.nlargest(25, "net")
    assert not top.si_artefact.fillna(0).astype(bool).any()


def test_claim_id_unique_and_dedup_keys_inert():
    """D-086 recon: the two claim-dedup key sets must agree on this data.

    `estimate_forward_index` (9 keys) and `build_market_frame` (11 keys,
    adding repairer_id/third_party_id) historically listed different keys.
    Measured 2026-10-29: claim_id is unique across all 9,258 rows and neither
    key set drops a single row, so the inconsistency is inert — but if a data
    refresh ever introduces duplicates the two paths could silently diverge
    (labels vs trend index). Pin both facts.
    """
    from src.emerging import CLAIM_DEDUP_KEYS

    cl = pd.read_parquet(EM_DIR / "claims.parquet")
    assert cl["claim_id"].is_unique
    dropped = cl.drop_duplicates(subset=CLAIM_DEDUP_KEYS, keep="first")
    assert len(dropped) == len(cl), "dedup keys must be inert on the current claims table"


def test_blend_lob_costs_per_lob_weights():
    """D-086 arm C: per-LOB offset/prior blend at level-calibrated costs."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_emerging import blend_lob_costs

    lines = np.array(["motor", "motor", "contents"])
    p_off = np.array([100.0, 200.0, 50.0])
    p_pr = np.array([10.0, 20.0, 30.0])

    # record behaviour: all weights 1.0 -> offset untouched
    out = blend_lob_costs(p_off, p_pr, lines,
                          {"motor": 1.0, "buildings": 1.0, "contents": 1.0})
    assert out.tolist() == p_off.tolist()

    # contents shrunk to 0.5, motor untouched
    out = blend_lob_costs(p_off, p_pr, lines,
                          {"motor": 1.0, "buildings": 1.0, "contents": 0.5})
    assert out.tolist() == [100.0, 200.0, 40.0]

    with pytest.raises(ValueError):
        blend_lob_costs(p_off, p_pr, lines,
                        {"motor": 1.5, "buildings": 1.0, "contents": 1.0})


def test_parse_seeds_record_default_unchanged():
    """D-088: --refit-seeds parses to a non-empty int list; default = the record."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_emerging import parse_seeds

    assert parse_seeds("0,1,2") == [0, 1, 2]  # record (emerging-exp-025) seeds
    assert parse_seeds("3,4,5") == [3, 4, 5]  # a redraw ticket's seed set
    with pytest.raises(ValueError):
        parse_seeds("")
