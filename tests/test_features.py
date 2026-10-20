"""Guardrails for feature engineering (AGENTS.md rule 6: strict lagging).

Anything derived from claims for a row must only use loss dates before that
row's `period_start`. Historical relativities are built from source rows with
`pe < row.ps`; these tests make leakage detectable on toy frames.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.features as features  # noqa: E402
from src.features import historical_relativities  # noqa: E402


@pytest.fixture(autouse=True)
def no_smoothing(monkeypatch):
    """Remove credibility smoothing so toy-frame expectations are exact."""
    monkeypatch.setattr(features, "PRIOR_EXPOSURE", 0.0)
    monkeypatch.setattr(features, "MIN_LEVEL_EXPOSURE", 0.0)


def _row(row_id, ps, pe, y, n, level):
    return {
        "row_id": row_id, "line": "motor", "ps": pd.Timestamp(ps), "pe": pd.Timestamp(pe),
        "y_ult": y, "n_claims": n, "exposure": 1.0, "lvl": level,
    }


def test_rel_hist_excludes_own_and_later_rows():
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2021-12-31", 100.0, 1, "a"),
        _row("r2", "2022-01-01", "2022-12-31", 0.0, 0, "a"),
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "a")])
    tr, te = historical_relativities(train, test, ["lvl"])
    # r2 sees only r1: rate 100. If it saw its own row, the rate would be 50.
    assert tr.loc[tr.row_id == "r2", "lvl_hist_rate"].iloc[0] == pytest.approx(100.0)
    # t1 sees both train rows: (100 + 0) / 2
    assert te["lvl_hist_rate"].iloc[0] == pytest.approx(50.0)


def test_rel_hist_excludes_overlapping_period_rows():
    """A source row whose period extends past ps must not contribute, even if
    it started earlier (its claims can be dated after our period_start)."""
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2021-12-31", 100.0, 1, "a"),
        _row("r2", "2021-07-01", "2022-06-30", 900.0, 4, "a"),  # pe after r3.ps
        _row("r3", "2022-01-01", "2022-12-31", 0.0, 0, "a"),
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "a")])
    tr, _ = historical_relativities(train, test, ["lvl"])
    # r3 may see r1 only (r2.pe = 2022-06-30 >= r3.ps = 2022-01-01)
    assert tr.loc[tr.row_id == "r3", "lvl_hist_rate"].iloc[0] == pytest.approx(100.0)


def test_rel_hist_unseen_level_falls_back_to_line_rate():
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2021-12-31", 100.0, 1, "a"),
        _row("r2", "2022-01-01", "2022-12-31", 0.0, 0, "zzz"),
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "qqq")])
    tr, te = historical_relativities(train, test, ["lvl"])
    line_rate = train["y_ult"].sum() / train["exposure"].sum()
    # r2's level has no history before its period_start -> line rate
    assert tr.loc[tr.row_id == "r2", "lvl_hist_rate"].iloc[0] == pytest.approx(line_rate)
    # test rows of a level never seen in training also fall back to the line rate
    assert te["lvl_hist_rate"].iloc[0] == pytest.approx(line_rate)


def test_rel_hist_frequency_matches_rate_structure():
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2021-12-31", 50.0, 3, "a"),
        _row("r2", "2022-01-01", "2022-12-31", 0.0, 0, "a"),
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "a")])
    tr, _ = historical_relativities(train, test, ["lvl"])
    assert tr.loc[tr.row_id == "r2", "lvl_hist_freq"].iloc[0] == pytest.approx(3.0)


def _prow(row_id, policy_id, cust, ps):
    return {
        "row_id": row_id, "policy_id": policy_id, "customer_id": cust,
        "ps": pd.Timestamp(ps),
    }


def test_denial_history_excludes_own_period_and_later():
    """D-031: a denied claim is outcome information about its own period.

    The row it belongs to must not see it; only strictly later rows may
    (`loss < ps`, same rule as the experience block).
    """
    import src.dataset as ds

    policies = pd.DataFrame([
        _prow("r1", "p1", "c1", "2021-01-01"),
        _prow("r2", "p1", "c1", "2022-01-01"),
        _prow("r3", "p2", "c1", "2023-01-01"),
    ])
    denied = pd.DataFrame([
        {"claim_id": "d1", "policy_id": "p1", "loss": pd.Timestamp("2021-06-01")},  # in r1's period
        {"claim_id": "d2", "policy_id": "p2", "loss": pd.Timestamp("2023-06-01")},  # in r3's period
    ])
    out = ds.denial_features(policies, denied).set_index("row_id")
    # r1's own denied claim must be invisible to r1
    assert out.loc["r1", "cust_denied_all"] == 0
    assert out.loc["r1", "days_since_denied"] != out.loc["r1", "days_since_denied"]  # NaN
    # r2 sees d1 (loss 2021-06-01 < ps 2022-01-01)
    assert out.loc["r2", "cust_denied_all"] == 1
    assert out.loc["r2", "days_since_denied"] == pytest.approx(214, abs=2)
    assert out.loc["r2", "pol_denied_all"] == 1
    # r3 must not see d2 (same period) and d1 is customer-level history
    assert out.loc["r3", "cust_denied_all"] == 1
    assert out.loc["r3", "pol_denied_all"] == 0


def test_denial_history_counts_are_zero_when_clean():
    import src.dataset as ds

    policies = pd.DataFrame([_prow("r1", "p1", "c1", "2021-01-01")])
    denied = pd.DataFrame(columns=["claim_id", "policy_id", "loss"])
    out = ds.denial_features(policies, denied).set_index("row_id")
    assert out.loc["r1", "cust_denied_all"] == 0
    assert out.loc["r1", "pol_denied_all"] == 0
