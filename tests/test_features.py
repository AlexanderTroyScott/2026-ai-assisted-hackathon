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
    # r2's level has no history before its period_start -> the strictly-lagged
    # line anchor as of r2.ps, which is r1 alone (D-084)
    assert tr.loc[tr.row_id == "r2", "lvl_hist_rate"].iloc[0] == pytest.approx(100.0)
    # test rows of a level never seen in training also fall back to the line rate;
    # by 2025 the whole training history qualifies -> (100 + 0) / 2
    assert te["lvl_hist_rate"].iloc[0] == pytest.approx(50.0)


def test_rel_hist_prior_is_strictly_lagged():
    """D-084: the credibility anchor itself must exclude outcomes from rows
    whose claims can be dated after the target row's period_start.

    Pre-D-084 the anchor was the full-period line rate, so every thin-cell
    feature carried a slice of the holdout's own outcomes.
    """
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2021-12-31", 100.0, 1, "a"),
        _row("r2", "2022-01-01", "2022-12-31", 900.0, 4, "zzz"),  # unseen -> anchor
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "qqq")])
    tr, te = historical_relativities(train, test, ["lvl"])
    # r2's anchor: only r1 qualifies (pe 2021-12-31 < ps 2022-01-01) -> 100,
    # NOT the full-period rate (100 + 900) / 2 = 500.
    assert tr.loc[tr.row_id == "r2", "lvl_hist_rate"].iloc[0] == pytest.approx(100.0)
    # t1's anchor: the whole training history qualifies -> 500
    assert te["lvl_hist_rate"].iloc[0] == pytest.approx(500.0)


def test_rel_hist_initial_prior_backoff_when_no_history():
    """D-084: a row with zero antecedent experience falls back to the labelled
    initial prior (the full-period line rate) - the only place the strict anchor
    may draw on the full period. On the real book these are AY2021-01-01
    inception rows only; holdout and test rows never draw on it.
    """
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2021-12-31", 100.0, 1, "zzz"),
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "qqq")])
    tr, _ = historical_relativities(train, test, ["lvl"])
    assert tr.loc[tr.row_id == "r1", "lvl_hist_rate"].iloc[0] == pytest.approx(100.0)


def test_rel_hist_legacy_prior_reproduces_pre_d084_anchor():
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2021-12-31", 100.0, 1, "a"),
        _row("r2", "2022-01-01", "2022-12-31", 900.0, 4, "zzz"),
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "qqq")])
    tr, _ = historical_relativities(train, test, ["lvl"], prior="legacy")
    # pre-D-084 anchor: the full-period line rate for every row
    assert tr.loc[tr.row_id == "r2", "lvl_hist_rate"].iloc[0] == pytest.approx(500.0)
    with pytest.raises(ValueError):
        historical_relativities(train, test, ["lvl"], prior="bogus")


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


# --- D-014: experience_features is the strongest feature block (data_findings
# section 7) and had no leakage coverage at all. These pin the `loss < ps`
# boundary in BOTH filters (customer-level and policy-level). Changing either
# to `<=` must fail here.

def _claim(claim_id, policy_id, loss, cost):
    return {"claim_id": claim_id, "policy_id": policy_id,
            "loss": pd.Timestamp(loss), "cost": float(cost)}


def test_experience_lag_is_strict_at_period_start():
    """A loss dated exactly on period_start belongs to that row's own period.

    It is outcome information for the row and must not leak in. `loss <= ps`
    would include it and is the single most likely regression in this file.
    """
    import src.dataset as ds

    policies = pd.DataFrame([_prow("r1", "p1", "c1", "2022-01-01")])
    claims = pd.DataFrame([
        _claim("on_start", "p1", "2022-01-01", 100.0),   # == ps: must be excluded
        _claim("day_before", "p1", "2021-12-31", 50.0),  # < ps: must be included
    ])
    out = ds.experience_features(policies, claims).set_index("row_id")
    assert out.loc["r1", "pol_claims_all"] == 1
    assert out.loc["r1", "pol_cost_all"] == pytest.approx(50.0)
    assert out.loc["r1", "cust_claims_all"] == 1
    assert out.loc["r1", "cust_cost_all"] == pytest.approx(50.0)


def test_experience_excludes_own_period_and_later_losses():
    import src.dataset as ds

    policies = pd.DataFrame([
        _prow("r1", "p1", "c1", "2021-01-01"),
        _prow("r2", "p1", "c1", "2022-01-01"),
        _prow("r3", "p1", "c1", "2023-01-01"),
    ])
    claims = pd.DataFrame([_claim("late", "p1", "2021-06-01", 100.0)])
    out = ds.experience_features(policies, claims).set_index("row_id")
    assert out.loc["r1", "pol_claims_all"] == 0, "own-period loss must not be visible"
    assert out.loc["r2", "pol_claims_all"] == 1
    assert out.loc["r3", "pol_claims_all"] == 1
    assert out.loc["r1", "cust_claims_all"] == 0


def test_experience_customer_history_cross_policy_is_lagged():
    """Customer-level history spans the customer's other policies, but the
    same `loss < ps` rule applies — and policy-level history must NOT pick up
    another policy's claim."""
    import src.dataset as ds

    policies = pd.DataFrame([
        _prow("r1", "p1", "c1", "2021-01-01"),
        _prow("r2", "p2", "c1", "2022-01-01"),  # same customer, different policy
    ])
    claims = pd.DataFrame([_claim("own", "p1", "2021-06-01", 100.0)])
    out = ds.experience_features(policies, claims).set_index("row_id")
    assert out.loc["r1", "cust_claims_all"] == 0
    assert out.loc["r1", "pol_claims_all"] == 0
    assert out.loc["r2", "cust_claims_all"] == 1, "cross-policy history is customer-level only"
    assert out.loc["r2", "pol_claims_all"] == 0, "policy history must stay on its own policy"
    assert out.loc["r2", "cust_cost_all"] == pytest.approx(100.0)


def test_experience_claim_in_later_period_is_invisible_to_earlier_row():
    """A claim dated after a row's period_start must never reach it, even when
    the two rows share a policy and a customer."""
    import src.dataset as ds

    policies = pd.DataFrame([
        _prow("r1", "p1", "c1", "2021-01-01"),
        _prow("r2", "p1", "c1", "2022-01-01"),
    ])
    claims = pd.DataFrame([_claim("future", "p1", "2022-06-01", 100.0)])
    out = ds.experience_features(policies, claims).set_index("row_id")
    for rid in ("r1", "r2"):
        assert out.loc[rid, "pol_claims_all"] == 0
        assert out.loc[rid, "cust_claims_all"] == 0


def test_experience_windows_bound_the_lookback():
    import src.dataset as ds

    policies = pd.DataFrame([_prow("r1", "p1", "c1", "2023-01-01")])
    claims = pd.DataFrame([
        _claim("a", "p1", "2022-06-01", 1.0),   # ~214d back -> inside 1y
        _claim("b", "p1", "2021-06-01", 2.0),   # ~579d back -> inside 2y only
        _claim("c", "p1", "2019-06-01", 4.0),   # ~1310d back -> inside 4y only
    ])
    out = ds.experience_features(policies, claims).set_index("row_id")
    assert out.loc["r1", "cust_claims_1y"] == 1
    assert out.loc["r1", "cust_claims_2y"] == 2
    assert out.loc["r1", "cust_claims_4y"] == 3
    assert out.loc["r1", "cust_claims_all"] == 3
    assert out.loc["r1", "cust_cost_all"] == pytest.approx(7.0)


def test_experience_keeps_distinct_claim_ids_with_identical_loss_and_cost():
    """D-084: two distinct claims sharing (customer, loss date, net cost) must
    both count. The pre-D-084 `(customer_id, loss, cost)` dedup merged them away
    (598 claims on the real book)."""
    import src.dataset as ds

    policies = pd.DataFrame([_prow("r1", "p1", "c1", "2022-01-01")])
    claims = pd.DataFrame([
        _claim("cl1", "p1", "2021-06-01", 100.0),
        _claim("cl2", "p1", "2021-06-01", 100.0),
    ])
    out = ds.experience_features(policies, claims).set_index("row_id")
    assert out.loc["r1", "cust_claims_all"] == 2
    assert out.loc["r1", "cust_cost_all"] == pytest.approx(200.0)
    legacy = ds.experience_features(policies, claims, legacy_dedup=True).set_index("row_id")
    assert legacy.loc["r1", "cust_claims_all"] == 1


# --- D-055: the two leakage boundaries that were not pinned at equality ----

def test_denial_history_pinned_at_exact_period_start():
    """D-031 / AGENTS.md rule 6: a denied claim dated exactly on `period_start`
    belongs to that row's own period and must be invisible to it.

    The existing denial test used mid-period dates, so a `loss < ps` -> `<= ps`
    regression would have slipped through. This is the boundary case.
    """
    import src.dataset as ds

    policies = pd.DataFrame([
        _prow("r1", "p1", "c1", "2021-01-01"),
        _prow("r2", "p1", "c1", "2022-01-01"),
    ])
    denied = pd.DataFrame([
        {"claim_id": "d_edge", "policy_id": "p1",
         "loss": pd.Timestamp("2022-01-01")},   # exactly r2.ps
    ])
    out = ds.denial_features(policies, denied).set_index("row_id")
    assert out.loc["r2", "pol_denied_all"] == 0, (
        "a denial dated exactly on period_start is the row's own outcome"
    )
    assert out.loc["r2", "cust_denied_all"] == 0
    # r1 is BEFORE the denial, so it must see nothing either
    assert out.loc["r1", "cust_denied_all"] == 0


def test_rel_hist_pinned_at_pe_equals_ps():
    """src/features.py: a source row whose period ends exactly on the target's
    `period_start` must be excluded.

    Its claims can be dated on or after the target's period_start, so including
    it is leakage. The existing test used `pe >> ps`, so a `searchsorted`
    `side="right"` regression would slip through.
    """
    train = pd.DataFrame([
        _row("r1", "2021-01-01", "2022-01-01", 100.0, 1, "a"),   # pe == r2.ps
        _row("r2", "2022-01-01", "2022-12-31", 0.0, 0, "a"),
    ])
    test = pd.DataFrame([_row("t1", "2025-01-01", "2025-12-31", 0.0, 0, "a")])
    tr, _ = historical_relativities(train, test, ["lvl"])
    # r2 must see only rows with pe < 2022-01-01, i.e. none of r1
    assert tr.loc[tr.row_id == "r2", "lvl_hist_rate"].isna().all() or \
           tr.loc[tr.row_id == "r2", "lvl_hist_rate"].iloc[0] != 100.0, (
        "a source row ending exactly on period_start must not contribute"
    )
