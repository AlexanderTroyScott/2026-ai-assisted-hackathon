"""Guardrails for the R-04 model diagnostics (Gini, normalised Gini, lift).

These are known-answer tests: each pins a property the source states explicitly,
so a change in implementation cannot silently redefine the metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diagnostics import (  # noqa: E402
    calibration,
    data_gini,
    double_lift,
    gini_gap,
    lift_chart,
    lift_summary,
    lorenz_curve,
    normalized_gini,
    weighted_gini,
)


def _sample(n=2000, seed=7):
    """A well-behaved loss sample: heavy-tailed severity, exposure ~1.

    The prediction is returned on the COST scale (not frequency) so that
    normalising by the mean prediction yields the O(1) quantities R-04 plots.
    """
    rng = np.random.RandomState(seed)
    exposure = rng.uniform(0.3, 1.0, n)
    freq = rng.uniform(0.05, 0.25, n)
    sev = rng.lognormal(mean=8.0, sigma=1.2, size=n)
    actual = (rng.uniform(size=n) < freq) * sev
    mean_sev = float(sev.mean())
    pred = freq * mean_sev          # expected cost per unit exposure
    return actual, exposure, pred


# --- R-04: Normalized Gini = Model Gini / Data Gini -------------------------

def test_perfect_ranking_scores_one():
    """R-04: "A saturated model... would have a normalized Gini equal to unity." """
    a, w, _ = _sample()
    assert normalized_gini(a, w, a) == pytest.approx(1.0, abs=1e-9)


def test_constant_prediction_scores_zero():
    """R-04: "A model where the predicted value is a constant, would have a
    normalized Gini equal to zero."

    With a random tiebreaker this holds in EXPECTATION, not exactly, so the
    assertion is on the seed-average.
    """
    a, w, _ = _sample()
    const = np.full_like(a, 123.0)
    vals = [normalized_gini(a, w, const, seed=s) for s in range(12)]
    assert abs(np.mean(vals)) < 0.08, f"constant ranking should average ~0, got {np.mean(vals):.4f}"
    assert max(abs(v) for v in vals) < 0.30, (
        f"individual draws swing with the tiebreaker but must stay bounded, got {max(abs(v) for v in vals):.4f}"
    )


def test_data_gini_is_the_ceiling():
    a, w, p = _sample()
    assert data_gini(a, w) > 0
    # an informative model cannot exceed the ranking that sorts by actuals
    assert weighted_gini(a, w, p) <= data_gini(a, w) + 1e-9


def test_informative_model_beats_noise():
    """A real signal must outperform a random ranking on the same sample."""
    a, w, p = _sample()
    rng = np.random.RandomState(3)
    noise = rng.random(len(p))
    assert normalized_gini(a, w, p) > normalized_gini(a, w, noise)


# --- R-04: the tiebreaker trap ---------------------------------------------

def test_tiebreaker_prevents_inflated_gini():
    """R-04: if the data was previously sorted by actuals and the sort is stable,
    a model with tied predictions inherits that order and the Gini comes out
    artificially high. The random tiebreaker must break that."""
    a, w, _ = _sample(n=800)
    # sort by actuals first: the "previously sorted" case R-04 warns about
    order = np.argsort(a)
    a_s, w_s = a[order], w[order]
    tied = np.ones_like(a_s)                      # model predicts a constant

    # without a tiebreaker a stable sort would preserve the actuals order and
    # the "constant" model would score like a perfect ranking.
    naive_order = np.argsort(tied, kind="stable")
    naive_lorenz = np.cumsum((a_s * w_s)[naive_order]) / (a_s * w_s).sum()
    wc = np.cumsum(w_s[naive_order] / w_s[naive_order].sum())
    naive = float((naive_lorenz[1:] * wc[:-1] - naive_lorenz[:-1] * wc[1:]).sum())

    # with the seeded random tiebreaker it collapses toward zero
    tied_vals = [weighted_gini(a_s, w_s, tied, seed=s) for s in range(12)]
    # The point is the CONTRAST: a stable sort on pre-sorted data scores like a
    # perfect ranking, while the random tiebreaker collapses tied predictions
    # toward zero. R-04: "the Gini coefficient and Lorenz curve might look
    # better than they ought to be."
    assert naive > 0.5, (
        f"sanity: a stable sort on pre-sorted data must be badly inflated (got {naive:.4f}); "
        "otherwise this test cannot detect the trap it is written for"
    )
    assert abs(np.mean(tied_vals)) < 0.15, (
        f"random tiebreaker must neutralise tied predictions, mean={np.mean(tied_vals):.4f}"
    )
    assert np.mean(tied_vals) < naive * 0.35, (
        f"tiebreaker must be far below the naive sort: {np.mean(tied_vals):.4f} vs {naive:.4f}"
    )


def test_result_is_seed_stable():
    a, w, p = _sample()
    g1 = weighted_gini(a, w, p, seed=11)
    g2 = weighted_gini(a, w, p, seed=11)
    assert g1 == g2, "same seed must reproduce"
    # different seeds only matter when there are ties; with continuous
    # predictions the value must be essentially unchanged
    assert weighted_gini(a, w, p, seed=12) == pytest.approx(g1, abs=1e-6)


# --- R-04: the train/holdout gap is the overfit measure ---------------------

def test_overfit_model_shows_a_wider_gap():
    """R-04's worked example: the worse-generalising model has the larger gap
    between train and holdout normalised Gini."""
    rng = np.random.RandomState(5)
    n = 3000

    def make(share_noise_between_pred_and_actual):
        """Model A tracks only the signal. Model B also tracks this sample's
        noise term, which is what overfitting looks like: it helps in-sample and
        is worthless out-of-sample."""
        latent = rng.uniform(0, 1, n)
        noise = rng.normal(size=n)
        exp = np.ones(n)
        actual = rng.lognormal(mean=7.5, sigma=1.0, size=n) * (1 + 2 * latent) * (1 + 0.4 * noise)
        pred_a = 1 + 2 * latent
        if share_noise_between_pred_and_actual:
            pred_b = 1 + 2 * latent + 0.30 * noise      # chases THIS sample's noise
        else:
            pred_b = 1 + 2 * latent + 0.30 * rng.normal(size=n)  # independent noise
        return actual, exp, pred_a, pred_b

    # train half: model B is fitted to that half's noise
    at, wt, pat, pbt = make(True)
    # holdout half: fresh draws, so model B's noise term is uncorrelated with it
    ah, wh, pah, pbh = make(False)

    clean = gini_gap((at, wt, pat), (ah, wh, pah))
    noisy = gini_gap((at, wt, pbt), (ah, wh, pbh))
    assert noisy["gap"] > clean["gap"], (
        f"expected the noisier model to generalise worse: clean gap {clean['gap']:.4f} "
        f"vs noisy {noisy['gap']:.4f}"
    )
    assert noisy["normalized_gini_train"] > noisy["normalized_gini_holdout"], (
        "an overfit model must score better in-sample than out-of-sample"
    )


def test_gini_gap_reports_both_sides():
    a, w, p = _sample()
    out = gini_gap((a, w, p), (a, w, p))
    assert set(out) == {"normalized_gini_train", "normalized_gini_holdout", "gap"}
    assert out["gap"] == pytest.approx(0.0, abs=1e-9)


# --- R-04: Lorenz curve and lift charts -------------------------------------

def test_lorenz_endpoints_and_monotonicity():
    a, w, p = _sample()
    lc = lorenz_curve(a, w, p)
    assert lc["cum_exposure"].iloc[0] == 0.0
    assert lc["cum_loss"].iloc[0] == 0.0
    assert lc["cum_exposure"].iloc[-1] == pytest.approx(1.0)
    assert lc["cum_loss"].iloc[-1] == pytest.approx(1.0)
    assert (lc["cum_exposure"].diff().dropna() >= 0).all()
    assert (lc["cum_loss"].diff().dropna() >= -1e-12).all()


def test_lift_bins_have_equal_exposure():
    a, w, p = _sample(n=5000)
    ch = lift_chart(a, w, p, bins=10)
    assert len(ch) == 10
    expo = ch["exposure"].to_numpy()
    # equal-exposure binning: within a tolerance from integer row boundaries
    assert (expo.max() - expo.min()) / expo.mean() < 0.25, (
        f"exposure should be near-equal across bins, got {expo}"
    )


def test_lift_of_an_informative_model_is_positive_and_monotone():
    a, w, p = _sample(n=5000)
    ch = lift_chart(a, w, p, bins=10)
    s = lift_summary(ch)
    assert s["lift_ratio"] > 1.0, "an informative model must lift the top bin above the bottom"
    # R-04: actuals should rise with the bins, small reversals tolerated
    act = ch["actual_norm"].to_numpy()
    assert act[-1] > act[0]
    # but not perfectly monotone necessarily; count only large reversals
    # reversals judged on the normalised scale (R-04: "some small reversals in
    # the pattern are fine"), relative to the spread across bins
    spread = act.max() - act.min()
    drops = np.diff(act)
    assert (drops < -0.25 * spread).sum() == 0, (
        f"large reversals indicate the model is not picking up the signal: {act}"
    )


def test_lift_of_a_useless_model_has_lift_near_one():
    a, w, _ = _sample(n=5000)
    noise = np.random.RandomState(1).random(len(a))
    ch = lift_chart(a, w, noise, bins=10)
    assert abs(lift_summary(ch)["lift_ratio"] - 1.0) < 0.35, (
        "a noise ranking should not separate the extremes meaningfully"
    )


# --- R-04: double lift ------------------------------------------------------

def test_double_lift_orders_by_prediction_ratio():
    a, w, pa = _sample(n=3000)
    pb = pa * np.random.RandomState(2).uniform(0.5, 1.5, len(pa))
    dl = double_lift(a, w, pa, pb, bins=10)
    assert len(dl) == 10
    r = dl["ratio_mean"].to_numpy()
    assert (np.diff(r) >= -1e-9).all(), "bins must be ordered by the prediction ratio"
    assert r[0] < r[-1], "first bin is where A dominates, last is where B dominates"
    assert set(dl.columns) >= {"bin", "n", "ratio_mean", "actual", "pred_a", "pred_b"}


def test_diagnostics_reject_mismatched_lengths():
    a, w, p = _sample(n=10)
    with pytest.raises(ValueError):
        weighted_gini(a, w, p[:-1])
    with pytest.raises(ValueError):
        double_lift(a, w, p, p[:-1])


def test_diagnostics_reject_all_zero_loss():
    w = np.ones(20)
    z = np.zeros(20)
    p = np.random.RandomState(0).random(20)
    with pytest.raises(ValueError):
        weighted_gini(z, w, p)


# --- D-054: calibration is the complement Gini/lift cannot see -------------

def test_calibration_of_a_well_levelled_model():
    """ratio 1.0 and slope 1.0 when the prediction is the true conditional mean."""
    a, w, p = _sample(n=4000)
    # p is the expected cost, so a perfectly levelled model has ratio ~1
    c = calibration(a, w, p)
    assert abs(c["calibration_ratio"] - 1.0) < 0.25, c
    assert abs(c["calibration_slope"] - 1.0) < 0.25, c


def test_calibration_detects_a_level_error_that_gini_cannot():
    """D-054: a uniform scale leaves ranking (Gini) unchanged but breaks level.

    This is exactly the exp-025 failure mode — it outranked exp-022 on
    normalised Gini and lift on both folds and still lost 0.011 on the
    leaderboard.
    """
    a, w, p = _sample(n=4000)
    scaled = p * 0.85                      # 15% under-prediction
    g_before = normalized_gini(a, w, p)
    g_after = normalized_gini(a, w, scaled)
    assert g_after == pytest.approx(g_before, abs=1e-9), (
        "a uniform scale must not move a rank-based metric"
    )
    c = calibration(a, w, scaled)
    assert c["calibration_ratio"] > 1.10, (
        f"a 15% under-prediction must show up as a ratio well above 1, got {c['calibration_ratio']:.4f}"
    )


def test_calibration_slope_detects_tail_misallocation():
    """A model that over-predicts the top decile has slope < 1."""
    a, w, p = _sample(n=4000)
    # stretch by RANK so the distortion lands on the top of the ranking,
    # not on whatever order the rows happen to sit in
    rank = pd.Series(p).rank(pct=True).to_numpy()
    stretched = p * (0.70 + 0.65 * rank)   # -30% at the bottom, +35% at the top
    c = calibration(a, w, stretched)
    assert c["calibration_slope"] < 0.95, (
        f"stretched predictions must show slope < 1, got {c['calibration_slope']:.4f}"
    )
