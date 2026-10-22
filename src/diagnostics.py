"""Model diagnostics: Gini, normalised Gini, Lorenz, lift and double-lift charts.

Implements the diagnostics specified in R-04 (CAS Institute, *Model Validation and
Holdout Data*), which identifies the **train-versus-holdout normalised-Gini gap** as
the overfit measure to watch rather than the level itself:

    "A model generalizes well if its normalized Gini in a testing or holdout sample
     is reasonably close to its normalized Gini in the training sample."

R-04's worked example: random forest 0.91358 train / 0.90295 holdout (gap 0.0106)
beats linear regression 0.88685 / 0.86674 (gap 0.0201) on both fit and generalisation.

Two implementation details from the note are load-bearing and are not optional:

1. **Sort by predicted, not actual.** The Lorenz/Gini of a rating plan sorts risks
   by the *model's* ranking and asks how much loss that ranking captures.
2. **Add a random tiebreaker to the sort key.** R-04: "If a model produces the same
   predicted value for large segments of the data, and if those segments had been
   previously sorted in ascending order of actual values, and if the sort function
   preserves that order, the Gini coefficient and Lorenz curve might look better than
   they ought to be." A seeded generator is used here so results are reproducible.

Normalised Gini = Model Gini / Data Gini, where Data Gini is the same statistic
computed on actuals. The denominator adjusts for the sample's own dispersion so
train and holdout are comparable. A saturated model scores 1.0 and does not
generalise; a constant model scores 0.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate(actual, weights, predicted):
    a = np.asarray(actual, dtype=float)
    w = np.asarray(weights, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if not (len(a) == len(w) == len(p)):
        raise ValueError("actual, weights and predicted must be the same length")
    ok = np.isfinite(a) & np.isfinite(w) & np.isfinite(p) & (w > 0)
    return a[ok], w[ok], p[ok]


def weighted_gini(actual, weights, predicted, seed: int = 0) -> float:
    """Exposure-weighted Gini of a ranking.

    `actual` is loss cost, `weights` is exposure, `predicted` is the model's
    estimate. Sorting is on `predicted` with a seeded random tiebreaker.
    """
    a, w, p = _validate(actual, weights, predicted)
    if len(a) == 0:
        raise ValueError("no valid rows")
    order = np.lexsort((np.random.RandomState(seed).random(len(p)), p))
    w = w[order]
    a = a[order]
    w_cum = np.cumsum(w / w.sum())
    total = (a * w).sum()
    if total <= 0:
        raise ValueError("total (actual x weight) must be positive")
    lorenz = np.cumsum(a * w) / total
    # Gini = area between the Lorenz curve and the diagonal, doubled.
    # Equivalent to sum(L[i]*x[i-1]) - sum(L[i-1]*x[i]) over consecutive pairs.
    return float((lorenz[1:] * w_cum[:-1] - lorenz[:-1] * w_cum[1:]).sum())


def data_gini(actual, weights, seed: int = 0) -> float:
    """The same statistic with the ranking driven by the actuals themselves.

    This is the ceiling for the sample: how well *any* perfect ranking would do.
    """
    return weighted_gini(actual, weights, actual, seed=seed)


def normalized_gini(actual, weights, predicted, seed: int = 0) -> float:
    """Model Gini / Data Gini (R-04).

    1.0 = saturated (does not generalise). 0.0 = constant. Compare across
    samples; the *gap* between train and holdout is the overfit signal.
    """
    dg = data_gini(actual, weights, seed=seed)
    if dg == 0:
        raise ValueError("data Gini is zero; sample has no dispersion")
    return weighted_gini(actual, weights, predicted, seed=seed) / dg


def gini_gap(train, holdout, seed: int = 0) -> dict:
    """Train-versus-holdout normalised-Gini comparison — the R-04 overfit measure.

    Each input is a (actual, weights, predicted) triple.
    """
    tr = normalized_gini(*train, seed=seed)
    ho = normalized_gini(*holdout, seed=seed)
    return {
        "normalized_gini_train": tr,
        "normalized_gini_holdout": ho,
        "gap": tr - ho,
    }


def lorenz_curve(actual, weights, predicted, seed: int = 0) -> pd.DataFrame:
    """Cumulative share of exposure against cumulative share of loss, sorted by
    prediction (R-04's "actuarial" Lorenz curve).
    """
    a, w, p = _validate(actual, weights, predicted)
    order = np.lexsort((np.random.RandomState(seed).random(len(p)), p))
    w, a = w[order], a[order]
    total = (a * w).sum()
    if total <= 0:
        raise ValueError("total (actual x weight) must be positive")
    cum_exp = np.concatenate([[0.0], np.cumsum(w / w.sum())])
    cum_loss = np.concatenate([[0.0], np.cumsum(a * w) / total])
    return pd.DataFrame({"cum_exposure": cum_exp, "cum_loss": cum_loss})


def lift_chart(actual, weights, predicted, bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Equal-exposure bins sorted by prediction, with mean predicted and actual.

    R-04: "Group your data in bins based on [sorting by prediction] and putting
    the same volume of exposure in each bin. For each bin, calculate the average
    predicted value and the average actual value." Always computed on holdout.

    Read for (a) monotonicity of actuals — small reversals fine, large ones are
    not; (b) lift = last bin minus first (or their ratio); (c) calibration = how
    closely predicted tracks actual.
    """
    a, w, p = _validate(actual, weights, predicted)
    order = np.lexsort((np.random.RandomState(seed).random(len(p)), p))
    w, a, p = w[order], a[order], p[order]
    cum = np.cumsum(w)
    total = cum[-1]
    # equal-exposure cut points
    targets = np.linspace(0.0, total, bins + 1)
    idx = np.searchsorted(cum, targets, side="left")
    idx = np.clip(idx, 0, len(w) - 1)
    rows = []
    for i in range(bins):
        lo, hi = idx[i], idx[i + 1]
        if hi <= lo:
            hi = min(lo + 1, len(w))
        seg = slice(lo, hi)
        rows.append({
            "bin": i + 1,
            "n": int(hi - lo),
            "exposure": float(w[seg].sum()),
            "actual": float((a[seg] * w[seg]).sum() / w[seg].sum()) if w[seg].sum() > 0 else np.nan,
            "predicted": float((p[seg] * w[seg]).sum() / w[seg].sum()) if w[seg].sum() > 0 else np.nan,
        })
    df = pd.DataFrame(rows)
    norm = (p * w).sum() / w.sum()
    df["actual_norm"] = df["actual"] / norm
    df["predicted_norm"] = df["predicted"] / norm
    return df


def calibration(actual, weights, predicted) -> dict:
    """Level and slope diagnostics — the complement to Gini/lift.

    Gini and lift measure DISCRIMINATION (can the model rank risks?) and are
    blind to level by construction. A model can discriminate well and still
    score badly if its level is wrong — measured directly: exp-025 outranked
    exp-022 on normalised Gini and lift on both folds and still lost 0.011 on
    the leaderboard (D-054). These three numbers close that blind spot.

    * ratio        sum(actual) / sum(predicted) over exposure weights. 1.0 = calibrated.
    * slope        regression of actual on predicted through the origin;
                   1.0 = calibrated, <1 over-predicts at the top.
    * intercept    mean residual, i.e. calibration-in-the-large.
    """
    a, w, p = _validate(actual, weights, predicted)
    if len(a) == 0:
        raise ValueError("no valid rows")
    aw = a * w
    ratio = aw.sum() / (p * w).sum() if (p * w).sum() else float("inf")
    denom = (w * p * p).sum()
    slope = float((w * p * a).sum() / denom) if denom else float("nan")
    return {
        "calibration_ratio": float(ratio),
        "calibration_slope": slope,
        "calibration_in_large": float((w * (a - p)).sum() / w.sum()),
    }


def lift_summary(chart: pd.DataFrame) -> dict:
    """Lift as defined in R-04: distance and ratio of the extreme bins."""
    first, last = chart.iloc[0], chart.iloc[-1]
    return {
        "lift_distance": float(last["actual_norm"] - first["actual_norm"]),
        "lift_ratio": float(last["actual_norm"] / first["actual_norm"])
        if first["actual_norm"] else float("inf"),
        "calibration_mae": float((chart["actual"] - chart["predicted"]).abs().mean()),
    }


def double_lift(actual, weights, pred_a, pred_b, bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Compare two models where they disagree most (R-04).

    Sort by the ratio of the two predictions, bin on equal exposure, plot both
    predictions plus actual. The first and last bins are where the models differ
    most, so that is where any real difference should show.
    """
    A = np.asarray(actual, float); W = np.asarray(weights, float)
    PA = np.asarray(pred_a, float); PB = np.asarray(pred_b, float)
    if not (len(A) == len(W) == len(PA) == len(PB)):
        raise ValueError("all inputs must be the same length")
    ok = np.isfinite(A) & np.isfinite(W) & np.isfinite(PA) & np.isfinite(PB) & (W > 0) & (PA > 0)
    A, W, PA, PB = A[ok], W[ok], PA[ok], PB[ok]
    if len(A) == 0:
        raise ValueError("no valid rows")
    ratio = PB / PA
    order = np.lexsort((np.random.RandomState(seed).random(len(ratio)), ratio))
    W, A, PA, PB, ratio = W[order], A[order], PA[order], PB[order], ratio[order]
    cum = np.cumsum(W)
    total = cum[-1]
    idx = np.clip(np.searchsorted(cum, np.linspace(0, total, bins + 1), side="left"), 0, len(W) - 1)
    rows = []
    for i in range(bins):
        lo, hi = idx[i], max(idx[i + 1], idx[i] + 1)
        s = slice(lo, hi)
        sw = W[s].sum()
        rows.append({
            "bin": i + 1,
            "n": int(hi - lo),
            "ratio_mean": float(ratio[s].mean()),
            "actual": float((A[s] * W[s]).sum() / sw) if sw else np.nan,
            "pred_a": float((PA[s] * W[s]).sum() / sw) if sw else np.nan,
            "pred_b": float((PB[s] * W[s]).sum() / sw) if sw else np.nan,
        })
    return pd.DataFrame(rows)
