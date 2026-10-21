"""Tweedie GLM benchmark (C-2, R-02 App. C / R-03 section 5.2).

A pure-premium GLM with log link and exposure weights: the standard actuarial
form, and the interpretable counterweight to the LGBM stack. Its purposes are

  1. an honest second opinion on the rating structure (relativities we can
     quote and defend),
  2. the second leg of the C-3 form ensemble,
  3. a check that the tree is not winning by exploiting something the
     actuarial structure cannot express.

Model form follows AGENTS.md: cost is modelled as `exposure x annual rate`, so
the response is the row's rate and the exposure is the weight. This makes the
Tweedie variance assumption apply to cost in the usual way.

D-043 note: log transforms are near-worthless for trees (which are invariant to
monotone transforms) but **material** here, where the linear predictor is
additive in the features. Heavy-tailed fields are logged and the `sum_insured`
artefact population is capped and flagged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Fields logged before entering the linear predictor. D-043 measured the gain
# on log-claim-severity: sum_insured +0.015 -> +0.316, written_premium
# +0.240 -> +0.303 (Pearson).
LOG_FIELDS = ("sum_insured", "written_premium", "distance_to_river_km",
              "premium_per_exposure")

# sum_insured has a dead zone 1M-10M and a separate population above 10M
# (premium/sum_insured 1000x lower). Cap and flag rather than delete (D-043).
ARTEFACT_CAP = 1e7


def _clean_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def prepare_design(df: pd.DataFrame, num: list[str], cat: list[str],
                   impute_from: pd.DataFrame | None = None
                   ) -> tuple[pd.DataFrame, list[str]]:
    """Design matrix for the GLM.

    Numeric columns are passed through (logged where listed) and median-imputed;
    categoricals are one-hot encoded with the first level dropped so the
    intercept is identifiable and coefficients read as relativities against
    that base. Object-typed numeric columns become their own "missing" level.

    `impute_from` supplies the imputation values. Pass the training frame when
    building a holdout/test matrix, so the holdout never contributes to the
    fill values (AGENTS.md rule 6 in spirit — no look-ahead).
    """
    ref = impute_from if impute_from is not None else df
    parts: list[pd.DataFrame] = []
    for c in num:
        if c not in df.columns:
            continue
        v = _clean_numeric(df[c])
        if c in LOG_FIELDS:
            v = np.log1p(v.clip(lower=0))
        med = _clean_numeric(ref[c]).median() if c in ref.columns else 0.0
        if not np.isfinite(med):
            med = 0.0
        parts.append(v.fillna(med).rename(c))

    if "sum_insured" in df.columns:
        si = _clean_numeric(df["sum_insured"])
        parts.append((si > ARTEFACT_CAP).astype(float).rename("si_artefact"))

    for c in cat:
        if c not in df.columns:
            continue
        parts.append(df[c].astype(str).fillna("other"))

    if not parts:
        raise ValueError("no usable columns for the GLM design")
    base = pd.concat(parts, axis=1)
    cat_present = [c for c in cat if c in base.columns]
    # Reindex dummies to the training levels so train/holdout agree on columns.
    dummies = pd.get_dummies(base[cat_present], drop_first=True, dtype=float)
    if impute_from is not None:
        ref_dummies = pd.get_dummies(
            pd.concat([ref[c].astype(str).fillna("other") for c in cat_present],
                      axis=1), drop_first=True, dtype=float)
        dummies = dummies.reindex(columns=ref_dummies.columns, fill_value=0.0)
    num_part = base[[c for c in base.columns if c not in cat]].astype(float)
    X = pd.concat([num_part, dummies], axis=1)
    return X, list(X.columns)


def fit_glm(X: pd.DataFrame, rate: pd.Series, exposure: pd.Series,
            power: float = 1.5, alpha: float = 1e-4, max_iter: int = 2000):
    """Tweedie GLM, log link, exposure weights.

    `rate` is cost/exposure; `exposure` is the weight. Predictions are
    `exposure * rate_hat`.

    Columns are standardised for the optimisation and the coefficients are
    converted back to original units afterwards. This is not optional: the raw
    design spans `driver_age` ~50 to `sum_insured` ~6e8, an ill-conditioned
    Hessian that makes L-BFGS stop at the starting point with every
    coefficient exactly zero (observed 2026-10-21, "converged in 1 iterations").
    """
    from sklearn.linear_model import TweedieRegressor

    mu = X.to_numpy(float).mean(axis=0)
    sd = X.to_numpy(float).std(axis=0)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    Xs = (X.to_numpy(float) - mu) / sd

    m = TweedieRegressor(power=power, link="log", alpha=alpha,
                         fit_intercept=True, max_iter=max_iter, tol=1e-8)
    w = exposure.to_numpy(float)
    y = rate.to_numpy(float)
    ok = np.isfinite(w) & np.isfinite(y) & (w > 0)
    m.fit(Xs[ok], y[ok], sample_weight=w[ok])

    # back-transform to original units:  log y = b0' + sum(b_i' x_i)
    m.coef_ = np.asarray(m.coef_).ravel() / sd
    m.intercept_ = float(m.intercept_ - np.sum(np.asarray(m.coef_) * mu))
    m._mu, m._sd = mu, sd
    m.scale_ = sd          # keep the transform visible for diagnostics
    return m


def predict_cost(model, X: pd.DataFrame, exposure: pd.Series,
                 floor: float = 1e-3) -> np.ndarray:
    """Row-level expected cost: exposure x predicted annual rate.

    `model.coef_` / `model.intercept_` are already back-transformed to original
    units by `fit_glm`, so the raw design is passed through unchanged.
    """
    rate = model.predict(X.to_numpy(float))
    return np.maximum(rate * exposure.to_numpy(float), floor)


def relativities(model, columns: list[str], num: list[str]) -> pd.DataFrame:
    """exp(coefficient) per term, for the write-up.

    Numeric terms are not directly comparable to a categorical relativity
    (they are per-unit), so they are reported separately with the coefficient.
    """
    coef = np.asarray(model.coef_).ravel()
    out = pd.DataFrame({"term": columns, "coef": coef,
                        "relativity": np.exp(coef)})
    out["kind"] = np.where(out["term"].isin(num), "numeric", "categorical")
    return out.sort_values("relativity", ascending=False).reset_index(drop=True)
