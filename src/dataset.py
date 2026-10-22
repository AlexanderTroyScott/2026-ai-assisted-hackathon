"""Dataset construction for the IAA hackathon.

Builds the modelling frame: one row per policy period, with a developed target and
lagged features. Every function here is pure (returns a new frame) so experiments
stay reproducible.

Key conventions (see docs/decisions.md):
  * exposure is always recomputed from dates (D-001)
  * claim amounts are attributed by loss date to exactly one row (D-002)
  * targets are developed to ultimate with LOB/accident-year LDFs (D-005)
  * any feature derived from claims is strictly lagged to period_start (D-014)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_raw"

LINES = ["motor", "buildings", "contents"]

# --- Development (D-005) -------------------------------------------------
# Valuation date of the incurred triangle: 31 Dec 2024.
EXTRACT_YEAR = 2024
ACCIDENT_YEARS = (2021, 2022, 2023, 2024)

# Volume-weighted age-to-age factors, docs/data_findings.md section 5.
# f[a] develops an accident year from age a to age a+1, where
# age = EXTRACT_YEAR - accident_year. Ages with no entry are fully developed.
AGE_TO_AGE = {
    "motor": {0: 1.223, 1: 1.027, 2: 1.003},
    "buildings": {0: 1.089, 1: 1.003},
    "contents": {0: 1.040, 1: 1.004},
}


def age_to_ultimate(factors: dict, age: int) -> float:
    """Product of the remaining age-to-age factors from `age` onwards (D-005).

    An accident year already at or beyond the last tabulated age is fully
    developed and gets 1.0 — the documented factors are the whole tail.
    """
    return float(np.prod([factors.get(a, 1.0) for a in range(age, max(factors) + 1)]))


# Age-to-ultimate development factors keyed by accident year. Derived, not
# typed in: age = EXTRACT_YEAR - AY, so 2024 -> age 0 (least developed) and
# 2021 -> age 3 (mature). tests/test_development.py pins these to the
# documented age-0 -> ultimate column.
LDF = {
    ln: {ay: age_to_ultimate(f, EXTRACT_YEAR - ay) for ay in ACCIDENT_YEARS}
    for ln, f in AGE_TO_AGE.items()
}

# --- 2025 level / trend (D-008, D-009) -----------------------------------
# Two independent documented indications, each a one-year 2025-vs-2024 step:
#   claims-index : 3-yr CAGR of the mix-adjusted severity index (D-007). The
#                  index holds the 2021 peril mix fixed, so mix shift does not
#                  contaminate the trend. data_findings.md section 6.
#   premium      : premium per exposure 2025 vs 2024 (data_findings.md section 6).
SEVERITY_INDEX = {
    "motor": {2021: 100.0, 2022: 105.0, 2023: 111.0, 2024: 112.0},
    "buildings": {2021: 100.0, 2022: 101.0, 2023: 108.0, 2024: 118.0},
    "contents": {2021: 100.0, 2022: 99.0, 2023: 106.0, 2024: 111.0},
}
PREMIUM_ANCHOR = {"motor": 1.027, "buildings": 1.053, "contents": 1.054}
CLAIMS_WEIGHT, PREMIUM_WEIGHT = 0.7, 0.3  # D-009 — status `proposed`, needs sign-off
TREND_SCENARIOS = ("premium", "low", "central", "high")


def forward_factors(scenario: str = "central") -> dict:
    """One-year 2025-vs-2024 level factor per LOB (applied at scoring, D-008).

    'central' is D-009's 70/30 blend of the claims-index CAGR and the premium
    anchor. 'low'/'high' bracket it with the two indications themselves, so the
    range never pretends to more precision than the two sources support.
    'premium' is the premium anchor on its own — the pre-D-034 level, kept so a
    submission can isolate a label change from a level change.

    D-009 is `proposed`: the *level* is a judgement call awaiting human sign-off
    (AGENTS.md), so callers must not treat 'central' as verified.
    """
    claims = {ln: (SEVERITY_INDEX[ln][2024] / SEVERITY_INDEX[ln][2021]) ** (1.0 / 3.0)
              for ln in LINES}
    if scenario == "central":
        return {ln: CLAIMS_WEIGHT * claims[ln] + PREMIUM_WEIGHT * PREMIUM_ANCHOR[ln]
                for ln in LINES}
    if scenario == "premium":
        return dict(PREMIUM_ANCHOR)
    if scenario == "low":
        return {ln: min(claims[ln], PREMIUM_ANCHOR[ln]) for ln in LINES}
    if scenario == "high":
        return {ln: max(claims[ln], PREMIUM_ANCHOR[ln]) for ln in LINES}
    raise ValueError(f"unknown trend scenario {scenario!r}; use one of {TREND_SCENARIOS}")


# Back-compat alias for the D-009 central case.
FORWARD_FACTOR = forward_factors("central")


# D-060: Bornhuetter-Ferguson scale factors, keyed (LOB, AY). BF ultimate =
# observed + (1 - 1/CDF) x ELR x earned_premium, ELR from the mature (2021-22)
# developed loss ratio. Scale = BF/chain-ladder ultimate, applied to labels.
# Measured: only 2024 is material (motor +1.7%, buildings -0.6%); every mature
# year has credibility ~0 so the scale is 1.0. Chain-ladder and BF agree to
# within ~2%, so this is a small refinement, not a level shift.
BF_SCALE = {
    "motor":     {2021: 1.000, 2022: 1.000, 2023: 1.000, 2024: 1.017},
    "buildings": {2021: 1.000, 2022: 1.000, 2023: 1.000, 2024: 0.994},
    "contents":  {2021: 1.000, 2022: 1.000, 2023: 1.000, 2024: 1.000},
}


def load_policies(line: str, split: str = "train") -> pd.DataFrame:
    name = f"policies_{line}" if split == "train" else f"test_{line}"
    df = pd.read_parquet(DATA / f"{name}.parquet")
    df["ps"] = pd.to_datetime(df["period_start"])
    df["pe"] = pd.to_datetime(df["period_end"])
    df["exposure"] = ((df["pe"] - df["ps"]).dt.days + 1) / 365.0
    # D-050: motor customer_id has lost its leading zeros on ~10% of rows, so
    # the customers join fails and every demographic field comes back NaN.
    # zfill(8) recovers 100% with zero collisions (measured 2026-10-22).
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str).str.zfill(8)
    # D-051: `written_premium` is an ANNUALISED rate, not the premium for this
    # row's period — corr(written_premium, exposure) is ~0, and a 1-day row
    # costs the same as a 365-day row. Earned premium for the row is the annual
    # rate pro-rated by exposure. Everything downstream must use this.
    df["earned_premium"] = df["written_premium"] * df["exposure"]
    df["acc_year"] = df["ps"].dt.year
    df["line"] = line
    df["is_test"] = split == "test"
    return df


DEDUP_KEYS = ["policy_id", "line", "peril", "loss_date", "report_date", "status",
              "paid_to_date", "incurred_to_date", "recoveries", "repairer_id",
              "third_party_id"]


def _dedup_claims(c: pd.DataFrame) -> pd.DataFrame:
    """D-051: 604 claims are exact duplicates differing only in claim_id.

    598 of the 604 extra copies carry 2022 loss dates, inflating 2022 claim cost
    by 1.63% and double-counting the training label for that policy row. Drop
    the later copy, keep the first.
    """
    keys = [k for k in DEDUP_KEYS if k in c.columns]
    n0 = len(c)
    c = c.drop_duplicates(subset=keys, keep="first").copy()
    dropped = n0 - len(c)
    if dropped:
        print(f"  D-051 dropped {dropped} duplicate claim rows")
    return c


def load_claims() -> pd.DataFrame:
    c = pd.read_parquet(DATA / "claims.parquet")
    c = _dedup_claims(c)
    c = c[c["status"] != "repudiated"].copy()
    c["loss"] = pd.to_datetime(c["loss_date"])
    c["rep"] = pd.to_datetime(c["report_date"])
    c["cost"] = np.maximum(0.0, c["incurred_to_date"] - c["recoveries"])
    return c[["claim_id", "policy_id", "line", "peril", "loss", "rep", "cost"]]


def load_denied_claims() -> pd.DataFrame:
    """Repudiated claims: zero cost by definition (D-004), but the *event* of
    having a claim refused is claim-occurrence information (D-031)."""
    c = pd.read_parquet(DATA / "claims.parquet")
    c = c[c["status"] == "repudiated"].copy()
    c["loss"] = pd.to_datetime(c["loss_date"])
    return c[["claim_id", "policy_id", "loss"]]


def denial_features(policies: pd.DataFrame, denied: pd.DataFrame) -> pd.DataFrame:
    """Strictly lagged denied-claim history (D-031).

    Same leakage rule as the experience block (D-014 / AGENTS.md rule 6): a row
    only sees denied claims with `loss < ps`. A denied claim inside the row's own
    period is outcome information about that period and must never be visible.
    """
    d = denied.merge(
        policies[["policy_id", "customer_id"]].drop_duplicates(), on="policy_id", how="inner"
    )[["customer_id", "policy_id", "loss"]].drop_duplicates()
    out = policies[["row_id"]].copy()

    cust = policies[["row_id", "customer_id", "ps"]].merge(
        d[["customer_id", "loss"]].drop_duplicates(), on="customer_id", how="left"
    )
    lag = cust[cust["loss"] < cust["ps"]]
    g = lag.groupby("row_id").agg(cust_denied_all=("loss", "size"), last_denied=("loss", "max"))
    out = out.join(g, on="row_id")

    pol = policies[["row_id", "policy_id", "ps"]].merge(d[["policy_id", "loss"]], on="policy_id", how="left")
    pol_lag = pol[pol["loss"] < pol["ps"]]
    g2 = pol_lag.groupby("row_id").agg(pol_denied_all=("loss", "size"))
    out = out.merge(g2, on="row_id", how="left")
    last = pd.to_datetime(out.set_index("row_id")["last_denied"])
    out["days_since_denied"] = (
        (policies.set_index("row_id")["ps"] - last).dt.days
    ).to_numpy()
    # counts default to 0; days_since_denied stays NaN for "no history" so the
    # model can distinguish it from "denied claim yesterday" (align fills -1).
    return out.drop(columns=["last_denied"]).fillna(
        {c: 0 for c in out.columns if c.startswith(("cust_denied", "pol_denied"))}
    )


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """D-001/D-quality: normalise dirty categorical levels."""
    if "construction_type" in df.columns:
        m = {
            "brick": "brick", "stone": "stone", "non_standard": "non_standard",
            "t": "timber", "timber": "timber",
        }
        df["construction_type"] = (
            df["construction_type"].astype(str).str.strip().str.lower().map(m).fillna("other")
        )
    if "smoke_alarms" in df.columns:
        df["smoke_alarms"] = df["smoke_alarms"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    return df


def write_orphan_log(claims: pd.DataFrame, policies: pd.DataFrame,
                     path: Path, expected: int = 414) -> pd.DataFrame:
    """Audit trail for D-002: claims whose loss date matches no policy row.

    Orphans are dropped from the target (they cannot be attributed to a policy
    period) but must remain visible, so write them out rather than discard
    silently.

    `policies` must be **all** lines concatenated and `claims` the **raw**
    table (all statuses): the verified count of 414 is book-wide and includes
    4 repudiated orphans, which `load_claims()` has already filtered out.
    `expected` is the verified figure from docs/data_findings.md — raise if it
    moves, because that means the join changed underneath us.

    The join is on (policy_id, line): policy_id is only unique within a line,
    so a policy_id-only merge fans out and invents orphans.
    """
    key = policies[["policy_id", "line", "ps", "pe"]].drop_duplicates()
    m = claims.merge(key, on=["policy_id", "line"], how="left")
    hit = (m["loss"] >= m["ps"]) & (m["loss"] <= m["pe"])
    matched_ids = set(m.loc[hit, "claim_id"])
    if (m.loc[hit].groupby("claim_id").size() > 1).any():
        raise SystemExit("FATAL: a claim matched more than one policy row (D-002)")

    orphans = claims.loc[~claims["claim_id"].isin(matched_ids),
                         ["claim_id", "policy_id", "line", "peril", "loss", "cost", "status"]]
    if expected is not None and len(orphans) != expected:
        raise SystemExit(
            f"FATAL: {len(orphans)} orphan claims, expected {expected} (D-002). "
            "The policy/claim join changed — re-verify docs/data_findings.md before proceeding."
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    orphans.to_csv(path, index=False)
    return orphans


def row_target(policies: pd.DataFrame, claims: pd.DataFrame, develop: bool = True,
               development: str = "lob") -> pd.DataFrame:
    """Attribute claims to rows and (optionally) develop to ultimate.

    development='lob'   uniform factor per (LOB, accident year) — D-005
    development='peril' level-preserving per-peril factors — src/development.py

    Orphan accounting (D-002) is book-level and lives in `write_orphan_log`;
    this function asserts the invariants that must hold for the rows it returns.
    """
    key = ["row_id", "policy_id", "ps", "pe", "acc_year", "line"]
    # Join on (policy_id, line), not policy_id alone: policy_id is only unique
    # within a line. Dropping claims["line"] and matching on policy_id happens
    # to work today (no policy_id spans two lines) but silently breaks if that
    # ever stops holding. Same key as write_orphan_log.
    merged = policies[key].merge(claims, on=["policy_id", "line"], how="inner")
    merged = merged[(merged["loss"] >= merged["ps"]) & (merged["loss"] <= merged["pe"])]

    # D-002: every claim must land on exactly one row.
    dup = merged.groupby("claim_id").size()
    if (dup > 1).any():
        raise SystemExit(f"FATAL: {(dup > 1).sum()} claims match more than one policy row")
    if develop and development == "peril":
        from src.development import claim_factors
        merged = merged.copy()
        merged["cost_dev"] = merged["cost"] * claim_factors(merged, LDF)
        agg = merged.groupby("row_id").agg(
            y=("cost", "sum"), y_ult=("cost_dev", "sum"),
            n_claims=("claim_id", "count"),
            n_theft=("peril", lambda s: int((s == "theft").sum())),
            n_large=("cost", lambda s: int((s > 10_000).sum())),
        )
        out = policies[["row_id"]].join(agg, on="row_id").fillna(
            {"y": 0.0, "y_ult": 0.0, "n_claims": 0, "n_theft": 0, "n_large": 0})
        return out
    agg = merged.groupby("row_id").agg(
        y=("cost", "sum"), n_claims=("claim_id", "count"),
        n_theft=("peril", lambda s: int((s == "theft").sum())),
        n_large=("cost", lambda s: int((s > 10_000).sum())),
    )
    out = policies[["row_id"]].join(agg, on="row_id").fillna({"y": 0.0, "n_claims": 0, "n_theft": 0, "n_large": 0})
    if develop:
        ldf = policies["line"].map({ln: pd.Series(LDF[ln]) for ln in LDF})
        # map accident-year factor per row
        factor = np.ones(len(policies), dtype=float)
        for ln in LDF:
            mask = policies["line"].to_numpy() == ln
            ay = policies["acc_year"].to_numpy()
            for year, f in LDF[ln].items():
                factor[mask & (ay == year)] = f
        # D-060: Bornhuetter-Ferguson blends the chain-ladder development with an
        # a-priori (premium x ELR) for immature years. The scale is ~1.0 except
        # 2024 (motor +1.7%, buildings -0.6%) because mature years have zero
        # credibility and the developed LR already sits near the mature ELR.
        if development == "bf":
            bf = policies["line"].map({ln: pd.Series(BF_SCALE[ln]) for ln in BF_SCALE})
            bf_arr = np.ones(len(policies), dtype=float)
            for ln in BF_SCALE:
                mask = policies["line"].to_numpy() == ln
                ay = policies["acc_year"].to_numpy()
                for year, f in BF_SCALE[ln].items():
                    bf_arr[mask & (ay == year)] = f
            factor = factor * bf_arr
        out["y_ult"] = out["y"] * factor
    else:
        out["y_ult"] = out["y"]
    return out


def experience_features(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Strictly lagged claim history (D-014): only losses before period_start.

    Built at customer level (cross-LOB) and at policy level.
    """
    cust = policies[["row_id", "policy_id", "customer_id", "ps"]].merge(
        claims[["claim_id", "policy_id", "loss", "cost"]], on="policy_id", how="left"
    )
    # customer-level history: any claim on any of the customer's policies
    cust_all = policies[["row_id", "customer_id", "ps"]].merge(
        claims[["claim_id", "policy_id", "loss", "cost"]].merge(
            policies[["policy_id", "customer_id"]].drop_duplicates(), on="policy_id", how="inner"
        )[["customer_id", "loss", "cost"]].drop_duplicates(),
        on="customer_id", how="left",
    )
    lagged = cust_all[cust_all["loss"] < cust_all["ps"]]
    for window, label in [(365, "1y"), (730, "2y"), (1460, "4y"), (10**9, "all")]:
        sel = lagged[(lagged["ps"] - lagged["loss"]).dt.days <= window]
        g = sel.groupby("row_id").agg(
            **{f"cust_claims_{label}": ("cost", "size"), f"cust_cost_{label}": ("cost", "sum")}
        )
        cust = cust.merge(g, on="row_id", how="left")

    # policy-level history
    pol_lag = cust[(cust["loss"] < cust["ps"])]
    g = pol_lag.groupby("row_id").agg(pol_claims_all=("cost", "size"), pol_cost_all=("cost", "sum"))
    out = policies[["row_id"]].join(g, on="row_id")
    for col in [c for c in cust.columns if c.startswith("cust_")]:
        out = out.merge(cust[["row_id", col]].drop_duplicates("row_id"), on="row_id", how="left")

    # recency
    rec = pol_lag.groupby("row_id")["loss"].max().rename("last_claim_date")
    out = out.merge(rec, on="row_id", how="left")
    out["days_since_claim"] = (
        (policies.set_index("row_id")["ps"] - out.set_index("row_id")["last_claim_date"]).dt.days
    ).to_numpy()
    return out.fillna({c: 0 for c in out.columns if c.startswith(("cust_", "pol_"))}).drop(
        columns=["last_claim_date"], errors="ignore"
    )


# Static rating attributes from the dimension tables. Never claim-derived, so
# no lag is required (AGENTS.md rule 6 applies to claims-derived quantities).
# `prior_claims_5y` is the customer's declared history as quoted at pricing time.
ADDRESS_COLS = [
    "region", "urban_density", "flood_zone", "distance_to_river_km", "elevation_m",
    "soil_type", "year_built", "property_type", "roof_type", "basement",
    "trees_within_5m", "listed_building", "bedrooms", "energy_rating", "solar_panels",
]
CUSTOMER_COLS = [
    "occupation", "marital_status", "household_size", "credit_band",
    "prior_claims_5y", "licence_points", "years_at_address", "customer_since_year", "marketing_segment",
]


def attach_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Join address and customer rating attributes onto a policy frame.

    The policy tables carry `address_id` / `customer_id` but not the attribute
    columns themselves; without this join every property-geo and demographic
    rating factor is silently absent from the model matrix (D-030).
    """
    addr = pd.read_parquet(DATA / "addresses.parquet", columns=["address_id"] + ADDRESS_COLS)
    cust = pd.read_parquet(DATA / "customers.parquet", columns=["customer_id"] + CUSTOMER_COLS)
    n = len(df)
    out = df.merge(addr, on="address_id", how="left").merge(cust, on="customer_id", how="left")
    assert len(out) == n, "attribute join must not duplicate policy rows"
    return out


def load_raw_claims() -> pd.DataFrame:
    """All claims including repudiated, for book-level audits like D-002.

    `load_claims()` drops repudiated rows (they carry zero cost, D-004), but
    4 of the 414 orphans are repudiated — an audit built on the filtered frame
    would report 410 and silently disagree with docs/data_findings.md.
    """
    c = pd.read_parquet(DATA / "claims.parquet")
    c["loss"] = pd.to_datetime(c["loss_date"])
    c["cost"] = np.maximum(0.0, c["incurred_to_date"] - c["recoveries"])
    return c


def build_frame(split: str = "train", development: str = "lob") -> pd.DataFrame:
    """Full modelling frame for one split, all LOBs concatenated."""
    claims = load_claims()
    if split == "train":
        # D-002: orphan audit is book-level — written once over all lines, not
        # per LOB, and over the raw claims table (4 of the 414 orphans are
        # repudiated and already filtered out of `claims`).
        all_pol = pd.concat(
            [load_policies(ln, split)[["policy_id", "line", "ps", "pe"]] for ln in LINES],
            ignore_index=True,
        )
        write_orphan_log(load_raw_claims(), all_pol, ROOT / "outputs" / "dropped_claims.csv")
    frames = []
    for ln in LINES:
        pol = attach_attributes(canonicalise(load_policies(ln, split)))
        feats = experience_features(pol, claims)
        feats = feats.merge(denial_features(pol, load_denied_claims()), on="row_id", how="left")
        df = pol.merge(feats, on="row_id", how="left")
        if split == "train":
            tgt = row_target(pol, claims, develop=True, development=development)
            df = df.merge(tgt[["row_id", "y", "y_ult", "n_claims"]], on="row_id", how="left")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
