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

# Age-to-ultimate development factors from the incurred triangle (D-005).
LDF = {
    "motor": {2021: 1.003, 2022: 1.005, 2023: 1.030, 2024: 1.259},
    "buildings": {2021: 1.002, 2022: 1.003, 2023: 1.092, 2024: 1.092},
    "contents": {2021: 1.002, 2022: 1.004, 2023: 1.044, 2024: 1.044},
}
# Forward severity factor to move 2024-basis labels/predictions to 2025 (D-008).
FORWARD_FACTOR = {"motor": 1.027, "buildings": 1.053, "contents": 1.054}


def load_policies(line: str, split: str = "train") -> pd.DataFrame:
    name = f"policies_{line}" if split == "train" else f"test_{line}"
    df = pd.read_parquet(DATA / f"{name}.parquet")
    df["ps"] = pd.to_datetime(df["period_start"])
    df["pe"] = pd.to_datetime(df["period_end"])
    df["exposure"] = ((df["pe"] - df["ps"]).dt.days + 1) / 365.0
    df["acc_year"] = df["ps"].dt.year
    df["line"] = line
    df["is_test"] = split == "test"
    return df


def load_claims() -> pd.DataFrame:
    c = pd.read_parquet(DATA / "claims.parquet")
    c = c[c["status"] != "repudiated"].copy()
    c["loss"] = pd.to_datetime(c["loss_date"])
    c["rep"] = pd.to_datetime(c["report_date"])
    c["cost"] = np.maximum(0.0, c["incurred_to_date"] - c["recoveries"])
    return c[["claim_id", "policy_id", "line", "peril", "loss", "rep", "cost"]]


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


def row_target(policies: pd.DataFrame, claims: pd.DataFrame, develop: bool = True,
               development: str = "lob") -> pd.DataFrame:
    """Attribute claims to rows and (optionally) develop to ultimate.

    development='lob'   uniform factor per (LOB, accident year) — D-005
    development='peril' level-preserving per-peril factors — src/development.py
    """
    key = ["row_id", "policy_id", "ps", "pe", "acc_year", "line"]
    merged = policies[key].merge(claims.drop(columns=["line"]), on="policy_id", how="inner")
    merged = merged[(merged["loss"] >= merged["ps"]) & (merged["loss"] <= merged["pe"])]
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


def build_frame(split: str = "train", development: str = "lob") -> pd.DataFrame:
    """Full modelling frame for one split, all LOBs concatenated."""
    claims = load_claims()
    frames = []
    for ln in LINES:
        pol = canonicalise(load_policies(ln, split))
        feats = experience_features(pol, claims) if split == "train" else experience_features(pol, claims)
        df = pol.merge(feats, on="row_id", how="left")
        if split == "train":
            tgt = row_target(pol, claims, develop=True, development=development)
            df = df.merge(tgt[["row_id", "y", "y_ult", "n_claims"]], on="row_id", how="left")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
