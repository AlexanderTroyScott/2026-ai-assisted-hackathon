#!/usr/bin/env python
"""Generate benchmark submissions for the IAA hackathon.

Kinds
-----
constant_rate : exposure x developed cost rate per LOB   (D-005 development applied)
premium_lr    : written_premium x developed LR per LOB   (D-015 premium benchmark)

Both are legitimate actuarial baselines, not strawmen: they are the level the
model has to beat. See docs/decisions.md and docs/data_findings.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data_raw"
OUT = ROOT / "outputs" / "submissions"

from src.dataset import LDF  # noqa: E402  single source of truth for D-005 factors

TEST_FILES = {
    "motor": "test_motor.parquet",
    "buildings": "test_buildings.parquet",
    "contents": "test_contents.parquet",
}
TRAIN_FILES = {
    "motor": "policies_motor.parquet",
    "buildings": "policies_buildings.parquet",
    "contents": "policies_contents.parquet",
}

# D-005 age-to-ultimate factors come from src.dataset.LDF (derived from the
# documented age-to-age table). Do not restate them here — a second copy
# drifted and over-developed 2023 buildings/contents labels.


def load_policies(line: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / TRAIN_FILES[line])
    df["ps"] = pd.to_datetime(df["period_start"])
    df["pe"] = pd.to_datetime(df["period_end"])
    df["exp_clean"] = ((df["pe"] - df["ps"]).dt.days + 1) / 365.0
    df["acc_year"] = df["ps"].dt.year
    return df


def load_claims(line: str) -> pd.DataFrame:
    c = pd.read_parquet(DATA / "claims.parquet")
    c = c[(c["line"] == line) & (c["status"] != "repudiated")].copy()
    c["loss"] = pd.to_datetime(c["loss_date"])
    c["cost"] = np.maximum(0.0, c["incurred_to_date"] - c["recoveries"])
    return c[["claim_id", "policy_id", "loss", "cost"]]


def attribute_claims(policies: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Sum claim cost onto policy rows by loss date. One row per policy row."""
    merged = policies[["row_id", "policy_id", "ps", "pe", "acc_year"]].merge(
        claims, on="policy_id", how="left"
    )
    matched = merged[(merged["loss"].isna()) | ((merged["loss"] >= merged["ps"]) & (merged["loss"] <= merged["pe"]))]
    dup = matched.dropna(subset=["claim_id"]).groupby("claim_id").size()
    if (dup > 1).any():
        raise SystemExit(f"FATAL: {(dup > 1).sum()} claims match more than one row")
    agg = matched.groupby("row_id").agg(y=("cost", "sum"), n=("claim_id", "count"))
    return policies[["row_id", "acc_year", "exp_clean", "written_premium"]].join(agg, on="row_id").fillna(
        {"y": 0.0, "n": 0}
    )


def developed_totals(attr: pd.DataFrame, line: str) -> dict[str, float]:
    by_year = attr.groupby("acc_year").agg(y=("y", "sum"), exp=("exp_clean", "sum"), prem=("written_premium", "sum"))
    dev = by_year["y"] * pd.Series(LDF[line]).reindex(by_year.index)
    return {
        "cost": float(dev.sum()),
        "exposure": float(by_year["exp"].sum()),
        "premium": float(by_year["prem"].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["constant_rate", "premium_lr"], required=True)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pieces = []
    for line in ["motor", "buildings", "contents"]:
        pol = load_policies(line)
        attr = attribute_claims(pol, load_claims(line))
        tot = developed_totals(attr, line)
        if args.kind == "constant_rate":
            rate = tot["cost"] / tot["exposure"]
            print(f"{line:10s} developed cost/exp = {rate:.4f}")
        else:
            lr = tot["cost"] / tot["premium"]
            rate = lr
            print(f"{line:10s} developed LR = {lr:.4f}  (prem/exp {tot['premium']/tot['exposure']:.2f})")

        test = pd.read_parquet(DATA / TEST_FILES[line])
        ps = pd.to_datetime(test["period_start"])
        pe = pd.to_datetime(test["period_end"])
        exp = ((pe - ps).dt.days + 1) / 365.0
        if args.kind == "constant_rate":
            pred = exp.to_numpy() * rate
        else:
            pred = test["written_premium"].to_numpy() * rate
        pieces.append(pd.DataFrame({"row_id": test["row_id"], "prediction": np.maximum(pred, 1e-3)}))

    sub = pd.concat(pieces, ignore_index=True)
    sample = pd.read_parquet(DATA / "sample_submission.parquet")
    assert len(sub) == len(sample) == 626_639, (len(sub), len(sample))
    assert set(sub["row_id"]) == set(sample["row_id"]), "row_id set mismatch"
    assert sub["prediction"].min() > 0, "predictions must be strictly positive"

    path = OUT / f"{args.kind}.csv"
    sub.to_csv(path, index=False)
    print(f"\nwrote {path}  rows={len(sub)}  mean={sub['prediction'].mean():.4f}  "
          f"p50={sub['prediction'].median():.4f}  max={sub['prediction'].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
