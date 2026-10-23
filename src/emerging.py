"""Part 2 — Emerging Market adaptation, transfer learning, and office premium build-up.

Key conventions (D-066):
  * Never join a `developed/` table to an `emerging/` table by ID (IDs can repeat
    across markets for unrelated records). Transfer operates strictly in the
    canonicalised feature/model space.
  * Recompute `exposure = (period_end - period_start + 1) / 365` in both markets.
  * Canonicalise `emerging/` schema shifts to the shared feature space:
      - `driver_age_band` / `policyholder_age_band` -> 5 age bands + midpoint `age_num`
      - 80 free-text `occupation` strings -> 20 canonical occupations + `is_self_employed`
      - local `construction_type` codes (`BRK`, `STN`, `WOD`, `MIX`) ->
        (`brick`, `stone`, `timber`, `non_standard`)
      - `sum_insured > 1e7` artefact population -> `si_artefact` + `sum_insured_clean`
  * Develop 2023/2024 emerging training labels to ultimate (`EM_LDF`).
  * Build office premiums for the 20 `reference_policies` from `emerging/expenses.parquet`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV_DIR = ROOT / "developed"
EM_DIR = ROOT / "emerging"
LINES = ["motor", "buildings", "contents"]

AGE_BANDS = ["17-24", "25-34", "35-49", "50-64", "65+"]
AGE_MAP = {"17-24": 21.0, "25-34": 30.0, "35-49": 42.0, "50-64": 57.0, "65+": 72.0}

CT_MAP = {
    "brick": "brick", "brk": "brick",
    "stone": "stone", "stn": "stone",
    "timber": "timber", "t": "timber", "wod": "timber",
    "non_standard": "non_standard", "mix": "non_standard",
}

# Developed market LDFs (D-005 / D-033)
DEV_LDF = {
    "motor": {2021: 1.0000, 2022: 1.0030, 2023: 1.0301, 2024: 1.2598},
    "buildings": {2021: 1.0000, 2022: 1.0000, 2023: 1.0030, 2024: 1.0923},
    "contents": {2021: 1.0000, 2022: 1.0000, 2023: 1.0040, 2024: 1.0442},
}

# Emerging market LDFs: 0->1 measured directly from emerging/claims.parquet
# (2023 origin year: motor 1.2518, buildings 1.0598, contents 1.0267),
# multiplied by the mature 1->ult tail (1.0301, 1.0030, 1.0040).
EM_LDF = {
    "motor": {2023: 1.0301, 2024: round(1.2518 * 1.0301, 4)},
    "buildings": {2023: 1.0030, 2024: round(1.0598 * 1.0030, 4)},
    "contents": {2023: 1.0040, 2024: round(1.0267 * 1.0040, 4)},
}


def clean_occupation(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Canonicalise free-text emerging occupations into the 20 standard occupations
    plus a binary `is_self_employed` indicator.
    """
    raw = s.fillna("other").astype(str).str.strip()
    is_se = raw.str.lower().str.startswith("self-employed").astype(float)
    clean = raw.str.lower().str.removeprefix("self-employed").str.strip()
    clean = clean.replace({"": "other", "nan": "other", "none": "other"})
    return clean, is_se


def canonicalise_construction(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(CT_MAP).fillna("other")


def build_market_frame(folder: Path, split: str = "train", ldf_tab: dict | None = None) -> pd.DataFrame:
    """Load and canonicalise one market ('developed' or 'emerging') without cross-market joins."""
    folder = Path(folder)
    cust = pd.read_parquet(folder / "customers.parquet")
    cust["customer_id"] = cust["customer_id"].astype(str).str.zfill(8)
    cust["occ_clean"], cust["is_self_employed"] = clean_occupation(cust["occupation"])
    if "gender" not in cust.columns:
        cust["gender"] = "U"
    addr = pd.read_parquet(folder / "addresses.parquet")
    w = pd.read_parquet(folder / "weather.parquet")
    w["year"] = w["year"].astype(int)

    claims = pd.read_parquet(folder / "claims.parquet")
    dedup_keys = [
        "policy_id", "line", "peril", "loss_date", "report_date", "status",
        "paid_to_date", "incurred_to_date", "recoveries", "repairer_id", "third_party_id",
    ]
    claims = claims.drop_duplicates(subset=[k for k in dedup_keys if k in claims.columns], keep="first")
    claims = claims[claims["status"] != "repudiated"].copy()
    claims["loss"] = pd.to_datetime(claims["loss_date"])
    claims["rep"] = pd.to_datetime(claims["report_date"])
    claims["cost"] = np.maximum(0.0, claims["incurred_to_date"] - claims["recoveries"])

    frames = []
    for ln in LINES:
        fname = f"policies_{ln}.parquet" if split == "train" else f"test_{ln}.parquet"
        df = pd.read_parquet(folder / fname)
        df["customer_id"] = df["customer_id"].astype(str).str.zfill(8)
        df["ps"] = pd.to_datetime(df["period_start"])
        df["pe"] = pd.to_datetime(df["period_end"])
        df["exposure"] = ((df["pe"] - df["ps"]).dt.days + 1) / 365.0
        df["earned_premium"] = df["written_premium"] * df["exposure"]
        df["acc_year"] = df["ps"].dt.year
        df["line"] = ln
        df["is_test"] = split == "test"

        if "driver_age" in df.columns:
            df["age_num"] = pd.to_numeric(df["driver_age"], errors="coerce").astype(float)
            df["age_band_5"] = pd.cut(
                df["age_num"], [0, 24, 34, 49, 64, 200], labels=AGE_BANDS
            ).astype(str)
        elif "driver_age_band" in df.columns:
            df["age_band_5"] = df["driver_age_band"].astype(str)
            df["age_num"] = df["driver_age_band"].map(AGE_MAP).astype(float)
        elif "policyholder_age" in df.columns:
            df["age_num"] = pd.to_numeric(df["policyholder_age"], errors="coerce").astype(float)
            df["age_band_5"] = pd.cut(
                df["age_num"], [0, 24, 34, 49, 64, 200], labels=AGE_BANDS
            ).astype(str)
        elif "policyholder_age_band" in df.columns:
            df["age_band_5"] = df["policyholder_age_band"].astype(str)
            df["age_num"] = df["policyholder_age_band"].map(AGE_MAP).astype(float)

        if "construction_type" in df.columns:
            df["construction_type"] = canonicalise_construction(df["construction_type"])
        if "smoke_alarms" in df.columns:
            df["smoke_alarms"] = df["smoke_alarms"].map(
                {"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}
            )

        df = df.merge(addr, on="address_id", how="left").merge(cust, on="customer_id", how="left")
        df["weather_year"] = np.where(df["is_test"], 2025, df["acc_year"])
        df = df.merge(
            w, left_on=["address_id", "weather_year"], right_on=["address_id", "year"], how="left"
        )

        # Strictly lagged policy-level claim experience (loss < ps)
        m_exp = df[["row_id", "policy_id", "ps"]].merge(
            claims[claims["line"] == ln][["policy_id", "loss", "cost"]], on="policy_id", how="left"
        )
        m_lag = m_exp[m_exp["loss"] < m_exp["ps"]]
        g_all = m_lag.groupby("row_id").agg(
            pol_claims_all=("cost", "size"),
            pol_cost_all=("cost", "sum"),
            last_loss=("loss", "max"),
        )
        g_1y = (
            m_lag[(m_lag["ps"] - m_lag["loss"]).dt.days <= 365]
            .groupby("row_id")
            .agg(pol_claims_1y=("cost", "size"), pol_cost_1y=("cost", "sum"))
        )
        df = df.join(g_all, on="row_id").join(g_1y, on="row_id")
        df["days_since_claim"] = (df["ps"] - df["last_loss"]).dt.days
        df = df.fillna(
            {"pol_claims_all": 0, "pol_cost_all": 0.0, "pol_claims_1y": 0, "pol_cost_1y": 0.0}
        ).drop(columns=["last_loss"])

        si = pd.to_numeric(df.get("sum_insured", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["si_artefact"] = (si > 1e7).astype(float)
        df["sum_insured_clean"] = np.where(si.notna(), np.minimum(si, 1e7), np.nan)
        df["ncb_band"] = pd.cut(
            df.get("ncb_years", pd.Series(0, index=df.index)).fillna(0),
            [-1, 0, 4, 200],
            labels=[0, 1, 2],
        ).astype(float)
        df["vvg_log"] = np.log1p(
            df.get("vehicle_value_group", pd.Series(0, index=df.index)).fillna(0)
        )

        if split == "train" and ldf_tab is not None:
            m_t = df[["row_id", "policy_id", "ps", "pe", "acc_year"]].merge(
                claims[claims["line"] == ln], on="policy_id", how="inner"
            )
            m_t = m_t[(m_t["loss"] >= m_t["ps"]) & (m_t["loss"] <= m_t["pe"])]
            agg = m_t.groupby("row_id").agg(y=("cost", "sum"), n_claims=("claim_id", "count"))
            df = df.join(agg, on="row_id").fillna({"y": 0.0, "n_claims": 0})
            df["y_ult"] = [y * ldf_tab[ln].get(ay, 1.0) for y, ay in zip(df["y"], df["acc_year"])]
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    wm = out.groupby(["line", "acc_year"]).apply(
        lambda g: (g["written_premium"] * g["exposure"]).sum() / g["exposure"].sum(),
        include_groups=False,
    )
    denom = pd.MultiIndex.from_frame(out[["line", "acc_year"]]).map(wm).to_numpy(float)
    out["wp_rel"] = out["written_premium"] / denom
    return out


def build_office_premiums(
    ref_policies: pd.DataFrame,
    predictions: pd.DataFrame,
    expenses: pd.DataFrame,
    market: str = "emerging",
) -> pd.DataFrame:
    """Compute the commercial / office premium build-up for `reference_policies`
    from predicted risk premiums and `emerging/expenses.parquet`.

    Actuarial grossing-up equation:
      P_office = (R * (1 + r_cat) + E_fixed) / (1 - v_exp - c_comm - t_tax - m_profit)
    where:
      * R        : predicted expected ultimate claim cost (risk premium)
      * r_cat    : catastrophe reinsurance loading (share of property risk premium:
                   0.12 in emerging, 0.06 in developed; 0.0 for motor)
      * E_fixed  : fixed expense per policy (85.00 CU in emerging, 45.00 CU in developed)
      * v_exp    : variable expense share of premium (0.15 emerging, 0.12 developed)
      * c_comm   : channel commission share of premium (broker 0.20, direct 0.00;
                   comparison_site fallback 0.10)
      * t_tax    : premium tax share of premium (0.08 emerging, 0.12 developed)
      * m_profit : target profit margin share of premium (0.08 emerging, 0.05 developed)

    Reports both the row-period office premium (pro-rated fixed expense `E_fixed * exposure`)
    and the annualised office premium (`R_annual = R / exposure` plus full `E_fixed`),
    which is directly comparable to the annualised `written_premium` column.
    """
    exp_map = expenses.set_index("item")[market].to_dict()
    dev_map = expenses.set_index("item")["developed"].to_dict()

    fixed_exp = float(exp_map["fixed_expense_per_policy"])
    var_exp = float(exp_map["variable_expense"])
    tax_rate = float(exp_map["premium_tax"])
    cat_rate_prop = float(exp_map["cat_reinsurance"])
    profit_margin = float(exp_map["target_profit_margin"])

    comm_map = {
        "direct": float(exp_map["commission_direct"]),
        "broker": float(exp_map["commission_broker"]),
        "comparison_site": float(
            exp_map["commission_comparison_site"]
            if pd.notna(exp_map["commission_comparison_site"])
            else dev_map["commission_comparison_site"]
        ),
    }

    df = ref_policies[["row_id", "policy_id", "line", "channel", "period_start", "period_end", "written_premium"]].copy()
    ps = pd.to_datetime(df["period_start"])
    pe = pd.to_datetime(df["period_end"])
    df["exposure"] = ((pe - ps).dt.days + 1) / 365.0
    df = df.merge(predictions[["row_id", "prediction"]].rename(columns={"prediction": "risk_premium_row"}), on="row_id", how="left")

    df["risk_premium_annual"] = df["risk_premium_row"] / df["exposure"]
    df["cat_reinsurance_rate"] = np.where(df["line"].isin(["buildings", "contents"]), cat_rate_prop, 0.0)
    df["cat_reinsurance_row"] = df["risk_premium_row"] * df["cat_reinsurance_rate"]
    df["cat_reinsurance_annual"] = df["risk_premium_annual"] * df["cat_reinsurance_rate"]

    df["fixed_expense_annual"] = fixed_exp
    df["fixed_expense_row"] = fixed_exp * df["exposure"]

    df["variable_expense_rate"] = var_exp
    df["commission_rate"] = df["channel"].map(comm_map).astype(float)
    df["premium_tax_rate"] = tax_rate
    df["profit_margin_rate"] = profit_margin
    df["total_prop_loading_rate"] = (
        df["variable_expense_rate"] + df["commission_rate"] + df["premium_tax_rate"] + df["profit_margin_rate"]
    )
    denom = 1.0 - df["total_prop_loading_rate"]

    # Row-period office premium (pro-rated fixed expense) and full-fixed-expense variant
    df["office_premium_row_prorated"] = (
        df["risk_premium_row"] + df["cat_reinsurance_row"] + df["fixed_expense_row"]
    ) / denom
    df["office_premium_row_unprorated"] = (
        df["risk_premium_row"] + df["cat_reinsurance_row"] + df["fixed_expense_annual"]
    ) / denom

    # Annualised office premium (directly comparable to annualised written_premium)
    df["office_premium_annual"] = (
        df["risk_premium_annual"] + df["cat_reinsurance_annual"] + df["fixed_expense_annual"]
    ) / denom
    df["variable_expense_annual"] = df["office_premium_annual"] * df["variable_expense_rate"]
    df["commission_annual"] = df["office_premium_annual"] * df["commission_rate"]
    df["premium_tax_annual"] = df["office_premium_annual"] * df["premium_tax_rate"]
    df["profit_margin_annual"] = df["office_premium_annual"] * df["profit_margin_rate"]
    df["office_vs_written_ratio"] = df["office_premium_annual"] / df["written_premium"]
    return df
