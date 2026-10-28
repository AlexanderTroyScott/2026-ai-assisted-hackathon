# Part 2 Actuarial Report — Emerging Market Expansion (`iaa-ai-assisted-hackathon-2026-part-2`)

**Status:** Verified empirically from `developed/` and `emerging/` Parquet tables (2026-10-23).
**Executable pipeline:** `src/emerging.py`, `scripts/run_emerging.py`, `tests/test_emerging.py`.
**Leaderboard:** best is `emerging-exp-025` at **0.16994**
Public on `iaa-ai-assisted-hackathon-2026-part-2` (rank 4 of 33 at 2026-10-28;
D-076: property NB/ren cohort ridge on top of the D-075 tuned offset layer.
Record chain: `emerging-exp-002` base 0.16297 → `emerging-exp-019` 0.16836
(D-069: 2025 forward index built into the pipeline — per-line/cohort trend +
transfer-book reconciliation at scoring; `emerging-exp-016`/`017` post-hoc
recipe scored 0.16833) → `emerging-exp-024` 0.16879 (D-075 tuning) →
`emerging-exp-025` 0.16994). Consistent with Part 1
experience, the leaderboard score sits above the 5-Fold CV estimate (+0.14045)
— CV folds train on 4/5 of two thin years, so the honest CV is the conservative
number.

---

## 1. Inventory & Structural Comparison (`developed/` vs `emerging/`)

| Dimension | `developed/` (Part 1) | `emerging/` (Part 2) | Adaptation Impact |
|---|---|---|---|
| **Training policy years** | 2021–2024 (4 years) | 2023–2024 (2 years) | 22× smaller training volume in `emerging/` |
| **Training rows** | 2,409,311 (M: 980k, B: 708k, C: 722k) | 109,296 (M: 47,283, B: 27,447, C: 34,566) | Thin per-LOB volume; unconstrained trees overfit |
| **Matched claims** | 162,736 (414 orphans) | 9,102 non-repudiated matched (36 orphans: 6 in 2021, 28 in 2022, 2 in 2023) | Only ~1,000 claims in Buildings and ~1,000 in Contents |
| **2025 Test rows** | 626,639 | 209,063 (M: 90,521, B: 52,295, C: 66,247) | 2025 test book is ~3.8× annual training volume due to broker portfolio takeover |
| **2025 New Business share** | 15.2% | **76.0%** (vs 13.8% in 2023–2024 train) | 68% of test rows are transferred policies starting `2025-01-02..2025-01-15` with zero insurer claim history |
| **Regions** | `R01`–`R12` | `E01`–`E06` | Market-specific geographic risk levels |
| **Age columns** | Continuous `driver_age`, `policyholder_age` | Banded `driver_age_band`, `policyholder_age_band` (`17-24`, `25-34`, `35-49`, `50-64`, `65+`) | Bin `developed/` ages into the exact 5 bands + midpoint `age_num` (`21, 30, 42, 57, 72`) |
| **Gender** | Not collected | `M` (49.97%) / `F` (50.03%) | Strong signal in `emerging/`: Motor `M` rate is +34% higher (`467` vs `348`, LR `0.76` vs `0.58`); Contents `F` rate is +34% higher (`102` vs `76`) |
| **Occupation** | 20 clean categories | 80 free-text strings (`'Nurse'`, `'nurse'`, `'Nurse '`, `'Self-employed nurse'`) | `.str.strip().str.lower().str.removeprefix('self-employed').str.strip()` maps **100%** of strings to the 20 `developed/` occupations + `is_self_employed` flag |
| **Construction type** | `brick`, `stone`, `timber`, `non_standard` | `BRK`, `STN`, `WOD`, `MIX` | 1-to-1 canonical mapping (`BRK→brick`, `STN→stone`, `WOD→timber`, `MIX→non_standard`) |
| **Dropped features** | `credit_band`, `annual_mileage`, `telematics_opt_in`, `telematics_score`, `date_of_birth` | Absent in `emerging/` | Excluded from the cross-market transfer specification |
| **Weather coverage** | 2021–2025 (100% join) | **2023–2025 (100% join, 136,914 addresses × 3 yrs)** | 2025 emerging weather has +65% `freezing_days` (`17.5 → 28.9`), −33% `hail_days` (`17.1 → 11.4`), −9% `rainfall_mm` (`1375 → 1251`) |

---

## 2. Data Quality & Current Pricing Misalignments in `emerging/`

1. **Same structural data defects as `developed/`:**
   - **Exposure corruption:** 114 training rows in `emerging/` (`47` motor, `33` buildings, `34` contents) have `exposure > 1.5` (recorded in days). Always recompute `exposure = (period_end - period_start + 1) / 365`.
   - **Annualised `written_premium`:** `corr(written_premium, exposure)` is `−0.031` to `+0.010` (~0). Earned premium for a row is `written_premium × exposure`.
   - **`sum_insured` >10M artefact:** 53 training and 104 test buildings rows (and 56 train / 134 test contents rows) sit above 10M (up to 568M) with a dead zone between 1M and 10M. Flagged via `si_artefact` and capped at `1e7` (`sum_insured_clean`).
2. **Severe misalignments in the insurer's current `emerging/` pricing (`written_premium`):**
   - **Contents `flood_zone` is unpriced:** `written_premium` per exposure is flat at `125–128 CU` across `flood_zone` 1 to 5, while observed claim cost per exposure rises **6.6×** from `63.2 CU` (`flood_zone=1`, LR `0.49`) to **`417.7 CU` (`flood_zone=5`, LR `3.33`)**.
   - **Motor `driver_age_band = 25-34` is underpriced:** Priced at `462 CU` (same as `35-49` at `456 CU` and `50-64` at `459 CU`), but carries `421.6 CU` observed claim rate (LR **`0.912`** vs `0.60–0.62` for `35-64`).
   - **Motor `third_party_fire_theft` is overpriced:** Priced at `764.7 CU` (above `comprehensive` at `596.6 CU`), but has less than half the claim frequency (`0.098` vs `0.204`, LR `0.438`).
   - **Motor `gender = M` is underpriced:** Priced only +3% above `F` (`617` vs `599 CU`), but carries +34% higher claim cost (`467.4` vs `347.7 CU`, LR `0.757` vs `0.580`).
3. **Claim-severity outlier review (D-078, 2026-10-28): real heavy tail, zero corruptions.**
   - **Integrity clean:** no negative amounts, `paid ≤ incurred` everywhere, closed claims exactly `paid == incurred`, all 120 repudiated claims carry zero amounts, no impossible dates (max report lag 653 days, nothing beyond the 2024-12-31 valuation).
   - **Concentration is genuine severity, not error:** buildings top-10 claims = **25.6%** of line cost (all fire total losses; max 361k vs p99 138.5k), motor top-10 = 5.7% (largest: open bodily_injury 155.7k incurred / 19.2k paid, 8.4× motor p99), contents top-10 = 12.7%. The top 10 training rows carry 8.3% of total `y_ult`.
   - **The giants are economics:** only 4 claims exceed their policy `sum_insured`, all marginally (fire at the SI limit ±0.05%, one flood +7.4%); **none of the 25 largest claims sits on a `si_artefact` row** — the corrupt-SI population carries no claim signal (same as `developed/`, D-043).
   - **Caution for development:** open-claim incurred is concentrated — contents top-3 open claims = 44.5% of open contents incurred (59 open claims), buildings 24.3%, motor 8.3% — so the line-level `EM_LDF` factors rest on a handful of property claims. This is residual label uncertainty, not fixable from this data (the sparsity behind D-072/D-074).
   - **Metric:** the top 10 rows carry 4.9% of total Tweedie deviance (buildings 13.9%); the 50 rows with `y_ult > 50k` carry 10.1% — part of board noise is irreducible claim randomness. **No capping/winsorisation** (trees are robust; capping would bias Σpred ≈ Σy); guardrail tests lock the integrity invariants.

---

## 3. Target Development (31 December 2024 Valuation Cutoff)

From `emerging/claims.parquet` (origin year 2023 reported in 2023 vs 2024):
- **Motor:** 2023 `age 0 = 6,522.8k`, `age 1 = 1,642.4k` $\implies f_{0\to 1} = 1.2518$. With mature tail $f_{1\to\text{ult}} = 1.0301$, $\text{LDF}_{2023} = 1.0301$ and $\text{LDF}_{2024} = 1.2895$.
- **Buildings:** 2023 `age 0 = 5,518.6k`, `age 1 = 330.1k` $\implies f_{0\to 1} = 1.0598$. With $f_{1\to\text{ult}} = 1.0030$, $\text{LDF}_{2023} = 1.0030$ and $\text{LDF}_{2024} = 1.0630$.
- **Contents:** 2023 `age 0 = 1,252.3k`, `age 1 = 33.4k` $\implies f_{0\to 1} = 1.0267$. With $f_{1\to\text{ult}} = 1.0040$, $\text{LDF}_{2023} = 1.0040$ and $\text{LDF}_{2024} = 1.0308$.

---

## 4. Transfer Learning + Credibility-Shrinkage Offset Adaptation (`D-066`)

Because `emerging/` has only 109,296 training rows (~1,000 claims in Buildings and Contents), training a gradient-boosted tree from scratch on `emerging/` alone severely overfits (`5-Fold Policy-Grouped CV D² = 0.10734`, underperforming even `written_premium × LOB LR` at `0.10965`).

We therefore implement a two-stage **Actuarial Transfer + Offset Adaptation** architecture (`src/emerging.py`, `scripts/run_emerging.py`):
1. **Stage 1 — `Developed` Prior Models (2.41M rows):** Fit Per-LOB Tweedie rate models on `developed/` restricted to the canonicalised shared feature set (`age_band_5`, `age_num`, `occ_clean`, `construction_type`, property/geo/weather/vehicle/excess factors), blending a `wp_rel`-informed prior (60%) and a pure physical/demographic prior without `written_premium` (40%), recalibrated to each `emerging/` LOB's total ultimate cost.
2. **Stage 2 — Credibility-Shrinkage Offset Adaptation (`init_score = log(dev_prior_cal)`):** Fit heavily regularized, ultra-shallow Per-LOB Tweedie offset trees (`num_leaves=3`, `min_data_in_leaf=600`, `lambda_l2=100`, `learning_rate=0.02`, `20–35` trees) on the emerging adaptation features (`region`, `gender`, `age_band_5`, `channel`, `cover_type`, `flood_zone`, `voluntary_excess`, `new_business`, `wp_rel`, `is_self_employed`, `pol_claims_all`, `prior_claims_5y`, `wildfire_index`, `hail_days`).

### 5-Fold Policy-Grouped Cross-Validation Results on `emerging/` (109,296 rows)

| Model | Overall 5-Fold OOF $D^2$ | Motor $D^2$ | Buildings $D^2$ | Contents $D^2$ |
|---|---:|---:|---:|---:|
| 1. Constant rate per LOB | `+0.08039` | `+0.02738` | `+0.01421` | `+0.03802` |
| 2. `written_premium × LOB LR` | `+0.10965` | `+0.04861` | `+0.06332` | `+0.06272` |
| 3. `Emerging`-only Per-LOB rate LGBM (`lv=6`) | `+0.10734` | `+0.06244` | `+0.03916` | `+0.07541` |
| 4. Zero-shot `Developed` Prior (recalibrated per LOB) | `+0.13855` | `+0.06964` | `+0.10151` | `+0.10478` |
| **5. Offset-Adapted Transfer Model (`emerging-exp-001`)** | **`+0.14045`** | **`+0.07080`** | **`+0.10515`** | **`+0.10450`** |

---

## 5. Commercial / Office Premium Build-Up for the 20 `reference_policies`

From `emerging/expenses.parquet`:
- Fixed expense per policy: $E_{\text{fixed}} = 85.00\text{ CU}$ (vs $45.00\text{ CU}$ in `developed`)
- Variable expense: $v = 15\%$ of office premium (vs $12\%$ in `developed`)
- Commission: $c_{\text{broker}} = 20\%$, $c_{\text{direct}} = 0\%$ of office premium
- Premium tax: $t = 8\%$ of office premium (vs $12\%$ in `developed`)
- Target profit margin: $m = 8\%$ of office premium (vs $5\%$ in `developed`)
- Catastrophe reinsurance: $r_{\text{cat}} = 12\%$ of property (`buildings` and `contents`) risk premium (vs $6\%$ in `developed`; $0\%$ for `motor`)

The grossing-up formula for the annualised Office Premium (directly comparable to the annualised `written_premium` column) and row-period pro-rated Office Premium is:

$$P_{\text{office, annual}} = \frac{R_{\text{annual}} \cdot (1 + r_{\text{cat}}) + 85.00}{1 - (0.15 + c_{\text{channel}} + 0.08 + 0.08)}, \qquad P_{\text{office, row}} = P_{\text{office, annual}} \times \text{exposure}$$

where the denominator is **$0.49$ for `broker`** ($51\%$ total proportional loadings) and **$0.69$ for `direct`** ($31\%$ total proportional loadings).

### Complete 20-Policy Office Premium Table (`outputs/metrics/emerging_reference_office_premiums.csv`)

| `row_id` | `policy_id` | `line` | `channel` | `exposure` | Predicted Row Risk Premium ($R_{\text{row}}$) | Annualised Risk Premium ($R_{\text{ann}}$) | Cat Reins ($12\%$ Prop) | Fixed Expense ($85\text{ CU}$) | Prop Loading Share | Pro-Rated Row Office Premium | Annualised Office Premium ($P_{\text{office, ann}}$) | Current `written_premium` | Ratio ($P_{\text{office, ann}} / \text{WP}$) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `M00161473` | `M92909747` | motor | broker | 0.9644 | 1,184.74 | 1,228.50 | 0.00 | 85.00 | 0.51 | 2,585.13 | **2,680.60** | 1,131.78 | 2.37× |
| `M00176777` | `M42023395` | motor | broker | 0.9918 | 1,090.55 | 1,099.59 | 0.00 | 85.00 | 0.51 | 2,397.65 | **2,417.52** | 443.61 | 5.45× |
| `M00185311` | `M83715550` | motor | direct | 0.9644 | 1,215.96 | 1,260.87 | 0.00 | 85.00 | 0.31 | 1,881.06 | **1,950.53** | 872.31 | 2.24× |
| `M00167496` | `M61457154` | motor | broker | 0.9836 | 775.96 | 788.93 | 0.00 | 85.00 | 0.51 | 1,754.20 | **1,783.52** | 463.95 | 3.84× |
| `M00129413` | `M41089337` | motor | broker | 0.3315 | 205.95 | 621.26 | 0.00 | 85.00 | 0.51 | 477.81 | **1,441.34** | 354.15 | 4.07× |
| `M00118569` | `M75811658` | motor | direct | 0.9397 | 915.83 | 974.57 | 0.00 | 85.00 | 0.31 | 1,443.06 | **1,535.61** | 1,095.48 | 1.40× |
| `M00153851` | `M47069725` | motor | broker | 0.9616 | 371.16 | 385.96 | 0.00 | 85.00 | 0.51 | 924.29 | **961.15** | 414.72 | 2.32× |
| `B00229821` | `B92110463` | buildings | broker | 0.9918 | 277.78 | 280.09 | 33.61 | 85.00 | 0.51 | 806.98 | **813.67** | 285.60 | 2.85× |
| `B00218282` | `B74876592` | buildings | broker | 0.9726 | 326.63 | 335.83 | 40.30 | 85.00 | 0.51 | 915.29 | **941.07** | 230.80 | 4.08× |
| `B00250679` | `B05817658` | buildings | broker | 0.1096 | 34.69 | 316.51 | 37.98 | 85.00 | 0.51 | 98.29 | **896.92** | 180.72 | 4.96× |
| `B00223369` | `B73680946` | buildings | broker | 0.9945 | 376.74 | 378.82 | 45.46 | 85.00 | 0.51 | 1,033.64 | **1,039.33** | 204.56 | 5.08× |
| `B00215809` | `B88623113` | buildings | direct | 0.9753 | 490.54 | 502.94 | 60.35 | 85.00 | 0.31 | 916.39 | **939.56** | 237.22 | 3.96× |
| `B00215707` | `B75172079` | buildings | broker | 0.9863 | 197.86 | 200.60 | 24.07 | 85.00 | 0.51 | 623.34 | **631.99** | 272.44 | 2.32× |
| `B00217584` | `B13799940` | buildings | broker | 0.9671 | 1,226.70 | 1,268.40 | 152.21 | 85.00 | 0.51 | 2,971.64 | **3,072.66** | 323.36 | 9.50× |
| `C00259568` | `C16955632` | contents | broker | 1.0000 | 33.09 | 33.09 | 3.97 | 85.00 | 0.51 | 249.10 | **249.10** | 83.18 | 2.99× |
| `C00283470` | `C11841593` | contents | broker | 0.9726 | 69.08 | 71.03 | 8.52 | 85.00 | 0.51 | 326.62 | **335.82** | 117.97 | 2.85× |
| `C00270331` | `C97318122` | contents | broker | 0.9726 | 33.61 | 34.56 | 4.15 | 85.00 | 0.51 | 245.55 | **252.46** | 140.29 | 1.80× |
| `C00314425` | `C17515168` | contents | broker | 0.0685 | 5.98 | 87.38 | 10.49 | 85.00 | 0.51 | 25.56 | **373.18** | 102.87 | 3.63× |
| `C00302985` | `C36495409` | contents | broker | 0.9918 | 45.71 | 46.09 | 5.53 | 85.00 | 0.51 | 276.53 | **278.83** | 115.88 | 2.41× |
| `C00286287` | `C12450281` | contents | broker | 0.9671 | 145.58 | 150.53 | 18.06 | 85.00 | 0.51 | 500.52 | **517.54** | 269.23 | 1.92× |
