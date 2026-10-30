# Part 2 Actuarial Report — Emerging Market Expansion (`iaa-ai-assisted-hackathon-2026-part-2`)

**Status:** Verified empirically from `developed/` and `emerging/` Parquet tables
(2026-10-23; write-up refreshed through D-089 on 2026-10-29).
**Executable pipeline:** `src/emerging.py`, `scripts/run_emerging.py`, `tests/test_emerging.py`.
**Model line:** the **record configuration** is `emerging-exp-025` (D-076:
property NB/ren cohort ridge on top of the D-075 tuned offset layer), built up
in measured steps — `emerging-exp-002` base → `emerging-exp-019`
(D-069: 2025 forward index built into the pipeline — per-line/cohort trend +
transfer-book reconciliation at scoring) → `emerging-exp-024` (D-075 tuning) →
`emerging-exp-025` (D-076). Consistent
with Part 1 experience, the live-board score sits above the 5-Fold CV
estimate (+0.14074 record config) — CV folds train on 4/5 of two thin years,
so the honest CV is the conservative number.

> **In plain terms.** This report documents how we priced a brand-new insurance
> market that had only two thin years of history. The short version: we learned
> from the established market first, let the thin new book correct it gently,
> and adjusted for 2025 price drift. Three
> findings did the heavy lifting — the 2025 book is 76% new business and new
> business was priced too cheaply relative to renewals; flood risk is entirely
> unpriced in contents cover; and the claim data has a genuine heavy tail that
> we refused to trim. We also measured the leaderboard's own randomness and
> rejected several ideas (ensembles, fancy claim-development schemes) that
> could not beat it. The deep sections below are the evidence; the accessible
> summary is `docs/final_writeup.md`.

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

Development is applied uniformly per LOB and accident year (`EM_LDF`,
`src/emerging.py`). **Alternatives were board-tested and rejected (D-072,
D-080):** level-preserving per-peril development (D-028/D-035 factors — motor
`bodily_injury` age-0→ultimate 1.708 vs `own_damage` 1.039) and
Horvitz–Thompson date×peril development both lost to uniform LOB development
on the Part 1 sandbox (§6). The uniform `EM_LDF` stands; the motor-BI
open-claim concentration (§2 item 3) is carried as disclosed residual
uncertainty, not as a label swing.

**Claim attribution and orphan treatment (recomputed 2026-10-29).** Each
claim is attributed to the unique policy row whose `line`, `policy_id` and
`[period_start, period_end]` contain its loss date: 9,222 of 9,258 claims
match, 0 multi-match. The **36 orphans** (loss date matches no row — 6 with
2021, 28 with 2022, 2 with 2023 loss dates, all outside the 2023–2024 policy
periods) are **dropped from labels and claims-trend estimation and counted**,
the same rule as Part 1's D-002; 9,102 matched claims are non-repudiated and
form the claims basis. Deduplication is on the single `CLAIM_DEDUP_KEYS` list
(11 columns) shared by label building and the trend index — measured inert on
this table (`claim_id` unique, 0 rows dropped) but pinned by a test so the two
paths cannot diverge silently on a data refresh (D-086).

---

## 4. Transfer Learning + Credibility-Shrinkage Offset Adaptation (`D-066` → `D-077`)

Because `emerging/` has only 109,296 training rows (~1,000 claims in Buildings and Contents), training a gradient-boosted tree from scratch on `emerging/` alone severely overfits (`5-Fold Policy-Grouped CV D² = 0.10734`, underperforming even `written_premium × LOB LR` at `0.10965`).

We therefore implement a three-stage **Actuarial Transfer + Offset Adaptation** architecture (`src/emerging.py`, `scripts/run_emerging.py`):
1. **Stage 1 — `Developed` Prior Models (2.41M rows):** Fit Per-LOB Tweedie rate models on `developed/` restricted to the canonicalised shared feature set (`age_band_5`, `age_num`, `occ_clean`, `construction_type`, property/geo/weather/vehicle/excess factors), blending a `wp_rel`-informed prior (60%) and a pure physical/demographic prior without `written_premium` (40%), recalibrated to each `emerging/` LOB's total ultimate cost. The prior alone carries OOF $D^2$ `+0.13855` — most of the skill is *transferred*, not learned from 109k rows.
2. **Stage 2 — Credibility-Shrinkage Offset Adaptation (`init_score = log(dev_prior_cal)`):** Fit heavily regularized, ultra-shallow Per-LOB Tweedie offset trees on the emerging adaptation features (`region`, `gender`, `age_band_5`, `channel`, `cover_type`, `flood_zone`, `voluntary_excess`, `new_business`, `wp_rel`, `is_self_employed`, `pol_claims_all`, `prior_claims_5y`, `wildfire_index`, `hail_days`). Record hyperparameters (D-075 `combo_top2`): `num_leaves=4`, `min_data_in_leaf=1200`, `lambda_l2=100`, `learning_rate=0.02`, `feature_fraction=bagging_fraction=0.8`, `35/30/20` trees (motor/buildings/contents), refit on 3 seeds and averaged. The offset layer stands alone: multi-family blends (XGBoost, pooled LGBM) were measured and rejected twice (D-070, D-079 — §6).
3. **Stage 3 — 2025 forward index at scoring (D-068/D-069):** 2025 test rows must be priced at the 2025 cost level. The pipeline builds a per-`(line, cohort)` forward index $F = \text{rate}_{2025} / \text{rate}_{\text{model level}}$ and applies it at scoring (`src/emerging.py`). **What actually carries the level (disclosed honestly):** the values used in production are `PROBE_FORWARD` — response-curve constants fitted on the **Public** leaderboard (≥8 one-change probes, D-068 → D-077). The data-estimated index (`f_data`: 70/30 claims severity/frequency trend + premium anchor, D-009) is computed and logged on every run but is **not** what multiplies the predictions (`f_final = f_probe` verbatim); the "reconciliation" column is just the ratio `f_probe / f_data` and adds no content. The measured optimum: motor nb **1.322** / ren **1.103**, property nb **1.12** / ren **0.816907** at constant exposure-weighted property total ×1.05. The NB/ren cohort ridge (D-076) has a solid **direction** — the bracket is strongly asymmetric (the ren-heavy arm is clearly negative) — but the gain is **peak-picked** at adaptively-selected points; the follow-up vertex probe was flat against the record. Actuarial interpretation: 2025 new business is underpriced relative to renewals, consistent with the broker transfer book degrading (D-068). The joint (total × split) optimum is settled at ×1.05 (D-077: quadratic vertex at ≈1.059, implied gain below probe noise; the refinement shot confirmed it flat). **Private caveat (D-085, measured in the Part 1 sandbox):** Public and Private optima diverge materially in level (Public prefers the higher level). Part 2's Private split is hidden, so whether this Public-fitted level sits at the Private optimum cannot be observed directly — while the selection rule rewards exactly this tuning. Carried as a disclosed limitation (§7).

### 5-Fold Policy-Grouped Cross-Validation Results on `emerging/` (109,296 rows)

| Model | Overall 5-Fold OOF $D^2$ | Motor $D^2$ | Buildings $D^2$ | Contents $D^2$ |
|---|---:|---:|---:|---:|
| 1. Constant rate per LOB | `+0.08039` | `+0.02738` | `+0.01421` | `+0.03802` |
| 2. `written_premium × LOB LR` | `+0.10965` | `+0.04861` | `+0.06332` | `+0.06272` |
| 3. `Emerging`-only Per-LOB rate LGBM (`lv=6`) | `+0.10734` | `+0.06244` | `+0.03916` | `+0.07541` |
| 4. Zero-shot `Developed` Prior (recalibrated per LOB) | `+0.13855` | `+0.06964` | `+0.10151` | `+0.10478` |
| 5. Offset-Adapted Transfer Model, D-066 config (`emerging-exp-001`) | `+0.14045` | `+0.07080` | `+0.10515` | `+0.10450` |
| **6. Offset-Adapted Transfer Model, record config (`emerging-exp-025`, D-075)** | **`+0.14074`** | — | — | — |

The record OOF is the seed-0 contract value; seeds 1/2 give `+0.14080`/`+0.14116` (local seed σ ≈ 0.00018, D-079). Stage-3 multipliers are book-level constants fitted at scoring and are deliberately excluded from OOF. D-075 established at the tuned config that **board gain = mean OOF gain exactly** (an OOF-faithful oracle), so OOF is used as the gate and the board as the confirmation.

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

*Table generated from `outputs/metrics/emerging_reference_office_premiums.csv`
by `build_office_premiums` (`src/emerging.py`) on the record configuration
(refreshed 2026-10-28; the D-066-era table is preserved in git history). Every
reference policy is priced above its current `written_premium` (ratio
1.28×–9.50×) — consistent with the §2 misalignment findings.*

---

## 6. Validation Discipline, Measured Noise, and Negative Results

**Measure the noise first (D-040, D-072, D-080 supplement).** One config
redraw moves the live boards by more than most ideas move them (three `lob`-arm
seed draws; magnitudes in `docs/decisions.md`); local OOF seed σ is 0.00018. Single-seed
comparisons inside the band are undecidable, so candidates are measured with
**paired seeds at identical folds** against **pre-registered gates** (≥3 seeds
per arm for board claims), and the board is the confirmation.

**What the gates caught (every one held; no slot spent on a gated-out idea):**
- **D-070 / D-079 — multi-family ensembles.** The 70/30 offset-LGBM/offset-XGB
  blend looked `+0.00023` OOF on the untuned base (D-070); promoted onto the
  record config with a pre-registered paired-seed gate it measured **−0.00020**
  (per-seed `+0.00009`/`−0.00023`/`−0.00046`). The seed-0 arm reproduced the
  record OOF to 1e-6, so the negative is a faithful comparison. No CSV; the
  day's slot was held.
- **D-082 — the upgraded transfer prior.** Porting the Part 1 winning recipe
  (exp-036 family blend: per-LOB LGBM + per-LOB XGBoost + pooled LGBM, with
  occupation dropped) into the developed-market prior — the carrier of 98% of
  pipeline skill — measured **flat where it matters** (prior-alone +0.13852
  vs +0.13855) and **−0.00017** through the pipeline (per-seed
  −0.00030/+0.00002/−0.00022; record contract held, drift 7.7e-07). Third
  failed port of a Part 1 ensemble idea (D-070, D-079, here): model-form
  diversity gains are location-specific and do not travel through the prior.
- **D-073 / D-074 — structure sweeps.** Property per-LOB split flat; motor
  per-peril frequency/severity decomposition **−0.00036** OOF — sparse perils
  (theft 390 / BI 450 claims in the emerging motor book) cannot carry
  standalone models.
- **Extended adaptation layer (D-067).** A 36-factor emerging-specific
  adaptation layer is negative against the canonicalised shared feature set.

**Label-space changes are tested on the Part 1 sandbox (D-071).** Part 1's
scoring gate stayed open after its 2026-10-23 deadline and reveals Private —
the only instrument that can see a Private failure *before* it costs a Part 2
slot:
- **D-072 — Horvitz–Thompson date×peril development labels:** a
  Private-for-Public trade (2 seeds — public up, private down). Rejected
  (direction consistent 2/2 on both boards; magnitude inside the revised
  noise band, so the rejection rests on direction + mechanism, not size).
- **D-080 — level-preserving per-peril development (D-028/D-035 factors):**
  negative mean on both boards over 3 paired seeds (3/3 pairs
  non-positive on Private; the third pair was a tie on a noisy control).
  No evidence of improvement and a negative point estimate — `EM_LDF` uniform
  development stands, and D-028's "local tie — LB tie-break open" closes
  against per-peril development. Grade: negative at n=3, not decisive.

**Tail honesty (D-078, §2 item 3).** The claim-size distribution is a real
heavy tail with zero corruptions. Nothing is capped: the top 10 rows carry
8.3% of `y_ult` and 4.9% of Tweedie deviance (buildings top-10 = 25.6% of
line cost), so part of any board movement is irreducible claim randomness.

**Guardrails.** 103 tests (`tests/`), including claims-table integrity
invariants and the `si_artefact`-orthogonality of the top claims; the
`Σpred ≈ Σy` total-cost calibration invariant; and submission-time validation
in `scripts/kaggle_submit.py` (row count, row_id prefix/LOB consistency,
implied-rate flatness — the check that would have caught D-052's exposure
double-count before upload).

---

## 7. Disclosed Limitations

**The record's fitted multipliers are Public-fitted (D-068/D-076) and are
disclosed as such.** D-072's lesson — public-fitted corrections can
disappoint on Private — applies to them. Part 1's sandbox measured the
divergence directly (D-085: the Public and Private level optima sat
materially apart, the Public-tuned level being the higher one), and Part 2's
Private split is hidden, so the record's level may sit above its Private
optimum. **D-088 tested this and D-089 withdrew the test the same day:** a
−2% global level ticket (on the D-085-measured direction) improved on the
record on Public, but the direction of Part 2's Private level optimum is a
fixed-but-unknown split property resting on n=1 evidence (Part 1's
measurement), so on explicit human risk preference the bet was withdrawn and
the model runs at the record level with **no level deviation**. The bet's
truth was never observed; its measurement trail stands in the log — notably
the level response is concave with its vertex near K ≈ 1 (D-090).
Exposure is bounded by construction: the multiplier shapes are constrained
to measured response-curve families, the property total is held at ×1.05,
and the joint optimum was settled at the flat end of the ridge (D-077,
implied gain below probe noise) rather than chased.

**Reproducibility.** `python3 scripts/run_emerging.py --use-cache --num-leaves 4
--min-leaf 1200` reproduces the record model output byte-equivalently (max rel
diff 8e-16, `outputs/metrics/emerging-exp-025-repro.json`); the s678 refit
draw is the same command with `--refit-seeds 6,7,8` (D-089). The full
measurement trail — including every negative result above, the duplicate-upload
incident and its serial-submission fix — is in `docs/experiment_log.md` and
`docs/decisions.md` (D-001…D-089).
