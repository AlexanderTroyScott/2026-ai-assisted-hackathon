# Data Findings — Verified Empirically

**Status:** authoritative reference for Week 1. Every number here was measured from
`data_raw/` (not inferred from the brief). Last verified 2026-10-20.

Load this file (or the `data-facts` skill) before any modelling, feature engineering or
target construction. Do **not** re-derive these facts from scratch.

---

## 1. Inventory & linkage

| File | Rows | Notes |
|---|---|---|
| customers | 545,240 | |
| addresses | 575,983 | |
| weather | 2,879,915 | 5 years × 575,983 addresses (2021–2025) |
| policies_motor | 979,968 | 308,109 distinct policy_ids |
| policies_buildings | 707,726 | 230,921 distinct policy_ids |
| policies_contents | 721,617 | 231,255 distinct policy_ids |
| claims | 163,150 | motor 112,606 / buildings 26,442 / contents 24,102 |
| test_motor | 254,986 | prefix `M` |
| test_buildings | 184,155 | prefix `B` |
| test_contents | 187,498 | prefix `C` |
| sample_submission | 626,639 | all-zero dummy; **parquet**, not CSV |

- Test row_id set **exactly equals** `sample_submission.row_id` (626,639 unique). Assert this before every submission.
- Row_id prefix encodes the line of business (`M`/`B`/`C`) — useful join/consistency check.
- All identifiers are text. Read as `str`, never numeric.
- `weather` covers **2025 for every test address** (join completeness = 1.0). Weather features are usable at scoring time under the documented assumption that annual climate indices are available at pricing time.
- Policy rows never overlap in time within a policy: consecutive same-year rows are exactly contiguous (140,149 pairs, gap = 0 days). Train periods end ≤ 2024-12-30; test periods start ≥ 2025-01-01 — **no temporal overlap between train and test rows**.

## 2. Target construction (verified)

Target = Σ over claims with `policy_id` match **and** `period_start ≤ loss_date ≤ period_end`
of `max(0, incurred_to_date − recoveries)`, excluding `status == 'repudiated'`.

Verified properties:

- Every claim matches **exactly one** policy row (162,736 of 163,150). Zero claims match >1 row.
- **414 claims are orphans** (loss date matches no row): 274 with 2020 loss dates (before the
  policy window), plus 156/115/69 in 2021/2022/2023 falling in gaps. Rule: drop + log. See `docs/decisions.md`.
- Repudiated claims carry `paid = incurred = recoveries = 0`, so the "exclude declined" filter is
  **numerically a no-op**. Document it explicitly rather than leaving it ambiguous.
- `incurred_to_date` is already net of policy excess. Do not deduct the excess again.
- Closed claims: `paid_to_date == incurred_to_date` exactly (share = 1.0) → closed = final amounts.
- `recoveries > incurred` never occurs; no negatives anywhere.

## 3. Exposure is corrupted — recompute it

> **Aggregation pitfall (cost us a wrong number once).** When joining claims to policy rows,
> a naive `merge` + filter drops policy rows whose claims fall in a *different* period
> (their merged claim rows fail the date filter and the row disappears). Exposure, row
> counts, frequency and premium-per-exposure must be aggregated over **all** policy rows,
> with claim amounts joined back afterwards. Severity (cost per claim) is unaffected.
> `tests/test_guardrails.py::test_exposure_sums_over_all_rows` guards this.

The supplied `exposure` column mixes three conventions and contains unit errors:

| Convention | motor | buildings | contents | test (all lines) |
|---|---|---|---|---|
| days / 365 (correct) | 958,703 | 694,632 | 706,957 | 616,486 |
| days / 360 (1.0139 for a full year) | 4,802 | 3,526 | 3,582 | 0 |
| days / 365.25 | 15,529 | 8,912 | 10,349 | 10,053 |
| **in days (unit error)** | **934** | **656** | **729** | **0** |

Test has no unit errors but still mixes 365 / 365.25 conventions.

**Rule: always recompute `exposure = (period_end − period_start + 1) / 365`.** The given column must
never be used as a GLM weight — a handful of rows at weight 365 will destroy a fit.

Test exposure mean is 0.78–0.82 (many partial-year rows); training mean is 1.08–1.12 *because* of
the unit errors. This alone is enough to explain apparent train/test exposure shift.

## 4. Data quality landmines

| Issue | Detail | Action |
|---|---|---|
| `construction_type` | 11 dirty levels in train: `BRICK`/`Brick`/`brick`, `NON_STANDARD`/`non_standard`, `STONE`/`stone`, `T`/`t`, `W`/`w`. Test has 9 (`T`/`t` absent). Dictionary only names brick/stone/timber/non-standard | lowercase + map `T/t→timber`, `W/w→other`; unseen → `other` |
| `smoke_alarms` | object dtype holding `'True'`/`'False'` strings | coerce to bool |
| `telematics_score` | 87.6% missing (train) / 86.7% (test) but perfectly consistent with `telematics_opt_in` (score present ⟺ opted in) | use opt-in flag + score with missing indicator; do not blind-impute |
| Nulls in customers | occupation / marital_status / household_size / years_at_address ≈ 2.5% each; `campaign_code` 51% missing (structural — no campaign) | explicit missing levels |
| Nulls in addresses | flood_zone 2.9%, distance_to_river_km / soil_type / roof_type / energy_rating ≈ 2.5% | explicit missing levels |
| `renewal_decision` | **training only** and is outcome information | never a feature — enforce with a deny-list test |
| 2025-only customers | `customer_since_year` max = 2025 | expected; experience features must fall back |

## 5. Development / IBNR — 2024 labels are understated

Claims are valued at 31 Dec 2024. Target is **ultimate**. The gap is material.

### Incurred triangle (net of recoveries, non-repaid; origin year × report year)

**Motor** (M CU):
| OY | age 0 | age 1 | age 2 | age 3 | ultimate |
|---|---|---|---|---|---|
| 2021 | 81.5 | 21.8 | 2.36 | 0.37 | 106.0 |
| 2022 | 92.2 | 21.5 | 3.19 | | 116.9 |
| 2023 | 100.1 | 17.3 | | | 117.4 |
| 2024 | 90.7 | | | | (develop ≈ 114) |

**Buildings:** 2021 50.0+4.0+0.1 = 54.1 | 2022 56.7+4.7+0.2 = 61.5 | 2023 59.1+5.9 = 65.0 | 2024 67.4 → develop ≈ 73.6

**Contents:** 2021 14.3+0.6+0.03 = 14.9 | 2022 15.5+0.5 = 16.1 | 2023 16.9+0.7 = 17.6 | 2024 17.5 → develop ≈ 18.3

### Age-to-age factors (volume-weighted, incurred)

| LOB | 0→1 | 1→2 | 2→3 | age-0 → ultimate |
|---|---|---|---|---|
| motor | 1.223 | 1.027 | 1.003 | **≈ ×1.26** |
| buildings | 1.089 | 1.003 | | **≈ ×1.09** |
| contents | 1.040 | 1.004 | | **≈ ×1.04** |

These are **lower bounds** — they capture late reporting but not IBNER on open claims.

### Other IBNR evidence

- Open-claim share by origin year: 0.7% (2021) → 1.6% (2022) → 3.2% (2023) → **20.4% (2024)**.
- 2024 open claims mean case ≈ 7.7k vs 21–27k for mature open claims → immature case estimates.
- Report lag: median 8–9 days, but max 1,361 days (motor). Share reported >365 days after loss:
  motor 1.07%, buildings 0.57%, contents 0.47%. **By peril: bodily_injury 10.2%, subsidence 3.9%**,
  everything else ≤ 0.5%.
- Q4-2024 truncation in loss counts (non-repaid): Dec 2024 = 2,264 vs Dec 2021–23 norm ≈ 4,001;
  Nov 2024 = 2,985 vs 3,415. Within-year reporting truncation, consistent with the short median lag.

**Consequences:** (a) never estimate trend on the raw all-claims series — 2024 "deflates";
(b) never validate against raw 2024 observed — it rewards underprediction;
(c) develop labels (esp. motor), not predictions; avoid double-counting.

## 6. Claims inflation — reverse-engineered

Method: `log(severity) ~ accident year`, cohort = **closed** claims with **report lag ≤ 30 days**
(final amounts, comparable maturity), 2021–2024.

| Peril | n | Inflation/yr | 95% CI |
|---|---|---|---|
| glass_hail | 15,206 | **+9.4%** | +7.4 … +11.5 |
| storm | 7,028 | **+8.2%** | +5.2 … +11.3 |
| third_party_prop | 24,096 | **+7.6%** | +5.9 … +9.2 |
| flood | 3,293 | +5.8% | +1.0 … +10.8 |
| theft | 13,237 | +4.7% | +2.5 … +7.0 |
| escape_of_water | 12,660 | +4.7% | +2.4 … +7.0 |
| own_damage | 44,088 | +3.2% | +1.8 … +4.5 |
| accidental_damage | 11,996 | +2.9% | +0.7 … +5.1 |
| fire | 962 | +1.1% | not significant |
| bodily_injury | 880 | −9.5% | **artefact — closure bias** |

Structural findings:

- **It is a scale shift, not a shape change.** 2024/2021 quantile ratios are roughly flat across
  q25–q95 within a peril (glass_hail 1.34/1.27/1.26/1.23/1.24; own_damage 1.12/1.11/1.06/1.06/1.10).
  A single per-peril scale index is defensible.
- **Not one factor** — spread is 3%/yr (own damage, contents AD) to 9%/yr (glass/hail, TP property).
- **Frequency is essentially flat** (exposure-corrected index over *all* policy rows: motor
  100/102/96/89*, buildings 100/103/103/100, contents 100/101/104/101; *2024 = IBNR
  artefact). Trend is severity-side.
- **Closure bias:** closed-only BI severity *falls* 9.5%/yr because large BI claims take years to
  close (n drops 287 → 76 by AY). Long-tail perils must be trended on **developed ultimates**, never
  closed-only cohorts.
- **Mix bias:** motor's raw mean severity falls in 2024 partly because BI's claim share dropped
  (1,825 → 1,038 closed). Use fixed-weight (mix-adjusted) indices.

Mix-adjusted severity index (fixed 2021 peril weights, closed lag ≤30): motor 100 → 105 → 111 → 112;
buildings 100 → 101 → 108 → 118; contents 100 → 99 → 106 → 111 → **≈ 3.5–5.5%/yr**.

Year-on-year rates are **not** constant (glass_hail +11.2%, +8.9%, +3.8%; storm +4.5%, +11.7%, +7.3%),
so no single deterministic factor reproduces the sequence. 2024 rates cooled (own_damage +0.0%,
glass_hail +3.8%) — treat extrapolation as a scenario, not a fact.

### Independent anchor: the insurer's own pricing

Premium per exposure (2021 = 100, exposure summed over **all** policy rows):

| LOB | 2021 | 2022 | 2023 | 2024 | 2025 | 2025 vs 2024 |
|---|---|---|---|---|---|---|
| motor | 100 | 113 | 117 | 121 | **124** | +2.7% |
| buildings | 100 | 117 | 123 | 130 | **137** | +5.3% |
| contents | 100 | 117 | 123 | 130 | **137** | +5.4% |

Absolute premium/exposure: motor 618 → 701 → 725 → 746 → 766; buildings 328 → 385 →
405 → 427 → 449; contents 141 → 165 → 174 → 183 → 193.

2025 prices already carry a ~3–5%/yr trended level. Use as sanity check **or** as a level anchor
(assume 2025 LR ≈ developed 2024 LR). Note the target is net of a **fixed nominal** excess
(motor 250, home 350 + voluntary 0/250/500/1000), so net-of-excess severity inflates faster than
gross — already embedded since we trend the net target.

## 7. Feature signal (measured on motor rows)

| Feature | Signal |
|---|---|
| **Lagged claim experience (customer, prior years)** | freq 0.083 → 0.389 (1 prior) → 0.486 (2+); cost/exp 323 → 1,686 → 1,750; **LR 0.47 → 2.37**. Strongest single lever. Available for 85% of test rows (customer seen in train); 20–26% of test rows have a customer with ≥1 historical claim |
| `written_premium` | decile freq 0.075 → 0.300; cost/exp 307 → 1,183; **LR roughly flat 0.71–0.81**. Excellent risk *ranker*; the interesting model finds where current pricing is wrong |
| `prior_claims_5y` (declared) | weak: mean actual claims 1.09 → 1.36 → 1.69 → 2.06 for declared 0→3→4. Use only as fallback for unseen customers |
| `driver_age` | U-shaped (young + old high) |
| `ncb_years` | monotone decreasing, saturates ~11 years |

Test/train shift: policy_id overlap 84.8%, customer overlap 84.8%, address overlap 81–84%.
`written_premium` per exposure in test is +3–6% vs 2024 (inflation, see §6). `renewal_count`
mean falls 6.60 → 5.87. `construction_type` loses `T/t` levels in test.

## 8. Metric notes

- Scored metric: **Tweedie D², power = 1.5**, relative to a "constant-rate baseline".
- sklearn's `mean_tweedie_deviance(power=1.5)` **requires strictly positive `y_pred`** — it raises
  otherwise. The brief says "zero or above" but the all-zero `sample_submission` is a dummy.
  **Floor predictions at 1e-3.**
- sklearn's `d2_tweedie_score` null is `mean(y)`; an actuarial "constant rate" null is
  `exposure × mean cost/exposure`. The brief's wording is ambiguous. Model cost as
  `exposure × annual rate` so you are sensible under either definition, and state the assumption.
- Whether D² is pooled across LOBs is unspecified. If pooled, LOB-level mean calibration is a large
  share of the score. Report D² both pooled and per LOB; recalibrate so Σpred ≈ expected total cost.
- **Leaderboard scale (verified 2026-10-20):** top 0.12996, 22 teams, 20th 0.11710. Use this to
  detect a metric-implementation mismatch — if local CV D² ≈ 0.4, our baseline definition is wrong.

## 9. Reproducibility

- `data_raw/` is gitignored. Do not commit parquets. Keep `docs/data_manifest.md` (file, rows, sha256)
  so the repo remains reproducible against the Kaggle download.
- All analysis in this file is reproducible with pandas over `data_raw/`.
