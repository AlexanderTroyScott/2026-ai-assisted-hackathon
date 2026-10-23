---
name: data-facts
description: Use when doing EDA, feature engineering, modelling or anything that reads developed/ parquet files in this actuarial hackathon repo. Loads the verified empirical findings (exposure corruption, target construction, IBNR triangle, inflation rates, feature signal, train/test shift, metric notes) so they do not need to be re-derived or re-read from scratch.
---

# Data Facts — verified findings (2026-10-20)

Canonical source: `docs/data_findings.md`. Everything below was measured from
`developed/` with pandas. **Do not re-derive these numbers.** If a figure is not
here and not in `docs/data_findings.md`, it is unverified.

## Inventory & linkage

- customers 545,240 · addresses 575,983 · weather 2,879,915 (2021–2025, every address)
- policies_motor 979,968 · policies_buildings 707,726 · policies_contents 721,617
- claims 163,150 (motor 112,606 / buildings 26,442 / contents 24,102)
- test_motor 254,986 (`M`) · test_buildings 184,155 (`B`) · test_contents 187,498 (`C`)
- `sample_submission` = 626,639 rows, **parquet**, all-zero dummy
- Test row_id set **exactly equals** sample_submission.row_id. Assert before every submission.
- IDs are text — always read as `str`.
- `weather` includes **2025 for 100% of test addresses** → usable at scoring time
  (assumption to document: annual climate indices known at pricing time).
- Train periods end ≤ 2024-12-30; test periods start ≥ 2025-01-01 → **no train/test
  time overlap**. Consecutive same-year rows are exactly contiguous (gap = 0 days).

## Target construction

`y = Σ max(0, incurred_to_date − recoveries)` over claims with matching `policy_id`
**and** `period_start ≤ loss_date ≤ period_end`, excluding `status == 'repudiated'`.

- Every claim matches **exactly one** row (162,736). **0 claims match >1 row.**
- **414 orphans** (274 pre-2021 + 156/115/69 in 2021–23). Rule: drop + log (D-002).
- Repudiated claims all carry `paid = incurred = recoveries = 0` → filter is a
  **numerical no-op**. State this in the write-up.
- `incurred_to_date` is already net of excess — never deduct excess again.
- Closed claims: `paid_to_date == incurred_to_date` (share = 1.0) → final amounts.

## Exposure is corrupt (D-001)

Always recompute `exposure = (period_end − period_start + 1) / 365`.

**Aggregation pitfall:** when joining claims to policy rows, a naive merge + date
filter drops policy rows whose claims fall in a different period (their merged rows
fail the filter and the row vanishes). Exposure / frequency / premium-per-exposure
must be summed over **all** policy rows and claim amounts joined back afterwards.
Severity per claim is unaffected. This produced wrong frequency and cost-per-exposure
numbers once — always aggregate exposure independently of claims.

| Convention | motor | buildings | contents | test |
|---|---|---|---|---|
| days/365 (correct) | 958,703 | 694,632 | 706,957 | 616,486 |
| days/360 (1.0139/yr) | 4,802 | 3,526 | 3,582 | 0 |
| days/365.25 | 15,529 | 8,912 | 10,349 | 10,053 |
| **days (unit error)** | **934** | **656** | **729** | 0 |

Train exposure mean 1.08–1.12 is an artefact of these errors; test is 0.78–0.82
(many partial-year rows). Never use the given column as a GLM weight.

## Data quality

- `construction_type`: 11 dirty levels in train (`BRICK`/`Brick`/`brick`,
  `NON_STANDARD`, `STONE`, `T`/`t`, `W`/`w`); test has 9 (`T/t` gone).
  Lowercase; map `T/t→timber`, `W/w→other`; unseen → `other`.
- `smoke_alarms` is object dtype holding `'True'`/`'False'` strings → coerce to bool.
- `telematics_score` 87.6% missing (train) / 86.7% (test); perfectly consistent with
  `telematics_opt_in`. Use flag + score with missing indicator.
- Nulls ≈2.5% in occupation/marital_status/household_size/years_at_address, and in
  flood_zone/distance_to_river_km/soil_type/roof_type/energy_rating;
  `campaign_code` 51% missing (structural).
- `renewal_decision` is train-only + outcome leakage → **deny-list feature** (D-011).
- `customer_since_year` reaches 2025 → experience features need a fallback.

## IBNR / development (D-005, D-006)

Age-0 → ultimate LDFs (incurred triangle, volume-weighted):

| LOB | 0→1 | 1→2 | 2→3 | to ultimate | 2024 observed → developed |
|---|---|---|---|---|---|
| motor | 1.223 | 1.027 | 1.003 | **≈ ×1.26** | 90.7M → ~114M |
| buildings | 1.089 | 1.003 | — | **≈ ×1.09** | 67.4M → ~73.6M |
| contents | 1.040 | 1.004 | — | **≈ ×1.04** | 17.5M → ~18.3M |

Lower bounds — they miss IBNER on open claims.

- Open-claim share by AY: 0.7% → 1.6% → 3.2% → **20.4% (2024)**.
- 2024 open mean case ≈ 7.7k vs 21–27k mature → immature case estimates.
- Report lag median 8–9 days, max 1,361. Share >365d: motor 1.07%, buildings 0.57%,
  contents 0.47%. **Peril: bodily_injury 10.2%, subsidence 3.9%**, others ≤0.5%.
- Q4-2024 loss counts truncated: Dec 2,264 vs Dec norm ≈ 4,001; Nov 2,985 vs 3,415.

**Never** estimate trend on raw all-claims severity (2024 "deflates") and never
validate against raw 2024 observed. Develop **labels**, not predictions.

## Claims inflation (D-007)

Cohort = closed claims, report lag ≤ 30 days, 2021–24. `log(severity) ~ AY`:

| Peril | n | /yr | 95% CI |
|---|---|---|---|
| glass_hail | 15,206 | **+9.4%** | +7.4 … +11.5 |
| storm | 7,028 | **+8.2%** | +5.2 … +11.3 |
| third_party_prop | 24,096 | **+7.6%** | +5.9 … +9.2 |
| flood | 3,293 | +5.8% | +1.0 … +10.8 |
| theft | 13,237 | +4.7% | +2.5 … +7.0 |
| escape_of_water | 12,660 | +4.7% | +2.4 … +7.0 |
| own_damage | 44,088 | +3.2% | +1.8 … +4.5 |
| accidental_damage | 11,996 | +2.9% | +0.7 … +5.1 |
| fire | 962 | +1.1% | n.s. |
| bodily_injury | 880 | −9.5% | **closure-bias artefact** |

- Scale shift, not shape change (2024/2021 quantile ratios ~flat q25–q95).
- Frequency is flat on clean exposure; trend is severity-side.
- Mix-adjusted index (fixed 2021 peril weights): motor 100→105→111→112,
  buildings 100→101→108→118, contents 100→99→106→111 → **3.5–5.5%/yr**.
- YoY rates are not constant; 2024 cooled (own_damage +0.0%, glass_hail +3.8%).
  Extrapolate as a **scenario**, not a fact. Long-tail perils → developed ultimates only.
- Premium/exposure index (2021=100, exposure over **all** policy rows): motor
  113/117/121/**124**; buildings & contents 117/123/130/**137** (2022/23/24/25).
  2025 vs 2024: +2.7% / +5.3% / +5.4%. Absolute premium/exposure: motor 618→701→725→746→766;
  buildings 328→385→405→427→449; contents 141→165→174→183→193.
  Usable as an independent level anchor (D-009, still `proposed`).
- Excess is fixed in nominal terms → net-of-excess severity inflates faster than
  gross. Already embedded since we trend the net target.

## Feature signal (motor, measured)

| Feature | Signal |
|---|---|
| **Lagged claim experience (D-014)** | freq 0.083 → 0.389 (1 prior) → 0.486 (2+); cost/exp 323 → 1,686 → 1,750; LR 0.47 → 2.37. **Strongest lever.** Available for 85% of test rows; 20–26% of test rows have a customer with ≥1 historical claim |
| `written_premium` (D-015) | decile freq 0.075 → 0.300; cost/exp 307 → 1,183; LR flat 0.71–0.81. Risk *ranker*; benchmark is `premium × LOB LR` |
| `prior_claims_5y` (declared) | weak (mean actual 1.09 → 2.06 across declared 0→4). Fallback only |
| `driver_age` | U-shaped |
| `ncb_years` | monotone decreasing, saturates ~11y |

Train/test shift: policy_id overlap 84.8%, customer 84.8%, address 81–84%;
`written_premium`/exposure +3–6% in test; `renewal_count` mean 6.60 → 5.87;
`construction_type` loses `T/t`.

## Metric (D-010)

- Tweedie D², p = 1.5. sklearn `d2_tweedie_score(y, y_pred, power=1.5)`.
- **Floor predictions at 1e-3** — deviance raises on `y_pred <= 0`.
- "Constant-rate baseline" is ambiguous (sklearn null = `mean(y)`; actuarial =
  `exposure × mean cost/exposure`). Model as `exposure × rate`; report both variants.
- Pooled vs per-LOB D² unspecified → report both; recalibrate each LOB so
  `Σpred ≈` expected total cost.
- **Leaderboard scale 2026-10-20:** top 0.12996, 20th/22 0.11710. If local CV D² ≈ 0.4,
  the baseline definition is wrong — fix before trusting local numbers.
