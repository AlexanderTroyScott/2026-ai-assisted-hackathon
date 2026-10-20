# Synthetic DGP — Reverse-Engineering Findings

**Status:** measured 2026-10-20 from `data_raw/`. Complements `docs/data_findings.md`
(that file is about data quality and the target; this one is about the generator).
These findings change *how* we model, not just what we clean.

---

## 1. Severity: Gamma scale family, not lognormal

- Closed claims, report lag ≤ 30 days, net cost > 0 (n = 133,649).
- `log(severity)` is **left-skewed** (skew −0.75 … −1.32) → not lognormal.
- **Gamma beats lognormal on AIC for all 11 perils** (e.g. own_damage 746,565 vs
  750,106; glass_hail 242,957 vs 245,803).
- **Gamma shape is stable by year** (own_damage 0.716–0.725; flood 0.753–0.793;
  glass_hail 0.96–1.01; third_party_prop 0.98–1.03) and **log-SD is constant across
  accident years for every peril** (Levene p = 0.098–0.96; log-SD 1.29–1.52).

**Implication:** severity is a fixed-shape Gamma whose *scale* shifts with
(accident year × peril). A per-peril, per-year scale factor is a sufficient
calibration — exactly the D-007 approach. Tweedie p=1.5 (Poisson–Gamma compound) is
a well-specified choice.

## 2. Frequency: NB marginally, Poisson given policy rate

- Unit-exposure rows: count variance/mean = motor **1.149**, buildings 1.041,
  contents 1.036. Row shares 0/1/2/3+ claims = 88.0/10.6/1.22/0.20% fit
  **NB(θ ≈ 0.92)** (P(0) = 0.880 predicted vs 0.880 observed).
- After a Poisson GLM on rating factors, Pearson dispersion = 1.08 → most
  overdispersion is rate heterogeneity, not extra noise.

**Implication:** a pure Poisson frequency model understates P(2+ claims). Fine for
Tweedie; relevant if we split frequency/severity.

## 3. `written_premium` is a decoded multiplicative formula (motor)

OLS on logs with exact categoricals: **R² = 0.9558**, residual SD = 0.080, residual
exactly normal in logs (KS = 0.002). Nothing hidden — adding customer/address
features does not improve R².

```
written_premium = base × θ_age × θ_ncb × θ_vvg × θ_cover × θ_region × θ_year
                  × exp(−1.04 × discount_applied) × Lognormal(0, 0.08²)
```

| Factor | Relativities |
|---|---|
| θ_age | 17–24: **1.00** · 25–69: **0.454** · 70+: **0.545** (sharp jumps at 25 and 70) |
| θ_ncb | 0: **1.00** · 1–4: **0.785** · 5+: **0.607** (3 flat bands) |
| θ_vvg | **1.030^(vvg − 1)** — log-linear, step 0.0295 per log-unit |
| θ_cover | comp 1.00 ≈ TPFT 0.999 · TPO **0.700** |
| θ_year | ×1.040 / 1.082 / 1.126 (≈ +4%/yr) |
| θ_region | 0.852 … 1.108 (12 levels) |

Buildings: HGB R² = 0.977 (flood_zone 1.46, sum_insured 0.30, distance_to_river 0.07,
elevation 0.05). Contents: R² = 0.955 (sum_insured 1.60, urban_density 0.13,
region 0.09). Residual SD 0.09–0.095.

## 4. Pricing and risk are generated separately — premium is *misaligned*

This is the most exploitable finding. The premium formula is not the risk formula:

| Misalignment | Evidence | Opportunity |
|---|---|---|
| **Experience is missing from price** | Price reacts to claims only via the NCB step; within-NCB premium change is flat (−0.001 to −0.026 by prior claims). Prior claimants have LR 2.3 but pay the same | **Largest single lever.** Add lagged experience to a premium-based model |
| **vehicle_value_group inverted** | Premium relativity rises 1.03/group while frequency relativity *falls* 1.36 → 0.96 | Correct the vvg slope |
| **cover_type mispriced** | TPFT premium ≈ comp (0.999) but frequency ratio 0.47, cost ratio 0.91 | TPFT overpriced → predict *lower* cost than premium implies |
| **LOB loss ratios differ** | motor 0.78 · contents 0.67 · **buildings 1.08** | Calibrate per LOB, don't assume one LR |
| **Non-stationary relativities** | ncb=0 freq rel 1.60 (2021) → 2.53 (2022/23); age 17–24 2.03 → 1.74; cover TPO/comp 0.476 → 0.411 | Allow year interactions rather than one static table |

Premium deciles: frequency 0.074 → 0.301 (4×) with **flat LR 0.72–0.81**. So premium
is an excellent *ranker* and a good *base*, but the residual structure is where the
D² comes from.

## 5. Weather genuinely drives perils (address-year level)

Rate ratio, top vs bottom weather quintile:

| Peril | Driver | RR |
|---|---|---|
| glass_hail | `hail_days` | **2.35** (monotone) |
| flood | `rainfall_mm` | **2.58** |
| flood | `distance_to_river_km` (near vs far) | **36×** |
| flood | `elevation_m` (low vs high) | **≈330×** |
| flood | `flood_zone` 1 → 5 | **190×** (0.00017 → 0.0325) |
| escape_of_water | `freezing_days` | **2.02** |
| storm | `wind_exposure_index` | **1.91** |
| subsidence | `dry_summer_index` | **3.03** |
| fire | `dry_summer_index` | 1.62 |

Negative control: theft vs `wildfire_index` RR = 0.99 (no link) — good, the signal is
peril-specific, not spurious. `flood_zone` is itself a river-distance/elevation
composite (mean distance 4.0 → 0.2 km, elevation 14.3 → 1.2 m), so don't put all
four in a linear model uncritically.

## 6. Recoveries and excess

- **Recoveries are rule-based and motor-only:** own_damage claims recover **exactly
  0.8 × incurred** (13,997 claims, 28.6% of OD claims) and theft claims recover
  **exactly 0.3 ×** (779 claims). Verified: `recoveries/incurred` has median = mean =
  0.8 (OD) and 0.3 (theft), min 0.778 / max 1.0. No recoveries on property perils.
- **Correction to an earlier draft:** `third_party_id` is *not* the trigger. It is
  present only on `bodily_injury` and `third_party_prop` claims (100% of each) and on
  **zero** recovering claims. The recovery trigger is not identifiable from any
  available field (rate is a flat 28.8% of own_damage regardless of `repairer_id`
  presence). Treat it as random from our point of view.
- **Excess is never applied to claim amounts:** net = gross in 87% (motor) and 100%
  (property) of claims; 18,948 motor claims with gross < excess still have net > 0.
  Confirms D-003 — do not deduct the excess in target construction.

## 6b. Accident size — what can and cannot be measured

There is **no accident identifier** in the data. `loss_date` is a plain calendar date
(1,473 distinct dates, ~110 claims per date), so claims cannot be clustered into
accidents. `third_party_id` links only 242 ids across more than one claim (85 across
more than one policy), so multi-claim accidents are not reconstructable.

What *is* measurable is severity structure — and it answers the "how big can an
accident get" question:

| peril | top 1% of claims = share of cost | mean | median |
|---|---|---|---|
| **bodily_injury** | **18.3%** | 21,582 | 11,525 |
| flood | 8.5% | 22,360 | 12,055 |
| fire | 7.7% | 51,685 | 28,553 |
| escape_of_water | 7.3% | 4,692 | 2,849 |
| theft | 7.1% | 4,942 | 3,086 |
| own_damage | 6.5% | 2,394 | 1,571 |
| all others | 5.7–6.4% | | |

**Bodily injury is uniquely catastrophic** — 18.3% of its cost sits in 1% of claims,
against ~6% for every other peril.

BI severity bands (motor):

| band | share of claims | share of BI cost |
|---|---|---|
| < 5k | 26.4% | 2.8% |
| 5–25k | 50.0% | 30.1% |
| 25–100k | 21.5% | 42.9% |
| **100k+** | **2.1%** | **24.2%** |

The log-histogram is **unimodal** (single peak at ~10⁴ = 10k) with a thin catastrophic
tail to 11.9M — so there is *no* separate "fatality = small fixed benefit" cluster
in this data. The hypothesis is not supported, but the underlying actuarial point
stands: BI is where accident *size* variance lives, and it dominates the Tweedie
deviance. Since motor currently has our **weakest per-LOB D²**, improving BI
handling is the highest-value modelling target.

Implication: do **not** winsorise or cap large claims — the metric (D² p=1.5)
rewards getting their expectation right. Model BI frequency and severity
separately if the single Tweedie cannot resolve the tail.

## 7. NCB is a stochastic transition kernel (not deterministic)

With 0 prior-period claims: +1 year 80.2%, no change 19.4%. With 1 claim: −2 in
55.8%, −1 9.9%, no change 17.1%, **+1 in 17.2%**. 2+ claims: −2 55.9%, −1 15.9%,
no change 24.3%. Premium reacts to the NCB *step*, not to claims directly.

## 8. What is NOT exploitable

- `make_model` (300 codes) is not a real vehicle taxonomy — codes do not determine
  body/engine/fuel (each spans 14–19 vgg, 20–23 engine sizes, all body types, all
  fuels). Split-half claim-rate correlation only 0.221, rate SD 0.0095 vs binomial
  SE 0.006. Weak signal at best; do not build a vehicle hierarchy on it.
- No severity ties or round-number heaping (top repeated value 6× in 151k) — nothing
  to exploit there.
- Premium residuals are iid across renewals within a policy (lag-1 corr 0.002) —
  the noise is redrawn per period, no persistence to exploit.

---

## How this changes the model

1. **Base = `written_premium`, not a flat rate.** It already encodes 95% of the
   rating structure. But do not treat it as the answer — see the misalignment table.
2. **Add the missing experience block** (D-014) — the price ignores it, so it is
   pure signal.
3. **Correct the known misalignments**: vvg slope, cover_type, per-LOB LR.
4. **Weather features by peril** for property lines; ignore them for motor theft.
5. **Model severity as a Gamma scale family** — per-peril, per-year scale factor
   (D-007) is sufficient; no need for a full severity distribution model.
6. **Do not deduct the excess** (D-003); do not build on `make_model`.
