# Decision Log

Append-only. Every entry: **date, decision, rationale, verification, status**.
Statuses: `proposed` → `accepted` → (optional) `superseded`. Nothing reaches
`accepted` without either an empirical check or an explicit human sign-off —
this is the "rigorous verification of AI-generated outputs" evidence trail.

---

## D-001 — Recompute `exposure` from period dates
**2026-10-20 · accepted · verified empirically**

The supplied `exposure` mixes days/365, days/360 and plain-days units
(934/656/729 rows in days across motor/buildings/contents). Using it as a GLM
weight would let a handful of rows dominate. Always recompute
`exposure = (period_end − period_start + 1) / 365`. Keep the original column
as `exposure_raw` for audit only.

## D-002 — Drop orphan claims (414) with an audit trail
**2026-10-20 · accepted · verified empirically**

414 claims have a loss date matching no policy row (274 pre-2021, 156/115/69 in
2021–2023). 0.25% of claims. Dropping them is immaterial to the level but leaving
them unmatched silently would create unattributed loss. Rule: exclude from
target construction, write ids to `outputs/dropped_claims.csv`, assert the count
in tests so a regression in matching logic is caught.

## D-003 — Target is net of excess as stated; do not deduct excess again
**2026-10-20 · accepted · verified empirically**

`incurred_to_date` is already net of policy excess. `max(0, incurred − recoveries)`
is the correct row cost. `recoveries > incurred` never occurs.

## D-004 — Excluding repudiated claims is a numerical no-op; document, don't hide
**2026-10-20 · accepted · verified empirically**

All 1,711 repudiated claims carry `paid = incurred = recoveries = 0`. The filter
must still be applied (it is part of the target definition and protects against a
data change), but the write-up should state that it changes no amounts.

## D-005 — Develop labels to ultimate; never develop predictions
**2026-10-20 · accepted · verified empirically**

2024 observed cost is understated (motor age-0→ultimate ≈ ×1.26, buildings ×1.09,
contents ×1.04, plus IBNER on open claims). Apply development to recent-year
**labels**, then model. Predictions are already ultimate-level — applying an
additional IBNR loading at scoring would double-count.

## D-006 — Validate against developed truth, never raw 2024
**2026-10-20 · accepted · follows from D-005**

A raw-2024 holdout rewards underprediction. Rolling-origin CV: train 2021–22 →
val 2023; train 2021–23 → val 2024; score against developed truth and report
raw-vs-developed side by side.

## D-007 — Per-peril severity trend, not one global inflation factor
**2026-10-20 · accepted · verified empirically**

Measured inflation spans +2.9%/yr (contents AD) to +9.4%/yr (glass_hail).
A single factor would be wrong for both ends. Long-tail perils (bodily_injury,
subsidence, fire) are trended on **developed ultimates**, never closed-only
cohorts — closure bias produces a fake −9.5%/yr on BI.

## D-008 — Trees: forward-factor only (Option B), not full de-trending
**2026-10-20 · accepted · judgement, flagged for review**

Train tree models on developed-but-not-de-trended labels and apply only the
forward factor `index(2025)/index(2024)` at scoring. Rationale: trees cannot
extrapolate a linear trend from a `year` feature, and full de-trending makes the
whole level depend on the index being exactly right. The GLM may use full
de-trending as the cleaner, explainable formulation. Revisit if backtests show
the partial approach underperforms.

## D-009 — 2025 level factor: blended claims-index + premium anchor (70/30)
**2026-10-20 · proposed · judgement — needs human sign-off**

Central 2025 factor = 0.7 × claims-index extrapolation (3-yr CAGR) + 0.3 ×
premium-implied factor (2025 premium/exposure vs 2024, under constant LR).
Low/high scenarios bracket it. **Open question for the team:** alternatives are
pure claims-index or pure premium/LR anchor (2025 prices are already trended
+5%/yr, arguably the more defensible single source).

## D-010 — Floor predictions at 1e-3
**2026-10-20 · accepted · verified against sklearn**

`mean_tweedie_deviance(power=1.5)` raises on `y_pred <= 0` despite the brief
saying "zero or above". Floor at 1e-3 in `build_submission`.

## D-011 — Feature deny-list enforced by test
**2026-10-20 · accepted**

`renewal_decision` is training-only **and** is outcome information. Also exclude
post-outcome fields if any appear. `tests/test_submission.py` asserts the deny-list
columns never appear in a model matrix.

## D-012 — Weather features are in scope, assumption documented
**2026-10-20 · accepted · verified empirically**

`weather` includes 2025 for 100% of test addresses. Assume annual climate indices
are available at pricing time. Pre-empt the leakage question in the write-up and
run a sensitivity using lagged (2024) weather.

## D-013 — Routine submissions during development
**2026-10-20 · accepted · user-directed**

Submit after every meaningful model change so the leaderboard acts as an external
test set for feature-level decisions. Budget 10/day. Every submission goes through
`scripts/kaggle_submit.py` (validation + logging) and is recorded in
`docs/experiment_log.md`. One change per submission where practical, otherwise the
feature attribution is confounded.

## D-014 — Experience features from `claims` are the top priority feature block
**2026-10-20 · accepted · verified empirically**

Lagged customer/policy claim history lifts motor frequency 0.083 → 0.486 and
cost/exposure 323 → 1,750 (LR 0.47 → 2.37). Must be lagged strictly (only
loss dates before the row's `period_start`). Fallback to `prior_claims_5y` for
customers unseen in history.

## D-016 — `customer_service_contacts` is outcome leakage — deny-list
**2026-10-20 · accepted · verified empirically**

A model using this single feature scores **D² = 0.628** on a 2024 holdout (motor),
against 0.001 for `written_premium`. Claimants average 3.28 contacts vs 0.41 for
non-claimants (ratio 8.1, max 36). Contacts are generated *during* the policy
period by the claim itself. The brief gives it no time qualifier — unlike
`missed_payments_12m` ("in the previous 12 months") — confirming it is post-outcome.
Excluded from every model. A leaked submission would score far above the public
leaderboard top (0.13) and would be disqualified on review.

Detection note: local D² near 0.6 is the tell. Anything above ~0.2 on this data
must be treated as leakage until proven otherwise.

## D-017 — `telematics_score` excluded from the primary model (ambiguous window)
**2026-10-20 · proposed · judgement — needs human sign-off**

"Driving score from the device" with no stated observation window; measured over
the policy period it would leak. Univariate signal is weak (mean 58.2 for
claimants vs 60.2, single-feature D² negligible) and it is 87% missing. Excluded
from the primary model; kept as a documented sensitivity variant only.

## D-015 — `written_premium` is a benchmark, not just a feature
**2026-10-20 · accepted · verified empirically**

Premium deciles rank risk 4x and LR is roughly flat (0.71–0.81). The baseline to
beat is `written_premium × LOB LR`; the value of a model is in finding where
current pricing is wrong (prior claimants LR 2.3, young/old drivers).

## D-018 — Local CV is a validated proxy for the leaderboard
**2026-10-20 · accepted · verified against 4 submissions**

Local D² (time split 2021–23 → 2024, developed truth, LOB-recalibrated, Tweedie
p=1.5) tracks the leaderboard with a consistent optimism gap:

| local | LB | gap |
|---|---|---|
| 0.0797 | 0.07293 | −0.0068 |
| 0.1207 | 0.11226 | −0.0084 |
| 0.1230 | 0.11339 | −0.0096 |
| 0.1250 | 0.11519 | −0.0098 |

Ranking is correct in all four cases. **Consequence:** iterate locally and spend a
submission slot only on a new local best, or when a hypothesis genuinely needs an
external test. Expected LB ≈ local D² − 0.009.

## D-019 — Feature attribution confirmed externally
**2026-10-20 · accepted · verified against the leaderboard**

Model vs flat LOB rate +0.0393; lagged experience +0.0011; weather + flags +0.0018.
The dominant gain is risk differentiation, not any single feature block. Experience
is sparse in training (7.7% of rows have history) but denser at 2025 scoring
(20–26% of test rows), so its true contribution is likely understated here.

## D-020 — Severity is a Gamma scale family (per peril, per year)
**2026-10-20 · accepted · verified empirically**

Gamma beats lognormal on AIC for all 11 perils; shape and log-SD are stable across
accident years. A per-peril, per-year scale factor is sufficient calibration
(D-007 approach). Tweedie p=1.5 (Poisson–Gamma compound) is well specified.

## D-021 — `written_premium` is a decoded formula with known misalignments
**2026-10-20 · accepted · verified empirically**

Motor log-premium R² = 0.956 (age 3 bands, ncb 3 bands, vvg log-linear, cover,
region, +4%/yr). Pricing and risk are generated separately: premium ignores claim
experience, vvg premium rises while frequency falls, TPFT is priced like comp but
has 0.47 the frequency, buildings LR 1.08 vs contents 0.67. Use premium as the
*base* and let the model correct these misalignments — not as the answer.

## D-022 — Recoveries are rule-based; the excess is inert
**2026-10-20 · accepted · verified empirically**

Motor own_damage with `third_party_id` → exactly 0.8 × incurred; theft with
`third_party_id` → exactly 0.3 ×. No recoveries on property perils. The policy
excess is never applied to claim amounts (net = gross in 87% motor / 100% property),
confirming D-003. Do not spend features on the excess columns for target purposes.

## D-025 — Local ranking is unreliable for near-ties; use the leaderboard for those
**2026-10-20 · accepted · verified against 6 submissions**

A user-requested population-shift test submitted two locally-inferior runs:

| exp | local | LB | verdict |
|---|---|---|---|
| exp-006 (premium-derived features) | 0.1242 (worse) | **0.11569** (better) | local ordering **flipped** |
| exp-005 (freq/sev split) | 0.1102 (much worse) | 0.10314 (much worse) | ordering **held** |

Rules that follow:
1. Do **not** drop a candidate because it is ≤0.002 D² worse locally — that is
   inside the 2024→2025 population-shift noise. Submit it, or at least re-run with
   several seeds before discarding.
2. Local differences ≳0.010 are trustworthy in direction.
3. Local D² is still the right *level* estimate (gap to LB is stable at
   −0.007 to −0.010 across six points).
4. Keep a small reserve of slots specifically for tie-breaking, since that is the
   only way to resolve them.

## D-026 — Frequency/severity split does not help on this book
**2026-10-20 · accepted · verified locally and on the leaderboard**

Poisson frequency × Gamma severity (exposure double-count corrected) scores
local D² 0.1102 vs 0.1250 for the single Tweedie, and LB 0.10314 vs 0.11519.
Cause: multiplying two noisy conditional expectations compounds error, and
count and severity are not independent given the features. A single Tweedie
(p=1.5, Poisson–Gamma compound — well specified per D-020) estimates E[cost|x]
directly and is the better choice. **Negative result, kept as evidence.**

## D-023 — Frequency/severity split as a model variant
**2026-10-20 · accepted for trial · verdict in D-026**

(exp-005's message referenced this ID before the entry was written — gap-filled
same day.) Poisson frequency × Gamma severity as an alternative to the single
Tweedie, motivated by the BI tail (DGP findings §6b). Mechanics: Poisson count
model exposure-weighted, Gamma severity on claims with cost > 0, combined as
E[cost] = E[count] × E[severity]. Verdict: loses to the single Tweedie — see
D-026. Kept as an actuarially standard cross-check for the write-up.

## D-024 — Premium-formula derived features
**2026-10-20 · accepted · verified on the leaderboard (exp-006)**

(exp-006's message referenced this ID before the entry was written — gap-filled
same day.) Expose the decoded premium structure (D-021) to the tree instead of
making it re-derive it: `premium_per_exposure`, `age_band` (3 bands), `ncb_band`
(3 bands), `vvg_log`. exp-006 tested this as one change: locally −0.0008 vs
exp-004 but **+0.0005 on the leaderboard** — the first D-025 near-tie flip.
Conclusion: the premium structure transfers to 2025 and the model benefits from
its components being exposed rather than re-learned.

## D-027 — Time-safe historical relativity features (strictly lagged)
**2026-10-20 · accepted · verified on the leaderboard (exp-007)**

Target-encoded historical relativities `<level>_hist_rate` and `<level>_hist_freq`
per categorical level (cost rate and claim frequency per exposure), credibility-
smoothed toward the line global with 200 exposure units of prior. Leakage rule
(AGENTS.md 6) enforced structurally: a query row only sees source rows with
`source.pe < row.ps` — since every claim attributed to a source row has
`loss <= source.pe`, all contributing loss dates precede the row's period_start.
Guardrailed by `tests/test_features.py` (own-row, overlapping-period and
unseen-level cases on toy frames).

Result: local +0.0006 / +0.0009 across seeds 0/1 vs exp-006 same-seed;
**LB exp-007 = 0.11637 (+0.00068 over exp-006)** — new best. The features give
the tree the historical level of each risk class so it can learn deviations
(price/risk misalignment per D-021) rather than raw category identity.

## D-028 — Per-peril development factors, level-neutral
**2026-10-20 · proposed · verified locally, no skill change — tie-break candidate**

Open question 3: distribute the accepted D-005 LOB×AY development level across
perils using their own age→ultimate factors (incurred triangle, volume-weighted
chain ladder from `scripts/build_development.py`): motor bodily_injury 1.535 and
buildings subsidence 1.306 at age 0 vs 1.01–1.05 for short-tail perils. The
cost-weighted mean factor within each (LOB, AY) is kept **exactly** equal to the
accepted D-005 factor (level-preserving normalisation, `tests/test_development.py`).

Local result: D² 0.1223 vs 0.1242 under the old truth, but the model-vs-baseline
skill gap is identical (0.0444 vs 0.0445) — the label redefinition moved truth
and predictions together. Within the D-025 tie band. Actuarially the per-peril
truth is the more defensible definition (BI really does develop more than OD);
kept as the proposed label definition and as a tie-break submission candidate.
Needs human sign-off before replacing D-005 labels in the primary stack.

## D-029 — 3-seed ensemble does not move the leaderboard
**2026-10-20 · accepted · verified locally and on the leaderboard (exp-009)**

Equal-weight average of 3 seeds on the exp-007 stack: local 0.1250 vs best single
0.1248 (mean single 0.1242) — variance reduction is real but small. Submitted as
exp-009: **LB 0.11632 vs exp-007 0.11637 (−0.00005)** — a tie. Second D-025
near-tie flip (this time against the ensemble). Consequence: seed averaging is
free robustness for a production model but is **not a scoring lever** here; do
not spend slots on ensembling again. The 2024→2025 population shift dominates
everything below the ~0.002 D² scale.

## D-030 — Join address and customer dimension tables; never drop features silently
**2026-10-20 · accepted · verified on the leaderboard (exp-010)**

`build_frame` never joined `addresses` or `customers`. The policy tables carry
only `address_id` / `customer_id`, so every property-geo and demographic rating
factor was **listed in the feature sets but silently absent from the model
matrix** (`align` dropped unknown names): `region`, `flood_zone`, `soil_type`,
`property_type`, `roof_type`, `energy_rating`, `urban_density`, `basement`,
`trees_within_5m`, `solar_panels`, `listed_building`, `elevation_m`,
`distance_to_river_km`, `year_built`, `bedrooms`, `prior_claims_5y`, and the
customer demographics. All models exp-002…exp-009 trained without them.

Fix: `attach_attributes` joins `ADDRESS_COLS` + `CUSTOMER_COLS` (static rating
attributes, never claim-derived, so no lag is required under AGENTS.md rule 6);
`align` now **raises** on any requested feature missing from the frame; guardrail
tests assert every listed feature exists in the built frame.

Result (exp-010 = exp-007 stack + the block): local D² 0.1248 → **0.1342**
(+0.0094, well outside the D-025 tie band), LB 0.11637 → **0.12113**
(**+0.00476**) — largest single gain since the model itself. Importances led by
`occupation`, `region`, `elevation_m`, `energy_rating`, `year_built`, matching
the DGP findings (flood_zone 1→5 RR 190×, elevation ≈330×, distance_to_river 36×
for flood; region/urban_density inside the decoded premium formula).

Gap note: local−LB widened from −0.0084 to −0.0131 — the block appears even more
informative on the 2025 population than the 2024 holdout, consistent with D-025.
