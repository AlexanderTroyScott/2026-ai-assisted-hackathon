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
post-outcome fields if any appear. `tests/test_guardrails.py` asserts the deny-list
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

## D-031 — Denied-claim history as strictly lagged features; hold the last slot
**2026-10-20 · accepted · verified by tests and local exp-011; human decided the slot call**

Repudiated claims carry zero cost (D-004) but the *refusal event* is
claim-occurrence information: a customer with a denied claim has demonstrably
had a loss event, and the premium formula ignores claim experience entirely
(D-022/DGP findings). `load_denied_claims` / `denial_features` in `src/dataset.py`
add `cust_denied_all`, `pol_denied_all`, `days_since_denied`, strictly lagged
(`loss < ps`) under the same leakage rule as the experience block (D-014,
AGENTS.md rule 6) — a row never sees denials inside its own period. Counts
default to 0; `days_since_denied` stays NaN for "no history" so the model can
tell that apart from a recent denial. Opt-in `denied` feature group.

Verification: 25 tests green, including two new ones asserting own-period
exclusion and the clean-history zero-defaults. exp-011 (exp-010 stack + the
block, same seed, same truth): local D² **0.1339 vs 0.1342** — −0.0003, inside
the D-025 tie band. No measurable gain.

Human decided (2026-10-20): do **not** spend the last quota slot (9 of 10 used)
on a local near-tie — D-025/D-029 show the leaderboard cannot resolve it. The
block stays implemented and tested but opt-in; promote it into the primary stack
only with evidence outside the tie band. Quota slot reserved for the next change
that clears the band.

## D-032 — Actuarial references layer; candidates gated like everything else
**2026-10-20 · accepted · human-provided sources, transcribed and verified**

Human supplied three CAS sources (Friedland reserving study note; Anderson et al.
GLM guide; Goldburd et al. GLM rating monograph) for two purposes: actuarial
credibility in the write-up and actionable method ideas before the deadline.
Recorded in `docs/references.md` as **R-01…R-03** with an R→D mapping, and
distilled into the `actuarial-methods` skill so agents apply the standard
practice without loading the PDFs.

Standing rule adopted with them: references justify **methods**, never numbers
(AGENTS.md rule 1 stands). Ideas from the sources enter the queue as **C-1…C-5**
(`proposed`), and each one that we act on goes through the normal flow — local
experiment → `D-xxx` → experiment-log row → quota slot only if it clears the
D-025 tie band or is explicitly accepted as a credibility play. Nothing in the
bibliography changes a frozen decision (D-005 LDFs, Tweedie p=1.5, freq/sev
result) without a new measurement. Sources that bear on existing findings are
reconciled in `docs/references.md` "Notes" (e.g. R-02 App. J says freq/sev and
Tweedie are often similar — our exp-005 measured it worse here; both recorded).

## D-033 — LDF table is derived, not typed; age is keyed off the valuation year
**2026-10-21 · accepted · correctness fix, verified against data_findings.md §5**

Review found the frozen LDF table in `src/dataset.py` (and a second copy in
`scripts/make_baseline.py`) mis-mapped accident year to age-to-ultimate factor.
The documented age-to-age factors are motor 0→1=1.223, 1→2=1.027, 2→3=1.003;
buildings 0→1=1.089, 1→2=1.003; contents 0→1=1.040, 1→2=1.004. At the 31 Dec
2024 valuation an accident year sits at `age = 2024 − AY`, so 2023 is **age 1**
and needs only the 1→2 (and 2→3) factors.

The old table gave 2023 the full **age-0** factor — buildings 1.092 instead of
1.003, contents 1.044 instead of 1.004 — overstating 2023 developed labels by
**~8.9%** and **~4.0%**. 2021/2022 were similarly, and more mildly, overdone.
Motor happened to be right at 2023 and wrong only in the mature years.

Changes:
- `AGE_TO_AGE` holds the documented age-to-age factors as the single source of
  truth; `age_to_ultimate()` multiplies out the remaining factors, so `LDF` is
  **derived**, never hand-typed. `scripts/make_baseline.py` now imports it —
  the duplicated table was what let the two drift.
- Mature accident years (2021, at age 3) get exactly 1.0: the documented
  factors are the whole tail, so inventing a residual loading would breach
  AGENTS.md rule 1.
- `tests/test_development.py` pins `AGE_TO_AGE` and the age-0→ultimate column
  to `data_findings.md` §5 and asserts mature years are undeveloped.

Reconciliation (`row_target`, all lines): observed totals match the documented
triangle to 0.1M (motor 105.9/116.7/117.3/90.7, buildings 54.1/61.5/64.8/67.4,
contents 14.9/16.0/17.5/17.5) and 2024 develops to 114.2 / 73.6 / 18.3 against
the documented ~114 / ~73.6 / ~18.3. 2023 buildings now develops 64.8 → 65.0,
not 64.8 → 70.8.

**Not fixed here:** `scripts/build_development.py` compares its
`DOCUMENTED_LOB_AGE0` (the 0→1 age-to-age factors) against an age-0→ultimate
quantity, so the check reports a permanent false mismatch and could never have
caught this. Its own re-derivation also lands at 1.18/1.06/1.03 vs the accepted
1.26/1.09/1.04 (see `experiment_log.md`, "different weighting window"). Left
open as a separate item rather than silently re-anchoring D-005.

## D-034 — 2025 level factor: implement D-009's blend, expose scenarios
**2026-10-21 · proposed · implements D-009, which itself is unratified**

Review found `FORWARD_FACTOR = {1.027, 1.053, 1.054}` was *exactly* the
premium/exposure 2025-vs-2024 column from `data_findings.md` §6 — i.e. only the
30%-weight component of D-009 — applied as a single hard-coded point and
presented as fact. D-009 is still `proposed`, and the accepted per-peril
severity trend (D-007) was not feeding it at all.

`src/dataset.py::forward_factors(scenario)` now builds the factor from the two
documented indications only:
- **claims-index** — 3-yr CAGR of the mix-adjusted severity index (D-007), which
  holds the 2021 peril mix fixed so mix shift cannot contaminate the trend;
- **premium anchor** — premium/exposure 2025 vs 2024 (`data_findings.md` §6).

`central` = 0.7 × claims-index + 0.3 × premium (D-009 as written). `low`/`high`
are the two indications themselves, so the bracket claims no more precision than
the sources support.

| scenario | motor | buildings | contents |
|---|---|---|---|
| low | 1.0270 | 1.0530 | 1.0354 |
| **central (D-009)** | **1.0350** | **1.0556** | **1.0410** |
| high | 1.0385 | 1.0567 | 1.0540 |

`scripts/run_experiment.py --trend {low,central,high}` selects the scenario and
records it in the run manifest, so a submission can never inherit an untracked
level choice.

**Still `proposed`, needs human sign-off.** The level moves total predicted cost
by roughly +0.8pp / +0.3pp / −1.3pp versus the old premium-only factors, and
`experiment_log.md` already notes local D² declines for factors beyond 1.05 —
buildings sits at 1.0556 central. Per AGENTS.md this stays `proposed` until a
human accepts it; until then treat `central` as the working assumption, not a
decision. Baseline scripts still do **not** apply any forward factor (a separate
finding), so model-vs-baseline comparisons remain on a mixed basis.

## D-035 — Triangle derivation: respect the valuation diagonal
**2026-10-21 · accepted · correctness fix, reproduces data_findings.md §5**

Review item 6 was reported as a `build_development.py` guardrail comparing
incompatible quantities. The real defect was upstream of the guardrail: the
triangle builder itself.

`build_cum` did `unstack(fill_value=0)` then `cumsum(axis=1)`. At the 31 Dec
2024 valuation accident year 2024 has reached **age 0 only** — no 2024 claim
has been reported at age 1 — but the fill-then-cumsum wrote 90.7M into that
unreached cell and every cell after it. `triangle_factors` then volume-weighted
over **all** AYs, so each immature AY contributed a phantom 1.0 ratio and
dragged every factor toward 1.0.

Corrected derivation, using only AYs that have actually reached both ages:

| | buggy | corrected | documented |
|---|---|---|---|
| motor f(0→1) | 1.166 | **1.221** | 1.223 |
| motor age-0→ult | 1.183 | **1.257** | 1.26 |

**The `data_findings.md` §5 numbers are correct and the script was wrong.** The
"D-028 re-derivation 1.18/1.06/1.03 vs documented 1.22/1.09/1.04 (different
weighting window)" note misdiagnosed this as a weighting-window difference. It
was the diagonal bug. No change to the findings document is required.

Changes:
- `build_cum` blanks cells past `age > EXTRACT_YEAR − AY`; `triangle_factors`
  drops any AY that has not reached both ages.
- `load_claims` restricts to the documented origin years 2021–2024. The 2020
  rows are pre-book orphans and are far too thin to anchor a factor — 2 claims
  of contents/flood swung f(1→2) to ~10x before this.
- The guardrail now compares **like with like**: age-to-age against the
  documented 0→1/1→2/2→3, and age-0→ultimate against the documented "to
  ultimate" column. It previously compared 0→1 factors to an age-0→ultimate
  quantity, so it reported a permanent false mismatch and could never have
  caught D-033.
- `PERIL_LDF` re-frozen from the corrected derivation (motor BI 1.535 → **1.708**,
  subsidence 1.306 → **1.464**; short-tail perils move ~1pp). Long-tail perils
  were the most under-developed, which is the direction the tail evidence
  predicts (bodily_injury 10.2% of claims report >365 days).
- `tests/test_development.py` pins the volume-weighted factors on the published
  motor triangle and asserts the fill-zero+cumsum variant dilutes f(0→1) to
  1.166 — the exact regression.

**Blast radius was contained**: D-028 normalises per-peril factors to the
accepted D-005 LOB level, so the **level never moved** and the D-005 table is
untouched (it came from the findings document, not this script). Only the
per-peril split was wrong, which affects `--development peril` alone.

Re-measured under D-028's protocol (skill-vs-baseline gap, same features/seed,
because a truth redefinition moves D² on its own and earns nothing):

| | model D² | baseline D² | gap |
|---|---|---|---|
| exp-012 peril (corrected) | 0.1280 | 0.0769 | 0.0511 |
| exp-013 lob | 0.1312 | 0.0791 | 0.0521 |

Gap −0.0010, inside the D-025 band. **D-028 stands as a tie**, now reading
slightly against per-peril rather than slightly for it. Per-peril remains
opt-in and unpromoted.

## D-036 — Baselines share the model's 2025 basis
**2026-10-21 · accepted · comparability fix**

`make_baseline.py` applied D-005 development but **no** forward factor, while
`run_experiment.py` applied one to every model submission. The two were on
different level bases: the baseline predicted 2025 at a 2024 level. That mixes
trend into any model-vs-baseline leaderboard gap and makes exp-001's 0.07293
incomparable with later entries.

Measured contamination is small — a uniform level miss of the cost-weighted
forward factor (1.0429) moves D² by **+0.00066** on the real 2024 holdout,
against a D-025 tie band of ~0.002. So the recorded conclusion that risk
differentiation dominates (+0.0393) survives; at most 0.0007 of it was level.
This is an attributability fix, not a restatement.

Baselines now take `--trend {low,central,high}` and apply the same
`forward_factors()` as the model, and the filename records the scenario
(`constant_rate-central.csv`).

## D-037 — Deny-list asserted on the model matrix
**2026-10-21 · accepted · guardrail fix**

The D-011 deny-list test grepped `run_experiment.py` source text and its first
assertion was `assert ... or True`, i.e. dead. A leak that reached the model
matrix by any route other than a quoted name in a feature list would have
passed.

- `align` now raises on any deny-listed column in the **returned** matrix —
  the thing the model actually sees, however the column got there.
- `tests/test_guardrails.py` asserts `align` refuses a request containing
  `renewal_decision`, that a clean request returns only what was asked, and
  that every documented feature group is disjoint from `DENYLIST`.
- Verified by mutation: removing the `align` guard fails the test.

## D-038 — Rolling-origin fold 2021-22 → 2023 (D-006) implemented
**2026-10-21 · accepted · validation completeness**

D-006 specified two rolling-origin folds; only the 2024 one was built.
`--fold {2024,2023}` now selects the holdout year and the manifest records it.

Stability read (exp-013 vs exp-014, same features/seed, D-028 gap protocol):

| fold | train AY | holdout | model D² | baseline D² | gap |
|---|---|---|---|---|---|
| 2024 | ≤2023 | 2024 | 0.1312 | 0.0791 | 0.0521 |
| 2023 | ≤2022 | 2023 | 0.1385 | 0.0878 | 0.0507 |

Gaps agree to 0.0014 (inside the D-025 band) and the top features are the same
(occupation, region, written_premium, exposure). This is **evidence of
stability for the write-up**, not a second tuning opportunity — per D-025 local
differences below ~0.002 do not rank on the leaderboard. The 2024 fold stays
primary because it is closest to the 2025 scoring population (D-018).

## D-039 — The IBNER gap is now exposed and measured
**2026-10-21 · accepted · negative result that pins an open quantity**

exp-015 (2026-10-21) isolated the D-033/D-035 label correction with the level
held at premium-only, so the slot tested the label change and nothing else. It
scored **0.11955 vs exp-010's 0.12113 (−0.00158)** — a correct fix that scores
worse. 1 of 10 slots used; best remains exp-010.

This is the first measurement bearing on the IBNER component, and it changes how
D-005 should be read:

- The age-to-age factors in `data_findings.md` §5 are correct *as chain-ladder
  factors*. D-035 now reproduces them exactly. That part is closed.
- They are documented **lower bounds** that "capture late reporting but not
  IBNER on open claims", and D-005's own wording requires "*plus IBNER on open
  claims*". That component has never been estimated.
- The old mis-keyed table applied the age-0 factor to accident years at age 1.
  Wrong — but it was accidentally supplying a loading in the direction of the
  missing IBNER. Removing the bug without replacing the loading leaves the
  training level short by roughly the documented gap.

Evidence the gap is real and material rather than noise: open-claim share is
**20.4% for 2024** against 3.2% for 2023; 2024 open claims carry a mean case of
~7.7k versus 21–27k for mature open claims. The immature year is exactly where
the bug over-developed and exactly where IBNER lives. Mapping −0.00158 D²
through the measured level sensitivity gives roughly a **−5% level miss**.

**No loading has been invented** (AGENTS.md rule 1). The gap is now a measured,
named quantity instead of an accidental one.

Consequences:
- C-1 (Bornhuetter-Ferguson / Cape Cod labels, `docs/references.md`) is no longer
  a speculative refinement — it is the candidate that closes a measured gap.
  Same gate as every other change: local experiment → `D-xxx` → experiment-log
  row → quota slot only if it clears the D-025 band.
- D-009's level question is entangled with this. The old accidental IBNER
  loading plus the premium-only level netted out near-correct for 2025; fixing
  either one alone leaves the level short. The two must be measured in separate
  slots to stay attributable.
- exp-015 is retained as a **valuable negative result**. It is the run that made
  the IBNER gap visible.

**Open question for the human:** whether to spend the next slot on (a) the label
correction plus an estimated IBNER component (C-1, the principled fix), or
(b) the label correction plus the D-009 central level (faster, but entangles
D-009). Both are single-primary-change if kept separate from each other.
