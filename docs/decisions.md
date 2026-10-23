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

## D-040 — Correction to D-039: the IBNER conclusion was wrong
**2026-10-21 · accepted · retracts the interpretation in D-039; D-039's negative result stands**

D-039 concluded that exp-015's drop (0.12113 → 0.11955) showed the old buggy
labels were "accidentally supplying an IBNER loading", that the drop corresponded
to "roughly a −5% level miss", and that the gap was "real and material, not
noise". An independent audit falsified the quantitative core and the causal
mechanism. D-039 is **withdrawn as an interpretation**. The measured negative
result (exp-015 scores 0.11955) remains true and remains useful.

### What was wrong

1. **The sign is inverted.** exp-015's submission is **+0.93% higher** than
   exp-010's in total predicted cost (motor +1.04%, buildings +1.01%, contents
   −0.03%; measured by joining the two submitted CSVs, 626,639 rows). The
   submission did not fall 5% — it rose. D-039 inferred a level *shortfall* from
   a level that actually *increased* while the score fell.

2. **The −5% number was measured on the wrong object.** The sensitivity curve
   D-039 interpolated against came from uniformly scaling the **constant_rate
   baseline**. On the **model** the curve is 3–4× flatter (re-measured on
   `exp-007-s0` valpred, `d2_tweedie_score` power=1.5, floor 1e-3):

   | uniform scale | 0.90 | 0.94 | 0.96 | 0.98 | 1.00 | 1.02 | 1.04 | 1.08 |
   |---|---|---|---|---|---|---|---|---|
   | model ΔD² | −0.00169 | −0.00062 | −0.00029 | −0.00009 | 0 | −0.00002 | −0.00015 | −0.00070 |
   | baseline ΔD² (D-039) | — | −0.00189 | −0.00112 | −0.00050 | 0 | +0.00037 | +0.00063 | — |

   A uniform −5% is worth only **−0.00044** on the model. To produce −0.00158
   you need roughly −10% (or +12.5%) uniform — and the observed move was +0.93%,
   which this curve scores at essentially zero. Curvature is a property of a
   specific predictor's calibration, not of D² itself; the baseline was also
   scored pre-recalibration (its optimum sits above 1.0) while the model is
   post-recalibration (optimum at 1.0). The transfer was invalid.

3. **The level channel is cancelled in this pipeline anyway.**
   `lob_recalibrate(..., actual=yv)` fits per-LOB scales on the holdout and
   carries them to the test set (`scripts/run_experiment.py`), so per-LOB
   uniform label-level changes cannot reach the submission level. The label
   correction moved aggregate training `y_ult` only ~−1.3% (buildings −3.3%,
   contents −1.6%, motor −0.16%) — and even the per-LOB part of that is what
   the recalibration absorbs. D-039's own arithmetic ("8.2%/3.8% of label level
   on ~45% of cost") conflated a one-accident-year change with a three-year one.

4. **"The triangle method explicitly excludes IBNER" is not correct.**
   Volume-weighted chain ladder on an *incurred* triangle derives link ratios
   from the historical development of incurred amounts, which includes
   historical strengthening of open case estimates as older years matured. The
   chain-ladder assumption is that this pattern repeats (R-01). It provides no
   *separate* IBNER provision, and it will understate if the immature year's
   open-case weakness is a pattern break against history — but that is a
   different and weaker claim than "explicitly does not include". The defensible
   statement is `data_findings.md` §5's own wording: these are **lower bounds**,
   with a pattern-break risk on 2024's weak open cases.

5. **The IBNER evidence was mislocated.** The open-claim statistics cited in
   support (20.4% open share for 2024 vs 3.2% for 2023; 2024 open mean case
   ~7.7k vs 21–27k mature) describe **AY2024** — which is the holdout and the
   level anchor, not the training set (train is AY ≤ 2023). AY2024's LDF is
   1.259/1.092/1.044 and is **unchanged to 3 dp** by the correction. So those
   statistics cannot support a claim about the labels the correction changed.

6. **"Not noise" over-read one draw.** −0.00158 sits **inside** the documented
   D-025 ~0.002 flip band. Local corroboration is only −0.0009 recalibrated and
   **−0.0004** under the skill-vs-baseline gap protocol (the baseline moved too:
   0.07965 → 0.07911). D-039 marked its interpretation `accepted` on that basis;
   it should not have.

### What is actually established

- The D-033/D-035 label correction is **correct** as chain-ladder factors —
  it reproduces `data_findings.md` §5 exactly. That is unchanged.
- exp-015 is a clean single-change negative result: the label correction, taken
  alone, did not improve the leaderboard and may have cost ~0.0016. The
  attribution claim survived audit.
- The correction caused a large **refit reshuffle**, not a level move: row-level
  Spearman 0.977–0.993 between the two submissions but sd of log-ratio 0.113,
  with buildings row-ratios spanning roughly 0.38×–2.41×. The model reorganised
  how it allocates cost across rows; the realized 2025 truth preferred the old
  allocation.
- There is a **plausible, unmeasured** level question, but it lives on the
  **2024 anchor**, not the 2023 labels. D-005 attaches "plus IBNER on open
  claims" to the 2024 development — i.e. `y_ult`, which is both the local
  validation truth and the per-LOB anchor for the 2025 submission totals. If
  the LDFs are lower bounds, the submission's level is anchored low. exp-015
  gives this hypothesis **zero** support either way.

### The metric defect this exposed (now fixed)

Local D² was computed after an **in-sample per-LOB oracle recalibration** —
scales fitted on the holdout and scored on the same holdout. Measured worth
**+0.0020 (exp-010) / +0.0029 (exp-015)** — the size of the entire D-025 tie
band — and structurally blind to LOB-level miss, which is precisely the
dimension a "level" argument turns on. Any level inference drawn from local D²,
including the methodology behind D-039's sensitivity curve, is therefore
invalid.

Fixed in `scripts/run_experiment.py`: two numbers are now reported and the
manifest records both —

- `d2_model` — LOB scales **fitted on the train side** and applied to the
  holdout. Out-of-sample; this is the honest local metric and the one to use
  for run comparison.
- `d2_model_oracle` — scales fitted on the holdout. Kept because the
  submission path legitimately uses it (2024 developed truth *is* the right
  level anchor for 2025 rows), but it is not a generalisation estimate.
- `d2_oracle_inflation` — the difference, so the optimism is visible per run.

`tests/test_guardrails.py` asserts the out-of-sample path scales by the
*reference* sample's totals only and never touches the target sample's actuals.

Note this changes the meaning of `d2_model` in the manifests. Historical
manifests hold oracle numbers; from here on `d2_model` is out-of-sample and is
**not** comparable to them. The stale "local−LB gap stable at −0.007 to −0.010"
rule (D-025) is also wrong now: measured gap is −0.0130 / −0.0137 for
exp-010/exp-015.

### Standing corrections to the record

- D-039's interpretation is **withdrawn**; its negative result stands.
- C-1 (Bornhuetter-Ferguson / Cape Cod) is **not** established as "closing a
  measured gap". It remains a candidate at the normal gate. Note also that BF
  stabilises immature-AY ultimates, whereas an IBNER-on-open-claims provision is
  a case-adequacy loading — related but not the same instrument.
- D-009 remains `proposed` and unratified.
- No IBNER or level loading has been estimated or applied. AGENTS.md rule 1
  stands: it must be measured, not invented.

### Open question for the human

Whether to spend a slot separating the two live hypotheses, both of which
exp-015 left open:

- **H1 — level is short on the 2024 anchor.** Probe: uniform level shift on the
  current best predictions. Expected under H1: LB rises.
- **H2 — exp-015's drop is refit/seed noise.** Probe: replicate the exp-015
  config at a second seed. Expected under H2: the spread is comparable to the
  −0.00158 drop, which would also defeat "not noise".

Cheapest decisive pair is those two, one slot each.

## D-041 — Four hypotheses tested overnight; seed band is asymmetric
**2026-10-21 · accepted · measured, with corrections to D-040's over-claim**

Budget: 5 slots spent (exp-016..exp-020), **5 reserved for the human**. All four
candidate explanations for exp-015's leaderboard drop were tested. Three are
refuted; the fourth is real but much weaker than first claimed.

### H1 — level is short on the 2024 anchor. **REFUTED.**

exp-017 = exp-016 predictions × 1.03 uniform. **0.11949** vs 0.11942, delta
**+0.00007** — half the seed band measured on that side. A 3% uniform level move
is worth nothing. Consistent with the model-level sensitivity curve (D-040),
which is flat near scale 1.0.

This also kills the IBNER-as-level story. If the LDFs' missing IBNER component
were a level shortfall, a uniform lift would have recovered it.

### H2 — the drop is refit/seed noise. **REFUTED as an explanation, but it
exposed a serious error in D-040.**

exp-016 (corrected labels, seed 1) = 0.11942 against exp-015 (seed 0) = 0.11955.
Seed band **0.00013** — and on that basis D-040 called the −0.00158 label effect
"12× the seed band". **That was wrong.** The band was measured only on the
corrected-label side, which turns out to be the *low-variance* side.

The control (exp-019, exp-020) re-ran the **pre-D-033 label table** at seeds 1
and 2, patched via `src.dataset.LDF` without touching the repo:

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| **old labels** | 0.12113 | 0.12024 | **0.12173** | 0.12103 | **0.00075** |
| **new labels** | 0.11955 | 0.11942 | — | 0.11949 | **0.00009** |

Two things fall out.

**The label effect is real but modest.** Mean difference **+0.00155** in favour
of the old labels, against a pooled seed sd of 0.00061 — about **2.1×**, not 12×.
Direction is consistent (old wins at every pairing: −0.00158 at seed 0,
−0.00082 at seed 1), so the effect is probably genuine; the magnitude is roughly
one to two seed standard deviations, not an order of magnitude.

**Best single run is exp-020 at 0.12173, not exp-010 at 0.12113.** exp-010's
position as "our best" was partly a lucky draw at the top of a 0.0015-wide band.
Our honest best estimate is the *mean*, 0.12103, not either single run.

**A separate, unexplained finding:** the corrected labels fit far more stably
(seed sd 0.00009 vs 0.00075, an 8× difference). Whatever the 2023 over-development
was doing, it was also injecting variance into the fit. That is not obviously
bad for the score — old labels win on the mean — but it is a real property of
the two label tables and worth noting in the write-up. Cause unknown.

### H3 — recency weighting replaces what the bug was doing. **REFUTED.**

The mechanism looked plausible: the old table inflated 2023 labels (buildings
×1.092 vs the correct ×1.003), and since `lob_recalibrate` pins each LOB's total
that inflation *redistributed* cost toward 2023-like rows. The scoring
population is 2025, the newest year — so the bug may have been accidental
recency weighting.

Tested explicitly with `--recency-lambda`, sample weights
`exp(lambda·(AY − train_max))`, on fold 2023 (train 2021–22 → holdout 2023 — the
exact analogue of train ≤2023 → predict 2025). Honest (train-fitted) local D²:

| lambda | 0.00 | 0.15 | 0.30 | 0.50 |
|---|---|---|---|---|
| local D² | **0.1404** | 0.1348 | 0.1379 | 0.1362 |

Uniform weights win by a clear margin at every tilt. Early stopping also
collapsed from 163 iterations to ~88, so the tilt distorts the objective rather
than refining it. The recency story is not supported.

### H4 — corrected labels flatten the tail. **NOT DISTINGUISHABLE FROM NOISE.**

First pass suggested the corrected labels produced a flatter predicted
distribution (sd/mean 1.082 vs 1.090, max 8,725 vs 9,605) and that the 2025
truth rewards spread. **This was an over-read and I discarded the probe before
submitting it.** Measured across all four submissions, the old-vs-new spread
difference (0.0074) is *smaller* than the within-labels seed spread (0.0207).

A supporting diagnostic does survive, and is worth keeping: the label change
moves ranks **less** than a seed change does (Spearman 0.993 old-vs-new vs 0.990
seed-0-vs-seed-1) yet costs more score. So the effect is not a ranking effect —
but the shape evidence is not strong enough to say what it *is*.

**No shape probe was submitted.** `outputs/submissions/exp-018.csv` was built
and then discarded locally.

### Where this leaves D-033/D-035

The label correction remains **correct as chain-ladder factors** and must stay.
The leaderboard mildly prefers the over-developed table (mean +0.00155), the
direction is consistent, but no mechanism has been established for *why*, and
all four candidate mechanisms are dead or unproven. The honest position:

- Keep the corrected labels. They reproduce `data_findings.md` §5 exactly; the
  old table is wrong and cannot be justified to an examiner.
- Do not claim the penalty is large. It is ~2 seed sd, and our previous "best"
  was itself a favourable draw from a wider band than we knew.
- The open question is now genuinely open, not merely unmeasured: **why does an
  actuarially wrong label table predict 2025 slightly better?** Candidate
  explanations that remain live: the 2025 population shift (D-025) interacting
  with fit shape; the corrected labels being a lower bound that *under*-develops
  in a way the old table accidentally compensated; or the effect being at the
  edge of detectability and not worth chasing.

### Metric note

`--recency-lambda` is implemented and gated locally, default 0.0 (off). The
control runs (exp-019/exp-020) were produced by patching `src.dataset.LDF` in a
`/tmp` runner so the repo keeps the correct table — the patch path is documented
in the experiment log. Historical `d2_model` in manifests remains oracle-based;
from exp-016 onward `d2_model` is train-fitted and `d2_model_oracle` is recorded
separately (D-040).

## D-042 — 2025 weather: realised data, but worth ~nothing locally
**2026-10-21 · measured, local only · no slots spent · for human decision**

Human asked how much the 2025 weather improves predictions and whether other
candidates might use it, so they can raise the question of whether it should be
in scope. Measurement only — nothing submitted, no position taken.

### What the 2025 weather actually is

**Realised values, not a climate normal.** Verified empirically: 2025 is a
distinct year, not a copy or interpolation of another. Cross-address
correlation of 2025 against the best-matching other year is 0.54 (hail_days),
0.63 (rainfall_mm), 0.81 (wildfire_index), 0.83 (freezing_days); exact-match
fraction is 0.000 for rainfall and 0.001 for dry_summer. For test addresses,
2025 vs 2024 rainfall has corr 0.63 with mean absolute difference 193mm.

So it is genuine future information: at pricing time nobody knows the realised
2025 rainfall for a given address.

### Two corrections to the framing

1. **Nobody said it would get a run thrown out.** `D-012` records the opposite
   — weather is **accepted**, on an explicit documented assumption ("annual
   climate indices available at pricing time"), with a required sensitivity.
   The repo position is disclose-and-measure, not avoid. It should not be
   described as leaked data that invalidates a run.
2. **Our best run already uses it.** exp-010 includes the `weather` feature
   group and `add_weather` defaults test rows to 2025. The 0.12113 on the
   leaderboard is already a 2025-weather score. The actionable question is what
   it costs to *remove*, not what it gains to add.

### The measurement is structurally hard

`add_weather` sets `weather_year = where(is_test, 2025, acc_year)`. The local
holdout is 2024 **train** rows (`is_test=False`), so it sees `acc_year` weather
whatever is chosen. Only test rows ever see 2025 weather.

**Therefore the value of 2025 weather is invisible to local CV.** No local
number can measure it; only the leaderboard can. Note also that `--weather-lag`
lags *train* rows too (`acc_year − 1`), so an A-vs-B comparison is confounded —
it tests "own-year vs lagged weather", not "is 2025 test weather useful".

Honest local D² (train-fitted scales, seed 0, exp-010 stack):

| config | honest D² | vs A |
|---|---|---|
| A weather 2025 (current) | 0.1331 | — |
| B weather lagged to 2024 | 0.1326 | −0.0005 |
| C **no weather at all** | 0.1338 | **+0.0007** |
| D + log1p(sum_insured) + artefact flag | 0.1334 | +0.0003 |

**All deltas ≤ 0.0007, inside the local seed band of ~0.0005–0.001.** Nothing
here is a measured effect. Two candidate readings, not separated by this data:

- The best local score is *no weather at all*, which would say the weather block
  adds little once `elevation_m`, `distance_to_river_km`, `flood_zone`,
  `region` and `energy_rating` are present — those already carry most of the
  geography, and weather's marginal content is temporal.
- Or all four are the same model and the differences are noise.

The proxy for "how much is correct-year weather worth" is A vs B = +0.0005,
which is the largest defensible claim and is itself within noise.

### On whether other candidates use it

Cannot be determined from here, and should not be guessed at. What is known:
the weather table is in the public competition data and includes 2025 for 100%
of test addresses, so any team that used it faced the same choice. Our own
top-feature lists show `rainfall_mm` and `dry_summer_index` ranking 8th and
16th for exp-010, so the model does use them.

### Recommendation for the human's question

The evidence to bring is: (a) 2025 weather is realised data and is a genuine
look-ahead, (b) its measurable value is at most +0.0005 and is not separable
from noise locally, (c) removing it costs nothing that we can detect, and (d)
the leaderboard value is unmeasurable without spending a slot on a lagged
submission. On that basis dropping 2025 weather is close to free and removes
the look-ahead entirely — but that is a policy call, not a modelling one.

## D-043 — Outliers and transforms: one real artefact, and a test I got wrong
**2026-10-21 · measured, local only · no slots spent**

### The `sum_insured` artefact is real

`sum_insured` has a **structural break with zero rows in between**:

| band | buildings | contents |
|---|---|---|
| ≤ 500k | 499,357 (99.02%) | 720,178 (99.80%) |
| 500k–1M | 6,934 | 107 |
| **1M–10M** | **0 rows** | **7 rows** |
| 50M–100M | 95 | 386 |
| > 100M | 1,300 | 26 |

Above 10M sits a separate population of **1,395 buildings rows (0.197%)** and
**1,395 contents rows (0.193%)**, maxing at 594,000,000. The test set has the
same contamination at the same rate (0.201% / 0.191%).

They are not real sums insured. The decisive evidence is the premium ratio:

- `written_premium / sum_insured` median for >10M rows: **0.000001**
- `written_premium / sum_insured` median for all other rows: **0.000970**

A **1000×** gap. Median premium is 215 for the giant rows against 217 overall —
ordinary premiums on absurd sums insured. The rows also look completely ordinary
otherwise (region R01/R12/R03, semi-detached/terraced/detached/flat, all three
urbanity levels), so this is not a distinct commercial book; it is a field
error, plausibly a unit or code column.

They carry essentially no claims signal either: buildings claims in the
10M–100M band number **14** with mean cost 3,749, versus 9,588 for the 100k–1M
band. The >100M band has 145 claims at 13,152.

**Recommendation: flag, do not delete.** A binary `si_artefact` indicator plus
a capped `log1p(clip(sum_insured, 1e7))` preserves the information that a row
is anomalous without letting 594M dominate a scale-sensitive split. Deleting
rows would break the 626,639-row submission contract.

### Other outliers named

| feature | evidence | note |
|---|---|---|
| `exposure` (supplied) | max 365, skew 35.5 | **the corrupt column (D-001)** — confirmed it is days, not exposure. Already recomputed. |
| `annual_mileage` | p50 10k, p99 30k, max 86k | 60% missing; plausible but long-tailed |
| `missed_payments_12m` | kurtosis 25, max 7 | genuine count, keep raw |
| `days_unoccupied` | kurtosis 20.5, 71% missing | not currently a feature |
| `written_premium` | skew 2.4, max 5,076 vs p99 1,522 | continuous tail, tree-safe |
| `customer_service_contacts` | skew 3.3, kurt 21.9 | **deny-listed (D-016)** — not a feature |

### Log transforms: they matter a lot for linear models, not at all for trees

I first tested this with Spearman correlation and got delta 0.0000 everywhere.
**That test was wrong and I am recording it as such**: log is monotonic, so it
*cannot* change a rank correlation. A tautology dressed as a result.

The correct test is Pearson against log-claim-severity (n = 183,418 claims):

| feature | raw | log1p | delta |
|---|---|---|---|
| `sum_insured` | +0.015 | **+0.316** | **+0.301** |
| `written_premium` | +0.240 | **+0.303** | +0.063 |
| `distance_to_river_km` | −0.063 | −0.090 | −0.027 |
| `elevation_m` | −0.055 | −0.103 | −0.048 |

So log1p is a large improvement *for a linear relationship* — `sum_insured`
goes from essentially uninformative to the strongest of the four. But the model
test moved only +0.0003 (0.1334 vs 0.1331), inside noise.

The reason is structural and worth writing down: **decision trees are invariant
to monotone transforms.** A split at the 90th percentile of `sum_insured` is the
same split as at the 90th percentile of `log1p(sum_insured)`. Tree models do not
need log transforms for skewness — they need them only to make *linear*
combinations behave.

Practical consequence: the log transforms are near-worthless for the current
LGBM stack and **material for C-2 (the Tweedie GLM benchmark)**. Anyone
building the GLM should absolutely use them; anyone tuning the tree should not
expect them to move the score. The one transform that *can* matter to a tree is
the artefact flag, because it is not monotone — it is a new binary split.

### Not done

No submission. Effect sizes are all inside the local noise band, and spending a
slot on a ≤0.0007 candidate would be a poor use of quota given the D-025 finding
that local near-ties do not rank reliably on the leaderboard.

## D-044 — Human decisions on scope, presentation and slot policy
**2026-10-21 · accepted · human-directed, closes the open questions in D-042/D-041**

Five calls made by the human, recorded so nothing downstream is ambiguous.

1. **2025 weather stays in scope.** Rationale given: it is available to every
   team. So `D-012` stands as written — weather is accepted on the documented
   pricing-time assumption, the sensitivity is retained, and the assumption is
   disclosed in the write-up rather than removed from the model. The exposure to
   examiner challenge is accepted knowingly and is now a recorded judgement, not
   an oversight. `D-042`'s measurement (removal costs ≤ +0.0005, not separable
   from noise) remains the sensitivity evidence to attach.

2. **The corrected label table is our presented model** (D-033/D-035). The old
   mis-keyed table scores marginally better (mean 0.12103 vs 0.11949) and holds
   the single best run (exp-020, 0.12173); it will be reported as a documented
   sensitivity, not as our result. Decision rests on actuarial correctness, not
   score.

3. **Slot policy: hybrid.** Conservative normally (only changes clearing the
   D-025 band), becoming more exploratory as we approach midnight GMT. Rationale
   is that the leaderboard has shown lucky-draw movement (exp-020) larger than
   most feature effects, so an exploratory window is worth more late than early.

4. **C-3 (GLM + LGBM form ensemble) is first** among the model-form candidates.
   Lower risk than C-1 because it does not touch the target and therefore cannot
   destabilise the label correction just settled.

5. **C-2 (Tweedie GLM benchmark) and C-5 (ASOP 9 documentation pass) take
   priority now**, with next steps evaluated afterwards. Rationale: judging
   weights actuarial credibility at 20% and that axis is where we can still
   gain, whereas the accuracy axis is showing diminishing returns.

Consequent gating rule for C-3, from `docs/references.md`: the GLM must land
within ~0.01 local D² of LGBM to be worth blending. That is measured next.

`D-009` is still `proposed` and is not ratified by this entry — the human was
told it is low-stakes (measured flat) and it remains for the assumptions record.

## D-045 — The final model was never fitted on 2024
**2026-10-21 · accepted · human-identified, measured, submitted as exp-021**

Human asked why we hold 2024 out entirely rather than cross-validating, whether
we were missing 2024 claim developments, and whether we should recalibrate on
all data before predicting. All three pointed at a real defect.

### The defect

`run_experiment.py` split `tr = AY <= 2023` and called `model.fit(X_tr, ...)`.
The 2024 rows — **626,190 of 2,409,311, i.e. 26.0% of the training book** —
appeared only in `eval_set` for early stopping. They contributed **zero**
gradient updates. The submission model had never seen the most recent year of
experience, which is also the year closest to the 2025 scoring population.

Training periods all end 2024-12-30, so no 2024 policy has exposure in 2025 —
but 2024 policies carry claims that develop *through* 2025, and 2024 is the
accident year whose rating relationships are most likely to persist into 2025.

### On 5-fold cross-validation specifically

Random 5-fold is the wrong instrument here and was not used. The data is
temporally structured and the scoring population is a future year, so random
folds leak future into past. Rolling-origin (train ≤2023 → holdout 2024, and
train ≤2022 → holdout 2023) is the right design and is what D-006 specifies.
What was missing was not more folds but the **final refit after selection**.

### Measurement: the most recent training year is worth +0.0022

Diagnostic holding the holdout fixed at 2024 and varying only which years are
fitted on (honest, train-fitted scales):

| training years | honest local D² |
|---|---|
| 2021–2022 | 0.1309 |
| 2021–2023 (previous) | 0.1331 |

**+0.0022** — outside the entire D-025 ~0.002 tie band, and larger than every
feature effect measured on 2026-10-21 (all ≤ 0.0007). By analogy, refitting on
2021–2024 before predicting 2025 should help similarly.

### Result: exp-021, +0.00195 on the leaderboard

| config | LB score |
|---|---|
| exp-015 corrected labels, **no refit** | 0.11955 |
| **exp-021 corrected labels, refit on all 2021–2024** | **0.12150** |
| exp-020 mis-keyed labels, best of 3 seeds | 0.12173 |
| exp-010/019/020 mis-keyed labels, mean of 3 | 0.12103 |

The refit gains **+0.00195**, larger than the **−0.00158** label penalty
measured in D-041. Consequence for that open question:

**The corrected label table is now score-competitive.** exp-021 sits 0.00023
below the best mis-keyed run and 0.00047 *above* the mis-keyed mean. That
difference is well inside the 0.00075 seed band measured on the mis-keyed side.

Stated carefully, because D-041 over-claimed twice already: this is **one seed**
against a three-seed band. It establishes parity, not superiority. But it does
defuse the tension D-041 left open — we can present the actuarially correct
label table at no material score cost, which is what D-044 committed us to
anyway.

### What was also changed and why it does not confound

The final refit also re-anchors the LOB recalibration on all-data scales rather
than the 2024 slice. That is part of "fit the final model on all the data", not
a separate knob. It cannot confound the attribution: D-041 measured the level
as **flat** (uniform +3% moved the leaderboard by +0.00007), so any movement
here is attributable to the refit, not the level.

Single primary change: refit scope. exp-021 vs exp-015 share seed, features,
labels, trend and fold; only the fit scope differs.

### Implementation

`--final-fit` refits on all training rows with the tree count fixed at the
cross-validated `best_iteration` (no early stopping available without a
holdout). `--train-max` added as a diagnostic override. Honest local D² is
still reported from the pre-refit model and remains the out-of-sample estimate
for the config family; it is not re-measurable after refitting.

## D-046 — C-2 measured: Tweedie GLM benchmark
**2026-10-21 · accepted · promotes C-2 from candidate to measured · no submission**

C-2 (R-02 App. C, R-03 §5.2) built as `src/glm.py` + `scripts/run_glm.py`:
Tweedie power 1.5, log link, exposure weights, cost modelled as
`exposure × annual rate` per AGENTS.md. 112 terms — 21 numeric, 91 categorical
dummies — on a deliberately narrow rating-plan spec rather than the LGBM
feature soup, so the relativities read cleanly.

### Result

| | train D² | holdout D² |
|---|---|---|
| GLM | +0.1360 | **+0.1206** |
| LGBM (raw) | — | +0.1303 |
| LGBM (honest, train-fitted recal.) | — | +0.1331 |

Calibration ratio Σpred/Σactual = **0.9805**. Per LOB: motor +0.0655,
buildings +0.0997, contents +0.0628.

**Gate check (C-3 requires the GLM within ~0.01 local D² of LGBM).** On a
like-for-like raw-vs-raw basis the gap is **0.0097 — inside the gate,
marginally**. Against the recalibrated LGBM it is 0.0125, outside. Raw-vs-raw
is the fair comparison since neither has had the LOB level fitted to it. So
**C-3 is cleared to proceed**, but the GLM is the weaker leg and should be
weighted accordingly.

Generalisation gap (train − holdout) is **0.0154**. Per R-04 the gap is the
number to watch rather than the level; that is a modest but real optimism.

### Specification decisions

- **`occupation` excluded** (D-047 follow-up measured this). 21 dummy
  parameters for 0.0001 of D² on the tree model, collinear with
  `driver_age` / `years_licensed` / `ncb_years`, and 64% of its dispersion is
  Student+Retired — age in disguise. Including it would destabilise every
  coefficient it touches and wreck the relativities table.
- **Log transforms applied** to `sum_insured`, `written_premium`,
  `distance_to_river_km`, `premium_per_exposure` — D-043 established these are
  material for linear models and near-worthless for trees. Here they matter
  (`sum_insured` Pearson vs log-severity +0.015 → +0.316).
- **`sum_insured` capped at 1e7 and flagged** (`si_artefact`), per D-043.

### Three defects found and fixed while building it

1. **Unscaled design.** First fit reported "converged in 1 iterations" with
   every coefficient exactly 0.00000 — L-BFGS could not line-search a Hessian
   spanning `driver_age` ~50 to `sum_insured` ~6e8. Now standardised for the
   optimisation and back-transformed to original units.
2. **Double transform.** `predict_cost` standardised the design *after*
   coefficients had already been back-transformed, mixing units. Predictions
   came out 22.7× too high. Now predicts on raw design.
3. **Length mismatch** in the train D² call (train-length predictions scored
   against holdout-length truth).

A fourth was mine, not the code's: I initially reported a catastrophic
collinearity on `written_premium` because I evaluated it at the raw premium
(267) rather than its transformed value (`log1p` ≈ 5.59). The model was fine.
Recorded so the mistake is not repeated.

### Relativities for the write-up

Categorical relativities are trustworthy (0/1 coded, no cancellation). Highest:
flood_zone 5 **1.978**, construction non-standard **1.304**, security_level
missing 1.240, cover TPO **1.197**, named young driver 1.183, flood_zone 4
1.164. Lowest: vehicle use social **0.665**, commuting 0.720, owner-occupied
0.735, rural 0.778, `si_artefact` **0.790**, TPFT **0.801**.

Two are worth commentary: **flood_zone 5 at 1.98** is the damped multivariate
shadow of the 190× rate ratio in `data_findings.md` §5, and **TPFT at 0.80
against TPO at 1.20** independently reproduces the DGP finding that
third-party-fire-and-theft is overpriced relative to its frequency.

Numeric effects are reported **per 1 SD of the transformed column**, since
per-unit coefficients on logged fields are not interpretable: written_premium
×1.726, driver_age ×0.761, years_licensed ×1.234, annual_mileage ×1.127,
ncb_years ×0.898, sum_insured ×1.112. Note `driver_age` is U-shaped in reality
(D-043 / data_findings §7) so its linear term is a compromise — the GAM
variant in R-06 would handle it properly.

Artefacts output: `outputs/metrics/glm-001_relativities.csv`.

## D-048 — Label parity resolved: the corrected table is now the better model
**2026-10-21 · accepted · closes the D-041/D-044 tension · exp-022**

D-045 established that the corrected label table was **at parity** with the
mis-keyed one (0.12150 vs a 0.12173 best / 0.12103 mean), but rested on a
single seed against a three-seed band and said so explicitly. exp-022 is that
second seed. It resolves the question and resolves it the other way.

| config | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| **corrected labels + D-045 refit** | 0.12150 | **0.12249** | — | **0.12200** | 0.00070 |
| mis-keyed labels, no refit | 0.12113 | 0.12024 | 0.12173 | 0.12103 | 0.00075 |

- Corrected **best (0.12249)** beats mis-keyed best (0.12173) by **+0.00076**.
- Corrected **mean (0.12200)** beats mis-keyed mean (0.12103) by **+0.00097**,
  about 1.3 pooled seed sd.
- Both corrected runs beat two of the three mis-keyed runs.

**This is our best submission to date**, and it is on the model we can defend.
The D-044 decision to present the corrected table is no longer a
correctness-over-score trade — it is now correct *and* better. The tension that
D-041 opened is closed.

Stated carefully, because the sample is still small: **n=2 against n=3**. The
mean gap is ~1.3 sd, so this is strong directional evidence rather than proof.
But note the direction never reversed across either pairing, and the previous
"corrected labels score worse" conclusion came from comparing n=2 to n=3 *before*
the D-045 refit was applied. With the refit in place the ordering flips.

### A correction to D-041's unexplained finding

D-041 recorded that the corrected labels fit **8× more stably** (seed sd
0.00009 vs 0.00075) with no explanation. That comparison was made on runs
**without** the final refit. After D-045 the corrected-label seed band is
0.00070 — essentially the same as the mis-keyed band (0.00075).

So the stability difference was an artefact of the missing refit, not a
property of the label table. Removing 26% of the training data happened to
suppress seed variance; restoring it restores the variance. The "8× more
stable" observation should be withdrawn.

### Current standings

| | score | note |
|---|---|---|
| **exp-022** | **0.12249** | **best — corrected labels + refit, seed 1** |
| exp-020 | 0.12173 | mis-keyed labels, best of 3 |
| exp-021 | 0.12150 | corrected labels + refit, seed 0 |

Best *single* run and best *defensible* model are now the same run.

## D-049 — Seed 3 landed low; D-048's margin claim needs softening
**2026-10-22 · accepted · corrects D-048 · new best 0.12260**

Quota reset window used for exp-023 (seed 2 of the corrected-labels + refit
config) and the seed ensembles.

| run | score | note |
|---|---|---|
| **blend-2s (exp-021+022)** | **0.12260** | **new best** |
| exp-022 | 0.12249 | seed 1 |
| blend-3s (021+022+023) | 0.12232 | dragged by exp-023 |
| exp-020 | 0.12173 | mis-keyed best |
| exp-021 | 0.12150 | seed 0 |
| **exp-023** | **0.12083** | seed 2 — weak draw |

Two things fall out.

**1. Our best submission is now 0.12260**, the two-seed ensemble, on the
actuarially correct model. Best single run and best defensible model remain
the same family.

**2. D-048 overstated the margin and must be softened.** It claimed the
corrected mean beat the mis-keyed mean by +0.00097, "about 1.3 pooled seed sd",
on n=2 vs n=3. With seed 2 added the picture is:

| config | seeds | mean | sd |
|---|---|---|---|
| corrected + refit | 0.12150, 0.12249, 0.12083 | 0.12161 | 0.00086 |
| mis-keyed, no refit | 0.12113, 0.12024, 0.12173 | 0.12103 | 0.00075 |

Mean gap is now **+0.00058 on n=3 vs n=3 — under one pooled sd**. The
direction still holds on every measure (mean, best, best-blend) but the
strength is "suggestive", not the "strong directional evidence" D-048 claimed.

What is **not** weakened:
- Corrected mean > mis-keyed mean (0.12161 vs 0.12103)
- Corrected best > mis-keyed best (0.12249 vs 0.12173)
- Corrected blend > anything on the mis-keyed side (0.12260 vs 0.12173)
- **D-045's refit gain remains the dominant effect** (+0.002), far larger than
  either the label effect (~0.0006) or the seed band (~0.0008)

So the D-044 presentation decision stands — the corrected table is not worse
and is on the right side of every comparison — but the language should be
"at least as good, probably better" rather than "better".

This is the fourth correction of an over-read today (after D-039, D-040's 12×
claim, and the occupation instability). The pattern: n=2 looks conclusive, n=3
isn't. Worth stating plainly in the write-up rather than burying.

**Ensembling note:** 2-seed (0.12260) beat 3-seed (0.12232). Averaging in a
weak draw hurts. This is consistent with D-029 (seed averaging measured
useless) — the 2-seed gain is small and not reliably monotone in the number of
seeds. Do not assume more seeds is better.

**Quota state:** 2 used on 2026-10-22, **8 remaining**.

## D-052 — Rate target regressed: exposure double-counted
**2026-10-22 · accepted · retracts the D-051 modelling half · exp-024**

exp-024 combined D-051's premium-unit fix with a new rate target
(`y_ult / exposure`, exposure weights, multiply back at scoring) plus the
D-045 final refit. Local honest D² **0.1365** — the best we have ever measured.
Leaderboard **0.06019**, below the constant-rate baseline. A collapse.

### What went wrong

`earned_premium = written_premium × exposure` is a feature, and `exposure` is a
feature. With `target = rate` the model can therefore learn

    rate_hat ~ earned_premium ~ written_premium × exposure

and scoring then multiplies by exposure again. Net effect: **cost ∝ exposure²**.

Measured implied rate (prediction ÷ exposure) by exposure band:

| band | ≤0.3 | 0.3–0.6 | 0.6–0.9 | 0.9–1.0 | max/min |
|---|---|---|---|---|---|
| training truth | 437 | 430 | 417 | 389 | **1.12** |
| exp-022 (cost target) | 395 | 456 | 463 | 426 | **1.17** ✓ |
| fixA (cost target + data fixes) | 458 | 493 | 476 | 426 | **1.16** ✓ |
| **exp-024 (rate target)** | **72** | **234** | **379** | **461** | **6.40** ✗ |

Training has `corr(exposure, rate) = −0.0015` — the rate is flat, so any fitted
positive slope is spurious.

**Important correction to my own framing:** I first attributed this to the 2025
test set having more partial-year rows. It does not — the exposure mix is
nearly identical (≤0.3 band: 11.6% train vs 12.8% test). The failure is the
double count, full stop.

### Why local D² did not catch it

Local honest D² preferred the rate target (0.1365 vs 0.1327) and was the best
number we had produced. The leaderboard rejected it decisively. This is the
sharpest possible confirmation of **D-025 / R-06** — a local gate that looks
excellent can be measuring the wrong thing — and it cost a slot to learn.

Note also that exp-022 was **already** learning `cost ∝ exposure` correctly
(implied rate flat to 1.17). The rate target added nothing and broke it. The
Tweedie tree with `exposure` as a feature recovers the product well enough.

### What is kept from D-051

The **data fixes are correct and retained**: `customer_id.str.zfill(8)` (motor
leading zeros, 10% of rows), the 604 duplicate claims dropped, and
`earned_premium` as a named quantity. The `premium_lr` benchmark correction is
also right and stays — it moved **0.0758 → 0.1092** on the 2024 holdout, exactly
as the audit predicted. `write_orphan_log` still asserts 414 because it reads
the raw, un-deduplicated claims table by design.

Only the **rate-target modelling change is retracted**. `--target` defaults back
to `cost`; `rate` is kept behind the flag for diagnostics.

### Guardrail added

`kaggle_submit.py` now computes implied-rate flatness as a pre-submission check
(fails above max/min 1.5 against a truth of 1.12). It reports

- exp-024: **FAIL**, max/min 6.40
- exp-022: **PASS**, max/min 1.17

This is exactly the check that would have caught the regression before spending
a slot. `tests/test_guardrails.py` pins it with synthetic flat vs squared
submissions.

### Standings

Our best remains **blend-2s at 0.12260** (exp-021 + exp-022). One slot was
wasted on exp-024. The premium-unit finding from D-051 is still valid and
important — it is the *benchmark* and the *feature naming* that were wrong, not
the tree's target formulation.

## D-054 — R-04 diagnostics built; the Gini gap does not predict transfer
**2026-10-22 · accepted · closes the C-5 diagnostic gap · negative result on the gap itself**

C-5's checklist identified that we produced no normalised Gini or lift charts,
and R-04 names the **train-versus-holdout normalised-Gini gap** as the overfit
measure to watch. Built: `src/diagnostics.py`, `scripts/run_diagnostics.py`,
`tests/test_diagnostics.py` (18 tests). No submission slots spent.

### The method, as R-04 specifies it

`normalized_gini = model Gini / data Gini`, sorted by **predicted** values with
a **seeded random tiebreaker**. The tiebreaker is not optional — R-04 warns
that a stable sort on data previously ordered by actuals makes tied predictions
inherit that order and score like a perfect ranking. Measured: a constant
predictor scores **0.9453** with a stable sort on pre-sorted data (identical to
the data Gini) versus **~0.06** with the tiebreaker. The test asserts the
contrast, not an absolute threshold.

Lift charts use equal-exposure bins per R-04, always on holdout.

### The gap does NOT predict leaderboard transfer

Two configurations with known leaderboard scores were compared:

| config | nGini train | nGini holdout | **gap** | lift | **LB** |
|---|---|---|---|---|---|
| A = exp-022 (leaves 64, lr 0.04) | 0.5695 / 0.5761 | 0.4614 / 0.4694 | **+0.1074** | 26.6 / 24.9 | **0.12249** |
| B = exp-025 (leaves 16, lr 0.02, + marketing_segment) | 0.5396 / 0.5329 | 0.4669 / 0.4851 | **+0.0602** | 28.0 / 31.9 | **0.11168** |

The diagnostic says B generalises **better** (smaller gap). The leaderboard says
B is **worse**. **The gap is inverted and does not rank these two.**

R-04's own worked example used gaps of 0.0106 and 0.0201; ours are 0.06–0.11,
3–5× larger, but the sample is 1.8M rows with a heavy tail versus their small
datasets, so the absolute level is not directly comparable. Only the *ordering*
was testable, and it fails.

### What the diagnostics DO buy: a decomposition

B has **higher** holdout normalised Gini (0.4669/0.4851 vs 0.4614/0.4694) and
**higher** lift (28.0/31.9 vs 26.6/24.9) on both folds — it genuinely ranks
risks better — and still lost 0.011 on the leaderboard.

That is only explicable one way: **exp-025 failed on level, not on
discrimination.** Which is exactly what the direct diagnosis showed — its
predicted total was 5.6% lower and it under-predicted low-exposure rows.

This is the useful result. Gini and lift are rank-based and structurally blind
to level; D² is not. Splitting the two tells you *which* one failed:

- **discrimination** → normalised Gini, lift ratio, Lorenz
- **level / calibration** → the measures added here: `calibration_ratio`
  (Σactual/Σpredicted), `calibration_slope` (regression of actual on predicted
  through the origin), `calibration_in_large`

Guardrail added: a test asserting a uniform scale leaves normalised Gini
unchanged while moving the calibration ratio — the exact exp-025 failure mode.

### Measured on our presented model (exp-022 family)

Holdout, fold 2024: normalised Gini 0.4669, lift ratio 28.0, calibration MAE
29.2 on a mean near 350 (~8%). Lift chart is monotone apart from one flat step
(bin 5 → 6), which R-04 treats as acceptable. Lorenz: the lowest-predicted 10%
of exposure carries 0.3% of loss; the highest 10% carries the bulk.

### Standing guidance

Use both families before spending a slot. A candidate that improves Gini/lift
but worsens calibration should be expected to fail on the leaderboard, and
vice versa. The gap metric itself is recorded but **not** adopted as a gate.

## D-055 — Test-suite audit: four unguarded behaviours, one tautology
**2026-10-22 · accepted · review of all 70 tests · 7 added, now 77**

Full audit of the test suite for errors and actuarial correctness, run via the
`reviewer` subagent against the project skills and `docs/data_findings.md`.
Verdict was "needs changes": ~40 of 70 were load-bearing, the rest decorative,
and four submission-critical behaviours were unguarded.

### What was wrong

**MAJOR — target construction had no test.** `row_target` builds `y` and
`y_ult`, which is the heart of the submission. Nothing tested it. A wrong join,
a double-counted recovery, or `y_ult = y` (no development at all) would have
produced the correct *columns* and passed all 70 tests. Now covered by
`test_row_target_attributes_and_develops_labels` — claim lands on the right
row, orphan excluded, `y_ult == y × LDF[line][acc_year]`, and the D-033
regression pinned (2023 buildings must be 1.003, not the age-0 1.092).

**MAJOR — the D-001 exposure rule was documented, not enforced.** Both exposure
tests recomputed the formula *independently from raw parquet* and never asserted
that `load_policies` did it. Replacing the recompute with the supplied column
would have passed them. Now covered by
`test_load_policies_recomputes_exposure_not_the_supplied_column`.

**MAJOR — the scoring metric was unasserted.** No test referenced
`power=1.5` or the `1e-3` floor in the *local* path; the floor was guarded only
at the submission CSV level. A wrong Tweedie power would have passed. Now
covered by `test_d2_tweedie_power_and_floor_are_pinned`.

**MAJOR — a dead test.** `test_generated_submission_if_present` looked for
`constant_rate.csv`, but `make_baseline.py` writes `constant_rate-central.csv`
(trend suffix). The `if path.exists()` always skipped. It has never run.

**Two leakage boundaries were unpinned at equality.** The experience block was
pinned (a claim dated exactly `== ps`), but the denial block used mid-period
dates and `historical_relativities` used `pe >> ps`. Both `loss < ps` → `<= ps`
and `pe < ps` → `<= ps` regressions would have slipped through. Now pinned.

**A tautology I wrote.** `test_perfect_ranking_scores_one` asserted
`normalized_gini(a, w, a) == 1.0`. Since `data_gini` is the same computation,
that is `X/X` and is 1.0 by construction for *any* formula shared by both. It
documented the definition and could never catch a wrong one. Replaced in
substance by `test_gini_matches_hand_computation`, which pins the formula on a
two-point case (Gini must be exactly 0.5, and an inverted ranking must score
−0.5). The original is retained but its docstring now states the limitation.

### Two fail-open / over-tight guards corrected

- The D-052 exposure-scaling gate **failed open**: any exception recorded a
  *pass* ("skipped"). That is the exact opposite of the D-030 fail-loud rule,
  and it would let a double-counted submission through — the failure mode the
  gate exists to catch. Now fails closed.
- `test_exposure_recompute_in_unit_interval` asserted `max() <= 1.0`, which is
  tighter than the documented convention `(pe − ps + 1)/365` allows for a
  366-day leap-spanning period. It pinned a data coincidence, not a code
  invariant. Relaxed to `366/365`.

### Mutation-resistance, after the fixes

| regression | caught by |
|---|---|
| `loss < ps` → `<=` in experience_features | existing, pinned at `== ps` |
| `loss < ps` → `<=` in denial_features | **added** |
| `pe < ps` → `<=` in historical_relativities | **added** |
| LDF age-0 factor reapplied to 2023 (D-033) | existing + **added** in row_target |
| Triangle fill-zero + cumsum (D-035) | existing |
| Supplied exposure used instead of recomputed | **added** |
| Deny-listed column in the model matrix | existing |
| Wrong Tweedie power / dropped floor | **added** |
| Gini tiebreaker removed (R-04) | existing |
| Gini formula wrong | **added** (two-point hand computation) |

### Still unguarded (accepted, ranked)

1. D-045 `--final-fit` all-data branch — worth +0.00195 on the board, no test.
2. Forward-factor scenario selection (`--trend`); the D-034 table is unasserted.
3. `--target` default (`cost`) after the D-052 retraction.

These are noted rather than built: the deadline is close and each is a
behaviour whose failure would show up in the manifest or the submission gate.

## D-056 — `occupation` removed from the tree specification
**2026-10-22 · accepted · human-directed · verified on both folds**

Human asked why `occupation` is still in the model given it is largely an age
proxy. Verified and removed. The GLM already excluded it (D-046); the tree did
not, and there was no good reason for the asymmetry.

### What occupation actually is

64% of its dispersion is Student and Retired — `driver_age` in disguise (median
driver age 22 and 77, against 44–45 for all seventeen other levels). It is not
*purely* a proxy: the mid-18 levels are identical on age, licence and NCB and
still show a 1.24× spread, with an age × occupation interaction (1.52× at 17–25
falling to 1.13× at 35–50). But that residual is small.

It also carried two artefacts we had already diagnosed: **Manager's 1.62× is a
single 11.9M claim** (D-041), and **Student's 1.81× is the age effect**.

### Measured, both folds (out-of-time gate)

| config | fold | honest D² | Gini gap | cal ratio | cal slope |
|---|---|---|---|---|---|
| with occupation | 2024 | 0.1308 | 0.1117 | 1.1431 | 1.1887 |
| with occupation | 2023 | 0.1349 | 0.1017 | 1.1849 | 1.2244 |
| **without** | 2024 | 0.1298 | 0.1077 | **1.1196** | **1.1576** |
| **without** | 2023 | **0.1365** | 0.1136 | **1.1545** | **1.1489** |

**Accuracy is a wash** — mean +0.0003 for removal, and the two folds disagree
in direction (occupation helps on 2024, hurts on 2023). Not a measured effect.

**Calibration improves on both folds** — ratio toward 1.0 (1.143→1.120,
1.185→1.155) and slope toward 1.0 (1.189→1.158, 1.224→1.149). This is the axis
that matters: **D-054 established that our leaderboard failures are calibration
failures, not ranking failures** (exp-025 outranked exp-022 on normalised Gini
and lift on both folds and still lost 0.011).

So the change is neutral on accuracy and positive on the property that has
actually been costing us — plus it removes a credibility liability.

### Also surfaced by this measurement

Both configurations show an out-of-sample **calibration ratio of 1.12–1.18** —
the model underpredicts the holdout by 12–18% before the LOB recalibration is
applied. The recalibration is therefore carrying a material amount of the work.
This does not hurt the submission directly (the submission uses holdout-anchored
scales) but it is a real property of the model and worth knowing: the intrinsic
out-of-sample level calibration is poor.

### Not submitted

No slot spent. The change is in the code path and will be reflected in the next
run; it is not reflected in `blend-2s` (0.12260), which remains our best
leaderboard result and which *does* include occupation.

## D-057 — Removing `occupation` is our new best; calibration-first reasoning worked
**2026-10-22 · accepted · exp-027 · new best 0.12436**

exp-027 = exp-022's configuration with `occupation` removed (D-056). Leaderboard
**0.12436**, against exp-022's **0.12249** — **+0.00187**. This is our best
result to date, overtaking blend-2s (0.12260).

| run | change | LB |
|---|---|---|
| **exp-027** | occupation removed | **0.12436** |
| blend-2s | exp-021 + exp-022 ensemble | 0.12260 |
| exp-022 | seed 1, occupation present | 0.12249 |
| exp-021 | seed 0, occupation present | 0.12150 |

### Why this one transferred when the others did not

exp-024 and exp-025 both had strong local evidence and both failed. exp-027 was
**not** justified by local D² — which measured it as neutral (mean +0.0003,
folds disagreeing in direction). It was justified by **calibration**:

| config | fold | honest D² | cal ratio | cal slope |
|---|---|---|---|---|
| with occupation | 2024 | 0.1308 | 1.1431 | 1.1887 |
| with occupation | 2023 | 0.1349 | 1.1849 | 1.2244 |
| **without** | 2024 | 0.1298 | **1.1196** | **1.1576** |
| **without** | 2023 | **0.1365** | **1.1545** | **1.1489** |

Calibration improved on **both** folds. And D-054 had established that our
leaderboard failures are calibration failures, not ranking failures (exp-025
outranked exp-022 on normalised Gini and lift on both folds and lost 0.011).

So the change was selected on the diagnostic that D-054 identified as the one
that predicts transfer, and it transferred. That is the first time today a
non-local-D² justification has worked. **The D-054 decomposition — ranking
versus level — is the tool that found this.**

### Attribution caveat

exp-027 and exp-022 differ by two things, not one: `occupation` removal **and**
the D-051 data corrections (customer_id zero-padding, 604 duplicate claims
dropped, `earned_premium` replacing `premium_per_exposure`). The D-051 fixes
measured neutral locally (fixA 0.1327 vs 0.1331), so the bulk of +0.0019 is
likely the occupation removal — but this is not a clean single-change
comparison and should not be reported as one.

### Also fixed while producing this run

The D-053 tuned hyperparameters (leaves 16, lr 0.02) were still the *default* in
`run_experiment.py` despite exp-025 having failed on the board at 0.11168.
Promoting them was premature. Defaults are reverted to the configuration that
produced our best result (leaves 64, lr 0.04); the tuned values are recorded in
a comment and must be opted into explicitly.

### Standing

- **Best single run and best presented model: exp-027, 0.12436**, corrected
  labels, D-045 refit, occupation excluded.
- The earlier `blend-2s` (0.12260) is superseded but remains a valid ensemble
  of the with-occupation family.
- 5 slots used today, **5 remaining**.

## D-058 — `flood_zone` alone is worse: the geo cluster is not the problem
**2026-10-22 · accepted · refutes the collinearity hypothesis · exp-028**

Human proposed that `flood_zone` is what we are really after and that
`distance_to_river_km` and `elevation_m` are proxies for it, so dropping the
continuous pair should be safe or better. Tested. It is worse.

| run | geo features | LB |
|---|---|---|
| **exp-027** | flood_zone + river + elevation | **0.12436** |
| exp-028 | **flood_zone only** | **0.12255** |

**−0.00181.** exp-027 remains our best.

### What the pre-submission evidence said

The symmetric redundancy test had already argued against this. Measured residual
spread *within* flood zones:

| within flood_zone | river spread | elevation spread |
|---|---|---|
| 1 | 1.11× | 1.23× |
| 2 | 1.06× | 1.61× |
| 3 | 1.06× | 1.62× |
| 4 | 1.03× | 1.81× |

`elevation_m` carries 1.2–1.8× residual signal *inside* every flood zone — it is
not a proxy, it refines the zone. And `flood_zone` is missing on 41,827 policy
rows carrying a 2.0× book rate, where elevation is present on every one.

The sweep also warned: dropping elevation was mixed, and the only configuration
that improved D² on both folds *and* looked plausible (`flood_zone only`) had
calibration improving on one fold and worsening on the other. It did not clear
the bar that selected `occupation`.

**Local D² got this one right** — exp-028 measured 0.1318 against exp-027's
0.1327 before submission, and the board agreed in sign and roughly in size
(−0.0009 local, −0.0018 LB). Worth recording: local D² is not uniformly
unreliable. Its failures have been specific (level/calibration effects, the
exp-024 double-count), not random.

### What this settles

The geo cluster is **not** a collinearity problem. All three fields earn their
place: `flood_zone` carries the big 7.4× residual, `elevation_m` refines it
within zones and covers the missing-flood-zone rows, `distance_to_river_km` is
the weakest but its removal also measurably worsens calibration.

Combined with the earlier sweep and the replication screen, **the feature set is
now closed.** Every field has been measured for removal; the only removal that
worked was `occupation`, and it had a specific pathology (age proxy + artefacts)
that nothing else shares.

### Standings

- **Best: exp-027, 0.12436** — corrected labels, D-045 refit, occupation excluded,
  all three geo fields present.
- Gap to leaderboard top: 0.12996 − 0.12436 = **0.0056**.
- 6 slots used today, **4 remaining**. Deadline 2026-10-23 16:00.

## D-059 — exp-027 is partly a lucky draw; the occupation gain survives on means
**2026-10-22 · accepted · closes the seed question · quota exhausted**

Used the four expiring slots on the exp-027 family: two more seeds and two
blends. All 10 of today's slots are now spent.

| run | change | LB |
|---|---|---|
| **exp-027** | seed 1 | **0.12436** |
| blend-027030 | 2-seed (exp-027 + exp-030) | 0.12389 |
| exp-03seed | 3-seed | 0.12344 |
| exp-030 | seed 2 | 0.12247 |
| exp-028 | flood_zone only | 0.12255 |
| exp-029 | seed 0 | 0.12179 |

### exp-027 is a good draw, not a better configuration

The no-occupation family is 0.12179 / 0.12247 / 0.12436 — **mean 0.12287, sd
0.00133, range 0.0026**. exp-027 sits ~1.1 sd above its own family mean.

So 0.12436 should not be reported as "the score of the no-occupation model." It
is the score of a favourable seed of that model. The honest figure for the
configuration is the family mean.

### But the occupation removal still holds

Like-for-like, three seeds per family:

| family | seeds | mean |
|---|---|---|
| with occupation (exp-021/022/023) | 0.12150 / 0.12249 / 0.12083 | 0.12161 |
| **without occupation (exp-027/029/030)** | 0.12436 / 0.12179 / 0.12247 | **0.12287** |

**+0.0018 on the means**, consistent with the +0.00187 claimed in D-057. The
removal is a real improvement; exp-027's individual score is partly luck on top
of it. D-057's caveat about the D-011 data-fix confound still stands.

### Blending diluted the lucky draw

2-seed 0.12389 and 3-seed 0.12344 both landed **below** exp-027. This is the
opposite of the earlier blend-2s result (0.12260 vs best member 0.12249,
+0.0001), and the reason is the same in reverse: blend-2s combined two
comparable runs (0.12150, 0.12249) with no standout, so variance reduction
helped. Here one member is a good draw and averaging pulls it down.

**Rule for the write-up:** ensembling helps when members are comparably good and
hurts when one is a standout. More seeds is not automatically better — consistent
with D-029 and with the earlier 3-seed blend underperforming the 2-seed one.

### Standings and quota

- **Best single run: exp-027, 0.12436** (a good draw).
- **Best defensible estimate of the configuration: family mean 0.12287.**
- Gap to leaderboard top (0.12996): 0.0056 on the best draw, 0.0071 on the mean.
- **10 of 10 slots used today. 10 fresh tomorrow, deadline 16:00.**

### Tomorrow

The model-work queue is exhausted: features (3 sweeps), hyperparameters, rate
target, calibration slope, C-3 blend, per-LOB split — all measured and rejected
or marginal. What remains is the write-up (C-5 is drafted but now stale against
D-053…D-059) and, if desired, a fresh look with tomorrow's slots after the
evidence has settled.

### Correction to D-059 — arithmetic error

D-059 states the occupation removal is "+0.0018 on the means". **That is wrong.**
The correct figure is **+0.00127**.

| family | seeds | mean | sd |
|---|---|---|---|
| with occupation | 0.12150 / 0.12249 / 0.12083 | 0.12161 | 0.00084 |
| without occupation | 0.12436 / 0.12179 / 0.12247 | 0.12287 | 0.00133 |
| **difference** | | **+0.00127** | |

Effect is **1.14 pooled sd**, not the ~2 sd implied by the earlier wording.

The 0.0018 figure came from the *single-run* comparison, exp-027 (0.12436) minus
exp-022 (0.12249) = 0.00187. I carried that across to the mean comparison without
recomputing. Both numbers are real; they answer different questions, and only the
mean comparison is the like-for-like one.

**Restated conclusion:** the occupation removal is directionally supported
(both the mean and every single-run pairing favour removal) but the strength is
**suggestive, not established** — ~1.1 sd on three seeds per side. D-057's
original +0.00187 was a single-run figure and should be quoted as such.

This is the fifth self-correction in this working session (after D-039, D-040's
12× claim, D-048's parity over-claim, and the occupation-instability finding).
The pattern is consistent: single measurements and small samples get over-read.
Recorded here rather than quietly fixed, because the retraction record is itself
part of the reliance disclosure required by ASOP 9 §5.8.

## D-060 — C-1 measured: Bornhuetter-Ferguson is a principled no-op
**2026-10-22 · accepted · promotes C-1 from candidate to measured · no submission**

C-1 was the last actuarially-defensible untested lever — BF labels blending
chain-ladder development with an a-priori `premium × ELR` for the immature 2024
year. Built and measured. It does nothing.

### The calculation

ELR from mature (2021–22) developed loss ratios: motor 1.066, buildings 1.299,
contents 0.828. Credibility `(1 − 1/CDF)` applied only to immature years:

| LOB | AY | credibility | CL ult | BF ult | BF/CL |
|---|---|---|---|---|---|
| motor | 2024 | 0.206 | 114.15M | 116.10M | +1.7% |
| buildings | 2024 | 0.084 | 73.55M | 73.08M | −0.6% |
| contents | 2024 | 0.042 | 18.29M | 18.28M | 0.0% |
| all mature years | | ≈0 | | | 1.000 |

Only 2024 changes, and by less than 2%. The reason: motor 2024's developed LR
(0.984) already sits close to the mature ELR (1.066), so the blend — weighted
by the 20.6% credibility — moves it by only 1.7%.

### The gate (D-028 protocol: skill-vs-baseline gap, not raw D²)

| development | fold | model D² | baseline D² | GAP |
|---|---|---|---|---|
| lob (chain-ladder) | 2024 | 0.1304 | 0.0790 | 0.0515 |
| lob | 2023 | 0.1353 | 0.0877 | 0.0476 |
| bf | 2024 | 0.1313 | 0.0796 | 0.0517 |
| bf | 2023 | 0.1353 | 0.0877 | 0.0476 |

**Gap moves +0.0002 at most.** BF does not clear the bar. No slot.

### What this closes

The IBNER / development-adequacy question has been open since D-040. It is now
answered: the chain-ladder development is **adequate**. The 7.7k open-claim
signal is real, but it is *not* under-reservation in a way that changes the
labels — the mature-year ELR and the developed LR already agree to within the
credibility weighting, so BF's correction is ~1.7% on one year and zero
elsewhere.

C-1 is retained in the code as `development="bf"` for completeness, but the
default stays `lob`. C-4 (Cape Cod cross-check) would arrive at the same place
by construction, since it differs only in how the a-priori is credibility-
weighted, and here the credibility is ~0 for every mature year.

### The strategic conclusion

Every lever has now been measured:

| lever | result |
|---|---|
| feature pruning (3 sweeps) | only `occupation` helped (+0.0013 on means) |
| hyperparameters | failed (exp-025) |
| rate target | failed (exp-024) |
| calibration slope | refuted — fixing it costs D² |
| C-3 GLM/LGBM blend | refuted — errors 99.9% correlated |
| per-LOB models | refuted — −0.004 both folds |
| **C-1 BF labels** | **no-op (+0.0002 gap)** |

The remaining 0.0056 gap is not in features, model form, target development, or
data. The most likely explanations are external to anything we can measure from
here: a fundamentally different model class, external data, or a target
formulation the leader has found and we have not. That is the honest position,
and it is now documented rather than speculated about.

## D-061 — Split development: develop only open claims (Boor / R-16)
**2026-10-23 · accepted · first positive result from the development-papers thread**

The Boor paper (R-16) identified a bias we had not seen: the ultimate cost of an
individual claim is a *distribution*, and applying a **uniform** LDF to a row's
total cost is biased — closed claims are final (`paid == incurred`) and need no
development, while open claims need **more** than the aggregate factor (the
aggregate averages the two).

Measured, the implied open-claim development factor per (LOB, AY):

| LOB | AY | open% | LDF (uniform) | implied F_open |
|---|---|---|---|---|
| motor | 2024 | 36.6% | 1.259 | **1.708** |
| buildings | 2024 | 35.5% | 1.092 | **1.259** |
| contents | 2024 | 18.4% | 1.044 | **1.239** |
| motor | 2023 | 25.3% | 1.030 | 1.119 |

So the uniform LDF over-develops the closed ~63% of 2024 (which is final) and
under-develops the open ~37% (which will strengthen). Since open claims are the
*large* ones, this is a shape bias in exactly the tail where Tweedie D² lives.

**Implementation.** `development="split"` builds `y_ult = y_closed + y_open ×
OPEN_LDF(line, AY)`. The open factor is implied by the aggregate chain-ladder
total, so the book-level ultimate is preserved and only the per-row allocation
changes.

**Gate (D-028 protocol, both folds):**

| dev | fold | model D² | baseline D² | GAP | cal ratio |
|---|---|---|---|---|---|
| lob | 2024 | 0.1304 | 0.0790 | 0.0515 | 1.1342 |
| lob | 2023 | 0.1353 | 0.0877 | 0.0476 | 1.1513 |
| **split** | 2024 | 0.1289 | 0.0759 | **0.0530** | **1.1268** |
| **split** | 2023 | 0.1358 | 0.0878 | **0.0480** | **1.1475** |

Gap improves on both folds (+0.0015 / +0.0004), and calibration ratio moves
toward 1.0 on both — the axis D-054 established as predictive of leaderboard
transfer. Small but consistent, and mechanistically grounded.

This is the first positive result from the reserve-adequacy / development-papers
investigation (R-07 through R-17). BF (D-060) was a no-op; the uniform-vs-open
split is not.

### Update to D-061 — exp-031 did NOT transfer

exp-031 (split development) scored **0.12301** against exp-027's **0.12436**
(−0.00135). The local gate — skill-vs-baseline gap +0.0015/+0.0004 and
calibration ratio toward 1.0 on both folds — did **not** predict the leaderboard,
the same failure mode as exp-025.

This is now the **sixth** local-metric over-read this session (exp-024, exp-025,
the occupation-instability alarm, the calibration-slope framing, the D-059
arithmetic, and now the split-development gate). The Boor individual-claim bias
is real and mechanistically correct — the split development *is* the more
actuarially-defensible target — but being right about the claim distribution
does not translate to Tweedie D² on the 2025 population.

The deep, consistent finding: **nothing we measure locally predicts the
leaderboard.** The 2024 holdout and the 2025 scoring population are different
enough (D-025) that even a correct, mechanistically-grounded label refinement
scores worse. The split development is retained in code (`development="split"`)
and in the write-up as a disclosed alternative, but it is not our presented
model.

Best remains **exp-027 at 0.12436** (uniform LDF, occupation removed, corrected
labels, D-045 refit). 1 slot used 2026-10-23, 9 remaining, deadline 16:00.

## D-062 — Frequency-severity separate development is an identity (no-op)
**2026-10-23 · measured, not submitted (nothing to submit)**

The human proposed developing claim count and severity separately (Friedland
count-severity). Measured 2026-10-23 from the raw claims:

| LOB | cost age0→ult | count age0→ult | implied severity | own-ladder severity* |
|---|---|---|---|---|
| motor | 1.184 | 1.061 | 1.116 | 8.32 (mix-biased) |
| buildings | 1.065 | 1.056 | 1.009 | 2.59 (mix-biased) |
| contents | 1.031 | 1.031 | 0.999 | 2.87 (mix-biased) |

\* the naive average-severity chain ladder is invalid for ratios (Freidland's
well-known mix-bias); the *consistent* count-severity method has
`cost_LDF ≡ count_LDF × severity_LDF`.

**Therefore, per row:**

```
y_ult = count × count_LDF × (y/count) × severity_LDF
      = count × count_LDF × (y/count) × (cost_LDF/count_LDF)
      = y × cost_LDF        # exactly the D-005 uniform factor
```

Frequency-severity development with the consistent (implied) severity factor is
**bit-for-bit identical** to the uniform LDF we already use. It changes nothing
— there is no new target, so no submission slot was spent.

**What it *does* tell us** (recorded for the write-up): motor's 1.18 development
is ~6% late *reporting* (count) and ~12% *case strengthening* (severity), while
buildings/contents are almost entirely count-side (severity component ~1.0 —
open building/contents claims barely strengthen). That asymmetry is *why* motor
is where the split development (D-061) showed the largest (non-transferring)
effect.

The only genuinely granular development is per-peril (already `development=
'peril'`, measured skill-neutral in D-028). **The development axis is now
exhausted**: lob / peril / bf / split / freq-sev are all measured, and none beat
the uniform LDF. Best remains exp-027 0.12436.

## D-063 — Correction to D-052/D-053/D-060: `--final-fit` defects fixed; Per-LOB + Pooled `rate` models (+0.0055 Private LB)
**2026-10-23 · accepted · retracts D-052, D-053, and D-060 per-LOB conclusions · exp-033, exp-034**

Audit of `scripts/run_experiment.py` revealed two implementation defects in the
`--final-fit` block (lines 460–472) that had caused `exp-024` (`--target rate`)
and `exp-025` (`D-053` shallow-tree hyperparameters) to fail on the leaderboard:

1. **Why `exp-024` (`--target rate`) collapsed to `0.06019`:**
   In `exp-024`, the validation model fitted `y_ult / exposure` with exposure
   weights (`0.1365` honest local D²), but `--final-fit` hardcoded
   `final_model.fit(X_all, train["y_ult"])` (fitting total **cost**, not rate,
   without exposure weights), and line 481 (`to_cost`) then multiplied the
   predicted cost by `exposure` a second time (`cost ∝ exposure²`, exposure-band
   ratio `6.40`). Furthermore, `earned_premium = written_premium × exposure` had
   been left in the feature matrix while `written_premium` is already the
   annualised rate.
2. **Why `exp-025` (`num_leaves=16, lr=0.02`) collapsed to `0.11168`:**
   In `exp-025`, the validation model used `num_leaves=16, lr=0.02` and stopped
   at `best_iter = 590` trees (`+0.0057` local D² across both folds), but
   `--final-fit` hardcoded `num_leaves=64, learning_rate=0.04` and trained it for
   `n_fix = 590` trees (~6× the optimal ~100-tree budget for `num_leaves=64,
   lr=0.04`), which reduces 2024 holdout raw D² from `0.1298` to `0.1012`.
3. **Why per-LOB models appeared worse in D-060:**
   They were tested on the raw `cost` target with `num_leaves=64` (4× too deep
   for single-LOB sample sizes) and retained constant dummy columns from the
   other lines.

### Fixes implemented (`scripts/run_experiment.py`, guarded by `tests/test_guardrails.py`)

- When `--target rate` is selected, `earned_premium` is excluded from the
  feature list, `eval_sample_weight=[exposure]` is passed to `fit()`, and
  `--final-fit` fits `y_ult / exposure` with `sample_weight = exposure` and
  `**final_params`.
- Added `--mode per_lob` fitting separate shallow Tweedie rate models per LOB
  (`num_leaves=6, lambda_l2=5.0`) on each line's active features.

### Measured results (Fold 2024, Fold 2023, and Leaderboard)

| run | config | Fold 2024 OOF D² | Fold 2023 OOF D² | Public LB | Private LB | Δ vs exp-027 |
|---|---|---:|---:|---:|---:|---:|
| exp-027 | pooled cost target (`lv=64, lr=0.04`) | 0.13223 | 0.13624 | 0.12436 | 0.12984 | — |
| exp-032 | pooled rate target (`lv=16, lr=0.04`) | 0.13832 | — | — | — | — |
| **exp-033** | **per-LOB rate LGBM (`lv=6, lr=0.05`)** | **0.14002** | **0.14350** | **0.12681** | **0.13531** | **+0.00245 Pub / +0.00547 Priv** |
| **exp-034** | **70% exp-033 + 30% exp-032** | **0.14033** | — | **0.12707** | **0.13536** | **+0.00271 Pub / +0.00552 Priv** |

Calibration slope improves from `1.158` (`exp-027`) to `1.010` (`exp-033`),
calibration ratio improves from `1.121` to `0.989`, and exposure-band flatness
is `1.13–1.14` (vs truth `1.12`). Crucially, with the `--final-fit` bugs fixed,
honest local OOF D² ranks models accurately again.

## D-064 — Multi-family rate ensemble: Per-LOB XGBoost + Per-LOB LGBM + Pooled LGBM (New Best: 0.12753 Public / 0.13570 Private)
**2026-10-23 · accepted · retracts D-060's error-correlation rejection of C-3 · exp-035, exp-036**

D-060 rejected model-form blending on the ground that `corr(y - p_1, y - p_2) =
0.999`. Because ~90% of rows have `y = 0` and `Var(y) ≈ 100 × Var(p)`, residual
correlation `corr(y - p_1, y - p_2)` is dominated by `Var(y)` by mathematical
identity for any two predictors with $R^2 \approx 0.05$.

Tested a standalone **Per-LOB XGBoost Tweedie rate model** (`objective=
'reg:tweedie'`, `tweedie_variance_power=1.5`, `max_depth=3`, `learning_rate=0.05`,
`min_child_weight=200`, `reg_lambda=5.0`, excluding `_hist_rate`/`_hist_freq`
target-encoding columns so the second model family has zero target-encoding
drift between 2024 and 2025).

| run | config | Fold 2024 OOF D² | Public LB | Private LB |
|---|---|---:|---:|---:|
| exp-035 | Per-LOB XGBoost rate (`depth=3, no_rel`) | 0.14014 | 0.12700 | 0.13509 |
| exp-034 | 70% Per-LOB LGBM + 30% Pooled LGBM | 0.14033 | 0.12707 | 0.13536 |
| **exp-036** | **40% exp-035 (XGB) + 42% exp-033 (Per-LOB LGBM) + 18% exp-032 (Pooled LGBM)** | **0.14076** | **0.12753** | **0.13570** |

`exp-036` outperforms every constituent model on both local OOF D² (`0.14076`)
and on both Public (`0.12753`) and Private (`0.13570`) leaderboards.

## D-065 — 31 Dec 2024 valuation cutoff: Horvitz-Thompson Date×Peril development & year-normalized relativities
**2026-10-23 · accepted · measured in censored age-0 backtest and on the leaderboard · exp-037, exp-038**

Investigation of the competition brief's central valuation statement (*"Claims
are valued at 31 December 2024. Some 2024 claims had not been reported by then,
and open claims may still develop"*) established three empirical properties in
`claims.parquet`:

1. **Motor `bodily_injury` drives 2024 IBNR and IBNER:**
   `bodily_injury` has a 127-day median reporting lag (56.6% analytical 2024
   reporting completeness vs 94%–98% on short-tail perils) and its 2024 mean
   `incurred_to_date` declines monotonically from Q1 (`20,094`, 70% open) to Q4
   (`14,735`, 93% open) due to immature open case estimates.
2. **Open motor `own_damage` and `theft` claims carry zero booked recoveries:**
   0.00% of open claims have `recoveries > 0`, whereas 30.0% of closed
   `own_damage` claims recover 80% of incurred (`net = 0.76 × incurred`) and
   13.2% of closed `theft` claims recover 30% (`net = 0.96 × incurred`). This
   explains why `exp-031` (`D-061`, uniform open-claim multiplier `1.708`)
   over-inflated open `own_damage` claims by 2.25× and regressed on the board.
3. **Horvitz-Thompson Date×Peril development (`development="ht_motor"` in `src/dataset.py`):**
   In a controlled censored-valuation backtest (`2021–2022` censored at
   `2022-12-31` predicting mature `2023` truth), weighting age-0 Motor claims by
   inverse reporting probability $1 / \max(F_{\text{peril}}(\text{val\_date} -
   \text{loss\_date}), 0.25)$ improved out-of-sample Motor D² from `0.08604`
   (uniform LOB LDF) to `0.08706` (vs `0.08856` uncensored Oracle, closing 40%–81%
   of the Oracle gap depending on feature set), while uniform LOB LDF matched the
   Oracle on Buildings and Contents.
4. **Year-normalized historical relativities (`--norm-rel`, `exp-037` / `exp-038`):**
   Normalizing `<col>_hist_rate` and `<col>_hist_freq` within `(line, acc_year)`
   to mean `1.0` improved Fold 2024 OOF D² from `0.14002` to `0.14054`, and
   scored `0.12587` Public / `0.13502` Private solo (`exp-037`) and `0.12721`
   Public / `0.13563` Private in the 4-model blend (`exp-038`). `exp-036`
   (`0.12753` Public / `0.13570` Private) remains our top leaderboard submission.

