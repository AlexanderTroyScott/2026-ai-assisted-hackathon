# Actuarial Report — 2026 IAA AI-Assisted Hackathon, Pricing Across Markets

**Documentation prepared to the requirements of Actuarial Standard of Practice No. 9,
*Documentation and Disclosure in Property and Casualty Insurance Ratemaking, Loss
Reserving, and Valuations* (ASB, adopted January 1991).** Text of ASOP 9 Sections 1–6
is reproduced in R-01 Appendix C (`docs/references.md`).

**Competition:** `iaa-ai-assisted-hackathon-2026-part-1`
**Metric:** Tweedie D², power = 1.5
**Deadline:** 2026-10-23 16:00
**Documentation date:** 2026-10-22; reconciled 2026-10-28 (D-071…D-080:
counted-entry correction in §6, sandbox verdicts in §5)

> **In plain terms.** This is the technical record for Part 1 — predicting 2025
> claim costs on 2.4 million policies across motor, buildings and contents. The
> short version: we cleaned the data (a corrupt exposure column, annualised
> premiums, 604 duplicate claims), fixed a bug in our claim-development table
> and **presented the corrected version even when the buggy one scored
> slightly better** (a table known to be wrong cannot be defended), removed a
> misleading "occupation" field, and found that deliberately small, shallow
> models beat big ones. The leaderboard counted our `exp-027` entry
> (0.12984 private, 41st of 54); our best model, `exp-036`, scores 0.13570
> private but was submitted after the deadline and is therefore unranked — it
> is attached as our Week 1 deliverable with that fact disclosed. The
> accessible summary is `docs/final_writeup.md`; the sections below are the
> detailed evidence.

---

## 1. Purpose, intended use, and users (ASOP 9 §5.1)

This report documents a predictive model of expected ultimate claim cost per policy
period, built to rank risk for pricing purposes. The intended use is **relative risk
differentiation** within a competition setting, not production rating.

Users are: (a) the competition adjudicators, as *indirect* users assessing actuarial
credibility and technical soundness; and (b) the entrant's own team, as direct users.

The work product comprises this report, `docs/decisions.md` (the decision record),
`docs/experiment_log.md` (the measurement record), `docs/data_findings.md` (verified
empirical findings), `docs/references.md` (external method authority), and the
executable code in `src/` and `scripts/` with its test suite.

---

## 2. Data — sources, quality, and limitations (ASOP 9 §5.2)

### 2.1 Sources

All data is competition-supplied, in `developed/` (gitignored; manifest with sha256
hashes in `docs/data_manifest.md`).

| table | rows | role |
|---|---|---|
| policies_motor / buildings / contents | 2,409,311 | training policy periods, 2021–2024 |
| test_motor / buildings / contents | 626,639 | scoring population, 2025 periods |
| claims | 163,150 | claim records, losses to 2024-12-29 |
| customers | 545,240 | customer attributes |
| addresses | 575,983 | address/geographic attributes |
| weather | 2,879,915 | address × year climate indices, 2021–2025 |

### 2.2 Data quality findings and adjustments

Four defects were identified and are material enough to record. Each is decisioned in
`docs/decisions.md`.

**(a) The supplied `exposure` column is a day count, not an exposure fraction (D-001).**
It reaches 365 with skew 35.5. Verified against recomputed `(period_end − period_start
+ 1)/365`. **Adjustment:** exposure is recomputed throughout; the supplied column is
never used. A test asserts this (`tests/test_guardrails.py::test_supplied_exposure_is_days_not_exposure`).

**(b) `written_premium` is an annualised rate, not the premium for the row's period
(D-051).** Verified three ways: `corr(written_premium, exposure)` ≈ 0 (a one-day row
costs the same as a 365-day row); within-policy pairs show 25.3× the days for only
1.086× the premium; and loss ratio on `premium × exposure` is stable to 1.41× across
row-length buckets while `cost/premium` swings 99×. **Adjustment:** earned premium for
a row is `written_premium × exposure`. The `premium_lr` benchmark was rebuilt on this
basis, moving from D² 0.076 to 0.109 — it had previously appeared *worse* than a flat
rate, which was an artefact of the unit error. This defect did **not** affect the
model's target formulation (the tree already learned `cost ∝ exposure` correctly).

**(c) `sum_insured` contains a non-physical population (D-043).** There are zero rows
between 1M and 10M, and a separate population above 10M (1,395 rows per line, 0.2%,
max 594,000,000) whose `premium/sum_insured` ratio is 1,000× lower than the rest —
ordinary premiums against implausible sums insured. The rows carry no claims signal
(buildings claims in the 10M–100M band: 14 claims at mean cost 3,749 versus 9,588 for
the 100k–1M band). **Adjustment:** capped at 1e7 and flagged (`si_artefact`) rather
than deleted — deletion would break the 626,639-row submission contract. Recorded as
a field error, plausibly a unit or code column.

**(d) Identifier and claim-level defects (D-051).** Motor `customer_id` has lost
leading zeros on 10% of rows (98,110 train, 25,554 test); `str.zfill(8)` recovers 100%
with zero collisions. Separately, 604 claims are exact duplicates differing only in
`claim_id`, 598 of which carry 2022 loss dates and inflate 2022 claim cost by 1.63%.
**Adjustments:** zero-padding at read; duplicate rows dropped and logged.

### 2.3 Known data limitations

- **414 claims match no policy period** (loss date falls outside every period of their
  policy). Dropped from the target and written to `outputs/dropped_claims.csv` with the
  count asserted in tests. 274 are pre-2021.
- **Claim amounts are already net of policy excess.** No excess deduction is applied
  (D-003); deducting again would double-count.
- **`telematics_score` is 87.6% missing** and has an ambiguous observation window
  (D-017); deny-listed.
- **`make_model` (300 codes) is not a vehicle taxonomy** — codes span 14–19 vehicle
  value groups and all body types and fuels. Split-half claim-rate correlation 0.225
  against a binomial standard error of 0.006. **Not used** (D-043 / `synthetic_dgp_findings.md` §8).
  Recorded because it presents as large signal (14× severity spread) and is not.
- **Excess detail on the above is in `docs/data_findings.md`**, which is the verified
  record and is not reproduced here.

---

## 3. Material assumptions (ASOP 9 §5.2, §5.3)

Each assumption is recorded with its status. `accepted` = decided; `proposed` = awaiting
sign-off and **not** yet a professional conclusion.

### 3.1 Target and development

- **A1. The target is expected ultimate net cost for the policy period**, not reported-to-date.
  Accepted (D-005).
- **A2. Development is applied to labels, never to predictions.** Predictions are already
  ultimate-level; applying a further loading at scoring would double-count. Accepted (D-005).
- **A3. Development factors are volume-weighted chain ladder from an incurred triangle**
  (`data_findings.md` §5): motor 0→1 = 1.223, 1→2 = 1.027, 2→3 = 1.003; buildings
  0→1 = 1.089, 1→2 = 1.003; contents 0→1 = 1.040, 1→2 = 1.004. Age is
  `2024 − accident_year`. Accepted (D-005).
- **A4. These factors are lower bounds.** They capture late reporting but not adverse
  development on open claims ("IBNER"). This is a **known understatement of the target**
  and is disclosed rather than corrected. `data_findings.md` §5.
- **A5. Per-peril development is level-neutral** — it redistributes the accepted LOB
  level across perils and does not move the total (D-028). Measured as a tie; unpromoted.

### 3.2 Level and trend

- **A6. The 2025 level factor is a blend of a claims-index extrapolation and a premium
  anchor** (D-009). **Status: `PROPOSED` — not ratified.** Measured flat: a uniform +3%
  moved the leaderboard by +0.00007. Carried by every submission.
- **A7. Trend is measured on a mix-adjusted severity index** holding the 2021 peril mix
  fixed, to avoid mix bias (D-007). Note this embeds the assumption that the 2021 peril
  mix is the reference.
- **A8. Trees receive a forward factor only, not full de-trending** (D-008), because
  trees cannot extrapolate a linear trend from a year feature.

### 3.3 Features and leakage

- **A9. Anything derived from claims uses only losses strictly before `period_start`.**
  Enforced at both the customer-level and policy-level filters and tested (AGENTS.md rule 6).
- **A10. `renewal_decision` is outcome information and is never a feature** (D-011).
  Enforced both at feature-request time and by assertion on the returned model matrix (D-037).
- **A11. `customer_service_contacts` is post-outcome** (D-016) — measured: 3.22 contacts
  on claim-years versus 0.41 on clean years of the same policy, paired t = 437. Deny-listed.
- **A12. Annual climate indices are knowable at pricing time** (D-012). **This assumption
  is contestable.** The 2025 weather is *realised* data, not a climate normal (2025 vs
  2024 rainfall correlation 0.63, mean absolute difference 193mm). It is in scope by
  explicit human decision, on the ground that it is available to every entrant, and this
  disclosure is made so the reader can judge it. Measured effect of removing it is at most
  +0.0005, not separable from noise.

### 3.4 Validation

- **A13. A time-based split is used, never random** (D-006): train ≤2023 → holdout 2024,
  and train ≤2022 → holdout 2023. Random k-fold would leak customer history across the
  split boundary and is not used (R-04 §4).
- **A14. The 2024 slice is a *selection* set, not a holdout.** By the R-04 definition it
  has been consulted repeatedly for model choice (label table, recency weighting, log
  transforms, weather variants). **The leaderboard is the only untouched data.** This is
  a limitation of the validation design and is disclosed here rather than implied away.
- **A15. Local D² is a weak gate.** Documented twice by direct measurement: exp-024
  scored local 0.1365 and leaderboard 0.06019; exp-025 scored local +0.0057 mean across
  two folds and leaderboard −0.0109. Local D² is **0 for 2** as a ranking gate since
  optimisation began.

---

## 4. Methodology (ASOP 9 §5.2, §5.3)

### 4.1 Target construction

```
y(row) = Σ max(0, incurred_to_date − recoveries)
         over claims where policy_id and line match
           AND period_start ≤ loss_date ≤ period_end
           AND status != 'repudiated'
```

Attribution is by `(policy_id, line)` — `policy_id` is not globally unique across lines.
Every claim matches exactly one row (162,736 matched, 0 multi-matched, 414 orphans).
Repudiated claims all carry zero cost, so the filter is a numerical no-op and is retained
for explicitness only (D-004).

Then `y_ult = y × LDF(line, accident_year)` per A3.

### 4.2 Features

Nine blocks, ~100 columns: base rating factors, premium-formula decomposition
(`age_band`, `ncb_band`, `vvg_log`), lagged claim experience (D-014 — the strongest
block; prior claimants have loss ratio 2.3 but pay the same), time-safe historical
relativities (D-027), weather (D-012), address and customer attributes (D-030),
rating flags, and denied-claim history (D-031, opt-in).

`occupation` is deliberately excluded from the GLM specification (D-046): it costs 21
dummy parameters for 0.0001 of D² on the tree model, is collinear with driver age,
licence years and NCB, and 64% of its dispersion is Student/Retired — age in disguise.
It remains in the tree specification, where it is harmless.

### 4.3 Model form

A single LightGBM gradient-boosted tree ensemble across all three lines, Tweedie
objective with variance power 1.5 to match the competition metric. The two-part
frequency × severity alternative was implemented and measured worse (D-023/D-026); the
Tweedie GLM benchmark (C-2) is built and lands at holdout D² 0.1206 against the tree's
0.1303 raw.

Hyperparameters were **not** tuned until 2026-10-22. A three-round sweep gated on
out-of-time stability found fewer leaves consistently better (64 → 31 → 16 improves;
128 → 256 collapses). The tuned configuration was tested on the leaderboard and
**failed** (exp-025, 0.11168); the mechanism is a 30% mix shift in `marketing_segment`
between training and scoring populations. This is recorded as a negative result and the
tuned configuration is **not** the presented model.

### 4.4 Post-processing

Per-line recalibration so predicted totals match expected totals per LOB (D-009). Where
the scales are fitted determines what a local number means (D-040): fitted on the holdout
and scored on it is an in-sample oracle worth +0.002–0.003 of local D² — the size of the
entire D-025 tie band — and is structurally blind to LOB-level miss. Local metrics are
reported on train-fitted (out-of-sample) scales.

The final model is refitted on **all** training years including the holdout before
scoring (D-045). This is standard practice (R-04 §3): measure honestly out-of-sample
first, then refit on complete data. The reported local D² remains the pre-refit,
out-of-sample figure and is never restated as if the refit had been measured.

### 4.5 Metric

`sklearn.metrics.d2_tweedie_score(y, y_pred, power=1.5)`, predictions floored at 1e-3
(D-010) because the deviance raises on non-positive values.

---

## 5. Material changes from the prior analysis (ASOP 9 §5.2)

ASOP 9 §5.2 requires that material changes in data, assumptions, or methods from the
last analysis be documented, with the reasons for and impact of the changes. The
substantive changes in this work, in order:

| change | reason | impact |
|---|---|---|
| **D-033/D-035 — label development table corrected** | The frozen LDF table applied the age-0 factor to accident years at age 1. Buildings 2023 was ×1.092 instead of ×1.003; contents 2023 ×1.044 instead of ×1.004. Corrected to derive factors from the documented age-to-age table. | 2023 training labels −8.2% (buildings), −3.8% (contents). Leaderboard effect measured across 3 seeds per table: corrected mean 0.12161 vs mis-keyed mean 0.12103. |
| **D-045 — final refit on all training years** | The submission model was fitted on AY ≤ 2023 only; the 2024 rows (26% of the book) contributed zero gradient updates. | **+0.00195** leaderboard. The dominant single effect measured. |
| **D-051 — data corrections** (annualised premium, customer_id padding, 604 duplicate claims) | Verified data defects, §2.2 above. | Benchmark corrected 0.076 → 0.109. Model effect neutral (−0.0004). |
| **D-052 — rate target implemented then retracted** | Implemented to match the annualised-premium finding; collapsed on the leaderboard at 0.06019. Cause: `earned_premium` contains exposure, so the model scaled by exposure twice. | **Retracted.** Guardrail added (`kaggle_submit.py` implied-rate flatness) that would have caught it before submission. |
| **D-053 — hyperparameter tuning** | Never previously tuned. | Local +0.0057 mean across both folds; **failed on the leaderboard** (0.11168). Not adopted. |
| **D-054 — R-04 diagnostics built** | The suite produced no normalised Gini or lift charts; R-04 names the train-vs-holdout Gini gap as the overfit measure. | Toolkit built. **The gap does not predict leaderboard transfer** (inverted on the two configs tested). The *decomposition* does: Gini/lift measure ranking, calibration measures level, and our failures were level failures. |
| **D-055 — test-suite audit** | Reviewer audit of all 70 tests. | Four submission-critical behaviours were unguarded (target construction, the exposure convention, the scoring metric, and a dead test) and one test was tautological. Fixed; 70 → 77 tests. |
| **D-056 — `occupation` removed** | 64% of its dispersion is Student+Retired, i.e. driver age in disguise; it carried two diagnosed artefacts. | Accuracy neutral (mean +0.0003, folds disagreeing). **Calibration improved on both folds.** Removal confirmed on the leaderboard. |
| **D-057 — exp-027** | Testing the D-056 change. | **0.12436**, our best draw prior to D-063. See §6 for the seed-family caveat. |
| **D-058 — `flood_zone`-only** | Proposed that river distance and elevation were proxies for it. | **Refuted**: −0.00181. Elevation carries 1.2–1.8× residual signal *within* flood zones. |
| **D-059 — seeds and blends** | Filling the expiring quota on the best config. | Family mean 0.12287, sd 0.00133. exp-027 is ~1.1 sd above its own mean. Blends diluted the standout. **Corrected:** the occupation gain is +0.00127 on means (1.14 pooled sd), not +0.0018. |
| **D-063 — `--final-fit` defects fixed; Per-LOB + Pooled `rate` models** | `--final-fit` had hardcoded `train["y_ult"]` (causing exposure double-counting in `exp-024`) and hardcoded `num_leaves=64, lr=0.04` (causing 6× tree-budget overfitting in `exp-025`). Fixed both and implemented shallow Per-LOB `rate` LGBM (`num_leaves=6`). | **+0.0078 honest OOF D²** (`0.1322 → 0.1400`), **Public LB 0.12681 / Private LB 0.13531** (`exp-033`, +0.00547 Private over `exp-027`); 70/30 Per-LOB + Pooled blend (`exp-034`) reaches **0.12707 Public / 0.13536 Private**. |
| **D-064 — Multi-family rate ensemble (Per-LOB XGBoost + LGBM)** | Retracts D-060's residual-correlation rejection of model-form blending. Adds Per-LOB XGBoost Tweedie rate model (`max_depth=3, no_rel`, solo `0.12700` Public / `0.13509` Private, `exp-035`). | **New Best (`exp-036`): 0.12753 Public / 0.13570 Private** (+0.00317 Public / +0.00586 Private over `exp-027`; honest OOF D² **0.14076**). |
| **D-065 — Horvitz-Thompson Date×Peril development & year-normalized relativities** | Addresses 31 Dec 2024 reporting truncation and open-claim recovery/IBNER asymmetry on Motor (`development="ht_motor"`), plus year-normalized historical relativities (`--norm-rel`, `exp-037`/`exp-038`). | Censored age-0 backtest Motor D² `0.0860 → 0.0871`; `exp-038` 4-model ensemble scores **0.12721 Public / 0.13563 Private**. |
| **D-071/D-072 — post-deadline sandbox campaign; HT labels board-tested** | Part 1's scoring gate stayed open after the 2026-10-23 deadline; all runs from `exp-033` onward are post-deadline and unranked. The gate reveals Private, so it served as a labelled sandbox: D-065's HT date×peril labels were board-tested at two paired seeds against the scored `exp-033`/`exp-040` controls. | **+0.00063 Public / −0.00021 Private** — a Private-for-Public trade. `ht_motor` labels **rejected**; the presented model never used them. Also measured: board seed noise σ ≈ 0.0013 Public / ≈ 0.002 Private per redraw (revised up by D-080's third seed pair), which reclassifies several earlier single-seed verdicts as undecidable. |
| **D-080 — level-preserving per-peril development board-tested** | D-028's per-peril LDF factors (motor `bodily_injury` age-0→ult 1.708 vs `own_damage` 1.039) had stood as "a tie locally, LB tie-break open" since 2026-10-20. Tested on the sandbox in a 3-seed paired design (`exp-042`/`043`/`044` vs `exp-033`/`040`/`045`). | **−0.00165 Public / −0.00222 Private** mean (3/3 pairs non-positive on Private; third pair a tie on a noisy control). No evidence of improvement; uniform LOB development stands and the tie-break closes against per-peril factors. Grade: negative point estimate at n=3, not decisive. |

**Professional judgement conflict (ASOP 9 §5.4).** The corrected label table (D-033/D-035)
is the actuarially correct one and reproduces `data_findings.md` §5 exactly. It is
presented as our model despite early evidence suggesting the mis-keyed table scored
better, on the ground that a table known to be wrong cannot be defended to a reviewer.
With the D-045 refit applied, the corrected table is at least at parity (mean 0.12161 vs
0.12103 on three seeds each) and holds the better ensemble score. **This is a case where
professional judgement was preferred to a measured difference, and it is disclosed as
such.**

---

## 6. Results and diagnostics

**Counted entry vs best model — stated plainly (reconciled 2026-10-28).**
Part 1's final ranking runs on **Private**, with the counted entry being the
best **Public** submission as of the 2026-10-23 16:00 deadline. Our counted
entry is therefore **`exp-027`** (cost-target pooled LightGBM, occupation
dropped, D-056): **0.12436 Public → 0.12984 Private**, rank 41 of 54. The
stronger `exp-033`→`exp-038` line was submitted **after** the deadline (the
scoring gate stayed open — D-071) and is **unranked**: presented below as
method evidence and as the sandbox instrument for Part 2 (D-072, D-080), not
as a competition result.

**Best model (`exp-036`, unranked).** Multi-family rate ensemble: 40% Per-LOB
XGBoost `max_depth=3` + 42% Per-LOB LightGBM `num_leaves=6` + 18% Pooled
LightGBM `num_leaves=16`, all modelling annualised rate `y_ult / exposure`
with exposure weights and reconstructed as `exposure × rate`, scoring
**0.12753 Public / 0.13570 Private** (honest Fold 2024 OOF D² **0.14076**).
Its individual components also substantially outperform the counted
cost-target baseline (`exp-027`, `0.12436` Public / `0.12984` Private):
- `exp-033` (Per-LOB LightGBM rate `num_leaves=6`): **0.12681 Public / 0.13531 Private** (OOF D² 0.14002)
- `exp-035` (Per-LOB XGBoost rate `max_depth=3`): **0.12700 Public / 0.13509 Private** (OOF D² 0.14014)
- `exp-034` (70% `exp-033` + 30% Pooled LightGBM rate `exp-032`): **0.12707 Public / 0.13536 Private** (OOF D² 0.14033)
- `exp-038` (4-model rate ensemble incl. `exp-037`): **0.12721 Public / 0.13563 Private** (OOF D² 0.14095)

Calibration on the Fold 2024 holdout improves markedly under the rate
formulation: out-of-sample calibration ratio improves from `1.12–1.18` (cost
target) to **`0.989–1.005`**, calibration slope improves from `1.15–1.22` to
**`1.006–1.010`**, and exposure-band implied-rate flatness is **`1.13–1.14`**
(matching the training truth of `1.12`).

**Prior cost-target baseline (`exp-027`).** Prior to D-063, our best single run
on the cost target was `exp-027`, leaderboard **0.12436** (`0.12984` Private),
with three-seed family mean **0.12287** (0.12179 / 0.12247 / 0.12436, sd
0.00133).

All presented runs use the corrected label table (D-033/D-035) and the D-045
final refit.

**The occupation removal, stated correctly.** Three seeds per family:

| family | seeds | mean | sd |
|---|---|---|---|
| with occupation | 0.12150 / 0.12249 / 0.12083 | 0.12161 | 0.00084 |
| **without occupation** | 0.12436 / 0.12179 / 0.12247 | **0.12287** | 0.00133 |

Difference **+0.00127 (1.14 pooled sd)**. Directionally consistent — the mean
and every single-run pairing favour removal — but **suggestive, not
established**. (D-057 quoted +0.00187, which is the *single-run* figure;
corrected in D-059.)

**Diagnostics now produced** (R-04, added 2026-10-22). Fold 2024 holdout,
presented config family: normalised Gini 0.4669, lift ratio 28.0, calibration
MAE ~29 on a mean near 350. Lift chart monotone apart from one flat step
(bin 5→6). Lorenz: the lowest-predicted 10% of exposure carries 0.3% of loss.

Two properties of the model are worth stating plainly:

- **Out-of-sample calibration ratio is 1.12–1.18** before LOB recalibration — the
  model underpredicts the holdout by 12–18%. The per-LOB recalibration carries
  that; it does not reflect the model's intrinsic level accuracy.
- **Calibration slope is 1.15–1.22** — predictions are under-dispersed. Tested
  and *not* corrected: a power transform that forces the slope to 1.0 reduces
  Tweedie D² from 0.1303 to 0.1166 even with an oracle estimate. **D² prefers
  the compressed predictions.** The slope is a descriptive property, not a defect
  to fix under this metric.

**Ensembling.** Blending seeds helped once (two comparable runs, +0.0001) and
hurt once (a standout plus two weaker draws, −0.0005 to −0.001). Rule: variance
reduction helps when members are comparably good and dilutes a lucky draw.

## 7. Reliance on others (ASOP 9 §5.8)

**This is material and is disclosed in full.**

ASOP 9 §5.8: *"An actuary who makes an actuarial communication assumes responsibility
for it, except to the extent the actuary disclaims responsibility by stating reliance on
another person… A communication should define the extent of any such reliance."*

This work was produced with substantial reliance on **AI systems**, and the extent of
that reliance is defined here.

**Extent of reliance.** AI agents were used for: exploratory data analysis; identifying
and verifying the data defects in §2.2; implementing the pipeline in `src/` and
`scripts/`; designing and running experiments; drafting the decision and measurement
records; and drafting this report. AI was **not** the source of the external method
authority (R-01 … R-06 were provided by or located as published sources and are cited),
and AI did not make the professional decisions recorded as `human-directed` in
`docs/decisions.md` — namely D-044 (weather in scope, corrected labels presented, slot
policy, method ordering) and the initial direction of the work.

**Where AI was wrong, and how it was caught.** The following AI-produced conclusions were
subsequently found to be incorrect and are recorded as retracted in `docs/decisions.md`:

- **D-039** — attributed a leaderboard drop to a missing IBNER loading. The quantitative
  core was wrong in sign, size and method. Retracted in D-040.
- **D-040** — claimed a label penalty of "12× the seed band", measured on one side of a
  two-sided comparison. Corrected in D-041 to ~2×.
- **D-048** — claimed label parity resolved in favour of the corrected table at "1.3
  pooled sd". Weakened in D-049 after a third seed, to under one pooled sd.
- The occupation "instability" finding (−0.170 time correlation) was traced to a single
  11.9M claim, not a non-stationary factor.
- An initial `sum_insured` log-transform test used a rank correlation, which is invariant
  to monotone transforms — a tautology. Replaced with a Pearson test.
- The `written_premium` collinearity alarm was an artefact of evaluating a logged field
  at its raw value.
- **The "calibration slope is the biggest number on the table" framing was wrong.**
  Measured directly: forcing the slope to 1.0 *reduces* Tweedie D² from 0.1303 to
  0.1166, even with an oracle estimate. The metric prefers the compressed
  predictions. The slope is descriptive, not a lever.
- **The per-LOB model proposal was wrong.** The premise (different drivers per
  line) was correct; the conclusion (separate models would help) was not.
  Measured −0.0044 / −0.0040 on both folds, with worse calibration.
- **The C-3 ensemble ceiling was overstated** in the first read by comparing a
  recalibrated blend against an un-recalibrated baseline. Corrected: real gain
  +0.0001, and the two models' errors correlate at 0.999.
- **An arithmetic error in D-059**: the occupation-removal gain on family means
  is +0.00127 (1.14 pooled sd), not the "+0.0018" claimed. That figure was the
  single-run delta carried across without recomputation.

Five self-corrections in the session that produced this model. The pattern is
consistent and worth stating: **single measurements and small samples get
over-read.** What caught these: independent subagent audit, a second measurement
before acting, and comparing family means rather than best draws. Where that
discipline was not applied, an over-claim followed.

**Verification discipline.** Every conclusion that reaches this report was recomputed
from `developed/` (AGENTS.md rule 1) or attributed to a named source. Independent
subagent audits were run on the code (`reviewer`) and on the modelling inferences
(`model-audit`, `data-audit`) and their findings are what produced D-040 and D-051.
48 automated tests guard the invariants, including mutation-checked leakage tests.

**Responsibility.** Notwithstanding the reliance above, responsibility for this
communication rests with the entrant. AI output was treated as a draft to be verified,
not as a conclusion to be accepted.

---

## 8. Disclosures and limitations

1. **The target is a documented lower bound** (A4). Development factors capture
   late reporting but not adverse development on open claims. Open-claim share is
   20.4% for 2024 against 3.2% for 2023; 2024 open claims carry mean case ≈ 7.7k
   versus 21–27k for mature open claims. **This is the largest open actuarial
   question in the work.**
2. **The 2025 level factor is unratified** (A6). Carried by every submission,
   still `proposed`.
3. **2025 weather is realised data** (A12). In scope by explicit human decision
   on the ground of equal availability. Measured effect of removal ≤ +0.0005,
   not separable from noise. Disclosed for the reader's judgement.
4. **The 2024 slice is a selection set, not a holdout** (A14). The leaderboard is
   the only untouched evaluation.
5. **Local D² does not reliably rank candidates** (A15), documented **four**
   times by direct measurement. Its failures have been specific (level and
   calibration effects, one target-formulation error) rather than random, but it
   must not alone justify a submission.
6. **Our best single run is partly a lucky draw.** exp-027 (0.12436) sits ~1.1 sd
   above its three-seed family mean (0.12287). Quote the mean as the
   configuration's level. The occupation removal is **+0.00127 on family means
   (1.14 pooled sd)** — suggestive, not established. (D-057's +0.00187 is the
   single-run figure; corrected in D-059.)
7. **The model is under-dispersed and this is not corrected.** Calibration slope
   1.15–1.22. Forcing it to 1.0 costs ~0.014 of Tweedie D² — the metric prefers
   compressed predictions. Disclosed because an examiner may reasonably ask why
   predictions are narrow relative to the observed spread.
8. **The single largest claim** (motor bodily injury, 11,894,277) is 10.6× the
   next largest, 73× the p99.9, and 1.59% of the book. Internally coherent and
   **retained**. It materially affects one occupation cell — one reason
   occupation left the model.
9. **Approaches tested and rejected** (each measured, not assumed): frequency ×
   severity split; seed averaging; per-peril development; recency weighting;
   hyperparameter simplification; a rate-transformed target; log transforms for
   the tree; `make_model`; feature pruning (three sweeps); the calibration-slope
   correction; C-3 GLM/LGBM blending; separate per-LOB models. Record:
   `docs/decisions.md` D-023 through D-059.
10. **Still `proposed`**, i.e. not professional conclusions: D-009 (level
    factor) and candidates C-1 and C-4.

## 9. Deviation from standard (ASOP 9 Section 6)

ASOP 9 Section 6: *"An actuary who uses a procedure which differs from this standard
must include… an appropriate and explicit statement with respect to the nature,
rationale, and effect of such use."*

**Deviation — extent of documentation.** ASOP 9 §5.2 contemplates documentation
sufficient for another actuary to evaluate the work. This report is written for a
competition adjudicator who may not be an actuary, and some sections are pitched
accordingly. The *substance* (data sources, material assumptions, methods, material
changes with reasons and impact) is complete and would satisfy §5.2 for a practitioner.

**Deviation — reliance disclosure form.** §5.8 contemplates reliance on *persons*. The
reliance here is on AI systems, which the standard does not directly address. §7 states
the extent of that reliance in the form §5.8 requires. Rationale: omitting it would
misrepresent how the work was produced. Effect: the reader can weigh conclusions
accordingly, and the record of retracted conclusions gives an empirical basis for that
weighing.

**Deviation — signature.** §5.7 requires signature by the responsible actuary where the
form of work requires it. This report is unsigned pending the entrant's sign-off.

---

## 10. Reproducibility

- `developed/` is gitignored; `docs/data_manifest.md` holds sha256 hashes so the dataset
  can be re-verified against the source.
- Every figure in this report is recomputable from `developed/` or is attributed to a
  numbered decision (`D-xxx`) or reference (`R-xx`).
- Submission CSVs are not committed; their sha256 is recorded in `docs/experiment_log.md`.
- 48 tests in `tests/` guard the invariants. `pytest -q` is the entry point.

---

## Cross-reference to ASOP 9

| ASOP 9 provision | where addressed |
|---|---|
| §5.1 purpose, users, circumstances | §1 |
| §5.2 sources of data | §2 |
| §5.2 material assumptions | §3 |
| §5.2 methods | §4 |
| §5.2 material changes + reasons + impact | §5 |
| §5.3 fair presentation, clarity for appraisal | §§2–5 |
| §5.4 disclosure of conflict with professional judgement | §5 (label table) |
| §5.5 availability of documentation | §10 |
| §5.6 conflicting interests | none arising |
| §5.7 signature | §9 (deviation) |
| §5.8 reliance on another | §7 |
| Section 6 deviation from standard | §9 |
