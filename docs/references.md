# Actuarial References

Bibliography of external actuarial sources behind the method choices in this
repo. IDs are `R-xx` and are cited from the write-up, the video and (where a
source changed a decision) from `docs/decisions.md` entries. Append-only.

**Rules (AGENTS.md rule 1).** These sources justify *methods*. They never supply
numbers that enter our findings or decisions — every figure in `docs/` is still
recomputed from `data_raw/`. Anything quoted from a source is attributed here
and stays clearly separated from our measured results.

Provided by the human, 2026-10-20, with two stated purposes: actuarial
credibility in the documentation **and** actionable method ideas before the
deadline.

---

## R-01 — Friedland, *Estimating Unpaid Claims Using Basic Techniques*
**CAS Study Note, Version 3, July 2010 · Jacqueline Friedland (KPMG), with R. Dutil & E. Lam**
<https://www.casact.org/sites/default/files/database/studynotes_friedland_estimating.pdf>

The CAS basic-techniques reserving text. Contents used here: development
triangles as a diagnostic (Ch. 5–6), the development technique (Ch. 7),
expected claims (Ch. 8), Bornhuetter-Ferguson (Ch. 9), Cape Cod (Ch. 10),
frequency-severity reserving (Ch. 11); appendices carry the *Statement of
Principles* and **ASOP 43** (unpaid claim estimates) and **ASOP 9**
(documentation and disclosure in ratemaking/reserving) — the standards that
define what "actuarially documented" means to a reviewer.

**Supports:** D-005 (develop labels with age→ultimate factors), D-007
(per-peril severity trend), D-028 (per-peril factors, level-neutral), and the
documentation discipline of `docs/decisions.md` / `docs/experiment_log.md`
(ASOP 9 is the external statement of the same idea).

## R-02 — Anderson et al., *A Practitioner's Guide to Generalized Linear Models*
**CAS Study Note, 3rd Edition, February 2007 · Anderson, Feldblum, Modlin, Schirmacher, Schirmacher, Thandi**
<https://www.casact.org/sites/default/files/database/studynotes_anderson9.pdf>

GLM theory and practice for actuaries: exponential family, link functions,
**offsets and prior weights** (§1), fitting and diagnostics (deviance, standard
errors, aliasing), factor/variates in the design matrix. Directly relevant
appendices: **App. C — the Tweedie distribution** (point mass at zero, variance
∝ μᵖ, 1<p<2 for pure premium), **App. J — frequency-severity vs Tweedie GLMs**
(Tweedie pure-premium fits "can often give very similar results" to the
two-model approach; the two-step approach buys interpretability of factor
effects, e.g. a factor moving frequency and severity in opposite directions).

**Supports:** D-023/D-026 (freq/sev split tested and measured worse locally —
the literature expectation of similarity is noted honestly alongside our
result), D-010 (deviance behaviour at the boundary), and the Tweedie p=1.5
choice behind the headline metric.

## R-03 — Goldburd, Khare, Tevet, Guller, *Generalized Linear Models for Insurance Rating*
**CAS Monograph Series No. 5, 2nd Edition (2025 revision)**
<https://www.casact.org/sites/default/files/2021-01/05-Goldburd-Khare-Tevet.pdf>

The rating-plan application guide: data preparation and **train/test/holdout
splitting and cross-validation** (§4.3) — held-out testing is called critical
and the split strategy is to be fixed before model building; **target choice:
frequency/severity vs pure premium** (§5.1.1); **weights and offsets**
(§2.5–2.6, incl. the frequency-model equivalence of exposure-as-offset vs
exposure-as-weight); distribution inventory incl. Tweedie (§2.7); model
validation (Ch. 7: lift/loss-ratio charts, Gini); **model documentation**
(Ch. 8, incl. "code as documentation"); ensembling competing models (§9.3).

**Supports:** D-001 (exposure handling), D-009 (LOB recalibration as a
target/level choice), D-018 (local D² as a validated proxy → time-based
holdout), D-029 (ensembles judged by whether they move the score, not by
taste), and the repo's guardrail/test culture (Ch. 8).

## R-04 — CAS Institute, *Study Note: Model Validation and Holdout Data*
**The CAS Institute, 13 January 2019 · 21pp**
<https://thecasinstitute.org/wp-content/uploads/2019/01/Exam-3-Study-Note-Model-Validation-01162019.pdf>

Provided by the human, 2026-10-21, specifically to check holdout design before
a submission. Distilled into the `model-validation` skill. The authority on the
one point where our repo is weakest.

Contents used here: the **three-way split** (training / test / holdout) and why
a repeatedly-consulted test set cannot serve as the objective metric;
**cross-validation is not a substitute for holdout** when humans choose the
model, because the *choice* is influenced by all the data even if the *fit* is
not; **refitting on complete data after measuring is standard practice**, with
the reporting rule that one may not claim better fit than was measured;
grouping **correlated observations** (policyholder across policy-years) into the
same split; **out-of-time validation** as the right design when time drives
correlation, with the caveat that time is often *not* the main dimension in
insurance and the recommended substitute — **residual stability by feature
level across the two time halves**; hash-based (MD5) rather than RNG splits for
reproducibility; and the diagnostics: **normalized Gini** (model Gini / data
Gini) as an overfit measure read via the *train-vs-test gap*, plus Lorenz
curves, lift charts on equal-exposure bins, and double lift charts.

**Supports:** D-045 (final refit on all years, explicitly endorsed), D-006
(out-of-time rolling-origin), and is the source for the *new* items it
introduces — the normalized Gini train/test gap and residual-stability-by-time,
neither of which we currently compute. It also supplies the language for a
disclosure we should make: our 2024 slice is a **test** set by this definition,
not a holdout, and the leaderboard is our only untouched data.

---

## R → D mapping

| Reference | Decisions / stages |
|---|---|
| R-01 | D-005, D-007, D-028, development tooling (`src/development.py`), ASOP 9 documentation posture |
| R-02 | D-010, D-023/D-026, Tweedie metric choice, any future GLM benchmark |
| R-03 | D-001, D-009, D-018, D-029, train/test discipline, weights vs offsets |
| R-04 | D-045, D-006, `model-validation` skill, normalized-Gini / residual-stability diagnostics |

---

## Method candidates (from the references, not yet adopted)

Each row is an idea the sources support that we have not yet measured. Nothing
here reaches a submission without the normal flow: local experiment →
`D-xxx` decision → experiment-log row → quota slot **only if** it clears the
D-025 tie band (~0.002 local D²) or is explicitly accepted as a credibility
play. Deadline 2026-10-23 16:00 — queue by expected value per remaining slot.

| # | Candidate | Source | Lever | Local gate before any slot |
|---|---|---|---|---|
| C-1 | **Bornhuetter-Ferguson labels** for immature AYs: blend chain-ladder development with expected claims (premium × LR) instead of pure LDFs | R-01 Ch. 9 | target/label definition (pairs with D-005/D-028) | compare skill-vs-baseline gap under both truths (same protocol as D-028); a truth change alone never earns a slot |
| C-2 | ~~Tweedie GLM benchmark~~ **MEASURED — see D-046** (Tweedie 1.5, log link, exposure weights, 112 terms) | R-02 App. C, R-03 §5.2 | benchmark/credibility; second leg of C-3 | holdout D² **0.1206** vs LGBM raw 0.1303 ⇒ gap **0.0097**, inside the 0.01 gate. Relativities in `outputs/metrics/glm-001_relativities.csv`. `occupation` excluded (D-047). |
| C-3 | **Form ensemble** (GLM + LGBM, R-03 §9.3) — not seed averaging (closed by D-029) but averaging across *model families* | R-03 §9.3 | prediction blend | blend weight swept locally; must beat exp-010's 0.1342 outside the tie band |
| C-4 | **Cape Cod / expected-claims cross-checks** on the development factors (credibility-weighted LDFs) | R-01 Ch. 8/10 | development diagnostics | documentation/credibility only unless it moves labels measurably |
| C-5 | ~~ASOP 9 documentation pass~~ **DONE — `docs/actuarial_report.md`** (2026-10-22), structured against ASOP 9 §§1–6 as reproduced in R-01 App. C | R-01 App. C | documentation (judging axis) | no model change. Includes the §5.8 reliance-on-AI disclosure and the Section 6 deviation statement. |

**Status:** all `proposed`. C-1 and C-3 are the only ones with a plausible
score lever; C-2/C-4/C-5 are credibility plays that do not compete for quota.

**Status 2026-10-21:** C-2 is **measured** (D-046) and clears the C-3 gate
marginally (gap 0.0097 raw-vs-raw), so C-3 is cleared to proceed with the
GLM as the weaker leg. C-4 and C-5 remain open.

---

## Notes on how these sources interact with our findings

- **Freq/sev (R-02 App. J, R-03 §5.1.1):** the literature says the two-step
  approach is usually *similar* to Tweedie with an interpretability benefit.
  Our exp-005 measured the split **worse** on this book (local 0.1102 vs
  0.1250, LB 0.10314 vs 0.11519). Both statements are kept: the reference
  explains the standard trade-off; the measurement decides here.
- **Holdout (R-03 §4.3):** externally confirms what D-018 measured internally —
  out-of-sample testing is required and the split must be fixed before fitting.
  Our split is by time (2021–23 → 2024), stricter than random CV and consistent
  with the 2025 scoring population.
- **Weights vs offsets (R-03 §2.5–2.6):** consistent with D-001 — exposure is a
  weight/amount question we solved by recomputing exposure and modelling cost
  as `exposure × annual rate`; the corrupt supplied column is audit-only.

## R-05 — Casotto (Akur8), *Penalized GLM: Between Credibility and GBMs*
**NAIC Call Materials · Mattia Casotto, Head of Product US, Akur8 · 62pp slide deck**
<https://content.naic.org/sites/default/files/call_materials/Handout%20-%20Akur8%20-%20Penalized%20Regression%20-%20Between%20Credibility%20and%20GBM.pdf>

Provided by the human, 2026-10-21. **Image-heavy deck** — most figures did not
survive text extraction, so the argument below is reconstructed from the slide
prose and should be checked against the figures before quoting specifics.

Closes the gap R-04 left open (credibility theory for C-3 weighting). Core
claims from the text:

- **Ridge regression *is* Bühlmann credibility** under a Gaussian assumption
  (cites Jewell 1974, *Credible Means are exact Bayesian for Exponential
  Families*, ASTIN Bull. 8(1)). **λ equals the credibility K parameter.**
  So a penalized GLM is not an ML technique bolted onto actuarial work — it is
  a credibility procedure by construction, and the deck states it meets
  **ASOP 25 (Credibility Procedures)**.
- **Lasso = credibility + effect selection**: the Laplace prior is "pointy" at
  zero, so it shrinks low-volume levels toward the grand average *and* zeroes
  what is immaterial — replacing p-value thresholding, which the deck criticises
  as binary, threshold-dependent, and unjustified when its hypotheses fail.
- **Derivative Lasso** — a Laplace prior on the *derivative* of an ordinal
  variable. Consecutive levels get pulled together unless the data separates
  them, which reproduces GBM's adaptive grouping while staying additive.
  Governing trade-off collapses to **one parameter (smoothness)** versus a GBM's
  several (trees, learning rate, depth…).
- **Warning on polynomial feature engineering:** a driver-age curve with a 4th
  degree polynomial + log term is fully statistically significant yet the
  upturn over age 80 is *"an artifact of the polynomial transformation more
  than material to the data"*, and two equally-significant specifications give
  wildly different tails. Significance is not evidence of a real tail.
- Cites **CAS Monograph §10.5** (Goldburd et al., *GLMs for Insurance Rating*,
  2nd Ed) as the reference treatment of penalized GLM.

**Supports:** a credibility-shrinkage fix for high-cardinality categoricals
(see the R-06 diagnostic), a principled rather than swept blend weight for C-3,
and the C-2 GLM should be **penalized** (elastic net), not unpenalized.

## R-06 — Fan, Luteijn & Tam, *Bridging transparency and predictive power:
integrating explainable ML into actuarial modelling*
**IFoA Big Pitch 2026, Staple Inn, 18 March 2026 · Research proposal, 17pp**
<https://actuaries.org.uk/media/xikgrpeg/bridging-transparency-and-predictive-power-integrating-explainable-ml-into-actuarial-modelling-fiona-fan-michiel-luteijn-jacky-tam.pdf>

Provided by the human, 2026-10-21. Life/H&C mortality setting (ILEC data,
273,402 claims, 160m life-years, Poisson deviance) — different line of
business, but the *methodological* comparison transfers and it is the most
direct external evidence we have on our two live questions.

Three hybrid approaches, all producing an **additive** final model:
1. **IBLM** — fit a GLM, then fit XGBoost on its residuals, then absorb the
   corrections back into a GLM-like structure.
2. **XGBoost-informed GLM** — use SHAP dependence plots to place spline knots
   (e.g. knots at 25/33/45/55/70 on the age curve) and hand-write the formula.
3. **Interaction detection** — XGBoost finds interactions, then a fresh additive
   model is built with just those terms.

**The result that matters to us: rankings do not transfer from a random split to
an out-of-time split.** They ran both — random 80/20, and 2012–2017 → 2018–2019
— and the table is explicit:

| model | random rank | 2018–19 rank |
|---|---|---|
| IBLM (depth 2) | 7 | **11** (worst) |
| IBLM (depth 3) | 6 | 10 |
| GLM baseline | 10 | **7** |
| GAM + XGB interactions | 3 | 3 |
| XGBoost (black box) | 1 | 1 |

Their own explanation: *"The performance ranking of IBLM dropped for the
out-of-time test data. This is hypothesised to be due to the inability of the
internal booster model to extrapolate an observation year effect."* That is
**D-025 reproduced externally**, and the mechanism is the one behind our D-008
choice (trees cannot extrapolate a year effect).

Two further points:
- The best hybrid is **interaction detection into a GAM**, stable at rank 3 on
  both splits. It beats IBLM out-of-time decisively. Its advantage is that
  XGBoost is used as a *diagnostic*, not as a predictive engine — the modeller
  keeps control of functional form (and can impose monotonic constraints).
- The hybrid keeps only part of the black-box gain: on the random split the
  final GAM captures ~80% of the XGBoost improvement; out-of-time only ~62%.
  Transparency costs accuracy, and costs *more* out-of-time.
- Governance frame: **TAS 100** principles 5 (models) and 7 (communication)
  make transparency a de facto requirement.

**Supports:** D-025 (our local gates are weak), D-008 (no tree year
extrapolation), and re-shapes C-3 — the better-evidenced version is
*interaction detection into the GLM*, not averaging GLM and LGBM predictions.

## Not relevant — recorded to prevent a wasted re-read

**McNichols & Rizzo, *Stochastic GBM Methods for Modeling Market Prices***
(CAS E-Forum, Summer 2012). "GBM" here is **Geometric Brownian Motion** —
market price simulation for FX, commodities and equity indices with a
Cox–Ingersoll–Ross rate generator and option-implied forward curves. Nothing to
do with gradient boosting machines or with insurance ratemaking. Supplied in
error via acronym collision.

**Status 2026-10-22 (final):** C-1 measured (D-060) — a principled no-op, gap
+0.0002. C-2 measured (D-046). C-3 measured (error correlation 0.999, recorded in
the D-059 session) — refuted. C-4 would follow C-1 by construction. C-5 done
(`docs/actuarial_report.md`). **The candidate queue is empty.**
