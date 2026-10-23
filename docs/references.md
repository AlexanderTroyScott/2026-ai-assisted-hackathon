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

## R-07 — DuVall, *Testing for Shifts in Reserve Adequacy*
**CAS Proceedings LXXIX, 1992 · Richard M. DuVall**
<https://www.casact.org/sites/default/files/database/proceed_proceed92_92001.pdf>

Regression models that test for shifts/trends in loss development factors, as an
objective alternative to the subjective Berquist-Sherman judgment. Models the
LDF as a function of age, then introduces dummy/trend terms on the LDF
parameters and tests significance; if a shift is detected, restates early
valuations to be consistent with recent reserving practice.

**Core diagnostic it inherits from Berquist-Sherman:** compare the growth rate
of the *per-claim reserve for open claims* against the growth rate of the
*per-claim cost for closed claims* over the experience years. If reserving is
consistent the two grow at the same rate; unequal rates indicate a shift in
reserve adequacy.

**Requires:** a loss triangle of reported losses at **annual valuation dates**.

## R-08 — Ghezzi, *Loss Reserving Without Loss Development Patterns — Beyond Berquist-Sherman*
**CAS Forum, 2001 · Thomas L. Ghezzi, FCAS, MAAA**
<https://www.casact.org/sites/default/files/database/forum_01fforum_01ff043.pdf>

For situations where case-reserving and/or settlement-rate changes have made
historical development patterns inappropriate for projecting immature years.
Where B-S *restates prior years* to the current basis, Ghezzi **restates the
current diagonal to the level implied by the older years' estimates**, on an
*implied ultimate* basis — eliminating the need to apply (possibly wrong)
development patterns to the less-mature years. Uses reported/closed/paid claim
counts in addition to loss triangles.

**Key empirical finding (Scenario with case-reserve strengthening):** the
**incurred** projection overstates ultimate ($796.0M vs $766.5M actual), while
the **paid** projection equals actual. Incurred development is contaminated by
reserve-adequacy shifts; paid development is the cleaner signal.

**Requires:** valuation-date history plus claim-count (reported/closed/paid)
triangles — i.e. `close_date`, which our dataset does not have.

## R-10 — Cook, *Trend and Loss Development Factors*
**CAS Proceedings LVII, 1970 · Charles F. Cook**
Classic on the conceptual confusion between trend and development factors — the
data base a factor is derived from versus the purpose it serves. Background for
the trend-vs-development separation we already follow (D-007/D-008).

## R-11 — Murphy, *Unbiased Loss Development Factors*
**CAS Forum, Spring 1994 · Daniel W. Murphy, FCAS, MAAA**
<https://www.casact.org/sites/default/files/database/forum_94spforum_94spf183.pdf>

Mathematical framework for whether the loss development technique is biased, and
which link-ratio estimator is best (we use volume-weighted). Relevant to whether
our chain-ladder LDFs are unbiased in the sense that matters.

## R-12 — CAS Tail Factor Working Party, *The Estimation of Loss Development Tail Factors*
**CAS E-Forum, Fall 2013 · Herman, Shapland (co-chairs) et al.**
<https://www.casact.org/sites/default/files/database/forum_13fforum_02-tail-factors-working-party.pdf>

Authoritative treatment of the tail beyond the observable triangle — exactly the
age-4+ development question raised by the human. Confirms the tail is a real,
separately-estimated quantity (we measured it negligible here: 6 claims / 0.1M
beyond age 3).

## R-13 — Spore, *Techniques for the Conversion of Loss Development Factors*
**CAS Forum, Spring 1998 · Louis Spore, ACAS, MAAA**
<https://www.casact.org/sites/default/files/database/forum_98sforum_98sf373.pdf>

Accident-year vs policy-year development factors and their conversion. Relevant
because our `acc_year` is keyed to policy start year.

## R-14 — Blumsohn & Laufer, *Unstable Loss Development Factors*
**CAS E-Forum, Spring 2009 · Gary Blumsohn, FCAS, Ph.D., Michael Laufer, FCAS, MAAA**
<https://www.casact.org/sites/default/files/database/forum_09spforum_01blumsohn.pdf>

How actuaries handle unstable triangles. Relevant to the declining motor f(0→1)
we observed (1.268 → 1.233 → 1.173).

## R-15 — Korn, *Strategies for Modeling Loss Development: Curve Fitting, Credibility, and Layer Adjustments*
**CAS · Uri Korn**
<https://www.casact.org/sites/default/files/2021-07/Strategies-Modeling-Loss-Korn.pdf>

Smoothing splines and credibility for loss development curves; layer-adjusted
development. Relevant to the size/layer dependence of development.

## R-16 — Boor, *[Best Paper 2017] — on individual-claim loss development distributions*
**CAS · Joseph Boor**
<https://www.casact.org/sites/default/files/2021-02/2017_Best-Paper_Boor.pdf>

**The one that matters for us.** *"The ultimate cost of an unpaid individual
claim follows a probability distribution… when actuaries apply loss development
factors to individual claims, they often create biased estimates of excess loss
costs."* Applying an aggregate LDF uniformly to individual claims is biased,
especially in the tail/layers — and our target applies `y_ult = y × LDF` to every
row's total, closed claims included.

## R-17 — Archer Actuarial, *How to Calculate Implied Development Patterns*
**Practitioner note, April 2025 · <https://archeractuarial.com>**
Worked example of the Ghezzi (R-08) implied-development-pattern method on a
Schedule P aggregate triangle: paid% and reported% of ultimate by maturity,
then age-to-age/age-to-ultimate for both paid and incurred. Confirms the paid
pattern is the cleaner signal, but requires the paid/IBNR triangle by maturity —
i.e. the valuation-date history our single-snapshot dataset does not provide.

---

## Curated bibliography (human-provided, 2026-10-23) — 40 items in 6 tiers

Recorded verbatim from the human's list. Several are already R-xx entries; the
rest are logged here as a working bibliography with a relevance flag for our
specific task (predict expected ultimate net cost per 2025 policy, Tweedie D²).

### Already in R-xx (no action)
Friedland (R-01) · GLM for Insurance Rating (R-02/R-03) · Tail Factor Working
Party (R-12) · Korn curve-fitting (R-15) · Ghezzi Beyond B-S (R-08) · DuVall
(R-07) · Colella-Jones four-algorithm ratemaking (digested, in R-06 discussion) ·
Kuo BMDN (R-09) · Boor individual-claim distributions (R-16) · Archer implied
patterns (R-17) · CAS Statement of Principles (R-01 appendix).

### New, but blocked by our single-snapshot data (need valuation history /
close_date / paid triangle)
- England & Verrall, *Stochastic Claims Reserving* (Mack, bootstrap, prediction
  error) — needs the triangle's process variance.
- IFoA *Practitioner's Introduction to Stochastic Reserving* — same.
- SOA *Comparison of IBNR Methods* (BF/Cape Cod/GLM survey) — methods need paid
  + incurred by maturity.
- Struhuss, *Using Best Practices to Determine a Best Reserve Estimate*.
- *Adjusting Incurred Loss for Simultaneous Shifts in Payment Patterns and Case
  Reserve Adequacy* (Berquist-Sherman / Fleming-Mayer) — the case-adequacy test
  we want, needs payment history.
- *On the Accuracy of Loss Reserving Methodology* — method comparison.
- Clark, *Reserving with Incomplete Exposure Information* (LDF ↔ Cape Cod ↔ BF
  link) — conceptual, we measured BF a no-op (D-060).
- Brosius, *Loss Development Using Credibility*.
- BF Initial Expected Loss Ratio Working Party.
- Improved Parametric Tail Factor Models (2025).
- Meyers Bayesian MCMC; Gesmann & Morris compartmental models; unifying
  triangle-based reserving; chainladder-in-Python guide.

### Directly relevant to OUR per-policy prediction (worth a look if time allows)
- Guszcza & Lommele, *Loss Reserving Using Claim-Level Data* — individual-claims
  reserving, the closest to our granular setting.
- Korn et al., *Comprehensive Non-Aggregated Stochastic Approach to Loss
  Reserving*.
- Richman & Balona, *The Actuary and IBNR Techniques: A Machine Learning
  Approach* — ML for individual-claim IBNR.
- Merz & Wüthrich, *Double Chain Ladder* — the classical ↔ granular link; the
  rigorous form of our split development (D-061).
- Werner & Modlin, *Basic Ratemaking* — exposure, trends, LDFs in pricing.
- *Alternative to Tweedie in Pure Premium GLM* (two-part frequency × severity) —
  we measured this worse (D-023), but the paper is the reference for why.
- *A Bayesian Model for Estimating Long Tailed Excess of Loss* — thresholds /
  excess layers; relevant because our metric is tail-dominated.

### Data handling
- *Completing and Using Schedule P* (paid/incurred net of salvage & subrogation)
  — maps to our "net of recoveries" convention (D-003).

**Note on the human's suggested skill split** (emergence features / baseline
ultimates / per-policy ultimates / GLM freq-sev): this is the right decomposition
for a full reserving *and* ratemaking pipeline. For our competition — predict
expected ultimate cost per 2025 policy — steps (3) and (4) are the relevant ones,
and step (3)'s individual-claims methods are the ones that need the payment
history we lack.
