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

---

## R → D mapping

| Reference | Decisions / stages |
|---|---|
| R-01 | D-005, D-007, D-028, development tooling (`src/development.py`), ASOP 9 documentation posture |
| R-02 | D-010, D-023/D-026, Tweedie metric choice, any future GLM benchmark |
| R-03 | D-001, D-009, D-018, D-029, train/test discipline, weights vs offsets |

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
| C-2 | **Tweedie GLM benchmark** (log link, exposure weight, top rating factors) as an interpretable second model form | R-02 App. C, R-03 §5.2 | benchmark/credibility; possible form-ensemble with LGBM | GLM must be within ~0.01 local D² of LGBM to be worth blending; report relativities in the write-up even if unsubmitted |
| C-3 | **Form ensemble** (GLM + LGBM, R-03 §9.3) — not seed averaging (closed by D-029) but averaging across *model families* | R-03 §9.3 | prediction blend | blend weight swept locally; must beat exp-010's 0.1342 outside the tie band |
| C-4 | **Cape Cod / expected-claims cross-checks** on the development factors (credibility-weighted LDFs) | R-01 Ch. 8/10 | development diagnostics | documentation/credibility only unless it moves labels measurably |
| C-5 | **ASOP 9 documentation pass** on the final write-up: data, assumptions, methods, diagnostics, disclosures checklist | R-01 App. C | documentation (judging axis, 15%) | no model change; checklist against `docs/` trail |

**Status:** all `proposed`. C-1 and C-3 are the only ones with a plausible
score lever; C-2/C-4/C-5 are credibility plays that do not compete for quota.

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
