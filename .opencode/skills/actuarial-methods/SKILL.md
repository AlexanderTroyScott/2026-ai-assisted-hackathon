---
name: actuarial-methods
description: Use when choosing or justifying modelling methods — GLM vs ML, Tweedie vs frequency-severity, exposure weights vs offsets, development/Bornhuetter-Ferguson/Cape Cod labels, holdout design, model documentation. Distilled from CAS sources (docs/references.md R-01..R-03) so agents apply standard actuarial practice without loading the PDFs.
---

# Actuarial Method Notes (CAS-sourced)

Distilled from `docs/references.md` (R-01 Friedland reserving, R-02 Anderson GLM
guide, R-03 Goldburd GLM rating monograph). Full annotations and method
candidates C-1..C-5 live there. Numbers below are from the **sources**, not from
our data — never paste them into `docs/` as findings (AGENTS.md rule 1).

## Model form (R-02, R-03)

- Pure premium (cost/exposure) has two standard routes: **Tweedie GLM** in one
  step, or **frequency × severity** models in two steps. R-02 App. J: results
  are often similar; the two-step route buys interpretability (a factor can move
  freq and sev in opposite directions). On *this* book the split measured worse
  (D-023/D-026, exp-005) — trust the measurement over the textbook default.
- Tweedie with 1<p<2 is the standard pure-premium family: point mass at zero
  plus Gamma-like positive tail (R-02 App. C). The competition metric is
  Tweedie deviance at **p=1.5** — keep that exact power in scoring code (D-010).
- **Exposure:** weight or offset depending on the target (R-03 §2.5–2.6); a
  count model with exposure offset equals a frequency model with exposure
  weight. Here we recompute exposure from dates (D-001) and model
  `exposure × annual rate` — do not use the corrupt supplied column.

## Validation (R-03 §4.3, Ch. 7)

- Fix the train/holdout split **before** fitting; never select on the training
  data. Our split is by time (2021–23 → 2024), stricter than random CV and
  matched to the 2025 scoring population (D-018).
- Judge candidate models on the holdout (lift/loss-ratio style diagnostics),
  but local near-ties do not transfer to this leaderboard (D-025: ~0.002 local
  D² band) — the leaderboard decides ties.

## Development / labels (R-01)

- Development technique (chain ladder) is the base; **Bornhuetter-Ferguson**
  (Ch. 9) blends developed + expected claims and stabilises immature years;
  **Cape Cod** (Ch. 10) is the credibility-weighted variant. Candidate C-1 is
  BF labels — gate on skill-vs-baseline gap, same protocol as D-028 (a truth
  redefinition alone moves D² and earns nothing).
- Per-peril development is legitimate and expected (long-tail perils develop
  more) — we keep level-neutral per-peril factors (D-028).
- Triangles are also a **diagnostic** (R-01 Ch. 5–6): age-to-age factors, volume
  weighted, with the mean preserved.

## Documentation (R-01 App. C = ASOP 9, R-03 Ch. 8)

- Data, assumptions, methods, diagnostics and disclosures all get written down;
  "code as documentation" is explicit in R-03. This repo's `docs/decisions.md` +
  `docs/experiment_log.md` + tests are that record — extend them, never rewrite.
