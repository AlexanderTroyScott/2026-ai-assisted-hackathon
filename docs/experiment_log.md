# Experiment Log

One row per submission **and** per local experiment that informs a submission
decision. Append-only. Mirrored to `outputs/experiment_log.csv`.

Grading note: this table is the "which features did and didn't help" evidence for
the write-up and for the video demo. Keep `hypothesis` and `Δ vs prev` honest —
negative results are as valuable as positive ones.

---

## Legend

- **local D²** — Tweedie D² (p=1.5) on the time-based holdout, scored against
  **developed** truth (see D-006). Always note which holdout year.
- **Δ vs prev** — change in leaderboard score against the previous submission.
- **attribution** — `clean` (one primary change) / `confounded` (several) / `baseline`.

---

## Submissions

| 2026-10-20 05:03 UTC | exp-004 | exp-003 + weather and property flags (D-012); LOB recalibration (D-009) | 0.1250 | 0.12996 | posted | 3c9cf1621af2 | Best local model. Weather adds +0.0020, LOB recal +0.0010 |

| 2026-10-20 05:02 UTC | exp-003 | exp-002 + lagged claim experience features (D-014) | 0.1230 | 0.12996 | posted | fdff74e6ab85 | Tests experience block: +0.0023 local D2 over exp-002 |

| 2026-10-20 05:02 UTC | exp-002 | LGBM Tweedie base model (rating factors + premium); D-016 leakage removed | 0.1207 | 0.12996 | posted | 12da874d8ce1 | Tests model vs constant_rate. Local D2 0.1207 vs 0.0797 baseline |

| 2026-10-20 03:01 UTC | exp-001 | Baseline: exposure x developed cost rate per LOB (D-005 LDFs applied) | n/a - no holdout truth | 0.12996 | posted | 7e1b0082099b | Marker submission to track quota reset; constant-rate baseline per D-015 |

| # | timestamp | tag | message | change | hypothesis | local D² | LB score | LB rank | Δ | attribution | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| — | 2026-10-20 | — | — | no submissions made prior to this log | — | — | — | — | — | — | Verified via `kaggle competitions submissions` → "No submissions found" |

---

## Local experiments (no leaderboard slot)

| date | experiment | result | decision |
|---|---|---|---|
| 2026-10-20 | exposure column conventions | 3 conventions + 2,319 rows in days units across train | D-001 recompute |
| 2026-10-20 | claim→row attribution | 162,736 matched uniquely; 414 orphans; 0 multi-matched | D-002 drop orphans |
| 2026-10-20 | repudiated claims amounts | all zero | D-004 filter is a no-op |
| 2026-10-20 | incurred triangle / LDFs | motor ×1.26, buildings ×1.09, contents ×1.04 (age 0 → ultimate) | D-005 develop labels |
| 2026-10-20 | open-claim share by AY | 0.7% → 20.4% (2024) | D-005/D-006 |
| 2026-10-20 | per-peril severity trend | +2.9%/yr … +9.4%/yr; BI −9.5% is closure artefact | D-007 per-peril factors |
| 2026-10-20 | quantile shift test | scale shift, not shape change | D-007 scale index OK |
| 2026-10-20 | premium/exposure index | 2025 already trended +3–6% vs 2024 | D-009 premium anchor |
| 2026-10-20 | experience feature deciles | freq 0.083 → 0.486; cost/exp 323 → 1,750 | D-014 top priority |
| 2026-10-20 | premium deciles | LR flat 0.71–0.81; freq 4x across deciles | D-015 premium as benchmark |
| 2026-10-20 | sklearn deviance with y_pred=0 | raises for power=1.5 | D-010 floor at 1e-3 |

---

## Open questions feeding experiments

1. D-009 — 2025 level factor: blended 70/30 vs pure claims-index vs pure premium/LR.
2. D-008 — forward-factor-only vs full de-trending for tree models.
3. Development: per-peril LDFs vs LOB-level (esp. motor bodily_injury).
4. Pooled vs per-LOB D² weighting — affects how much to calibrate each LOB's level.
5. Weather: 2025 indices vs lagged 2024 (sensitivity for the leakage question).
