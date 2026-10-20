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

| 2026-10-20 13:07 UTC | exp-005 | freq/sev split: Poisson freq x Gamma severity (D-023) | 0.1102 | 0.12996 | posted | 6101af03a7b2 | Population-shift test: locally 0.1102 vs exp-004 0.1250 (worse). Tests whether local ranking transfers. |

| 2026-10-20 13:06 UTC | exp-006 | exp-004 + premium-formula derived features (D-021/D-024) | 0.1242 | 0.12996 | posted | fac8f9ceb4ba | Population-shift test: locally 0.1242 vs exp-004 0.1250 (slightly worse). If LB ranks it higher, 2024 holdout != 2025 population. |

| # | timestamp | tag | change | hypothesis | local D² | LB score | Δ vs prev | attribution | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-20 03:01 | exp-001 | `exposure ×` developed cost rate per LOB (D-005) | establishes the metric scale and a quota-reset marker | 0.0797 | **0.07293** | — | baseline | `7e1b0082099b` |
| 2 | 2026-10-20 05:02 | exp-002 | LGBM Tweedie on rating factors + premium, leakage removed (D-016) | a real risk model should beat a flat LOB rate substantially | 0.1207 | **0.11226** | **+0.0393** | clean | `12da874d8ce1` |
| 3 | 2026-10-20 05:02 | exp-003 | exp-002 + lagged claim experience block (D-014) | experience is missing from price, so it is pure signal | 0.1230 | **0.11339** | **+0.0011** | clean | `fdff74e6ab85` |
| 4 | 2026-10-20 05:03 | exp-004 | exp-003 + weather + property flags (D-012) + LOB recalibration (D-009) | weather drives property perils (see DGP findings §5) | 0.1250 | **0.11519** | **+0.0018** | clean | `3c9cf1621af2` |

**Leaderboard at time of writing:** top 0.12996 (22 teams). Our best 0.11519.

### Metric validation — local D² is a reliable proxy

| experiment | local D² | LB score | gap |
|---|---|---|---|
| exp-001 | 0.0797 | 0.07293 | −0.0068 |
| exp-002 | 0.1207 | 0.11226 | −0.0084 |
| exp-003 | 0.1230 | 0.11339 | −0.0096 |
| exp-004 | 0.1250 | 0.11519 | −0.0098 |

Local D² is consistently **optimistic by 0.007–0.010** and ranks models correctly.
That validates the whole CV setup (time split 2021–23 → 2024, developed truth,
Tweedie p=1.5, LOB recalibration) as a trustworthy offline oracle. Going forward
we can iterate locally and spend a slot only on a new best.

### Feature attribution (measured on the leaderboard)

| block | Δ LB | verdict |
|---|---|---|
| Model vs flat LOB rate | **+0.0393** | the dominant gain — risk differentiation is where the score is |
| Lagged claim experience | +0.0011 | real but modest; sparse in training (7.7% of rows) and it will be denser at 2025 scoring (20–26% of test rows have history) |
| Weather + property flags | +0.0018 | real; weather features rank highly for property lines |

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
| 2026-10-20 | **leakage probe on policy-period fields** | `customer_service_contacts` alone → D² 0.628; claimants 3.28 contacts vs 0.41 | **D-016 deny-list** |
| 2026-10-20 | exposure aggregation pitfall | naive merge+filter dropped policy rows with claims in other periods; exposure understated 787k→636k on motor | D-001 note + regression test |
| 2026-10-20 | DGP: severity family | Gamma beats lognormal on AIC for all 11 perils; log-SD constant across years (Levene p 0.098–0.96) | per-peril scale index is sufficient |
| 2026-10-20 | DGP: premium formula | motor log-premium R² = 0.956, fully decoded (age 3 bands, ncb 3 bands, vvg log-linear, cover, region, +4%/yr) | use premium as base, correct its misalignments |
| 2026-10-20 | DGP: price vs risk misalignment | premium ignores claim experience entirely; vvg premium ↑ while freq ↓; TPFT ≈ comp price but freq 0.47 | experience block is pure signal |
| 2026-10-20 | DGP: weather→peril links | hail→glass RR 2.35; flood_zone 1→5 RR 190×; freezing→water 2.02; dry_summer→subsidence 3.03 | add weather features by peril |
| 2026-10-20 | DGP: recoveries / excess | recoveries rule-based (OD+TP = 0.8×incurred, theft+TP = 0.3×); excess never applied to claim amounts | confirms D-003; no excess feature needed |
| 2026-10-20 | LOB recalibration | model underpredicts buildings by 15% before recalibration | D-009 apply per-LOB scale |
| 2026-10-20 | forward-factor sensitivity | D² flat for factors 0.98–1.03, declines beyond 1.05 | level is close; keep factor modest |
| 2026-10-20 | **local D² vs leaderboard** | gap −0.0068 / −0.0084 / −0.0096 / −0.0098 across 4 submissions | local CV is a validated proxy (D-018) |

---

## Open questions feeding experiments

1. D-009 — 2025 level factor: blended 70/30 vs pure claims-index vs pure premium/LR.
2. D-008 — forward-factor-only vs full de-trending for tree models.
3. Development: per-peril LDFs vs LOB-level (esp. motor bodily_injury).
4. Pooled vs per-LOB D² weighting — affects how much to calibrate each LOB's level.
5. Weather: 2025 indices vs lagged 2024 (sensitivity for the leakage question).

### Population-shift check (user-requested, 2026-10-20 13:0x)

Both "worse-locally" runs were submitted to test whether local ranking transfers
to the 2025 population. **It does not transfer for near-ties.**

| exp | what | local D² | LB score | local rank | LB rank | Δ LB vs exp-004 |
|---|---|---|---|---|---|---|
| exp-004 | base best (weather + flags) | 0.1250 | 0.11519 | 1st | 2nd | — |
| exp-006 | + premium-derived features | 0.1242 | **0.11569** | 2nd | **1st** | **+0.0005** |
| exp-005 | freq/sev split (Poisson × Gamma) | 0.1102 | 0.10314 | 3rd | 3rd | −0.0121 |

**Conclusions (D-025):**

1. **Near-ties do not rank reliably.** exp-006 was 0.0008 *worse* locally and 0.0005
   *better* on the leaderboard. Local differences below ~0.002 D² are within the
   population-shift noise and must not be treated as real.
2. **Large local differences do transfer directionally.** exp-005's −0.0148 local
   became −0.0121 on the leaderboard — same ordering, similar magnitude. The
   freq/sev split is genuinely worse, not a measurement artefact.
3. **The optimism gap is stable** (−0.0068 to −0.0098 across all six points), so
   local D² remains useful for *level* even when ordering is unreliable near the top.
4. **exp-006 is the new best (0.11569).** The premium-formula derived features
   (D-024) help on the 2025 population in a way the 2024 holdout could not detect —
   consistent with D-021: the premium structure transfers and the model benefits
   from having its components exposed rather than re-derived.

| exp | local | LB | gap |
|---|---|---|---|
| exp-001 | 0.0797 | 0.07293 | −0.0068 |
| exp-002 | 0.1207 | 0.11226 | −0.0084 |
| exp-003 | 0.1230 | 0.11339 | −0.0096 |
| exp-004 | 0.1250 | 0.11519 | −0.0098 |
| exp-005 | 0.1102 | 0.10314 | −0.0071 |
| exp-006 | 0.1242 | 0.11569 | −0.0085 |

**Leaderboard now:** top 0.12996 · our best **0.11569** · 6 of 10 slots used in the
current window (4 remain).
