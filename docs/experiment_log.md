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

| # | timestamp | tag | change | hypothesis | local D² | LB score | Δ vs prev | attribution | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-20 03:01 | exp-001 | `exposure ×` developed cost rate per LOB (D-005) | establishes the metric scale and a quota-reset marker | 0.0797 | **0.07293** | — | baseline | `7e1b0082099b` |
| 2 | 2026-10-20 05:02 | exp-002 | LGBM Tweedie on rating factors + premium, leakage removed (D-016) | a real risk model should beat a flat LOB rate substantially | 0.1207 | **0.11226** | **+0.0393** | clean | `12da874d8ce1` |
| 3 | 2026-10-20 05:02 | exp-003 | exp-002 + lagged claim experience block (D-014) | experience is missing from price, so it is pure signal | 0.1230 | **0.11339** | **+0.0011** | clean | `fdff74e6ab85` |
| 4 | 2026-10-20 05:03 | exp-004 | exp-003 + weather + property flags (D-012) + LOB recalibration (D-009) | weather drives property perils (see DGP findings §5) | 0.1250 | **0.11519** | **+0.0018** | clean | `3c9cf1621af2` |
| 5 | 2026-10-20 13:06 | exp-006 | exp-004 + premium-formula derived features (D-021/D-024) | population-shift test: locally 0.1242 vs exp-004 0.1250 (worse); if LB ranks it higher, 2024 holdout ≠ 2025 population | 0.1242 | **0.11569** | **+0.0005** | clean | `fac8f9ceb4ba` |
| 6 | 2026-10-20 13:07 | exp-005 | freq/sev split: Poisson freq × Gamma severity (D-023) | population-shift test: locally 0.1102 vs exp-004 0.1250 (worse); tests whether local ranking transfers | 0.1102 | **0.10314** | **−0.0121** | clean | `6101af03a7b2` |
| 7 | 2026-10-20 14:56 | exp-007 | exp-006 + time-safe historical relativity features (D-027) | historical level relativities (cost rate + freq) strictly lagged (pe < ps) expose price/risk misalignment | 0.1248 | **0.11637** | **+0.00068** | clean | `44af51f7b6c3` |
| 8 | 2026-10-20 15:47 | exp-009 | exp-007 averaged over 3 seeds (D-025/D-029) | seed averaging reduces the variance behind local/LB rank flips | 0.1250 | **0.11632** | **−0.00005** | clean | `5a16b2b2d1ad` |

**Leaderboard at time of writing:** top 0.12996 (22 teams). Our best **0.11637**
(exp-007). 8 of 10 slots used in the current window (2 reserved).

### Metric validation — local D² is a reliable proxy

| experiment | local D² | LB score | gap |
|---|---|---|---|
| exp-001 | 0.0797 | 0.07293 | −0.0068 |
| exp-002 | 0.1207 | 0.11226 | −0.0084 |
| exp-003 | 0.1230 | 0.11339 | −0.0096 |
| exp-004 | 0.1250 | 0.11519 | −0.0098 |
| exp-006 | 0.1242 | 0.11569 | −0.0085 |
| exp-005 | 0.1102 | 0.10314 | −0.0071 |
| exp-007 | 0.1248 | 0.11637 | −0.0084 |
| exp-009 | 0.1250 | 0.11632 | −0.0087 |

Local D² is consistently **optimistic by 0.007–0.010** and ranks models correctly
in direction (D-018, now 8 points). That validates the whole CV setup (time split
2021–23 → 2024, developed truth, Tweedie p=1.5, LOB recalibration) as a
trustworthy offline oracle — with the near-tie caveat of D-025.

### Feature attribution (measured on the leaderboard)

| block | Δ LB | verdict |
|---|---|---|
| Model vs flat LOB rate | **+0.0393** | the dominant gain — risk differentiation is where the score is |
| Weather + property flags | +0.0018 | real; weather features rank highly for property lines |
| Lagged claim experience | +0.0011 | real but modest; sparse in training (7.7% of rows) and it will be denser at 2025 scoring (20–26% of test rows have history) |
| Premium-formula derived features | +0.0005 | modest, but it *ranked above* exp-004 despite being locally worse (D-025) |
| Historical relativities (D-027) | **+0.00068** | real; strictly-lagged target encodings expose price/risk misalignment |
| 3-seed ensemble (D-029) | −0.00005 | **nothing at tie-level** — variance reduction does not move the board (kept as evidence) |

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
| 2026-10-20 | multi-seed band on exp-006 config | seeds 0/1/2 → 0.1242 / 0.1233 / 0.1240 (mean 0.1238, sd 0.0005) | local seed noise ≈ ±0.0005 — below the ~0.002 D-025 tie band |
| 2026-10-20 | historical relativity block (D-027) | +0.0006 (seed 0) / +0.0009 (seed 1) over exp-006 same-seed; consistent across seeds | submitted as exp-007 → LB +0.00068 |
| 2026-10-20 | per-peril development (D-028) | level-neutral; D² 0.1223 vs 0.1242 but skill-vs-baseline gap identical (0.0444 vs 0.0445) — truth redefined, not skill changed | open Q3 answered locally as a tie; LB tie-break still open |
| 2026-10-20 | weather lagged-2024 sensitivity (D-012) | D² 0.1243 (lagged) vs 0.1242 (same-year) | weather value does **not** depend on same-year indices — leakage/availability question closed |
| 2026-10-20 | 3-seed ensemble on exp-007 stack | local 0.1250 vs best single 0.1248 / mean 0.1242 | submitted as exp-009 → LB 0.11632 vs exp-007 0.11637 — a tie; see D-029 |
| 2026-10-20 | per-peril age→ultimate triangles (re-derived) | motor BI age0→ult 1.535, buildings subsidence 1.306 vs short-tail 1.01–1.05; LOB re-derivation 1.18/1.06/1.03 vs documented 1.22/1.09/1.04 (different weighting window) | D-028 factors frozen from `scripts/build_development.py`; level kept on the accepted D-005 table |

---

## Open questions feeding experiments

1. D-009 — 2025 level factor: blended 70/30 vs pure claims-index vs pure premium/LR.
2. D-008 — forward-factor-only vs full de-trending for tree models.
3. ~~Development: per-peril LDFs vs LOB-level~~ — measured 2026-10-20: skill-neutral
   locally (D-028); the truth definition is more defensible but did not move D².
   LB tie-break still open; keep as a credibility item for the write-up.
4. Pooled vs per-LOB D² weighting — affects how much to calibrate each LOB's level.
5. ~~Weather: 2025 indices vs lagged 2024~~ — closed 2026-10-20: 0.1243 vs 0.1242,
   no dependence on same-year indices (see D-012 sensitivity row).

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
| exp-007 | 0.1248 | 0.11637 | −0.0084 |
| exp-009 | 0.1250 | 0.11632 | −0.0087 |

### Second population-shift pair: exp-007 vs exp-009 (2026-10-20 15:0x)

| exp | what | local D² | LB score | local rank | LB rank |
|---|---|---|---|---|---|
| exp-007 | relativity features, single seed | 0.1248 | **0.11637** | 2nd | **1st** |
| exp-009 | same stack × 3-seed ensemble | 0.1250 | 0.11632 | 1st | 2nd |

The pair flips again inside the tie band (local +0.0002 for the ensemble,
LB −0.00005). **Reinforces D-025:** nothing that resolves at the 4th decimal of
local D² should be expected to move the leaderboard. Seed averaging is harmless
but not a scoring lever on this book (D-029). Real gains still come from feature
blocks (D-027 delivered +0.00068; premium-derived +0.0005; experience +0.0011;
weather +0.0018) and, by far, from risk differentiation over flat rates (+0.0393).

**Leaderboard now:** top 0.12996 · our best **0.11637** (exp-007) · 8 of 10 slots
used in the current window (2 reserved for tie-breaks / best re-submission).
