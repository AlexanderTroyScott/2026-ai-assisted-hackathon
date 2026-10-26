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
| 9 | 2026-10-20 16:57 | exp-010 | exp-007 + address/customer rating attributes (D-030) | the dimension tables were never joined, so the property-geo and demographic rating factors were listed but silently absent from every model — restoring them should help substantially, especially property lines | 0.1342 | **0.12113** | **+0.00476** | clean | `0b06b8e14c23` |
| 10 | 2026-10-21 02:24 UTC | exp-015 | exp-010 stack + corrected 2023 labels (D-033/D-035); premium-only level (D-036) | isolates the label correction; level held at premium-only so D-009 stays out of it | 0.1332 | - | — | clean | `155f5cc7198f` |
| 11 | 2026-10-21 12:54 UTC | exp-016 | exp-015 config, seed 1 (labels corrected, premium level) - LB noise band probe [D-040] | H2 probe: measures unmeasured LB seed band; also tests whether exp-015's -0.00158 was refit noise | 0.1294 | - | — | clean | `0d4d1f9992c4` |
| 12 | 2026-10-21 13:35 UTC | exp-017 | H1 level probe: exp-016 predictions x1.03 uniform (corrected labels, premium level) [D-040] | single primary change = level. Tests whether the 2025 level is short (D-005 anchor / IBNER). Expected under H1: LB > 0.11942 | 0.1294 | - | — | clean | `f624564534d6` |
| 13 | 2026-10-21 14:28 UTC | exp-019 | exp-010 config, seed 1, PRE-D-033 label table (control) - measures old-labels seed band [D-041] | control run via /tmp patch of src.dataset.LDF only. Isolates whether exp-010's 0.12113 is a lucky seed draw. | 0.1337 | - | — | clean | `271770ec27ef` |
| 14 | 2026-10-21 14:49 UTC | exp-020 | exp-010 config, seed 2, PRE-D-033 label table (control) - third point on old-labels seed band [D-041] | FINAL slot today (5 reserved for the human). Completes the seed-band estimate on the old-labels side where variance is largest. | 0.1346 | - | — | clean | `e28fcebeaac8` |
| 15 | 2026-10-21 19:54 UTC | exp-021 | D-045 final refit on ALL training years incl. 2024 (was fitting AY<=2023 only) | single primary change = refit scope. Diagnostic: adding most recent train year (2022->2023) is +0.0022 honest local D2, outside the D-025 band. 2024 was 26% of training rows and unused in fit. | 0.1331 | - | — | clean | `a68c5a47d9a4` |
| 16 | 2026-10-21 23:20 UTC | exp-022 | exp-021 config, seed 1 - second seed to firm up the label-parity claim [D-045][D-048] | identical to exp-021 apart from seed. Tests whether 0.12150 was a lucky draw; the parity claim in D-045 rests on n=1 vs a 3-seed band. | 0.1328 | - | — | clean | `510e0e367a6a` |
| 17 | 2026-10-21 23:58 UTC | blend-2s | 2-seed ensemble of exp-021+exp-022 (corrected labels + D-045 refit) | seed average of the two best runs; variance reduction | 0.1329 | - | — | clean | `97fb01260e96` |
| 18 | 2026-10-22 00:14 UTC | exp-023 | exp-021 config, seed 2 - third seed, corrected labels + D-045 refit | third seed of best config; human approved fresh window | 0.1320 | - | — | clean | `357fd1e58866` |
| 19 | 2026-10-22 00:14 UTC | blend-3s | 3-seed ensemble: exp-021+022+023 (corrected labels + D-045 refit) | variance reduction across 3 seeds of the best config | 0.1326 | - | — | clean | `e6aa82c9f4e8` |
| 20 | 2026-10-22 01:55 UTC | exp-024 | D-051 rate target + premium unit fix + earned_premium; D-045 refit | written_premium is ANNUANALISED not pro-rated. Fit y_ult/exposure with exposure weights, multiply back at scoring. +0.0038 honest D2 vs cost target. motor +0.0070. | 0.1365 | - | — | clean | `977c2bf34918` |
| 21 | 2026-10-22 12:00 UTC | exp-025 | D-053 tuned hyperparams (leaves 16, lr 0.02) + marketing_segment; D-045 refit | biggest stable local gain: +0.0057 mean across both folds (2024 0.1352 / 2023 0.1420). Both folds rank it #1. Simpler model: fewer leaves consistently better. Guardrail passes (implied rate 1.44). | 0.1360 | - | — | clean | `3ae436d19cc3` |
| 22 | 2026-10-22 16:01 UTC | exp-027 | D-056 occupation removed from the tree spec; exp-022 config otherwise | analysis run. Single change from exp-022: occupation dropped. Accuracy measured neutral (mean +0.0003, folds disagree); calibration better on both folds. Tests whether the calibration gain transfers. | 0.1327 | - | — | clean | `9539335aba0d` |
| 23 | 2026-10-22 20:32 UTC | exp-028 | D-058 flood_zone only (drop river distance + elevation); D-045 full refit + all-data recalibration | analysis run. Single change from exp-027: drop distance_to_river_km and elevation_m, keep flood_zone. Tests the collinearity hypothesis - all three encode flood risk. Local D2 slightly lower (0.1318 vs 0.1327) but local has been unreliable all day. | 0.1318 | - | — | clean | `be48df10ac4e` |
| 24 | 2026-10-22 22:21 UTC | exp-029 | exp-029: exp-027 config, seed 0 (occupation removed, corrected labels, D-045 refit) | second/third seed of the best config family; enables seed blending | 0.1324 | - | — | clean | `36826703681c` |
| 25 | 2026-10-22 22:21 UTC | exp-030 | exp-030: exp-027 config, seed 2 (occupation removed, corrected labels, D-045 refit) | second/third seed of the best config family; enables seed blending | 0.1324 | - | — | clean | `07b012a56c75` |
| 26 | 2026-10-22 22:24 UTC | blend-027030 | 2-seed ensemble: exp-027 + exp-030 (best two, occupation removed) | blend of the two best seeds of the exp-027 family | 0.1322 | - | — | clean | `de9ed2efeae2` |
| 27 | 2026-10-22 22:24 UTC | blend-03seed | 3-seed ensemble: exp-027 + exp-029 + exp-030 (occupation removed) | full seed ensemble of the exp-027 family | 0.1321 | - | — | clean | `a478bded65cf` |
| 28 | 2026-10-23 01:06 UTC | exp-031 | D-061 split development: develop open claims only (Boor R-16) | first positive result from the development-papers thread. y_ult = y_closed + y_open x OPEN_LDF. Gap +0.0015/+0.0004 on both folds, cal ratio toward 1.0. Human-directed fresh-window submission. | 0.1309 | **0.12301** (Priv 0.12928) | −0.00135 | clean | `d85a2c1969dd` |
| 29 | 2026-10-23 19:50 UTC | exp-033 | D-063 per-LOB rate LGBM (lv=6, lr=0.05) + final-fit rate bugfix | D-063 fixes final_model rate target & params; per-LOB rate LGBM lv=6 (+0.0078 honest D2 over exp-027, motor +0.0114) | 0.1400 | **0.12681** (Priv **0.13531**) | **+0.00245** vs exp-027 (+0.00547 Priv) | clean | `7ceacd8f551e` |
| 30 | 2026-10-23 19:59 UTC | exp-034 | D-063 70/30 blend of per-LOB rate LGBM (exp-033) + pooled rate LGBM (exp-032) | 70% exp-033 (per_lob lv=6 rate) + 30% exp-032 (pooled lv=16 rate); honest OOF D2 0.14033 | 0.1403 | **0.12707** (Priv **0.13536**) | **+0.00026** (+0.00005 Priv) | clean | `6fc43e4e5fbd` |
| 31 | 2026-10-23 20:18 UTC | exp-036 | D-064 multi-family blend: 40% per-LOB XGBoost (depth=3, no_rel) + 42% per-LOB LGBM (lv=6) + 18% pooled LGBM (lv=16) | D-064 adds per-LOB XGBoost rate model (0.14014 solo) to exp-034 LGBM rate blend; honest OOF D2 0.14076 | 0.1408 | **0.12753** (Priv **0.13570**) | **+0.00046** (+0.00034 Priv) | clean | `0039b77f4d8a` |
| 32 | 2026-10-23 21:20 UTC | exp-037 | D-065 per-LOB rate LGBM (lv=6) + year-normalized historical relativities | D-065 normalizes _hist_rate and _hist_freq within (line, acc_year) to mean 1.0, removing inflation & IBNR drift (+0.00052 honest D2 over exp-033) | 0.1405 | **0.12587** (Priv 0.13502) | −0.00094 vs exp-033 | clean | `57b3aaa127c6` |
| 33 | 2026-10-23 21:20 UTC | exp-038 | D-065 4-model ensemble: 35% exp-037 (norm-rel per-LOB LGBM) + 35% exp-035 (per-LOB XGB) + 15% exp-033 + 15% exp-032 | D-064/D-065 ensemble combining year-normalized per-LOB LGBM, per-LOB XGBoost (depth=3), and pooled LGBM rate models | 0.1410 | **0.12721** (Priv 0.13563) | −0.00032 vs exp-036 | clean | `f2418464525e` |
| 34 | 2026-10-23 21:21 UTC | exp-035 | D-064 per-LOB XGBoost rate model (max_depth=3, no_rel) | Standalone per-LOB XGBoost rate model (tweedie p=1.5, max_depth=3, no relativity cols) | 0.1401 | **0.12700** (Priv 0.13509) | standalone XGB | clean | `a354d7148c88` |
| 35 | 2026-10-23 22:42 UTC | emerging-exp-001 | D-066 Part 2 transfer learning + credibility-shrinkage offset adaptation (5-Fold CV D2 0.14045) | Part 2 emerging market: Developed 2.41M-row prior + Offset Per-LOB LGBM (lv=3, min_leaf=600, l2=100). First 4 attempts errored (host evaluator still expected Part 1's 626,639 rows); organizers fixed it and the 2026-10-24 06:34 UTC retry scored. | 0.1405 | **0.16170** (Part 2 Public, rank 9/20 at 2026-10-26) | first Part 2 submission | clean | `477fa66d8c22` |
| 36 | 2026-10-26 23:41 UTC | emerging-exp-002 | D-066 trial: offset-only adaptation w_off=1.0 (drop prior blend; OOF 0.14045 vs 0.14015) | primary change vs exp-001: blend weight 0.75/0.25 -> 1.0 offset | 0.1405 | **0.16297** (new best) | **+0.00127** vs exp-001 | clean | `24a2d2a66f46` |
| 37 | 2026-10-26 23:41 UTC | emerging-exp-004 | D-066 trial: per-LOB prior shrinkage (contents w_off=0.5, motor/buildings 1.0) | primary change vs exp-002: contents blended 50/50 with calibrated dev prior | 0.1405 | 0.16293 | +0.00123 vs exp-001 (−0.00004 vs exp-002: local +0.00005 did not rank) | clean | `529c805122f8` |

**Leaderboard at time of writing:** top 0.12996 (22 teams). Our best **0.11637**
(exp-007). 8 of 10 slots used in the current window (2 reserved).

**Status at 2026-10-26 (supersedes the note above):** Part 1 best is `exp-036`
at **0.12753 Public / 0.13570 Private** (row 31). Part 2 best is
`emerging-exp-002` (row 36) at **0.16297** Public — offset-only adaptation
(D-067); rank 8 of 20 teams, top is `applause_bin` at 0.17303.

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
| exp-010 | 0.1342 | 0.12113 | −0.0131 |

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
| Address/customer attributes (D-030) | **+0.00476** | largest single block since the model itself; occupation/region/elevation_m/energy_rating were absent from every prior model |

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
| 2026-10-20 | categorical code-drift probe (align train vs test level sets) | no drift across all 13 frame categoricals | latent bug, no score effect; align level sets stay consistent |
| 2026-10-20 | **feature-presence audit of the built frame** | 7 listed categoricals + 4 flags + 4 geo numerics + prior_claims_5y absent — `addresses`/`customers` never joined | **D-030** — biggest single fix of the session |
| 2026-10-20 | exp-010: exp-007 + address/customer block | local 0.1342 vs 0.1248 (+0.0094); occupation/region/elevation_m/energy_rating lead importance | submitted → LB 0.12113 (+0.00476) — new best |
| 2026-10-20 | per-peril age→ultimate triangles (re-derived) | motor BI age0→ult 1.535, buildings subsidence 1.306 vs short-tail 1.01–1.05; LOB re-derivation 1.18/1.06/1.03 vs documented 1.22/1.09/1.04 (different weighting window) | D-028 factors frozen from `scripts/build_development.py`; level kept on the accepted D-005 table |
| 2026-10-20 | **exp-011: exp-010 + denied-claim history (D-031)** | local 0.1339 vs exp-010 0.1342 (−0.0003, same seed/truth — inside the D-025 tie band) | **not submitted** — human held the last quota slot (1 of 10 left) for a change that clears the tie band; block stays opt-in |

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
| exp-010 | 0.1342 | 0.12113 | −0.0131 |

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

**Leaderboard now:** top 0.12996 · our best **0.12113** (exp-010) · 9 of 10 slots
used in the current window (1 reserved). Rank 20/22 → gap to top −0.0088.

### D-030 — the attribute-join bug (exp-010, 2026-10-20 16:5x)

`build_frame` never joined `addresses` or `customers`, so 7 categoricals
(`region`, `flood_zone`, `soil_type`, `property_type`, `roof_type`,
`energy_rating`, `urban_density`), 4 property flags (`basement`,
`trees_within_5m`, `solar_panels`, `listed_building`), the geo numerics
(`elevation_m`, `distance_to_river_km`, `year_built`, `bedrooms`) and
`prior_claims_5y` were **listed as features but silently dropped** by `align`
(drop-if-absent). Every model exp-002…exp-009 trained without them.

| | exp-007 | exp-010 | |
|---|---|---|---|
| local D² | 0.1248 | **0.1342** | +0.0094 |
| LB | 0.11637 | **0.12113** | **+0.00476** |
| gap | −0.0084 | −0.0131 | |

Top-10 importances in exp-010 are led by `occupation` (1264) and `region` (492)
with `elevation_m` (295), `energy_rating` (197), `year_built` (168) — exactly
the recovered block. The DGP findings predicted this (flood_zone RR 190×,
elevation ≈330×, distance_to_river 36× for flood; region/urban_density inside
the premium formula). The gap widening to −0.0131 is consistent with the block
being *more* informative on the 2025 population than on the 2024 holdout —
same direction as the D-025 shift, larger scale.

Guardrail added: `align` now raises on any requested feature absent from the
frame (was: silent drop), and `tests/test_guardrails.py` asserts every listed
feature exists in the built frame.

### D-031 — denied-claim history (exp-011, local only, 2026-10-20 19:0x)

Repudiated claims carry zero cost (D-004) but the refusal event is
claim-occurrence information. `denial_features` adds `cust_denied_all`,
`pol_denied_all`, `days_since_denied`, strictly lagged (`loss < ps`) under the
same leakage rule as the experience block (D-014 / AGENTS.md rule 6) — own-period
denials are invisible to their own row, asserted in `tests/test_features.py`.

| | exp-010 | exp-011 | |
|---|---|---|---|
| local D² | **0.1342** | 0.1339 | −0.0003 |

Same seed, same truth, single change: inside the D-025 tie band — no measurable
gain. **Not submitted.** Human decided to hold the last quota slot (9 of 10 used)
for a change outside the band rather than spend it confirming a local near-tie
(D-025/D-029: local 4th-decimal movement does not rank on the leaderboard).
The block remains implemented and tested but **opt-in** (`denied` feature group);
promote it only with evidence outside the tie band.

**Leaderboard at time of writing:** top 0.12996 · our best **0.12113** (exp-010) ·
rank 20/22 · 9 of 10 slots used today (**1 remaining, deliberately unspent**).

### D-033 / D-034 — actuarial review fixes (2026-10-21, code only, not submitted)

First pass of the `reviewer` subagent against the `actuarial-methods`,
`actuarial-target` and `data-facts` skills. Two label/level defects and two
missing guardrails. **No leaderboard slot spent** — these are correctness fixes
to the labelling and scoring path, not a model change, so nothing was submitted.

**D-033 — LDF age mis-key (fixed).** The frozen table applied the age-0 factor
to 2023 (which is age 1 at the 2024 valuation). 2023 labels were overstated
buildings **+8.9%**, contents **+4.0%**; 2021/2022 mildly overstated. `LDF` is
now derived from the documented age-to-age factors via `age_to_ultimate()`, and
`scripts/make_baseline.py` imports it instead of keeping a second copy — the
duplication is what let the drift survive.

Observed vs `data_findings.md` §5 after the fix (row_target, all lines):

| AY | motor | buildings | contents |
|---|---|---|---|
| 2021 | 105.9M | 54.1M | 14.9M |
| 2022 | 116.7M | 61.5M | 16.0M |
| 2023 | 117.3M | 64.8M | 17.5M |
| 2024 | 90.7M | 67.4M | 17.5M |
| 2024 → dev | 114.2M | 73.6M | 18.3M |

2023 buildings now develops 64.8 → 65.0 (was 64.8 → 70.8).

**D-034 — forward factor was premium-anchor-only (fixed, `proposed`).**
`FORWARD_FACTOR` was exactly the premium/exposure 2025-vs-2024 column — the 30%
leg of D-009's blend — hard-coded as fact while D-009 is still `proposed`.
`forward_factors(scenario)` now implements the 70/30 blend and brackets it:

| scenario | motor | buildings | contents |
|---|---|---|---|
| low | 1.0270 | 1.0530 | 1.0354 |
| central | 1.0350 | 1.0556 | 1.0410 |
| high | 1.0385 | 1.0567 | 1.0540 |

`--trend` selects it and writes the choice into the run manifest. **Level is
`proposed` and unratified** — see D-034. Note the old sensitivity row above
("D² declines beyond 1.05") puts buildings central 1.0556 in the soft zone;
worth a local check before the next submission.

**Guardrails added.** `experience_features` (D-014, the strongest feature block)
had **no** leakage test; `loss < ps` is now pinned at both the customer-level and
policy-level filters. Orphan audit (D-002) now writes `outputs/dropped_claims.csv`
and asserts the documented **414** — note 4 of them are repudiated, so an audit
over the post-filter claims frame reports 410 and drifts from the findings doc.
The join is `(policy_id, line)`: `policy_id` is not globally unique, and a
policy_id-only merge manufactures orphans.

Mutation-checked: flipping either lag filter to `<=` fails `tests/test_features.py`;
re-introducing the age-0 factor on 2023 fails `tests/test_development.py`. 39 tests
green (was 25).

**Still open from the review:** baseline submissions omit the forward factor
(model is trended, baselines are not — mixed basis); `build_development.py`'s
guardrail compares 0→1 factors against age-0→ultimate so it can never catch a
table drift; `renewal_decision` deny-list is asserted at source-text level, not
on the aligned model matrix; rolling-origin 2021-22→2023 fold (D-006) unimplemented.

### D-035..D-038 — review items 2-4 + baseline basis (2026-10-21)

Continuation of the actuarial review. Four items evaluated with measurement
before touching anything. No submission; quota unchanged (1 slot held).

**Item 2 — the triangle bug (D-035).** Reported as a guardrail comparing the
wrong quantities. The defect was upstream: `build_cum` did
`unstack(fill_value=0)` + `cumsum`, writing the 2024 age-0 amount across every
later age. Accident year 2024 has reached **age 0 only** at the 31 Dec 2024
valuation; those cells are unreached, not zero. Volume-weighting then averaged
in a phantom 1.0 ratio from each immature AY.

| | buggy | corrected | documented |
|---|---|---|---|
| motor f(0→1) | 1.166 | **1.221** | 1.223 |
| motor age-0→ult | 1.183 | **1.257** | 1.26 |

**This corrects the note in the measurements table above** ("LOB re-derivation
1.18/1.06/1.03 vs documented 1.22/1.09/1.04 — different weighting window").
That was a misdiagnosis: it was not a weighting-window difference, it was the
diagonal bug. `data_findings.md` §5 was right all along and needs no change.

Also dropped 2020 origin rows (pre-book orphans) from the derivation — 2 claims
of contents/flood swung f(1→2) to ~10x on their own. The guardrail now compares
age-to-age against age-to-age and age-0→ult against the "to ultimate" column;
previously it compared 0→1 factors to an age-0→ult quantity and so could never
have caught the D-033 table bug either.

`PERIL_LDF` re-frozen. Long-tail perils were the most under-developed, which is
the direction the tail evidence predicts (bodily_injury 10.2% of claims report
>365 days):

| peril | was | now |
|---|---|---|
| motor bodily_injury | 1.535 | **1.708** |
| buildings subsidence | 1.306 | **1.464** |
| motor glass_hail | 1.026 | 1.037 |

Blast radius was contained — D-028 normalises to the accepted D-005 LOB level,
so the level never moved and `--development peril` was the only consumer.

Re-measured under D-028's own protocol (skill-vs-baseline gap, not raw D²,
because a truth redefinition moves D² by itself and earns nothing):

| | model D² | baseline D² | gap |
|---|---|---|---|
| exp-012 peril (corrected) | 0.1280 | 0.0769 | 0.0511 |
| exp-013 lob | 0.1312 | 0.0791 | 0.0521 |

Gap −0.0010, inside the D-025 band. **D-028 remains a tie**, now leaning
slightly against per-peril. Still opt-in, still unpromoted.

**Item 1 — baseline mixed basis (D-036).** Measured first: a uniform level miss
of the cost-weighted forward factor (1.0429) moves D² by only **+0.00066** on
the real holdout, against the 0.002 D-025 band. So no recorded conclusion is
contaminated — the "+0.0393 is risk differentiation" claim survives, with at
most 0.0007 of it being level. Fixed for attributability: `make_baseline.py`
now takes `--trend` and applies the same `forward_factors()` as the model, and
the filename records the scenario.

**Item 3 — deny-list (D-037).** The gap was narrower than reported:
`feature_cols` already raised at runtime. The genuine hole was that the test
grepped source text and its first assertion was `assert ... or True` (dead).
`align` now raises on any deny-listed column in the **returned** matrix, and the
tests assert on `align` output and on every feature group. Verified by mutation.

**Item 4 — D-006 second fold (D-038).** `--fold {2024,2023}` implemented.

| fold | train AY | holdout | model D² | baseline D² | gap |
|---|---|---|---|---|---|
| 2024 | ≤2023 | 2024 | 0.1312 | 0.0791 | 0.0521 |
| 2023 | ≤2022 | 2023 | 0.1385 | 0.0878 | 0.0507 |

Gaps agree to 0.0014 and the top features are identical (occupation, region,
written_premium, exposure). **Stability evidence for the write-up**, not a
second tuning opportunity (D-025). 2024 stays primary (D-018).

**Tests: 43 green (was 25 at the start of the review).** Mutation-checked on
the triangle factors, both lag filters, the LDF age key, and the `align`
deny-list guard.

**Still open:** D-009 is `proposed` and needs human sign-off before the next
submission; the buildings central factor 1.0556 sits just past the point where
the earlier sensitivity row found D² starting to decline.

### exp-015 — corrected labels, premium-only level (2026-10-21 02:24)

| # | timestamp | tag | message | change | hypothesis | local D² | LB score | Δ | attribution | sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 2026-10-21 02:24 | exp-015 | exp-010 stack + corrected 2023 labels (D-033/D-035); premium-only level (D-036) | **one primary change: the label correction** | correcting the 2023 over-development should improve calibration | 0.1332 | **0.11955** | −0.00158 | clean | `155f5cc7198f70f8` |

Config: exp-010's exact stack (`base,premium,experience,weather,flags,relativity,address,customer`, tweedie, seed 0, `development=lob`, fold 2024). Level deliberately held at **premium-only** so D-009 stays out of the comparison — this slot isolates the label change and nothing else. 1 of 10 slots used today.

**Result: worse by 0.00158** (exp-010 0.12113 → exp-015 0.11955). Just inside the D-025 ~0.002 band, so taken alone it is not conclusive — but the direction is consistent with the level arithmetic, and this is the first leaderboard evidence bearing directly on the open D-009 level question.

Local D² moved only −0.0010 (0.1342 → 0.1332) — a local tie that the leaderboard resolved as a real drop. Another instance of D-025.

**Why a correct fix can score worse — the important finding.** The D-033/D-035 correction is right: it reproduces `data_findings.md` §5 exactly (motor f(0→1) 1.221 vs documented 1.223). But it *removes* level from the training labels:

| | old (buggy) | new (correct) | level removed |
|---|---|---|---|
| buildings 2023 | ×1.092 | ×1.003 | **−8.2%** |
| contents 2023 | ×1.044 | ×1.004 | **−3.8%** |
| motor 2023 | ×1.030 | ×1.030 | 0 |

The bug was applying the age-0 factor to an accident year sitting at age 1. That was wrong — but it was accidentally supplying a loading the triangle method **explicitly does not include**. From `data_findings.md` §5: the age-to-age factors are "*lower bounds — they capture late reporting but not IBNER on open claims*", and D-005 states the target needs "*…plus IBNER on open claims*". The over-development was masking part of a documented gap.

Supporting evidence that the gap is real and material: open-claim share is **20.4% for 2024** against 3.2% for 2023, and 2024 open claims carry a mean case of ~7.7k versus 21–27k for mature open claims. The immature year is exactly where the bug was over-developing, and exactly where IBNER lives.

The measured sensitivity on the 2024 holdout maps −0.00158 D² to roughly a **−5% level miss** — consistent with removing 8.2%/3.8% of label level on ~45% of cost without replacing it with an IBNER loading.

**Conclusion for the write-up, and the state of D-009.** The chain-ladder factors are now correct *as chain-ladder factors*. What they are not is ultimate factors — the IBNER-on-open-claims component has never been estimated. That is the identified gap, it is documented in the findings as a lower bound, and it is the C-1 candidate (Bornhuetter-Ferguson / Cape Cod labels) already queued in `docs/references.md`. Per AGENTS.md rule 1 it must be **measured, not invented** — no ad-hoc loading is being added here.

Practical read for the next slot: the old table's accidental IBNER loading plus the premium-only level happened to net out close to right for 2025. Both corrections are independently justified; doing one without the other leaves the level short. The natural next experiment is the label correction **with** a properly estimated IBNER component (C-1), or the label correction with the D-009 central level, measured as separate slots so the two effects stay attributable.

Our best remains **exp-010 at 0.12113**. exp-015 is retained as a negative result — it is the evidence that pins the IBNER gap.

### CORRECTION to the exp-015 entry above (2026-10-21) — see D-040

The measured result stands: exp-015 **0.11955** vs exp-010 **0.12113**
(−0.00158), clean single-change attribution, sha256 `155f5cc7198f70f8`. The
**interpretation** that followed it is withdrawn in full. Specifically:

- **The "−5% level miss" is wrong in sign, size and method.** exp-015's
  submission is **+0.93% higher** than exp-010's in total predicted cost
  (motor +1.04%, buildings +1.01%, contents −0.03%; joined the two submitted
  CSVs, 626,639 rows). It rose, not fell. The −5% figure came from
  interpolating a sensitivity curve measured on the **constant_rate baseline**;
  on the model that curve is 3–4× flatter (−5% ⇒ only **−0.00044**, and −0.00158
  would need ~−10% uniform). Method, sign and size all wrong. **Withdrawn.**
- **"Removing the bug leaves the training level short"** — the level channel is
  cancelled in this pipeline: `lob_recalibrate` fits per-LOB scales on the
  holdout and carries them to the test set, so per-LOB uniform label-level
  changes cannot reach the submission level. Aggregate training `y_ult` moved
  only ~−1.3%, and the label change is one accident year of three. **Withdrawn.**
- **"A loading the triangle method explicitly does not include"** — incorrect
  for incurred chain ladder. Link ratios embed historical case strengthening;
  CL provides no *separate* IBNER provision and is a documented lower bound, but
  "explicitly excludes" overstates it. **Corrected** to the weaker, defensible
  claim.
- **The IBNER evidence was mislocated.** 20.4% open share and 7.7k vs 21–27k
  case means describe **AY2024** — the holdout and level anchor, not training
  (train ≤ 2023) — whose LDF is unchanged to 3 dp by the correction. Those
  statistics cannot support a claim about the 2023 labels. **Re-attributed** to
  the 2024 anchor, where D-005 actually puts "plus IBNER on open claims".
- **"Real and material, not noise"** — over-read one draw **inside** the D-025
  ~0.002 flip band. Local corroboration is only −0.0009 recalibrated and
  −0.0004 on the skill-gap protocol. **Demoted.** D-039's `accepted` status for
  this interpretation was premature.

**What survives:** the label correction is correct as chain-ladder factors
(reproduces `data_findings.md` §5 exactly); exp-015 is a clean negative result;
and the correction caused a large **refit reshuffle** rather than a level move —
row-level Spearman 0.977–0.993 but sd of log-ratio **0.113**, buildings
row-ratios spanning roughly 0.38×–2.41×. The model reorganised how it allocates
cost across rows and the realized 2025 truth preferred the old allocation.

**Metric defect found and fixed (D-040).** Local D² was computed after an
in-sample per-LOB oracle recalibration, worth **+0.0020 (exp-010) /
+0.0029 (exp-015)** — the size of the whole tie band — and structurally blind to
LOB-level miss. `run_experiment.py` now reports `d2_model` (train-fitted scales,
out-of-sample — use this for comparison) and `d2_model_oracle` (holdout-fitted;
correct for the submission path, not a generalisation estimate) plus
`d2_oracle_inflation`. Historical manifest `d2_model` values are oracle numbers
and are **not** comparable to the new ones. The D-025 "local−LB gap stable at
−0.007 to −0.010" rule is stale: measured is −0.0130 / −0.0137.

**Level-removed table, restated** (the original omitted 2021/22): buildings 2023
×1.092→×1.003 (−8.15%) and contents 2023 ×1.044→×1.004 (−3.83%) materially;
2021/2022 all LOBs −0.2…−0.4% marginally; 2024 unchanged to 3 dp (drifts
+0.015%…+0.063% from rounding, per review NIT).

Sensitivity curve underlying the above is recorded here for reproducibility
(AGENTS.md rule 1) — sample `outputs/metrics/exp-007-s0_valpred.parquet`,
626,190 rows, `d2_tweedie_score(y, max(s·pred, 1e-3), power=1.5)` minus the
s=1.0 value, Σpred/Σy = 1.0000:

| s | 0.90 | 0.92 | 0.94 | 0.96 | 0.98 | 1.00 | 1.02 | 1.04 | 1.06 | 1.08 |
|---|---|---|---|---|---|---|---|---|---|---|
| ΔD² | −0.00169 | −0.00108 | −0.00062 | −0.00029 | −0.00009 | 0 | −0.00002 | −0.00015 | −0.00038 | −0.00070 |

Our best remains **exp-010 at 0.12113**. exp-015 stays logged as a valuable
negative result — it is the run that exposed both the refit-reshuffle effect and
the oracle-recalibration defect.

### Overnight session 2026-10-21 — hypothesis testing (D-041). 5 slots used, 5 reserved.

| # | timestamp | tag | message | change | local D² (raw) | LB score | Δ vs control | attribution | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 11 | 12:54 | exp-016 | exp-015 config, seed 1 | seed only | 0.1294 | **0.11942** | −0.00013 vs exp-015 | clean | `0d4d1f9992c4a36b` |
| 12 | 13:35 | exp-017 | H1 level probe: exp-016 ×1.03 uniform | level only | 0.1294 | **0.11949** | +0.00007 vs exp-016 | clean | `f624564534d690a1` |
| 13 | 14:28 | exp-019 | exp-010 config, seed 1, PRE-D-033 labels | seed only (old labels) | 0.1319 | **0.12024** | −0.00089 vs exp-010 | clean | `271770ec27ef9349` |
| 14 | 14:49 | exp-020 | exp-010 config, seed 2, PRE-D-033 labels | seed only (old labels) | 0.1327 | **0.12173** | +0.00060 vs exp-010 | clean | `e28fcebeaac86844` |

**The headline correction to D-040.** D-040 called the label penalty "12× the
seed band". That was an artefact of measuring the band on one side only. Full
design:

| | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| old labels | 0.12113 | 0.12024 | 0.12173 | **0.12103** | **0.00075** |
| new labels | 0.11955 | 0.11942 | — | **0.11949** | **0.00009** |

- Label effect **+0.00155** (old better) vs pooled seed sd 0.00061 ⇒ **~2.1×**,
  not 12×. Direction consistent across both pairings (−0.00158, −0.00082), so
  probably real, but modest.
- **Best single run is exp-020 at 0.12173**, not exp-010. Our previous "best"
  was a favourable draw from a band 0.0015 wide. Honest best estimate is the
  **mean 0.12103**.
- Unexplained and worth keeping: corrected labels fit **8× more stably**
  (seed sd 0.00009 vs 0.00075). Cause unknown.

**H1 level — refuted.** exp-017 = exp-016 ×1.03 → +0.00007, half a seed band.
Uniform level is worth nothing; the IBNER-as-level story dies with it.

**H2 noise — refuted as an explanation**, but see the band correction above.

**H3 recency weighting — refuted.** `--recency-lambda` sample weights
`exp(lambda·(AY−train_max))`, swept on fold 2023 (train 2021–22 → holdout 2023):

| lambda | 0.00 | 0.15 | 0.30 | 0.50 |
|---|---|---|---|---|
| honest local D² | **0.1404** | 0.1348 | 0.1379 | 0.1362 |

Uniform wins at every tilt; early stopping collapses 163 → ~88 iterations.

**H4 tail flattening — not distinguishable from noise. NOT SUBMITTED.**
`exp-018.csv` was built (gamma power-stretch, sum-preserving) then discarded
locally after measurement showed old-vs-new spread diff 0.0074 < within-labels
seed spread 0.0207. Keeping this in the log as a negative result so nobody
rebuilds it.

One diagnostic that does survive: label change moves ranks **less** than a seed
change (Spearman 0.993 vs 0.990) yet costs more score — so it is **not** a
ranking effect.

**Control-run method.** exp-019/exp-020 were produced by a `/tmp` runner that
patches `src.dataset.LDF` back to the pre-D-033 table before calling
`run_experiment.main()`. The repo keeps the correct table; nothing in `src/`
was reverted. Reproducible: see `/tmp/opencode/run_old_ldf.py` pattern — patch
the module global, then call main().

**State at handover.** 5 slots remain for the human. Our best *single* run is
exp-020 at 0.12173 (old, wrong labels). Our best *defensible* model is the
corrected-labels config at mean 0.11949. D-009 still `proposed`. The open
question is why an actuarially wrong label table predicts 2025 slightly better —
no mechanism established after four tests.

### D-045 — exp-021, final refit on all training years (2026-10-21 19:54)

| # | timestamp | tag | message | change | local D² (honest) | LB score | Δ | attribution | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 19:54 | exp-021 | D-045 final refit on ALL training years incl. 2024 | **refit scope only** | 0.1331 | **0.12150** | **+0.00195** vs exp-015 | clean | `a68c5a47d9a48eaa` |

**Human-identified defect.** The submission model was fitted on AY ≤ 2023
only. 2024 rows (626,190 = 26.0% of the training book) were used for early
stopping and contributed zero gradient updates — the most recent year of
experience never entered the fit, and it is the year closest to the 2025
scoring population.

Diagnostic, holdout fixed at 2024, varying only fit scope (honest D²):
train 2021–2022 = 0.1309 · train 2021–2023 = 0.1331 ⇒ **+0.0022** for one
additional recent year. Outside the D-025 ~0.002 band and larger than every
feature effect measured today (all ≤ 0.0007).

**On 5-fold CV:** random 5-fold is wrong for temporally structured data scored
on a future year — it leaks. Rolling-origin is the right design (D-006) and
was already in place. What was missing was the *final refit after selection*,
not more folds.

**Result.** exp-021 (corrected labels, refit on 2021–2024) = **0.12150**.

| | LB |
|---|---|
| exp-015 corrected labels, no refit | 0.11955 |
| **exp-021 corrected labels, refit** | **0.12150** |
| exp-020 mis-keyed labels, best of 3 | 0.12173 |
| mis-keyed labels, mean of 3 | 0.12103 |

**This defuses the D-041 label puzzle.** The refit gain (+0.00195) exceeds the
label penalty (−0.00158). The corrected-labels model is now 0.00023 below the
best mis-keyed run and 0.00047 above the mis-keyed mean — both inside the
0.00075 seed band.

Careful wording, given D-041 over-claimed twice: **one seed against a
three-seed band**. This establishes **parity**, not superiority. But it means we
can present the actuarially correct label table (D-044) at no material score
cost, which removes the tension between correctness and score that D-041 left
open.

Attribution is clean: exp-021 vs exp-015 share seed, features, labels, trend
and fold; only fit scope differs. The LOB recalibration anchor also moved to
all-data scales as part of the same change, but D-041 measured level as flat
(+0.00007 at uniform +3%), so any movement is the refit, not the level.

Random 5-fold CV was **not** used and should not be: it leaks future into
past. Rolling-origin remains the design.

**State:** 6 slots used today, 4 remaining. Best single run exp-020 at 0.12173
(mis-keyed labels). **Best defensible run exp-021 at 0.12150 (corrected
labels).**

### exp-022 — second seed, parity resolved (2026-10-21 23:20)

| # | timestamp | tag | message | change | local D² (honest) | LB score | Δ | attribution | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 23:20 | exp-022 | exp-021 config, seed 1 | **seed only** | 0.1328 | **0.12249** | **+0.00099** vs exp-021 | clean | `510e0e367a6a10e3` |

D-045 said the corrected label table was "at parity" with the mis-keyed one but
rested on n=1 against a 3-seed band. This is that second seed.

| config | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| corrected + refit | 0.12150 | **0.12249** | — | **0.12200** | 0.00070 |
| mis-keyed, no refit | 0.12113 | 0.12024 | 0.12173 | 0.12103 | 0.00075 |

**Parity resolves in our favour.** Corrected mean beats mis-keyed mean by
+0.00097 (~1.3 pooled sd); corrected best beats mis-keyed best by +0.00076.
**exp-022 at 0.12249 is our best submission to date** and it is on the
actuarially correct model — D-044's presentation decision is now correct *and*
better, not a trade.

n=2 vs n=3, so strong directional evidence rather than proof. But the direction
never reversed in either pairing, and the earlier "corrected labels are worse"
reading came from comparing before the D-045 refit was applied.

**Withdraws part of D-041.** The "corrected labels fit 8× more stably" note was
measured on runs *without* the final refit. After D-045 the corrected band is
0.00070 against the mis-keyed 0.00075 — no meaningful difference. The
stability was an artefact of dropping 26% of the training data, not a property
of the label table.

**State:** 7 slots used, 3 remaining. Best single run and best defensible model
are now the same run (exp-022, 0.12249).

### exp-027 — occupation removed (D-056); NEW BEST (2026-10-22 16:01)

| # | timestamp | tag | change | local D² (honest) | LB | Δ vs exp-022 | sha256 |
|---|---|---|---|---|---|---|---|
| 17 | 16:01 | exp-027 | **occupation removed** | 0.1327 | **0.12436** | **+0.00187** | `9539335aba0d0bce` |

**New best leaderboard result**, overtaking blend-2s (0.12260).

Selected on **calibration**, not local D² — which called it neutral (mean
+0.0003, folds disagreeing). Calibration improved on both folds (ratio
1.143→1.120 / 1.185→1.155; slope 1.189→1.158 / 1.224→1.149), and D-054 had
established that our leaderboard failures are calibration failures rather than
ranking failures. It transferred. **First time today a non-local-D²
justification has worked.**

Attribution caveat: exp-027 differs from exp-022 by occupation removal *and*
the D-051 data corrections. Those measured neutral locally (fixA 0.1327 vs
0.1331), so most of the +0.0019 is likely occupation — but it is not a clean
single-change comparison.

Also reverted the D-053 tuned hyperparameters as the default (leaves 16, lr
0.02, which failed at 0.11168) back to the scoring configuration (leaves 64,
lr 0.04).

**Standings:** exp-027 **0.12436** (best) · blend-2s 0.12260 · exp-022 0.12249 ·
exp-021 0.12150. 5 slots used, 5 remaining. Deadline 2026-10-23 16:00.

### exp-028 — flood_zone only (2026-10-22 20:32)

| # | tag | change | local D² (honest) | LB | Δ vs exp-027 | sha256 |
|---|---|---|---|---|---|---|
| 18 | exp-028 | drop river distance + elevation, keep flood_zone | 0.1318 | **0.12255** | **−0.00181** | `be48df10ac4e5e46` |

Human proposal: flood_zone is the real signal and the other two are proxies for
it. **Refuted.** Keeping all three is better by 0.00181; exp-027 stays best.

The redundancy test had predicted this — elevation carries 1.2–1.8× residual
spread *within* flood zones, and flood_zone is missing on 41,827 rows carrying a
2.0× book rate where elevation is present. The collinearity story (removing any
one of the three helps) did not survive contact with the leaderboard.

Local D² agreed in sign and roughly in size (−0.0009 local, −0.0018 LB), so
local was reliable here. Its failures have been specific — level/calibration
effects and the exp-024 double-count — not random.

**Standings:** exp-027 **0.12436** (best) · exp-028 0.12255 · blend-2s 0.12260 ·
exp-022 0.12249. 6 slots used, 4 remaining. Deadline 2026-10-23 16:00.

### Seeds and blends of the exp-027 family (2026-10-22 22:21–22:25) — quota exhausted

| # | tag | change | LB | Δ vs exp-027 |
|---|---|---|---|---|
| 19 | exp-029 | seed 0 | 0.12179 | −0.00257 |
| 20 | exp-030 | seed 2 | 0.12247 | −0.00189 |
| 21 | blend-027030 | 2-seed (027+030) | 0.12389 | −0.00047 |
| 22 | exp-03seed | 3-seed (027+029+030) | 0.12344 | −0.00092 |

**Family: 0.12179 / 0.12247 / 0.12436 → mean 0.12287, sd 0.00133.** exp-027 is
~1.1 sd above the mean — a good draw. Report the mean as the configuration's
true level, not the best draw.

**The occupation removal survives like-for-like:** old family mean 0.12161
(exp-021/022/023) vs new family mean **0.12287** (exp-027/029/030) =
**+0.0018**, consistent with D-057's +0.00187.

**Blending diluted the standout.** Both blends landed below exp-027. Compare the
earlier blend-2s (0.12260, above both members 0.12150/0.12249) where variance
reduction helped. Rule: ensembling helps when members are comparably good and
hurts when one is a standout.

**Quota: 10 of 10 used on 2026-10-22.** 10 fresh tomorrow, deadline 2026-10-23
16:00. Best remains exp-027 (0.12436); best defensible estimate is the family
mean 0.12287. Model-work queue exhausted — see D-059.

### D-063 / D-064 / D-065 — `--final-fit` rate & hyperparameter bugfixes, Per-LOB rate models, and XGBoost + LGBM multi-family ensemble (2026-10-23)

Audit of `scripts/run_experiment.py` identified two critical defects in the
`--final-fit` block (lines 460–472) that had caused `exp-024` (`--target rate`)
and `exp-025` (`D-053` shallow-tree hyperparameters) to collapse on the
leaderboard (`0.06019` and `0.11168`):
1. `final_model.fit(X_all, train["y_ult"])` ignored `args.target == "rate"`,
   fitting total cost instead of annualised rate `y_ult / exposure` and omitting
   exposure weights, after which `to_cost` multiplied by `exposure` a second
   time (`cost ∝ exposure²`).
2. `final_model` hardcoded `learning_rate=0.04, num_leaves=64` instead of
   inheriting `**params`, so `exp-025` trained `num_leaves=64, lr=0.04` for
   `n_fix = 590` trees (~6× over-budget).

Fixing both defects, excluding `earned_premium` when `target == "rate"`,
fitting shallow per-LOB rate models (`num_leaves=6, lambda_l2=5.0`), and
blending across model families (Per-LOB LightGBM + Pooled LightGBM + Per-LOB
XGBoost `max_depth=3`) produced our largest leaderboard gain of the competition:

| # | tag | config | Fold 2024 OOF D² | Public LB | Private LB | Δ vs exp-027 (Public / Private) |
|---|---|---|---:|---:|---:|---:|
| 22 | exp-027 | prior best (pooled cost target, `lv=64`) | 0.13223 | 0.12436 | 0.12984 | — |
| 29 | exp-033 | Per-LOB rate LGBM (`lv=6, lr=0.05`) + `--final-fit` fix | 0.14002 | 0.12681 | 0.13531 | +0.00245 / +0.00547 |
| 30 | exp-034 | 70% exp-033 (Per-LOB LGBM) + 30% exp-032 (Pooled LGBM `lv=16`) | 0.14033 | 0.12707 | 0.13536 | +0.00271 / +0.00552 |
| 34 | exp-035 | Per-LOB rate XGBoost (`max_depth=3`, `no_rel`) solo | 0.14014 | 0.12700 | 0.13509 | +0.00264 / +0.00525 |
| **31** | **exp-036** | **40% exp-035 (XGB) + 42% exp-033 (Per-LOB LGBM) + 18% exp-032 (Pooled LGBM)** | **0.14076** | **0.12753** | **0.13570** | **+0.00317 / +0.00586** |
| 32 | exp-037 | Per-LOB rate LGBM (`lv=6`) + `--norm-rel` | 0.14054 | 0.12587 | 0.13502 | +0.00151 / +0.00518 |
| 33 | exp-038 | 35% exp-037 + 35% exp-035 + 15% exp-033 + 15% exp-032 | 0.14095 | 0.12721 | 0.13563 | +0.00285 / +0.00579 |

**New best:** `exp-036` at **0.12753 Public / 0.13570 Private** (+0.00317 Public
and +0.00586 Private over `exp-027`).

