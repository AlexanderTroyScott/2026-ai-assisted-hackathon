---
name: actuarial-target
description: Use when building the prediction target, claim-to-row attribution, development/IBNR factors, claims inflation or trend indices, or when deciding how to validate models against ultimate cost. Covers the exact target formula, the triangle/LDF results, the inflation estimates and the labelling rules.
---

# Actuarial Target, Development & Trend

Canonical sources: `docs/data_findings.md` §2/§5/§6, `docs/decisions.md` D-002…D-009.
Use this when constructing labels or extrapolating to 2025.

## Target definition

```
y(row) = Σ  max(0, incurred_to_date − recoveries)
         over claims where policy_id matches
           AND period_start ≤ loss_date ≤ period_end
           AND status != 'repudiated'
```

- `incurred_to_date` is **already net of policy excess** — do not deduct again.
- Every claim matches exactly one row (162,736 matched; 0 multi-matched). Assert this.
- 414 orphans (loss date matches no row) → drop + write to `outputs/dropped_claims.csv`
  + assert the count in tests.
- Repudiated claims all have zero amounts → the filter is a numerical no-op. Say so
  in the write-up rather than implying it changed anything.

## Labelling pipeline (order matters)

```
y_observed  →  y_ultimate = y_observed × LDF(line, peril_mix, AY)   [D-005]
            →  apply forward factor index(2025)/index(2024) at scoring  [D-008]
```

**Develop labels, never predictions.** Predictions are already ultimate-level;
adding an IBNR loading at scoring double-counts.

## Development factors (verified)

| LOB | age-0 → ultimate | 2024 observed → developed |
|---|---|---|
| motor | ≈ ×1.26 | 90.7M → ~114M |
| buildings | ≈ ×1.09 | 67.4M → ~73.6M |
| contents | ≈ ×1.04 | 17.5M → ~18.3M |

Lower bounds — they capture late reporting, not IBNER on open claims.

Why it is needed: open-claim share 20.4% for 2024 (vs 3.2% for 2023), 2024 open mean
case 7.7k vs 21–27k mature, Dec-2024 loss counts 2,264 vs norm ≈ 4,001.

Tail location: share of claims reported >365 days after loss — motor 1.07%,
buildings 0.57%, contents 0.47%; **bodily_injury 10.2%, subsidence 3.9%**.
Concentrate per-peril factors there; short-tail perils need almost nothing.

## Trend / inflation (D-007)

Estimate on **closed claims with report lag ≤ 30 days** (final amounts, comparable
maturity). Never on the raw all-claims series — 2024's immaturity looks like deflation.

Measured `log(severity) ~ AY`: glass_hail +9.4%/yr, storm +8.2%, third_party_prop
+7.6%, flood +5.8%, theft +4.7%, escape_of_water +4.7%, own_damage +3.2%,
accidental_damage +2.9%, fire +1.1% (n.s.).

Traps:
- **Closure bias** — closed-only BI gives a fake −9.5%/yr. Trend long-tail perils on
  **developed ultimates**, not closed cohorts.
- **Mix bias** — motor's raw mean severity falls when BI's claim share drops. Use
  fixed-weight (mix-adjusted) indices: motor 100→111→112, buildings 100→108→118,
  contents 100→106→111 (2021→23→24).
- Scale shift, not shape change → a single per-peril scale factor is defensible.
- YoY is not constant (2024 cooled). Extrapolate as **low / central / high** scenarios.

Premium anchor (independent): premium/exposure index 2025 vs 2024 = motor +3.2%,
buildings +5.1%, contents +5.6%. D-009 proposes blending 70% claims-index / 30%
premium-anchor for the central factor — **status `proposed`, needs human sign-off**.

## Validation rules (D-006)

- Time-based splits only: train 2021–22 → val 2023; train 2021–23 → val 2024.
- Score against **developed** truth. Report raw-vs-developed side by side.
- Report D² pooled and per LOB; recalibrate each LOB so `Σpred ≈` expected total cost.
- Floor predictions at `1e-3` before scoring.

## Excess note

Voluntary excess is 0/250/500/1000 and compulsory is fixed (motor 250, home 350).
Because the excess is fixed in nominal terms, net-of-excess severity inflates faster
than gross. Since the target is net and we trend the net target, this is already
embedded — just don't expect it to match a published CPI.

## If asked "is there IBNR to find?"

For **2025 test rows: no** — there are no 2025 claims in the data at all
(max loss date 2024-12-29); the target is *expected ultimate*, so unreported claims
are inside the expectation. For **2021–2024 training rows: yes**, material, and it
cannot be "found" claim-by-claim — only estimated in aggregate via the triangle.
