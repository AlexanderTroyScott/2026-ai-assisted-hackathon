# 2026 AI-Assisted Actuarial Hackathon — Pricing Across Markets

**Competition:** `iaa-ai-assisted-hackathon-2026-part-1` · **Deadline:** 2026-10-23 16:00
**Metric:** Tweedie D², power = 1.5 · **Budget:** 10 submissions/day · **Teams:** 22

Predict the **expected ultimate claim cost** per 2025 policy-period row for three
lines — private motor, home buildings, home contents. Net of policy excess and
recoveries, excluding declined claims, by loss date in the row's period.

Judged on four axes: pricing performance & adaptation (40%), effective use of AI
(25%), trust/validation/guardrails (20%), communication & documentation (15%).

---

## Where things live

| Path | Purpose |
|---|---|
| `docs/data_findings.md` | **Authoritative empirical findings.** Read before modelling |
| `docs/decisions.md` | Decision log (D-001 …). Judgement calls flagged, nothing unverified |
| `docs/ai_workflow_log.md` | AI interaction evidence trail for the judging panel |
| `docs/submission_protocol.md` | Format rules, metric notes, submission cadence |
| `docs/experiment_log.md` | Every submission + local experiment, with hypotheses |
| `AGENTS.md` | Working conventions for AI agents on this repo |
| `src/` | Reusable Python modules (pipeline code) |
| `scripts/` | Baseline generation + Kaggle submission tooling |
| `tests/` | Automated tests / guardrails |
| `outputs/` | Submission CSVs, charts, manifests |
| `data_raw/` | Parquet files as received (**gitignored** — see `data_raw/MANIFEST.md`) |
| `.opencode/` | Skills, agents and commands for context-efficient work |

---

## The pipeline

Every stage feeds the prediction CSV, the write-up and the video.

```
 1. load & validate     src/io.py           schema, ID dtypes, manifest check
 2. data quality        src/data_quality.py exposure recompute, category canonicalisation
 3. target              src/target.py       claim → row attribution, y_observed
 4. development         src/development.py  triangles, LDFs → y_ultimate   (IBNR/IBNER)
 5. inflation           src/inflation.py    per-peril trend, 2025 level factor
 6. features            src/features.py     experience lags, premium, risk factors
 7. models              src/models.py       GLM (p=1.5) + LightGBM, per LOB
 8. validate            src/validate.py     calibration, lift, residuals, model cards
 9. submit              scripts/kaggle_submit.py
```

### Stage notes (the short version — details in `docs/data_findings.md`)

- **Exposure is corrupt.** Three conventions in one column, plus 2,319 rows recorded
  in *days*. Always recompute `(period_end − period_start + 1) / 365`.
- **Target:** `max(0, incurred_to_date − recoveries)` matched on `policy_id` **and**
  `period_start ≤ loss_date ≤ period_end`. Every claim matches exactly one row.
  414 orphans dropped (D-002). Repudiated filter is a numerical no-op (D-004).
- **2024 labels are understated** — IBNR + immature case estimates. Develop labels
  (motor ≈ ×1.26, buildings ×1.09, contents ×1.04), never develop predictions
  (D-005). Validate against developed truth (D-006).
- **Inflation is per-peril**, +2.9%/yr to +9.4%/yr, extrapolated to 2025 as a
  scenario. Long-tail perils trended on developed ultimates only (D-007).
  Trees get a forward factor `index(2025)/index(2024)`, not full de-trending (D-008).
- **Strongest feature block:** lagged claim experience from `claims`
  (freq 0.083 → 0.486, LR 0.47 → 2.37). Built strictly lagged (D-014).
- **Benchmark to beat:** `written_premium × LOB LR` — premium already ranks risk 4x
  and LR is flat at 0.71–0.81 (D-015).

---

## Models

Per line of business:

1. **Tweedie GLM** (p=1.5, log link, exposure weights) — the explainability anchor.
2. **LightGBM** (`objective='tweedie'`, `tweedie_variance_power=1.5`) — accuracy engine.
3. Optional frequency/severity split as a credibility cross-check.

Ensemble where it helps; always recalibrate per LOB so `Σpred ≈` expected total cost.

### Validation

Rolling-origin time splits (train 2021–22 → val 2023; train 2021–23 → val 2024),
scored against **developed** truth, reported raw-vs-developed side by side.
Diagnostics: actual-vs-predicted by decile and by premium decile, lift/gains,
residuals by year/region/key factors, monotonicity checks, feature-importance
stability, model card per LOB. Never a random split.

---

## Submissions — routine, attributable, logged

The leaderboard is our external test set. We submit after every meaningful change
so feature-level gains are visible (user-directed, D-013).

```
python scripts/make_baseline.py --kind constant_rate
python scripts/kaggle_submit.py --input outputs/submissions/<name>.csv \
    --message "D-014 experience features" --tag exp-014
```

Rules (full detail in `docs/submission_protocol.md`):

- One primary change per submission; confounded changes are marked as such.
- Always validate first: 626,639 rows, exact row_id set, all predictions ≥ 1e-3.
- Log local D² + leaderboard score/rank into `docs/experiment_log.md` immediately.
- Baselines (constant rate, premium×LR) go on the board before any model.
- Reserve ~2 slots/day for re-submission of the current best after refactors.

### Leaderboard reference (2026-10-20)

| | score |
|---|---|
| Top | 0.12996 |
| 20th / 22 | 0.11710 |

If local CV D² lands near 0.4 while the board tops out at 0.13, our baseline
definition is wrong — fix before trusting local numbers.

---

## AI-assisted workflow

AI handles speed and boilerplate: profiling, scaffolding, diagnostics code, first
drafts. Actuaries handle judgment: target definition, development factors, trend
assumptions, which AI outputs are accepted. Every AI interaction that reached a
decision is logged in `docs/ai_workflow_log.md`; every AI-produced number in
`docs/` is either recomputed or marked unverified.

Context is managed with `.opencode/` skills and agents so heavy reference material
is loaded on demand rather than carried in every prompt:

| Skill | Loads when |
|---|---|
| `data-facts` | Modelling, EDA, feature work — the empirical findings |
| `kaggle-submit` | Any submission or leaderboard check |
| `actuarial-target` | Building labels, development or trend factors |

| Agent | Role |
|---|---|
| `data-audit` | Profiling, data-quality checks, schema drift (read-only) |
| `model-audit` | Diagnostics, metric verification, leakage checks |
| `experiment-runner` | Baseline → validate → submit → log |

Commands: `/submit`, `/leaderboard`, `/log-experiment`.

---

## Week 2 — Expansion (framework pre-built)

New market data arrives at end of Week 1. The response is pre-scripted so we spend
the week on judgement, not plumbing:

1. **Diff** — schema, new/dropped levels, distribution shift (PSI), null patterns,
   exposure/claim-rate shift.
2. **Adapt** — unseen-category fallback (already tested on `construction_type`),
   credibility-weighted blend toward the GLM if volume is thin, recalibration.
3. **Stress assumptions** — the `written_premium`-as-proxy assumption and the
   claims-history features are the two most likely to break in a new market
   (no history for new customers).
4. **Communicate** — where the model is least reliable, what would help, what
   remains uncertain.

---

## Reproducibility

- `data_raw/` is gitignored; `data_raw/MANIFEST.md` records files, row counts and
  sha256 so the dataset can be re-verified against the Kaggle download.
- Submission CSVs are hashed on upload so any leaderboard entry is reproducible.
- Tests guard the invariants: exposure bounds, claim-row uniqueness, orphan count,
  feature deny-list, submission format.
