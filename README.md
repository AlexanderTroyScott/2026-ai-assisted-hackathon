# 2026 AI-Assisted Actuarial Hackathon — Pricing Across Markets

Solution write-up for the IAA 2026 AI-Assisted Hackathon (Parts 1 and 2): predicting
expected ultimate claim cost per policy-period row across a developed market and a
newly entered emerging market. This README is the front door for the judging panel —
the detailed evidence lives in `docs/`, the executable solution in `src/`, `scripts/`
and `tests/`.

**Metric:** Tweedie D², power = 1.5 (higher is better) ·
**Target:** expected ultimate claim cost per 2025 row — total, not annualised — net of
policy excess and recoveries, including IBNR, excluding repudiated claims, allocated by
loss date within the policy period.

## Results snapshot (2026-10-28)

| Competition | Submission of record | Score | Standing |
|---|---|---|---|
| **Part 2** — emerging market (live; closes 2026-10-31 03:59 UTC) | `emerging-exp-025` | **0.16994** Public | **4th of 33 teams** — 0.00178 off 3rd (HS 0.17303, SnorlaxCode 0.17192, Actuarial NOOB 0.17172, us 0.16994, ManuelRodríguez 0.16914) |
| **Part 1** — developed market (closed 2026-10-23 16:00; final ranking on Private) | `exp-027` (best Public at the deadline) | 0.12984 Private | 41st of 54 — our best model `exp-036` (0.12753 Pub / 0.13570 Priv) was built *after* the deadline and is not ranked |

Part 2 is the live competition and the primary workstream. Part 1's scoring gate
remained open after its deadline, which made it a fully labelled sandbox for
validating Part 2 ideas — the source of several measured accept/reject verdicts below.

---

## 1. The problem

Two markets, one modelling question:

- **Part 1 (`developed/`)**: four accident years (2021–2024, 2.4M rows) of private
  motor, home buildings and home contents business; price the 2025 book (626,639 rows).
- **Part 2 (`emerging/`)**: a new market with two thin years (2023–2024, 109,296 rows,
  ~9,100 matched claims) and a 2025 test book of 209,063 rows that is **76% new
  business** — a broker portfolio takeover where most test policies have zero insurer
  claim history and start on 2025-01-02..2025-01-15.

The Part 1 data cannot be joined to the Part 2 data by policy or claim ID — **IDs
collide across the two datasets**. All transfer happens in *feature space*: developed
market models are fitted on harmonised features and evaluated on emerging rows to form
a prior. This rule is enforced in code and in the test suite.

The actuarial content: claim counts are sparse and heavy-tailed (Tweedie variance power
1.5), the 2025 book is dominated by transferred new business with no experience, and
claims valued at 31 Dec 2024 require development to ultimate (IBNR/IBNER) before the
target is constructed.

## 2. Solution architecture

### Part 2 — transfer-learned offset boosting with a measured 2025 forward index
(`src/emerging.py`, `scripts/run_emerging.py` — submission `emerging-exp-025`)

1. **Developed-market transfer prior.** Models fitted on `developed/` predict emerging
   rates in feature space; the prior blends a written-premium-weighted and an unweighted
   version (0.6 / 0.4) and is recalibrated to emerging training totals.
2. **Ultra-shallow offset layer, per LOB.** A LightGBM Tweedie model (p = 1.5) is fitted
   on emerging data with `log(prior)` as `init_score` offset, correcting the transfer
   prior for emerging-specific structure. Deliberately tiny — 35/30/20 trees (motor /
   buildings / contents), learning rate 0.02, `num_leaves` 4, `min_data_in_leaf` 1200,
   `lambda_l2` 100 — so it adapts the prior without overwriting it. The offset layer
   stands alone (blend weights sweep up to 1.0 on the offset; D-067).
3. **2025 forward index applied at scoring.** A per line × cohort (new business vs
   renewal) level/trend multiplier captures the 2025 book's shift: motor nb 1.322 /
   ren 1.103, buildings & contents nb 1.120 / ren 0.817 (exposure-weighted total ×1.05
   held at 2025 shares). Every factor is measured on the board via response-curve
   brackets (D-068/069/076/077), never assumed.
4. **Refit on all training rows, predictions averaged over 3 seeds.**

5-fold GroupKFold (by customer) D² ladder — the transfer prior is the backbone, the
shallow offset adds the emerging-specific signal:

| Model | OOF D² |
|---|---:|
| Per-LOB constant rate | +0.0804 |
| Per-LOB premium × loss ratio | +0.1097 |
| Emerging-only LightGBM (no transfer) | +0.1073 |
| Transfer prior alone (no adaptation) | +0.1386 |
| **Transfer prior + offset adaptation (final)** | **+0.1407** |

### Part 1 — multi-family rate blend
(`src/features.py`, `src/glm.py`, `scripts/run_experiment.py`; best model `exp-036`)

Rate models (claim cost per unit of exposure, with premium, experience, weather,
relativity, address and customer feature families) across three model families —
per-LOB LightGBM, pooled LightGBM, per-LOB XGBoost — blended 42 / 18 / 40
(`exp-036`, honest OOF D² 0.1408, 0.12753 Pub / 0.13570 Priv). Labels are developed to
ultimate with line-level LDFs, with open/closed split development tested (D-061).

*The entry that actually counts for Part 1 ranking* is `exp-027` (pooled cost-target
LightGBM, occupation dropped, D-056) — our best Public submission as of the deadline
(0.12436 Pub → **0.12984 Private** on the final board). The `exp-033`→`exp-036` line
of work landed in the hours after the deadline; the gate still scored it, so it is
fully measured, but it is not ranked. Deadline discipline is now part of the protocol.

## 3. What actually moved the needle

Measured on the board unless noted. Full record: `docs/experiment_log.md` (65 logged
submissions), judgement calls and evidence: `docs/decisions.md` (D-001…D-080).

**Part 2 record progression** (Public):

| Step | Submission | Score | Lever |
|---|---|---:|---|
| Base | `emerging-exp-002` | 0.16297 | transfer-prior offset pipeline |
| + level/trend response curve | `emerging-exp-019` | 0.16836 | 2025 forward index, pipeline-native (D-068/069) |
| + shrinkage-pair tuning | `emerging-exp-024` | 0.16879 | `num_leaves` 4 / `min_leaf` 1200, paired-seed confirmed (D-075) |
| **+ property cohort ridge** | **`emerging-exp-025`** | **0.16994** | new business runs hotter than renewals in property (D-076) |

- **The 2025 forward index is the largest single lever** (base → record, +0.0070
  combined with tuning): the transferred book's new-business share and the 2025 level
  shift dominate the error, and correcting for them is worth more than any model
  capacity change.
- **New business runs hotter than renewals in both motor and property** — two
  independently measured ridges (motor 1.322/1.103; property 1.120/0.817 at constant
  total). The property ridge alone is +0.00115, and its quadratic top is flat
  between the two bracket arms, so the axis is closed at the measured point.
- **Negative results are first-class findings.** Each was tested, logged and rejected
  on evidence: multi-family ensemble port (+0.00023 OOF, below the noise gate, D-070);
  Horvitz–Thompson development labels (trades Private for Public, mean Private
  −0.00021 in a 2×2 seed design, D-072); buildings/contents split (flat, D-073);
  motor per-peril decomposition (−0.00036 OOF; sparse perils can't carry standalone
  models, D-074); single-knob hyperparameter moves (flat, D-075).

## 4. Validation, guardrails and trust

- **Pre-registered adoption gate:** a change is submitted only if honest out-of-fold D²
  improves by more than +0.0003 (D-025), and effects near the gate get paired-seed
  confirmation before adoption. Bracket probes use two arms at constant total plus a
  quadratic vertex, stopping when the vertex gain falls below probe noise.
- **OOF is a faithful oracle, and that is measured, not assumed:** the exp-024 board
  gain equalled its mean OOF gain to four decimals. Board seed noise ≈ 0.0012 Public
  (2×2 seed replication, D-072) vs local OOF seed σ 0.00011 — so the score moved off
  model changes, not luck.
- **89 guardrail tests** (`tests/`, run before every commit): the no-ID-join rule,
  per-peril labels summing to each row's `y_ult`, held cohort totals, forward-index
  ridge assertions, label development arithmetic, feature and submission-format checks.
- **Reproducibility:** the record submission is regenerated end-to-end by the pipeline
  to float precision (max relative difference 8e-16) — the leaderboard number is not a
  post-hoc artefact:
  `python3 scripts/run_emerging.py --use-cache --num-leaves 4 --min-leaf 1200`.
- **Submission guardrails** (`scripts/kaggle_submit.py`, the only upload path):
  validates schema, row count (209,063), positivity and finiteness before anything is
  uploaded; refuses to submit blind when quota is unavailable; records sha256 for every
  CSV; requires explicit human confirmation for Part 2 submissions and quotes the
  authorisation in the log; one primary change per submission; submissions are issued
  **serially** (a parallel pair once produced a duplicate upload — logged honestly, and
  the rule was added to `docs/submission_protocol.md` §4).

## 5. Effective use of AI

The workflow is an AI-driven measurement loop with human authorisation gates: the agent
designs and runs experiments, computes honest OOF evidence, proposes adopt/reject
verdicts, and may only upload on explicit human confirmation (standing authorisation
for the sandbox market only). Every claim in the logs is tied to a measurement; every
judgement call is flagged as such. The full interaction trail — including negative
results, the duplicate-upload incident and its process fix — is in
`docs/ai_workflow_log.md` and the decision log.

## 6. Where things live

| Path | Purpose |
|---|---|
| `docs/actuarial_report.md` | Part 1 actuarial report (ASOP 9 documentation standard) |
| `docs/emerging_market_report.md` | Part 2 market analysis and adaptation methodology |
| `docs/data_findings.md` | Authoritative empirical findings (structural comparison, feature harmonisation) |
| `docs/decisions.md` | Decision log D-001…D-080 — judgement calls flagged, nothing unverified |
| `docs/experiment_log.md` | Measurement record — every submission, score, verdict, authorisation, sha256 |
| `docs/submission_protocol.md` | Format/metric rules, cadence policy, serial-submission rule |
| `docs/ai_workflow_log.md` | AI interaction evidence trail for the judging panel |
| `docs/references.md` | Actuarial bibliography (R-01…) and method candidates (C-1…) |
| `src/emerging.py` | Part 2: market frames, transfer priors, forward index |
| `scripts/run_emerging.py` | Part 2 end-to-end pipeline → submission CSV |
| `scripts/run_emerging_tune.py`, `run_emerging_peril.py` | D-075 tuning sweep, D-074 per-peril trial |
| `scripts/kaggle_submit.py` | Only submission path (validate → upload → log) |
| `tests/` | 89 guardrail tests (`python3 -m pytest tests/ -q`) |

## 7. Reproducing the submission of record

```bash
python3 -m pytest tests/ -q            # 89 guardrails
python3 scripts/run_emerging.py --use-cache --num-leaves 4 --min-leaf 1200
# -> outputs/submissions/emerging-exp-025-repro.csv
#    byte-equivalent predictions to the 0.16994 submission (max rel diff 8e-16)
```

## 8. Mapping to the judging axes

| Axis | Where the evidence is |
|---|---|
| Pricing performance & adaptation (40%) | §2–§3 — transfer-prior + offset adaptation, measured forward index, 0.16994 (4th of 33) |
| Effective use of AI (25%) | §5, `docs/ai_workflow_log.md`, `docs/decisions.md` |
| Trust, validation & guardrails (20%) | §4 — pre-registered gates, noise measurement, 89 tests, exact reproducibility |
| Communication & documentation (15%) | this README, `docs/actuarial_report.md`, `docs/emerging_market_report.md`, `docs/experiment_log.md` |
