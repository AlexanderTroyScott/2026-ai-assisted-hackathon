# 2026 AI-Assisted Actuarial Hackathon — Pricing Across Markets

We built insurance pricing models for two markets — a mature one with four years of
history, and a brand-new one with only two thin years — to predict the claim cost of
every 2025 policy. Our emerging-market model finished **4th of 33 teams**.

This README is the front door. If you read one document, read
**`docs/final_writeup.md`** (the concise five-page write-up). The detailed evidence
lives in `docs/`; the runnable solution in `src/`, `scripts/` and `tests/`.

*In plain terms: for each policy we predict how much the insurer will pay in claims —
big fires and all — and we score with Tweedie D², where 0 means "just predict one flat
average for everyone" and 1 means "perfect".*

## Results at a glance (2026-10-28)

| Competition | Our entry | Score | Standing |
|---|---|---|---|
| **Part 2** — emerging market (closes 2026-10-31) | `emerging-exp-025` | **0.16994** | **4th of 33** — within 0.0018 of the podium |
| **Part 1** — developed market (closed 2026-10-23) | `exp-027` | 0.12984 (hidden board) | 41st of 54 — our best model `exp-036` (0.13570) came just after the deadline and is unranked (disclosed in §2) |

One nice coincidence of the rules: Part 1's scoring gate stayed open after its
deadline and reveals both scores — so it became a free, fully labelled **laboratory**
for testing Part 2 ideas before spending Part 2 submissions. Several ideas died there
cheaply (§3).

---

## 1. The challenge

- **Part 1 (developed market):** 2.4 million policy rows over 2021–2024 — motor,
  buildings and contents. Predict 2025 claim costs for 626,639 policies.
- **Part 2 (emerging market):** a new country with two thin years (109,296 rows,
  ~9,100 claims) — and a 2025 test book nearly four times a normal year, because the
  insurer took over a broker portfolio. **76% of it is business the insurer has never
  seen before.**

Two traps were built into the data. First, the two markets' policy and claim IDs
**collide** — joining them by ID is forbidden and would be silently wrong; knowledge
can only travel through *shared customer characteristics*. Second, claims counted at
31 December 2024 are unfinished: a claim reported in December is still growing, so
every training label has to be **developed to its final value** before we can learn
from it.

## 2. What we built

**Part 2 — learn from the old market, let the new one correct it.** With only ~9,100
claims, a model trained from scratch memorises noise. So the model starts from a
*prior*: strong models trained on the 2.4M-row developed market, recalibrated to the
new market's cost level. A deliberately tiny correction layer (20–35 shallow trees per
product) then learns only the *differences* the prior cannot know. Finally, a 2025
adjustment — measured on the live board, not assumed — accounts for cost drift and for
the fact that new business and renewals behave differently.

The measured quality ladder (cross-validated, higher is better):

| Model | OOF D² |
|---|---:|
| One flat rate per product | +0.0804 |
| Premium × loss ratio | +0.1097 |
| Emerging data alone (no transfer) | +0.1073 |
| Transfer prior alone | +0.1386 |
| **Transfer prior + correction layer (final)** | **+0.1407** |

**Part 1 — one careful model per product.** Rate models (claim cost per unit of
cover) blending three model families (LightGBM and XGBoost), on cleaned data with
properly developed labels.

**One disclosure.** Part 1's leaderboard counted our `exp-027` entry — the best we had
by the deadline (0.12984 on the hidden board, 41st of 54). In the hours after the
deadline we built `exp-036`, a family blend that scores **0.13570** on the hidden
board. The gate still scored it, so it is fully measured, but it is unranked. We
attach `exp-036` as our Week 1 prediction file with that fact stated openly rather
than buried.

## 3. What moved the needle — and what didn't

The Part 2 climb (public score): **0.1630** transfer baseline → **0.1684** the 2025
forward adjustment → **0.1688** model restraint → **0.1699** charging new business and
renewals differently (the 2025 book is underpriced on new business — measured twice,
in motor and in property, independently).

Equally important, the ideas we **killed with evidence** — each one logged:

- **Ensembles** — a winning idea in Part 1, tested three times in Part 2
  (including as an upgrade to the transfer "starting point" model itself), and
  it lost every time. Each test was pre-registered, so each rule stopped the
  submission before it cost us anything.
- **Fancy claim-development schemes** — two alternatives to our simple development
  factors were tested on the Part 1 laboratory. One traded hidden-board score for
  public-board score; the other lost on both. Simple won.
- **Occupation** — looked predictive, but it was mostly driver age in disguise. Gone.
- **Aggressive tuning** — won locally, lost on the board. Restraint transferred;
  cleverness didn't.

## 4. Keeping it honest

- **We measured the noise before claiming signal.** Re-running the same model with a
  different random seed moves the score by ~0.0013 (hidden board: ~0.002). Any claim
  smaller than that needs more evidence — and we say so even when it weakens our story.
- **Rules before results.** Each experiment declared its pass/fail bar in advance
  ("the blend must beat the baseline by 0.0002 across three matched seeds"), which is
  why a good-looking ensemble stayed unsubmitted.
- **100 automated guardrail tests** run before every commit: leakage rules, label
  arithmetic, calibration invariants, submission format checks.
- **Reproducibility.** One command regenerates the Part 2 submission of record
  byte-for-byte (differences below 1e-15) — the leaderboard number is not an artefact.
- **One submission, one explained change**, uploaded only through a validating script,
  with an explicit human go-ahead for Part 2 every time. A one-off double upload
  (two parallel calls raced) is logged honestly and became a permanent serial-submission
  rule.

## 5. How AI was used

The AI agent did the hands-on analytical work — exploring data, writing the pipelines,
running experiments, computing diagnostics, drafting documentation — inside a
governance frame built by the human: authorisation gates on every submission, every
number recomputed from source data before it could appear in a report, judgement calls
flagged and escalated, negative results kept on the record. The full interaction trail
(including where AI was wrong and how it was caught) is in `docs/ai_workflow_log.md`
and `docs/decisions.md`.

## 6. Where to find things

| Path | What it is |
|---|---|
| `docs/final_writeup.md` | **The concise five-page write-up — start here** |
| `docs/emerging_market_report.md` | Part 2 deep report (market analysis, adaptation method) |
| `docs/actuarial_report.md` | Part 1 deep report (ASOP 9 documentation standard) |
| `docs/decisions.md` | Every judgement call, dated (D-001…D-080) |
| `docs/experiment_log.md` | Every experiment and submission, with scores and file hashes |
| `docs/ai_workflow_log.md` | AI interaction evidence trail |
| `docs/submission_protocol.md` | Submission rules and safety rails |
| `src/`, `scripts/` | The pipelines (`scripts/kaggle_submit.py` is the only upload path) |
| `tests/` | 100 guardrail tests (`python3 -m pytest tests/ -q`) |

Reproducing the Part 2 (`emerging-exp-025`) and Part 1 (`exp-036`) submissions of record:

```bash
python3 -m pytest tests/ -q            # 100 guardrails
# Part 2 submission of record (0.16994 Public, 4th of 33):
python3 scripts/run_emerging.py --use-cache --num-leaves 4 --min-leaf 1200
# -> outputs/submissions/emerging-exp-025-repro.csv (byte-equivalent, max rel diff 8e-16)

# Part 1 multi-family rate blend (0.12753 Public / 0.13570 Private).
# --legacy-features reproduces the pre-D-084 feature construction exactly
# (full-sample relativity prior + (customer_id, loss, cost) claim dedup);
# omit it for the leak-free D-084 pipeline (strict per-row credibility anchor).
python3 scripts/run_experiment.py --name exp-032 --mode tweedie --target rate --num-leaves 16 --lr 0.04 --final-fit --legacy-features \
    --features base,premium,experience,weather,flags,relativity,address,customer
python3 scripts/run_experiment.py --name exp-033 --mode per_lob --target rate --num-leaves 6 --lr 0.05 --final-fit --legacy-features \
    --features base,premium,experience,weather,flags,relativity,address,customer
python3 scripts/run_experiment.py --name exp-035-xgb --mode per_lob_xgb --target rate --lr 0.05 --final-fit --legacy-features \
    --features base,premium,experience,weather,flags,address,customer
python3 scripts/ensemble_blend.py --weights 0.40,0.42,0.18 \
    --out outputs/submissions/exp-036-xgb-lgb-blend.csv \
    outputs/submissions/exp-035-xgb.csv outputs/submissions/exp-033.csv outputs/submissions/exp-032.csv
```

The same recipe **without** `--legacy-features` is `exp-046` (D-084, leak-free
pipeline): **0.12751 Public / 0.13577 Private** — impact-neutral vs exp-036
(−0.00002 / +0.00007, both boards inside noise) and marginally our best Private
score. The strict pipeline is the production pipeline; `--legacy-features`
exists only to reproduce pre-D-084 runs.

**For the judging panel** — where each axis is evidenced: pricing performance &
adaptation → §2–§3 and `docs/emerging_market_report.md`; effective use of AI → §5 and
`docs/ai_workflow_log.md`; trust, validation & guardrails → §4 and `tests/`;
communication & documentation → this README and `docs/final_writeup.md`.
