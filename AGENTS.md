# AGENTS.md — Working Conventions

You are working on an actuarial pricing hackathon repo. Score is judged on
accuracy, actuarial credibility, AI-workflow evidence and validation guardrails.
Read `README.md` for the plan and `docs/data_findings.md` before doing any
modelling work — the findings there are verified and must not be re-derived.

## Load on demand (context economy)

Do not carry the full data dictionary or findings in your prompt. Load the
relevant skill instead:

| Situation | Load |
|---|---|
| Modelling, EDA, features, anything touching `developed/` | `data-facts` skill |
| Submitting to Kaggle, checking leaderboard | `kaggle-submit` skill |
| Building labels, development factors, trend/inflation | `actuarial-target` skill |
| Choosing a split, judging overfit, is this number a valid gate | `model-validation` skill |

For large read-only investigations, delegate to the `data-audit` or `model-audit`
subagents and bring back only the conclusions.

## Hard rules

1. **Never invent numbers.** Anything that reaches `docs/` or a decision must be
   recomputed from `developed/` or explicitly marked `unverified`.
2. **Never submit to Kaggle** without going through `scripts/kaggle_submit.py`
   and getting explicit human confirmation. Submissions are irreversible and
   capped at 10/day.
3. **Never print secrets.** Check credential *names/lengths*, never values.
4. **Never commit `developed/`** or submission CSVs. `developed/` is gitignored.
5. **One primary change per submission** so leaderboard movement is attributable.
6. **Lag features strictly** — anything derived from `claims` for a row must only
   use loss dates before that row's `period_start`. Leakage is a hard fail.
7. **Enforce the feature deny-list:** `renewal_decision` is never a feature
   (training-only + outcome leakage).

## Modelling conventions

- Model cost as `exposure × annual rate`; use recomputed exposure only
  (`(period_end − period_start + 1) / 365`) — the supplied column is corrupt.
- Train/validate by **time**, never randomly. Score against **developed** truth.
- Metric: sklearn `d2_tweedie_score(y, y_pred, power=1.5)`. Floor predictions at
  `1e-3` (deviance raises on non-positive `y_pred`).
- Separate models per LOB (motor / buildings / contents); recalibrate each so
  `Σpred ≈` expected total cost.
- Keep `docs/decisions.md` current: append, never rewrite history. Judgement calls
  stay `proposed` until a human accepts them.

## Repo layout

```
src/        reusable modules (io, data_quality, target, development, inflation,
            features, metrics, models, validate)
scripts/    make_baseline.py, kaggle_submit.py
tests/      pytest guardrails
docs/       findings, decisions, ai_workflow_log, submission_protocol, experiment_log
outputs/    submissions/, charts/, manifests
developed/   parquet as received (gitignored)
.opencode/  skills, agents, commands
```

## Version control

Commit early and often so the development of the solution is legible to the
judging panel ("well-structured code repository with version control").

- **Commit after every meaningful unit of work**: a doc update, a feature block,
  a model change, a test, or a submission. Not every file save.
- **Push to `origin/main` after each commit** — this is the shared record while
  humans are away.
- Message format: `<area>: <what changed> [D-xxx]` e.g.
  `target: develop 2024 labels with LOB LDFs [D-005]`, `docs: fix exposure
  aggregation pitfall [D-001]`, `submit: exp-002 premium_lr baseline`.
- Reference decision IDs where one exists, and the experiment tag for submissions,
  so commits map to `docs/experiment_log.md` rows.
- Never commit `developed/` or files under `outputs/submissions/` (gitignored).
- Do not rewrite history, amend published commits, or force-push.
- If a commit is made and then a correction is needed, add a follow-up commit —
  the trail is evidence, not clutter.

## Definition of done

- Code: runs end-to-end from `developed/` to a valid submission CSV.
- Tests: `pytest -q` passes; new behaviour has a test.
- Docs: `docs/experiment_log.md` updated if anything was submitted or measured;
  `docs/decisions.md` updated if a choice was made.
- **Committed and pushed** with a message referencing the decision/experiment ID.
- No unexplained AI output in the diff.
