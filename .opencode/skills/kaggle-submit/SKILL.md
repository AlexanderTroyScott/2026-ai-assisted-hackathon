---
name: kaggle-submit
description: Use when submitting predictions to the Kaggle leaderboard, checking or logging leaderboard scores, preparing a submission CSV, or deciding whether a change is worth a submission slot. Covers the submission format rules, the Tweedie D² metric checks, the 10/day budget policy, and the exact commands.
---

# Kaggle Submission Protocol

Competition: `iaa-ai-assisted-hackathon-2026-part-1`
Deadline 2026-10-23 16:00 · **10 submissions/day** · 22 teams · irreversible.

Canonical source: `docs/submission_protocol.md`. Log every submission in
`docs/experiment_log.md`.

## Never do this

- Never run `kaggle competitions submit` directly — always via
  `scripts/kaggle_submit.py`, and always with explicit human confirmation.
- Never submit a CSV you cannot explain (hypothesis + expected direction required).
- Never submit two unrelated changes in one slot — attribution matters more than
  speed here, because the leaderboard is our external test set for feature decisions.
- Never print credential values. `KAGGLE_USERNAME`/`KAGGLE_KEY` are env vars;
  `~/.kaggle/kaggle.json` also exists.

## Format gate (the script enforces all of these)

| Rule | Value |
|---|---|
| Format | CSV, header `row_id,prediction` (sample_submission is **parquet**) |
| Row count | **626,639** = motor 254,986 + buildings 184,155 + contents 187,498 |
| row_id set | must equal `sample_submission.row_id` exactly |
| prediction | **strictly positive** — floor at `1e-3` (deviance raises on 0) |
| NaN | none |
| Prefix | `M`/`B`/`C` must match the source test file |

## Metric checks before spending a slot

- Local metric: `sklearn.metrics.d2_tweedie_score(y_true, y_pred, power=1.5)`.
- Score against **developed** truth on a time-based holdout (see `actuarial-target`).
- Also compute the exposure-rate-baseline variant and record both.
- Compare with leaderboard scale: top 0.12996, 20th 0.11710 (as of 2026-10-20).
  Local D² near 0.4 ⇒ our baseline definition is wrong.

## Commands

```bash
# build a baseline
python scripts/make_baseline.py --kind constant_rate     # or: premium_lr

# validate + submit + log (asks before upload)
python scripts/kaggle_submit.py \
    --input outputs/submissions/<name>.csv \
    --message "D-014 experience features: freq+cost lags" \
    --tag exp-014

# read-only status
kaggle competitions leaderboard -c iaa-ai-assisted-hackathon-2026-part-1 --show
kaggle competitions submissions -c iaa-ai-assisted-hackathon-2026-part-1
```

## Cadence policy (D-013, user-directed)

1. Submit after every **meaningful** change: feature block, development/trend
   change, calibration change, model swap. Not for refactors or docs.
2. One primary change per submission. Confounded submissions are marked `confounded`.
3. Baselines first: constant-rate and premium×LR must be on the board before any
   model submission.
4. Log local D² + LB score/rank into `docs/experiment_log.md` in the same session.
5. Reserve ~2 slots/day to re-submit the current best after any refactor.

## What to write into the log

```
| # | timestamp | tag | message | change | hypothesis | local D² | LB score | LB rank | Δ | attribution | notes |
```

Negative results are valuable — keep them. The write-up needs to show *which
features did and did not help*.

## Version control around a submission

1. Commit the code that generates the submission **before** uploading, so the
   leaderboard entry maps to a commit: `submit: exp-014 experience features [D-014]`.
2. Push to `origin/main` after the commit.
3. After the score lands, commit the log update: `docs: log exp-014 result`.
4. Never commit the CSV itself (`outputs/submissions/` is gitignored) — record its
   sha256 in `docs/experiment_log.md` instead.
