# Submission Protocol

Competition: **`iaa-ai-assisted-hackathon-2026-part-1`**
Deadline: **2026-10-23 16:00** · Budget: **10 submissions/day** · Teams: 22

Every submission is made through `scripts/kaggle_submit.py`. No manual CLI uploads.
The point of routine submissions is **feature attribution** — the leaderboard is our
external test set, so change one thing at a time where practical.

---

## 1. Format rules (hard)

| Rule | Detail |
|---|---|
| File | CSV with header `row_id,prediction` (output is CSV even though `sample_submission` ships as parquet) |
| Rows | Exactly 626,639 — the union of `test_motor` (254,986) + `test_buildings` (184,155) + `test_contents` (187,498) |
| row_id set | Must equal `sample_submission.row_id` exactly (no extras, no missing) |
| prediction | **Strictly positive** — floor at `1e-3`. `mean_tweedie_deviance(power=1.5)` raises on `y_pred <= 0` even though the brief says "zero or above" |
| NaN | None allowed |
| LOB consistency | `row_id` prefix must match its source test file (`M`/`B`/`C`) |

`scripts/kaggle_submit.py` enforces all of the above and refuses to upload otherwise.

## 2. Metric

Tweedie **D², power = 1.5**, vs a "constant-rate baseline". Higher is better.

- Use sklearn `d2_tweedie_score(y_true, y_pred, power=1.5)` locally.
- Also report the exposure-rate-baseline variant (`null = exposure × mean cost/exposure`)
  so we know which definition Kaggle used. Compare against the leaderboard scale.
- **Leaderboard scale (2026-10-20):** top 0.12996, 20th 0.11710. If local CV D² is
  ~0.4 our baseline definition is wrong — fix before trusting any local number.
- Report D² **pooled and per LOB**; recalibrate each LOB so Σpred ≈ expected total cost.

## 3. Cadence policy (user-directed)

1. **Submit after every meaningful change** — a feature block, a development/trend
   change, a calibration change, a model swap.
2. **One primary change per submission.** If two changes ship together, the log must
   say so and the attribution is marked `confounded`.
3. **Never burn a slot on a submission you can't explain.** Minimum content for a
   submission entry: hypothesis, change, local CV D² (developed truth), expected
   direction of movement.
4. **Baselines first.** Constant-rate and premium×LR baselines must be on the board
   before any model submission, so all later gains are measured against them.
5. **Log immediately.** Leaderboard rank/score goes into `docs/experiment_log.md`
   within the same session as the submission.
6. Reserve at least **2 slots per day** for the current best model after any
   refactor, in case a bug regresses the score silently.
7. **Confirmation policy (human-directed, 2026-10-27; Part 1 reaffirmed
   2026-10-28: "You do not need any permission anymore for part 1 submissions
   since the number of daily submissions is 100"):** every **Part 2**
   submission needs explicit human confirmation at the time of submission.
   **Part 1** submissions carry standing human authorization and may be
   submitted without asking, subject to rules 1–6 above and the
   `scripts/kaggle_submit.py` gate. Part 1 serves as the validation sandbox
   (D-071): its gate scored byte-identical reproductions four days past the
   deadline and it reveals Private scores. All other rules (one primary change
   per submission, logging, tag/message) apply unchanged to both markets.

## 4. Procedure

```
python scripts/make_baseline.py --kind constant_rate   # or: premium_lr / model
python scripts/kaggle_submit.py \
    --input outputs/submissions/<name>.csv \
    --message "D-014 experience features: freq+cost lags" \
    --tag exp-014
```

The script: validates → uploads via `kaggle competitions submit` → fetches the
leaderboard → appends a row to `docs/experiment_log.md` (and
`outputs/experiment_log.csv`).

Before uploading, the script also records sha256 of the CSV so a submission can be
reproduced later.

**Submit serially.** Never run two `kaggle_submit.py` invocations concurrently:
on 2026-10-28 a parallel bracket pair produced a duplicate upload (same file
and description twice, refs 57741577/57741578, 27 ms apart), which consumed an
extra daily slot. One submission at a time; wait for the score before the next.

## 5. Leaderboard snapshot format

Log after every submission:

```
| # | timestamp | tag | message | local D² | LB score | LB rank | Δ vs prev | notes |
```

Fetch with:
```
kaggle competitions leaderboard -c iaa-ai-assisted-hackathon-2026-part-1 --show
kaggle competitions submissions -c iaa-ai-assisted-hackathon-2026-part-1
```

## 6. What counts as "meaningful change" (submission-worthy)

| Worth a slot | Not worth a slot |
|---|---|
| New feature block (e.g. experience lags) | Refactor with no behavioural change |
| Development/trend factor change | Doc/test-only change |
| Model swap or blend weight change | Hyperparameter micro-tweak (batch these) |
| Calibration change (LOB re-centering) | Logging/formatting changes |
| Handling of unseen categories / missing | Renaming columns |

## 7. Week 2 note

Week 2 data arrives at the end of Week 1. Same protocol applies; add an entry with
the dataset-diff summary and the adaptation decision. Keep the Week 1 baseline
scores so the two datasets are comparable.
