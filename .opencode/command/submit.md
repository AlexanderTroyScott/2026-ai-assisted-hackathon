---
description: Prepare and submit a Kaggle submission with validation and logging. Stops for human confirmation before upload.
---

Run the submission workflow for the current experiment using the
`kaggle-submit` skill and the `experiment-runner` agent.

1. Load `.opencode/skills/kaggle-submit/SKILL.md`.
2. Ask me which change this submission tests (one primary change) and the
   hypothesis, unless already stated in the conversation.
3. Build or locate the prediction CSV in `outputs/submissions/`.
4. Run `python scripts/kaggle_submit.py --input <csv> --validate-only` and report
   the validation summary.
5. Compute local D² (pooled and per LOB) on the time-based holdout against
   developed truth.
6. Show me: tag, message, hypothesis, local D², sha256, submissions used today.
7. **Stop and ask for confirmation before uploading.**
8. On confirmation, upload, fetch the leaderboard and append the result to
   `docs/experiment_log.md` and `outputs/experiment_log.csv`.

Arguments: $ARGUMENTS
