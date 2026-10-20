---
description: Fetch the Kaggle leaderboard and our submission history, compare against the best local model, and update the status table in docs/experiment_log.md.
---

Check leaderboard status. Load `.opencode/skills/kaggle-submit/SKILL.md` first.

1. Run read-only commands only:
   - `kaggle competitions leaderboard -c iaa-ai-assisted-hackathon-2026-part-1 --show`
   - `kaggle competitions submissions -c iaa-ai-assisted-hackathon-2026-part-1`
2. Report: our best score and rank, the top-5 table, submissions used today vs the
   10/day budget, and any score movement since the last logged entry.
3. Compare against our best **local** D² and flag any gap that suggests a metric
   implementation mismatch (local ≈ 0.4 vs board ≈ 0.13).
4. Append a status line to `docs/experiment_log.md` if anything moved.

Do not submit anything. Never print credential values.

Arguments: $ARGUMENTS
