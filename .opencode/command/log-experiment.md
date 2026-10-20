---
description: Record a local experiment result in docs/experiment_log.md without making a submission.
---

Log a local experiment. Use this whenever something was measured but no Kaggle
slot was spent — the write-up needs these as evidence of what worked and what did
not.

1. Ask for (or infer from the conversation): experiment name, what was changed,
   the result (numbers), and the decision it produced.
2. Append a row to the **Local experiments** table in `docs/experiment_log.md`:
   `| date | experiment | result | decision |`
3. If the result produced a decision, append an entry to `docs/decisions.md` with
   status `proposed` (or `accepted` if I confirm), following the existing format.
4. Keep it to one or two lines per row — no methodology prose in the log.

Arguments: $ARGUMENTS
