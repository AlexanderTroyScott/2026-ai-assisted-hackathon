# AI-Assisted Workflow Log

Judging criteria explicitly reward *"meaningful integration of AI throughout the
actuarial workflow"* and *"rigorous verification of AI-generated outputs"* (25%),
with AI-assisted screening of videos, documentation and code as a first layer.
This log is the evidence trail.

Format per entry: **what was asked · what the AI did · how it was verified ·
what a human decided**. Anything AI-produced that reached a decision or a file is
noted with its verification status.

---

## Session 1 — 2026-10-20 · Challenge review + full data profile

**Asked.** Review the competition brief and the existing README battle plan; look
at the data in `developed/`; give suggestions.

**AI did.**
- Read the brief and README; mapped the four judging weights to deliverables.
- Profiled all 11 parquet files (shapes, dtypes, nulls, value levels).
- Verified row_id coverage vs `sample_submission` (exact match, 626,639).
- Built the claim→row attribution and tested uniqueness.
- Found and quantified the `exposure` unit-error contamination.
- Found dirty categorical levels (`construction_type` case variants, `T/t/W/w`),
  string booleans, structural telematics missingness.
- Built the origin-year × report-year incurred triangle and derived LDFs.
- Measured the experience-feature and premium-decile signals.

**Verified.** All figures recomputed directly from `developed/` with pandas; no
number in `docs/data_findings.md` is estimated or carried over unverified.
Contradictions between the brief and the data were checked twice (e.g. repudiated
claims carry zero amounts — confirmed before calling the filter a no-op).

**Human decided.** Prioritise experience features (D-014), treat premium as the
benchmark (D-015), recompute exposure (D-001), add automated tests for the
data-quality landmines (D-011).

**Output.** Review delivered as chat; findings then consolidated into
`docs/data_findings.md` and `docs/decisions.md`.

---

## Session 2 — 2026-10-20 · IBNR on existing records

**Asked.** Could there still be new claims on the existing records in 2025 that we
would need to find (IBNR)?

**AI did.** Separated the question into test rows vs training rows and tested both:
- Confirmed zero claims have a 2025 loss date (max 2024-12-29) and every claim
  attaches to a training policy row → nothing to "find" for test rows.
- Quantified training-side development: triangle LDFs, open-claim share by origin
  year, report-lag tail by peril (bodily_injury 10.2% >365 days), Q4-2024 loss-count
  truncation (Dec 2,264 vs norm ≈ 4,001).
- Confirmed closed claims are final (`paid == incurred`) so development is only
  (a) unreported claims and (b) open-case development.

**Verified.** Attribution rule checked for double-matching (0 claims match >1 row);
loss-date max re-read from source; LDFs derived from an independently constructed
triangle.

**Human decided.** Develop labels, not predictions (D-005); validate against
developed truth (D-006); concentrate factors where the tail is; state residual
uncertainty rather than presenting the LDF as known.

**Output.** Discussion → D-005/D-006 and §5 of `docs/data_findings.md`.

---

## Session 3 — 2026-10-20 · Reverse-engineering claims inflation

**Asked.** Many countries experienced inflation 2020–2025 — is it possible to
reverse-engineer an inflation factor?

**AI did.**
- Designed cohorts that avoid development and closure bias (closed, report lag ≤30).
- Fitted `log(severity) ~ accident year` per peril with standard errors → rates from
  +2.9%/yr to +9.4%/yr.
- Tested whether the shift is multiplicative (quantile ratios ~flat across q25–q95
  → scale shift, not shape change).
- Built mix-adjusted (fixed-weight) indices to neutralise changing peril mix.
- Demonstrated the closure-bias artefact on bodily_injury (−9.5%/yr) and mix bias
  in raw motor severity.
- Cross-checked with premium/exposure indices (2025 already trended +5%/yr).

**Verified.** Regression coefficients reported with 95% CIs and sample sizes;
backtest design specified (fit 2021–23, predict 2024) but not yet executed —
recorded as an open task, not presented as done.

**Human decided.** Per-peril factors (D-007), forward-factor-only for trees
(D-008), blend with the premium anchor (D-009 — still `proposed`, needs sign-off).

**Output.** §6 of `docs/data_findings.md`, `src/inflation.py` design in the plan.

---

## Session 4 — 2026-10-20 · Implementation plan + Kaggle access check

**Asked.** Write up an implementation plan; can we make submissions / see the
leaderboard using the env-var credentials?

**AI did.**
- Wrote the phased implementation plan (scaffold → data quality → target →
  development → inflation → features → models → guardrails → submissions) with
  file layout, function signatures and test table.
- Checked credentials **without printing secrets** (names and lengths only),
  verified `kaggle` CLI 2.2.4 and `kagglehub` 1.0.2.
- Found the competition (`iaa-ai-assisted-hackathon-2026-part-1`, 22 teams, entered),
  read the public leaderboard (top 0.12996, 20th 0.11710) and confirmed zero prior
  submissions. Read-only only — no submission made in plan mode.

**Verified.** Leaderboard parsed from CLI output; submission count read from
`kaggle competitions submissions` ("No submissions found").

**Human decided.** Routine submissions during development (D-013); documentation,
README revision and opencode skills/agents for context management.

---

## Session 5 — 2026-10-20 · Documentation, README, skills & agents

**Asked.** Update documentation with everything discussed; revise the README;
document conversations; create associated agent files and skills to manage the
context window; encourage routine submissions.

**AI did.** Wrote `docs/` (findings, decisions, this log, submission protocol,
experiment log), rewrote `README.md`, wrote `AGENTS.md`, created the opencode
project config with three skills, three agents and three commands, and built
`scripts/` for baselines and Kaggle submission.

**Verified.** Config validated against the published opencode schema
(<https://opencode.ai/config.json>); skill/agent/command file locations match the
documented layout; submission script runs its validation gate before any upload.

**Human decided.** Skill/agent split for context economy (heavy reference material
lives in loadable skills, not in the prompt), and the submission-cadence policy.

---

## Working rules this log implies

1. Never let an unverified AI number reach `docs/` — recompute or mark it `unverified`.
2. Record what was *checked* (e.g. "0 claims match >1 row"), not just what was built.
3. Judgement calls are flagged as such and stay `proposed` until a human accepts them.
4. Each submission gets an entry in `docs/experiment_log.md` with the hypothesis it
   tests, so leaderboard movement is attributable to a change.

---

## Session 6 — 2026-10-20 (overnight, autonomous) · Actuarial control cycle

**Asked.** Follow the actuarial control cycle unattended: build a model, revise
assumptions, test, submit, use results to improve. Make one submission now as a
quota-reset marker; before each later submission check the quota and **hold off if
it has reset to 10**. Also look for exploitable patterns in the synthetic DGP.
Make relevant git commits and pushes as we go.

**AI did.**
- Submitted the constant-rate baseline as the marker (exp-001, LB **0.07293**).
  The Kaggle CLI reports "N submissions remaining today", so the reset is directly
  observable. Quota never reset during the session (checked before every upload).
- Delegated a DGP reverse-engineering investigation to a subagent and consolidated
  it into `docs/synthetic_dgp_findings.md` (severity is a Gamma scale family;
  the motor premium formula is decoded at R² = 0.956; pricing and risk are
  generated separately; weather drives perils at large rate ratios).
- Built `src/dataset.py` + `scripts/run_experiment.py` (LightGBM Tweedie, time
  split, LOB recalibration) and ran the feature ladder locally.
- **Ran a leakage audit before submitting.** A model scoring local D² = 0.675 was
  treated as suspicious rather than a win; probing each field found
  `customer_service_contacts` alone explains D² = 0.628 (claimants average 3.28
  contacts vs 0.41). Deny-listed (D-016) with a regression test. This is the
  single most important judgement of the session — a leaked submission would have
  looked like a spectacular result and been disqualified on review.
- Also caught and fixed an aggregation pitfall in the earlier analysis (a naive
  merge dropped policy rows whose claims fell in other periods), correcting the
  frequency / cost-per-exposure / premium-per-exposure figures in
  `docs/data_findings.md`.
- Submitted the attribution ladder exp-002/003/004, one primary change each.

**Verified.**
- 11 pytest guardrails pass, including the new leakage and aggregation tests.
- Every submission went through `scripts/kaggle_submit.py` validation gate first.
- Local D² validated against 4 leaderboard points (gap −0.007 to −0.010, ranking
  correct) → the CV setup is now a trusted offline oracle (D-018).
- Quota checked before each of the 4 uploads; it never returned to 10, so no hold
  was triggered. 6 slots remain in the window.

**Human decided (pre-authorized before sleep).**
- Routine submissions as part of the control cycle, with the quota-reset guardrail.
- Git commit + push after each meaningful unit of work.

**Output.** Leaderboard 0.07293 → **0.11519** (top is 0.12996). 4 submissions,
each with a stated hypothesis and a measured local D². Full attribution in
`docs/experiment_log.md`; decisions D-016 to D-022 recorded.

---

## Session 7 — 2026-10-20 (afternoon, autonomous) · Control cycle: relativity features, per-peril development, ensemble

**Asked.** Continue the actuarial control cycle from `docs/experiment_log.md`;
check the Kaggle quota before any submission and hold if it has reset to 10.

**AI did.**
- Checked the quota through the Kaggle API (`num_today=6`, `num_allowed_now=4`) —
  not a fresh window, so the hold rule did not trigger. Re-checked before every
  upload; each upload printed its own quota line.
- Fixed a logging bug found on the way: `fetch_leaderboard` recorded the board
  *maximum* as our score (0.12996) — replaced with our own latest submission
  score, and added a mechanical hold gate to `kaggle_submit.py` that refuses to
  upload when the daily quota reads 10 unless `--ack-quota-reset` is given.
- Measured the local noise band first (D-025 operationalised): exp-006 config at
  seeds 0/1/2 → 0.1242 / 0.1233 / 0.1240 (sd ≈ 0.0005). Anything below ~0.002
  local D² is a tie.
- Built `src/features.py`: historical relativity target encodings with the
  strict lag `source.pe < row.ps` (AGENTS.md rule 6), credibility-smoothed, with
  toy-frame guardrail tests for own-row, overlapping-period and unseen-level
  leakage (4 new tests).
- Built `scripts/build_development.py` + `src/development.py`: per-peril
  age→ultimate factors (BI 1.535, subsidence 1.306 vs short-tail 1.01–1.05),
  level-neutral against the accepted D-005 table, with level-preservation tests
  (5 new tests). 20 pytest guardrails pass.
- Ran the ladder: exp-007 (relativities) local 0.1248 — consistent +0.0006 to
  +0.0009 across seeds; exp-008 (per-peril development) 0.1223 with identical
  skill-vs-baseline gap; weather-lag sensitivity 0.1243 vs 0.1242 (D-012
  question closed: no dependence on same-year indices); 3-seed ensemble 0.1250.
- Submitted exp-007 (relativity block) and exp-009 (seed ensemble), one primary
  change each, through `scripts/kaggle_submit.py`. 2 slots kept in reserve.

**Verified.**
- Quota read from the API before each submission (6→7 used; never reset to 10).
- exp-007 = **0.11637** (new best, +0.00068 over exp-006); exp-009 = 0.11632 —
  the second near-tie flip (D-025), now measured in both directions.
- The optimism gap holds at −0.0084 / −0.0087 for the two new points (8 points
  total, −0.007 to −0.010 band).
- 20 tests green before either submission; every upload passed the format gate
  (626,639 rows, exact row_id set, floor ≥ 1e-3) and recorded sha256.

**Human decided (standing pre-authorization, restated in the task).**
- Routine submissions as part of the control cycle, with the quota-reset hold.
- The user's message named exp-004 (0.11519) as best; the live board showed
  exp-006 (0.11569) from the earlier population-shift test — the log and the
  board were treated as ground truth and the discrepancy flagged in the report.

**Output.** Leaderboard 0.11569 → **0.11637**. Local evidence: seed-noise band,
relativity gain, per-peril tie, weather-lag robustness, ensemble tie. Decisions
D-023/D-024 gap-filled and D-027/D-028/D-029 recorded. 2 quota slots reserved.

## Session 8 — 2026-10-20 (evening) · Crash recovery + denied-claim history (D-031)

**Asked.** Session crashed; reconstruct where things stood, report the remaining
quota, and proceed with everything except spending the last submission slot
(the user judged exp-011's local near-tie not worth it).

**AI did.**
- Reconstructed state from git, `docs/experiment_log.md`, `docs/decisions.md`,
  `outputs/metrics/` and the live quota check — reported it before touching
  anything.
- Found uncommitted D-031 work at the crash point: `denial_features` (strictly
  lagged denied-claim history), the `denied` feature group, two leakage tests,
  and a trained exp-011 (local D² 0.1339).
- Re-ran the guardrail suite (pytest had dropped out of the venv; reinstalled,
  then 25 tests green including the new own-period-exclusion assertions).
- Logged D-031 in `docs/decisions.md`, exp-011 in `docs/experiment_log.md` and
  `outputs/experiment_log.csv`, and this session entry. Committed and pushed.

**Verified.**
- Quota read live from the API: 9/10 used, 1 remaining — matches the nine
  recorded submissions (exp-001…007, 009, 010).
- exp-011 = exp-010 stack + denied block, same seed, same truth: 0.1339 vs
  0.1342 (−0.0003, inside the D-025 tie band). No leaderboard claim made from it.
- 25/25 tests pass after the change; nothing was uploaded to Kaggle.

**Human decided.**
- exp-011 is not worth the last quota slot (D-025/D-029: 4th-decimal local
  movement does not rank). The slot is held for a change outside the tie band.
- Otherwise proceed: document and commit the D-031 work as opt-in.

**Output.** State fully recovered and documented. Leaderboard unchanged at
**0.12113** (exp-010, rank 20/22). 1 quota slot reserved.

## Session 9 — 2026-10-20 (late) · Actuarial references layer (R-01…R-03)

**Asked.** Human proposed incorporating actuarial links/references into the
development, for documentation credibility *and* method influence; agreed the
links would be pasted and that agents should get a distilled skill.

**AI did.**
- Downloaded the three CAS PDFs and identified each from metadata + contents:
  Friedland, *Estimating Unpaid Claims Using Basic Techniques* (R-01); Anderson
  et al., *A Practitioner's Guide to GLMs* (R-02); Goldburd et al., *GLMs for
  Insurance Rating*, CAS Monograph 5 (R-03).
- Extracted the method-relevant sections (triangles/dev technique/BF/Cape Cod +
  ASOP 9; Tweedie App. C and freq/sev App. J; train-test splitting, weights vs
  offsets, validation, documentation chapters) and wrote `docs/references.md`
  with annotations, an R→D mapping, method candidates C-1…C-5, and an honest
  reconciliation of the one place the literature and our measurement differ
  (freq/sev vs Tweedie).
- Distilled an `actuarial-methods` skill for agents; cross-linked from
  `actuarial-target`; added the README table row; logged D-032.

**Verified.** Every claim about the sources is traceable to a section/appendix
read in the PDFs this session (TOC + page extracts); no source number was
merged into our findings; frozen decisions untouched. Candidate queue is
explicitly `proposed` and gated by the D-025 tie band before any quota spend.

**Human decided.** References serve both documentation and modelling; agents
should load a skill rather than the PDFs; the three CAS links are the sources.

**Output.** `docs/references.md`, `docs/decisions.md` D-032, `actuarial-methods`
skill. No model change, no submission — leaderboard unchanged at **0.12113**,
1 quota slot still reserved.

## Session 10 — 2026-10-20 (late) · Reviewer subagent

**Asked.** Create a reviewer subagent on `deepseek-v4-pro` that reviews the
repo's programs and has access to the project skills.

**AI did.** Added `.opencode/agent/reviewer.md`: read-only subagent
(`edit: deny`, `bash: ask`) on `litellm/deepseek-v4-pro` (the model is
registered under the `litellm` provider in the global config). Its prompt
requires loading `data-facts`, `actuarial-target`, `actuarial-methods` and
`kaggle-submit` before judging, and reviews against AGENTS.md hard rules
(leakage, deny-list, exposure recompute, metric floor) first, then correctness,
actuarial soundness, guardrails and style — citing `D-xxx`/`R-xxx` IDs. Output
is a verdict + severity table, same posture as the `data-audit`/`model-audit`
agents.

**Verified.** Frontmatter fields match the documented agent schema (`description`,
`mode`, `model`, `permission`); model ID confirmed against the live provider
config; file placed in the auto-discovered `.opencode/agent/` directory. Takes
effect on the next opencode restart.

**Human decided.** Second-opinion code review as a standing workflow step, on a
different model family than the primary agent.

## Session 11 — 2026-10-23 · Code & actuarial audit: `--final-fit` bugfixes, Per-LOB rate models, and multi-family ensemble (`D-063`..`D-065`)

**Asked.** Audit the repository, challenge prior assumptions, investigate the
31 December 2024 valuation cutoff (*"Some 2024 claims had not been reported by
then, and open claims may still develop"*), and implement and submit
improvements.

**AI did.**
- Audited `scripts/run_experiment.py` and found two critical implementation bugs
  in `--final-fit` (lines 460–472) that had caused `exp-024` (`--target rate`)
  and `exp-025` (`D-053` shallow-tree hyperparameters) to collapse on the
  leaderboard (`0.06019` and `0.11168`):
  1. `final_model.fit(X_all, train["y_ult"])` ignored `args.target == "rate"`
     (fitting total cost instead of rate `y_ult / exposure` and omitting
     exposure weights), after which `to_cost` multiplied by `exposure` a second
     time (`cost ∝ exposure²`).
  2. `final_model` hardcoded `num_leaves=64, learning_rate=0.04` instead of
     inheriting `**params`, causing `exp-025` to train `num_leaves=64, lr=0.04`
     for `590` trees (~6× over-budget).
- Audited `claims.parquet` around the 31 Dec 2024 cutoff: verified that open
  motor `own_damage` and `theft` claims carry `0.00%` booked recoveries (versus
  `30% × 80% = 24%` on closed `own_damage`), explaining why `exp-031` (`D-061`
  uniform open-claim multiplier) over-inflated open `own_damage` claims by
  2.25×; implemented Horvitz-Thompson Date×Peril inverse reporting probability
  + open-claim recovery/IBNER development (`development="ht_motor"` in
  `src/dataset.py`).
- Implemented `--mode per_lob` (shallow `num_leaves=6` Tweedie rate models per
  LOB on active line features, dropping `earned_premium` when `target == "rate"`)
  and a Per-LOB XGBoost Tweedie rate model (`max_depth=3`, `no_rel`), plus a
  regression test in `tests/test_guardrails.py`.

**Verified.** Tested across both rolling-origin folds (`Fold 2024` OOF D²
`0.13223 → 0.14076`; `Fold 2023` OOF D² `0.13624 → 0.14350`), exposure-band
flatness gate (`1.13–1.14` vs truth `1.12`), and confirmed on the live Kaggle
leaderboard (`exp-033` `0.12681` Public / `0.13531` Private; `exp-034` `0.12707`
Public / `0.13536` Private; `exp-035` `0.12700` Public / `0.13509` Private;
**`exp-036` `0.12753` Public / `0.13570` Private** — +0.00317 Public and
+0.00586 Private over `exp-027`).

**Human decided.** Approved implementing the fixes, submitting to Kaggle,
updating documentation, and pushing commits to GitHub.

**Output.** `src/dataset.py`, `scripts/run_experiment.py`,
`tests/test_guardrails.py`, `docs/decisions.md` (D-063..D-065),
`docs/experiment_log.md`, `docs/actuarial_report.md`, and submissions
`exp-033`..`exp-038`.

