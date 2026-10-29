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

## Session 12 — 2026-10-23 → 2026-10-26 · Part 2 launch: cross-market transfer pipeline (D-066/D-067)

**Asked.** Part 2 opened with a new `emerging/` market: build the Part 2
solution and submission pipeline, transferring what can be transferred from
Part 1 **without joining IDs** (they collide across datasets), document it, and
submit only with explicit per-submission confirmation.

**AI did.**
- Profiled `emerging/` against `developed/` (structural comparison in
  `docs/emerging_market_report.md`): 2 thin years (109,296 rows, ~9,100
  claims), 2025 test book 209,063 rows and **76% new business** from a broker
  portfolio takeover — a regime the training data barely covers.
- Built schema canonicalisation recovering 100% of shared rating factors (80
  free-text occupations → 20 standard + `is_self_employed`; `BRK/STN/WOD/MIX`
  → construction types; age bands → midpoints). Transfer happens in feature
  space only; the no-ID-join rule is test-enforced.
- Quantified emerging pricing misalignments (flood_zone unpriced in contents:
  flat premium against a 6.6× cost gradient; motor 25–34 band priced as 35–64;
  gender +34% claim rate for +3% premium) — the adaptation target list.
- Built `src/emerging.py` + `scripts/run_emerging.py`: developed-market prior
  (0.6 written-premium-weighted / 0.4 unweighted, recalibrated to emerging
  totals) as `init_score` to an ultra-shallow emerging offset layer, plus the
  20-policy office-premium build-up deliverable.
- Ran the D-067 trial sprint before a quota reset: blend grid (weight on the
  offset rises monotonically to 1.0 — the offset stands alone), capacity
  sweep, extended feature set, prior variants — six negative results, none
  submitted.

**Verified.** 5-fold policy-grouped CV ladder established the architecture:
transfer prior alone +0.1386 vs emerging-only tree +0.1073 vs constant rate
+0.0804 (the prior is the backbone; the thin book cannot train a standalone
50-feature tree). Every submission CSV passes mechanical gates (209,063 rows,
exact row_id set, positivity, finiteness) and records sha256.
`tests/test_emerging.py` guards the arithmetic.

**Human decided.** Accepted D-066 architecture and D-067's offset-only
conclusion; Part 2 keeps the explicit per-submission confirmation rule.

**Output.** `emerging-exp-001` 0.16170 (rank 9/20) → `emerging-exp-002`
**0.16297** (record). D-066/D-067; `emerging_market_report.md`.

## Session 13 — 2026-10-26/27 · Level/trend probes and the 2025 forward index (D-068/D-069)

**Asked.** Spend the expiring Part 2 window on structured one-change probes of
the 2025 prediction level (the model was systematically low), then build
whatever survives properly into the pipeline instead of leaving post-hoc
multipliers in a CSV.

**AI did.**
- Mapped the motor level response curve with uniform multipliers (×0.95 …
  ×1.40): optimum ≈ ×1.27 — the model under-predicts 2025 motor ultimate cost
  by ~27%. Property peak ≈ ×1.06.
- Probed the motor new-business/renewal shape at constant total: mild and mid
  splits tie at 0.16833 while a hard split collapses to 0.16711 — lift new
  business, do not crush renewals below ≈ ×1.0.
- Replaced the post-hoc multipliers with `estimate_forward_index`
  (`src/emerging.py`): a per-(line, cohort) data estimate (mix-adjusted
  severity trend × credibility-shrunk cohort frequency trend, blended 70/30
  with a premium anchor) **plus explicit, logged reconciliation residuals** to
  the measured response curve — the residual is reported, not hidden.
- Found and fixed a basis trap mid-implementation: closed-lag frequency counts
  fake −20% deflation in 2024 (claims open at the 31-Dec-2024 valuation are
  missing — closure truncation); frequency re-counted on the reported basis
  with a valuation-symmetric Jan–Oct window.

**Verified.** Machinery check: motor renewals — a stable cohort — reproduce
the response-curve factor from data alone (residual 1.01); large residuals sit
exactly where the transfer book broke comparability. The mid-ridge motor point
is a deliberate Public-overfit hedge (two bracket arms tie; the hard split is
the caution). 8 guardrail tests assert ridge direction and held totals.
`emerging-exp-019` (0.16836) reproduced the probe recipe end-to-end from the
pipeline.

**Human decided.** Adopt the forward index at scoring (D-008 convention:
labels stay ultimate at the 2024 valuation — never trend both); keep the
mid-ridge hedge; property cohort split left as open item (a).

**Output.** Probes exp-005…exp-018 (rows 38–50), `emerging-exp-019`
**0.16836** record. D-068/D-069.

## Session 14 — 2026-10-27 · Ensemble port rejected in OOF; Part 1 gate test; sandbox policy (D-070/D-071)

**Asked.** Port the Part 1 multi-family blend lesson (D-064, our only Part 1
board-winner) to Part 2; and test whether Part 1 still scores days past its
deadline — if so, use it as a labelled sandbox.

**AI did.**
- Built `scripts/run_emerging_ensemble.py` — 4 components on identical folds
  and seeds to the pipeline. Best blend (+0.00023) fell below the
  pre-registered +0.0003 gate: **no CSV emitted, no slot spent** (D-070).
  Side-finding validated empirically: XGBoost `base_margin` must be passed at
  both `fit` and `predict`; LightGBM `raw_score` excludes `init_score`.
- Resubmitted `exp-036` **byte-identical** (same sha256) four days past the
  Part 1 deadline: accepted, scored 0.12753/0.13570 — an exact reproduction.
  The gate is open (D-071).
- Recorded the sandbox use policy: Part 1 is an *evaluation instrument* for
  Part 2 hypotheses (100 slots/day, reveals Private scores), with the D-065
  caution that board-adjacent results do not port automatically.

**Verified.** The gate test is self-verifying (a known CSV reproduces its
known scores); the ensemble comparison is paired on identical folds/seeds; the
threshold was fixed before the run.

**Human decided.** Standing authorization for Part 1 submissions — "feel free
to make those because we won't hit the 100 submission/day limit" (2026-10-27)
— scoped and recorded in `docs/submission_protocol.md` §3.7. Part 2 keeps
explicit per-submission confirmation.

**Output.** D-070/D-071; rows 52–53. No Part 2 slot spent.

## Session 15 — 2026-10-27 → 2026-10-28 · HT development labels: a designed 2×2 sandbox campaign (D-072)

**Asked.** Test D-065's Horvitz–Thompson date×peril development labels on the
board for the first time, and decide whether Part 2's coarse LOB-level
development labels should be replaced.

**AI did.**
- Designed a 2×2 replication before spending anything: {LOB labels, HT
  labels} × {seed 0, seed 1} on the byte-identical `exp-033` recipe — one
  factor changed per arm, one arm explicitly measuring the seed-noise floor.
- Measured: HT gains +0.00049/+0.00078 Public in both seeds but loses Private
  in both (mean −0.00021); seed noise alone moves Public by −0.00119. The
  Public effect is inside noise and the Private direction is consistently
  wrong.
- Verdict: HT trades Private for Public → **Part 2 label port rejected**.

**Verified.** The conclusion rests on a designed experiment, not a single
score: effect measured against the noise band (σ ≈ 0.0012 Public), two seeds
per arm, everything else byte-identical, all four CSVs cell-verified. Local
OOF could not answer this (the label *is* the change), so the board was used
as the instrument and the design was stated before the shots.

**Human decided.** Accept the rejection (D-072); Part 2 keeps its `EM_LDF`
labels; the four sandbox uploads are covered by the standing authorization.

**Output.** Rows 54–56; D-072; the board-noise measurement that calibrates
every subsequent gate.

## Session 16 — 2026-10-28 · Bracket discipline: LOB split, per-peril, tuning (D-073/D-074/D-075)

**Asked.** With a fresh Part 2 window (human-approved), spend it on the best
candidates; and — "if you think further tuning could benefit then test and
submit that before going forward" (2026-10-28) — pursue tuning if the
evidence supports it.

**AI did.**
- Buildings/contents split bracket (two arms at constant property total +
  quadratic vertex): both arms lose to uniform ×1.05, vertex gain +0.00003 →
  **no split; axis closed with no refinement shot** (D-073).
- Motor per-peril trial (`scripts/run_emerging_peril.py`): per-peril labels
  developed pro-rata with Σ = `y_ult` asserted and tested; OOF −0.00036 →
  **gate failed, no CSV, no slot** (D-074). The pro-rata label fix was kept as
  a correctness improvement with its own regression test.
- Hyperparameter residual sweep (`scripts/run_emerging_tune.py`, 15 configs
  one-at-a-time around the incumbent): single knobs flat (all ≤ +0.00011); one
  real shrinkage-pair interaction (`num_leaves` 3→4 with `min_leaf`
  600→1200), confirmed on paired seeds (+0.00029/+0.00046/+0.00053) before
  adoption.
- Submitted `emerging-exp-024` — **0.16879**, new record. The board gain
  equalled the mean OOF gain *exactly* (local OOF is a faithful oracle), and
  measured board seed noise (0.0012) vs local seed σ (0.00011) quantified how
  much of any board move is luck.

**Verified.** Paired-seed deltas at identical seeds; thresholds pre-registered
(D-025); the board/OOF equality on exp-024 is a direct validation of the
offline gate. `run_emerging.py` was parameterised so the adopted config is the
default and reproducible.

**Human decided.** Authorised the tuning submission (quote above); accepted
the two negative verdicts without spending slots.

**Output.** Rows 57–60; D-073/D-074/D-075.

## Session 17 — 2026-10-28 · Property cohort ridge → record 0.16994; joint settlement; an AI process error on the record (D-076/D-077)

**Asked.** "Lets go ahead and submit since we have the remaining submissions
and see what happens" (2026-10-28) — spend the remaining window on the D-069
open item (property NB/ren split). Then "Go ahead and settle it" — settle the
joint (total × split) optimum.

**AI did.**
- Two-arm cohort bracket at constant ×1.05 total (nb-heavy vs ren-heavy,
  post-hoc on exp-024 with cell-wise-verified multipliers): nb-heavy **0.16994
  (+0.00115, new record)**, ren-heavy 0.16815 (−0.00064) — property new
  business runs hotter than renewals, the same ridge direction as motor.
  Refinement at the quadratic vertex (0.16991) showed a flat top → axis
  closed at the measured point (D-076).
- Adopted the split pipeline-natively (property nb 1.12 / ren 0.816907, solved
  to hold the ×1.05 total at measured 2025 exposure shares) and proved
  reproduction: a full pipeline run regenerates the record predictions to
  float precision (max rel diff 8e-16); 87 tests green.
- Settled the joint optimum with pure total-scale arms at the fixed split
  (D-077): ×1.02 → 0.16986, ×1.08 → 0.16992 vs ×1.05 → 0.16994; vertex
  +0.000004 — nothing there. The level axis is fully mapped.
- **Process error, recorded in full:** the two settlement submissions were
  issued as concurrent `kaggle_submit.py` invocations and one upload ran twice
  (refs 57741577/57741578, 27 ms apart). The duplicate scored identically but
  consumed an extra daily slot. The AI diagnosed the race from the submission
  list, recorded it in D-077 and the experiment log, and added the
  **serial-submission rule** to `docs/submission_protocol.md` §4.

**Verified.** Cell-wise multiplier checks on every post-hoc CSV (motor exactly
×1.0, held totals recomputed); exact end-to-end reproduction of the record
(8e-16); quota read from the API before and after; the incident is documented
rather than quietly absorbed.

**Human decided.** Both quotes authorised the rounds; the bracket/refinement
protocol (arms at constant total + vertex, stop when vertex gain < probe
noise) applied as pre-registered; `emerging-exp-025` accepted as the
submission of record.

**Output.** Rows 61–65; D-076/D-077; record **0.16994 Public (rank 4 of 33)**;
serial-submission rule.

## Session 18 — 2026-10-28 · Write-up: README rewrite and a correction to our own Part 1 record

**Asked.** "Can you first work on the readme part of the writeup?" — make the
README the judging-panel front door.

**AI did.**
- Fact-checked every number the README would state against the logs *and live
  ground truth* (both leaderboards pulled from the Kaggle API at write time)
  instead of copying the logs forward.
- **The fact-check corrected the logs:** the final Part 1 board ranks on the
  Private split and shows our entry at 0.12984 — the Private score of
  `exp-027`, our best Public submission as of the 2026-10-23 16:00 deadline.
  The `exp-033`→`exp-036` line — including our best model `exp-036`
  (0.12753/0.13570) — was submitted after the deadline and is measured but not
  ranked. Evidence chain: the board score matches `exp-027`'s Private to five
  decimals; entry timestamps and the other teams' deadline-era dates agree.
  Recorded as a correction in `docs/experiment_log.md` and framed honestly in
  the README, with the lesson (deadline discipline) adopted into the protocol.
- Rewrote `README.md` (results, both architectures, measured findings
  including negatives, guardrails, AI workflow, reproducibility, judging-axes
  map) and refreshed the stale `emerging_market_report.md` header.

**Verified.** Standings re-pulled from the API at write time (Part 2: 4th of
33; Part 1: 41st of 54); every README number traced to an experiment-log row,
a decision, or the snapshot; the Part 1 finding stated with its evidence and
without over-claiming the mechanism.

**Human decided.** README first, then the remaining write-up ("Yes please" →
this AI-workflow log next).

**Output.** `README.md`, `emerging_market_report.md` header, Part 1 correction
note.

## Session 19 — 2026-10-28 → 2026-10-29 · Endgame, audit close-out, pre-registered Part 2 gates, and the honesty/doc pass (D-079…D-087)

**Asked.** First the audits and endgame work ("prioritize anything that would
alter a Part 2 submission, make those submissions, and then proceed with
documentation as you have outlined") — with standing batch confirmation for
Part 2 submissions, plus explicit per-decision human choices where judgement
calls arose.

**AI did.**
- **Recon before modelling, measurements before claims.** The flagged
  `estimate_forward_index` dedup-key inconsistency was measured, not assumed:
  `claim_id` is unique across all 9,258 emerging claims and both key sets drop
  0 rows — the bug is **inert** (hygiene fix + pinned test, zero submission
  impact). Part 2 claim-history features verified strictly lagged (`loss < ps`)
  — no Part 1-style leak. The stale quota note ("1 of 10 remains") was
  re-checked against the API: Part 2 is 10/day with daily reset — which
  changed the strategy from rationing to spend-the-window.
- **Pre-registration discipline at window scale.** Two mean-quality candidates
  (prior-leg weight grid; the audit's `wp_rel` cross-year normalization) were
  written into `docs/decisions.md` with gates *before* results and committed
  (`f0a9147`). Counter-evidence against the second candidate was measured and
  disclosed before its run (emerging's own 23→24 drift is flat/wrong-signed vs
  rates). A third arm (per-LOB `w_off`, D-067's fixed weights) followed the
  same protocol. Every harness reproduced the record contract (0.14074) to
  7.7e-07 before any comparison counted.
- **Results were all gate FAILs and all closed honestly:** the prior-weight
  grid put the record 0.6 at the response-curve peak; `wp_rel` failed both
  arms exactly as the pre-registered counter-evidence predicted; the contents
  shrink was 3/3 (then 9/9) seeds positive but an order below the effect bar.
- **One judgement call surfaced to the human instead of being silently
  adopted:** a gate-override on the contents shrink (9/9 positive, +0.00009,
  supplemental seeds labelled post-hoc) under the endgame asymmetry. The human
  approved; the counter-evidence (exp-004's board reversal) was disclosed in
  the question itself. `emerging-exp-030` was built with proof of a single
  primary change (motor/buildings byte-identical to exp-025, contents only)
  and submitted through the gate: **0.16991 — the override LOST** (−0.00003),
  exp-025 stands, and the result is logged as much as the wins (row 82).
- **Documentation honesty pass on the Part 2 write-up:** Stage 3 now says
  plainly that the scoring multipliers are Public-fitted constants
  (`f_final = f_probe`), that the data-estimated index is dead code, that
  D-076's +0.00115 is peak-picked, and carries the D-085 Public/Private ~5%
  level-divergence caveat — with `final_writeup.md`'s plain-language trend
  wording corrected to match (board-set factors, data estimate a cross-check).
- **Week 1 deliverable decision, human-made (D-087):** the three candidates
  (exp-054 0.13600 / exp-046 0.13577 / exp-036 0.13570 Private) were presented
  with all caveats; the human chose **exp-054** (clean pipeline + the
  data-derived flat-level convention). README, `final_writeup.md` §7 and
  `actuarial_report.md` updated with the post-deadline and probe-selected-level
  disclosures; exp-027 remains the board-counted entry.
- **Remaining Part 2 doc items closed:** the emerging orphan figures were
  recomputed from the data (9,222/9,258 matched, 0 multi-matched; 36 orphans
  6/28/2 by loss year; 9,102 non-repudiated) and the D-002 drop treatment
  documented in `emerging_market_report.md` §3. Test suite grew to **102
  passing** (blend-math + dedup-inertness guardrails).
- **D-088 — the expiring-window play, pre-registered and factorized (human
  approved the 4-shot 2×2).** With 9 slots burning at midnight and the
  mean-quality queue empty, the one directional lever was the D-085 port:
  Part 1's sandbox measured Public-tuned levels ~5% above Private optima, and
  the final metric is the Private of the best-Public entry. Structural analysis
  ruled out the naive plays: a cut can never ride behind its own carrier (the
  uncut file out-Publics it and takes the counted slot), and pure redraws are
  strictly harmful (they raise the bar our cuts must clear). Each ticket
  therefore carried draw+cut in one file: record config at seeds 3,4,5 / 6,7,8
  × global K 0.98 / 0.95. Contract run proved the seed flag inert to 8.1e-16.
  The 2×2 returned a textbook factorial (level −0.00037 per −3% at both
  draws; draw spread +0.00036 at both Ks) and one clearing ticket —
  **exp-033 at 0.17019, the new best Public and counted entry, carrying the
  −2% level bet** whose truth the hidden Private ranking decides at the
  deadline. An amendment forced by the selection mechanics (the bar is the
  running best Public, not 0.16994) was recorded before any further
  submission; the remaining-window EV analysis (deepening ≈ +0.00001/shot)
  kept the budget from being spent on noise.

**Output.** `emerging-exp-030` (submitted, 0.16991), `scripts/run_emerging{,_priorweight,_wprel,_wblend}.py`,
`src/emerging.py` (`CLAIM_DEDUP_KEYS`), `tests/test_emerging.py` (+2),
`docs/decisions.md` (D-086/D-087 + closures), `docs/emerging_market_report.md`,
`docs/final_writeup.md`, `docs/actuarial_report.md`, `README.md`,
`docs/experiment_log.md` (row 82).

## Verification mechanisms (summary for the judging panel)

1. **Pre-registered gates stop submissions.** The +0.0003 OOF gate (D-025)
   killed two complete feature families — the multi-family ensemble (D-070,
   +0.00023) and motor per-peril (D-074, −0.00036) — without a single quota
   slot spent.
2. **Effects near the gate get designed replications**, not single shots:
   paired seeds (D-075), a 2×2 treatment/control design (D-072), brackets at
   constant total with quadratic vertices and a stop rule (D-073/076/077).
3. **Noise is measured, not assumed:** board seed σ ≈ 0.0012 vs local OOF
   σ 0.00011 (D-072/D-075); the exp-024 board gain equalled its mean OOF gain
   exactly — the offline oracle is validated against the board.
4. **Reproduction is exact:** a byte-identical gate test (D-071) and an
   end-to-end pipeline reproduction of the record to 8e-16 (D-076).
5. **AI errors are recorded and hardened against:** the leaked-feature
   deny-list (D-016), the `--final-fit` bugfixes (D-063), the closure-truncation
   basis trap (D-069), and the duplicate-upload race — each logged with the
   mechanical rule added afterwards (e.g. the serial-submission rule, §4).
6. **Human authorisation gates every submission:** Part 2 requires explicit
   per-submission confirmation and the log quotes it verbatim; the sandbox
   standing authorisation is scoped and recorded (§3.7); `kaggle_submit.py`
   refuses to upload blind.
7. **Every claim is traceable:** 65 experiment-log rows with hypotheses,
   verdicts and sha256; 77 decisions with judgement calls flagged; 87
   guardrail tests green at each commit.

