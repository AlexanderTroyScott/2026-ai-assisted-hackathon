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
at the data in `data_raw/`; give suggestions.

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

**Verified.** All figures recomputed directly from `data_raw/` with pandas; no
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
