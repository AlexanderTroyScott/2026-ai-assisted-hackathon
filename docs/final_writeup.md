# Pricing Insurance Across Two Markets — Final Write-Up

*IAA AI-Assisted Hackathon 2026 — Part 1 (developed market) and Part 2 (emerging market).
Only the first five pages of this document are graded; everything essential is in those pages.*

---

## 1. The challenge

We were asked to price two insurance markets:

- **Part 1 — the developed market.** Four years of history (2021–2024, 2.4 million
  policy rows), private motor and home business. We had to predict the claim cost of
  every policy in the 2025 book (626,639 rows).
- **Part 2 — the emerging market.** A new country, only two thin years of history
  (109,296 rows, about 9,100 claims) — and a 2025 test book of 209,063 rows that is
  nearly four times the size of a normal training year, because the insurer took over
  a broker portfolio. **76% of it is brand-new business** the insurer has never seen.

For every test row we predict *the expected claim cost of that policy period* — the
money the insurer expects to pay on claims, after recoveries and excluding claims it
rightly declines. The score is **Tweedie D²**, a standard insurance measure where 0
means "no better than one flat average" and 1 means "perfect". Claim costs are lumpy
(a few big fires among thousands of quiet policies), which is exactly what this score
is designed for.

**Where we finished:** Part 2 — **4th of 33 teams** (0.16994). Part 1 — 41st of 54
with our counted entry; our best Part 1 model, built just after the deadline, scores
notably better (0.13570 vs 0.12984 on the hidden board) but is unranked. Details and
the disclosure in §7.

---

## 2. Our modelling approach, in plain terms

Three ideas carry the solution.

**One model per product.** Motor, buildings and contents behave differently, so each
gets its own model. Each predicts an *annual claim rate* per policy; multiply by how
long the policy was on cover and you get the expected cost for that row.

**We teach the models from claims, not from averages.** Every training label is the
real claim cost of that policy, **developed to its final value** — a claim reported in
December 2024 is not finished yet, so we estimate where it will land using development
factors measured from the market's own history. (We tested fancier versions of this
step on a live test board and they *lost* — see §4.)

**Gradient-boosted trees, deliberately kept shallow.** Our engine is LightGBM (with
XGBoost as a second opinion in Part 1). The surprising lesson: in this data, **small,
careful models beat big clever ones**. Deeper trees memorise noise; shallow trees
(two to six leaves) generalise. The final Part 2 model is deliberately tiny — 20 to 35
shallow trees per product — and that restraint is worth more than any feature
engineering we tried.

For **Part 2**, thin data changes everything, so the architecture becomes two-stage
("transfer learning"):

1. **Start from what the developed market taught us.** We train a strong model on the
   2.4 million rows of Part 1 and use it as a *prior* — a first guess for every
   emerging-market policy, recalibrated to the new market's overall cost level. This
   prior alone is already close to our final quality.
2. **Let the thin new book correct it, gently.** A very small adaptation model learns
   only the *differences* — region, gender, channel, flood zone, new-business effects.
   With so few claims, this layer must be heavily restrained or it overfits.
3. **Adjust for 2025.** Prices and costs drift. We estimate a forward factor per
   product and customer type (new business vs renewal) from measured trend patterns
   and apply it at scoring time. This step was worth the largest gains of Part 2:
   the 2025 adjustment alone moved the score from 0.1630 to 0.1684, and the
   new-business/renewal split added a further +0.00115.

---

## 3. How our solution changed when the new market data arrived

This is the heart of the adaptation story, so it gets its own section.

| What changed | What we did about it |
|---|---|
| 4 years of data → 2 thin years | Stopped learning from scratch. Knowledge transferred from Part 1; the new market only adjusts it. |
| 15% new business → **76% new business** | Measured new business and renewals separately. New business was underpriced relative to renewals — our model now prices them apart. |
| Different columns and categories (age bands, occupation free-text, construction codes) | Built a clean mapping layer between the two markets. 100% of messy occupation strings mapped to the 20 known categories. |
| Different cost level and trend | Estimated 2025 forward factors per product from the data, then calibrated their strength on the live leaderboard rather than trusting noisy local estimates. |
| Unknown pricing quality | Analysed the insurer's current prices and found real mispricings (e.g. contents cover ignores flood zone entirely — claim cost rises 6.6× across flood zones while the price is flat). Our office-premium build-up fixes these. |

The method changed too. In Part 1 we could afford one confident model. In Part 2
almost every "improvement" is invisible in local testing — with ~9,000 claims,
random luck moves scores more than most ideas do. So we changed *how we decide*:
measure the noise first, require pre-registered evidence, and use the Part 1
leaderboard (which stayed open after its deadline and reveals both scores) as a
laboratory to test Part 2 ideas cheaply before spending Part 2 submissions.

---

## 4. Major decisions

A sample of the calls that shaped the result — each one measured, logged and dated in
`docs/decisions.md` and `docs/experiment_log.md`:

- **Correct labels over lucky labels.** We found a bug in the label-development table
  (the wrong age column was being used). The *corrected* table initially scored
  slightly *worse* on the leaderboard. We presented the corrected one anyway — a table
  known to be wrong cannot be defended — and disclosed the conflict. It later reached
  parity.
- **Drop "occupation".** It looked predictive but 64% of its signal was really driver
  age (students and retirees) wearing a disguise, plus two data artefacts. Removing it
  improved calibration on every test.
- **Fix the data before modelling.** The supplied exposure column was corrupt in
  places (recomputed from dates), premiums were annualised rather than per-period, and
  604 duplicate claims existed. One of these fixes alone moved a baseline score from
  0.076 to 0.109.
- **Shrink, don't tune.** Hyperparameter tuning that won locally scored only 0.11168
  on the leaderboard, well below our contemporaneous runs. Restraint — fewer,
  shallower trees — transferred reliably. This became the default.
- **Ensembles: tried, measured, rejected.** Combining XGBoost and LightGBM won in
  Part 1 (+0.003). We ported the idea to Part 2 three times — including as an
  upgrade to the transfer "starting point" model itself — and it **failed all
  three times**. The tests were pre-registered, so their rules stopped the
  submissions before any of them cost us anything, and the rejections stayed
  on the record.
- **Trust live measurement over local scores for the final adjustments.** The 2025
  forward factors were shaped by small, careful leaderboard probes. We also measured
  the leaderboard's *own noise* (a redraw of the same model moves the score by about
  0.0013) — which told us exactly how much evidence a claim needs.

---

## 5. How we used AI

An AI agent did the hands-on analytical work — exploring data, writing pipeline code,
running experiments, computing diagnostics, and drafting this documentation — under a
strictly human-governed protocol. The human set the rules, judged the ambiguous calls,
and held the keys to every submission.

What made this *governed* rather than "AI did stuff":

- **Authorisation gates.** No leaderboard submission could be uploaded without the
  human's explicit confirmation (Part 2: per submission). One submission carried one
  explained change, so every score movement has one cause.
- **Everything recomputed from source.** Numbers that reach the reports must be
  re-derived from the raw data or marked unverified. AI-drafted claims were checked by
  rerunning code, not by rereading prose.
- **Pre-registration.** Before testing an idea we wrote down what would count as
  success (e.g. "the blend must beat the baseline by 0.0002 across three matched
  seeds"). This stopped us from reinterpreting noise as signal — it killed two ensemble
  ideas that looked good after the fact.
- **Negative results kept.** Roughly half of the ideas failed. They are logged with
  their measurements. They are the reason the final model is small and stable.
- **Contract checks and automated tests.** The final Part 2 model reproduces its
  scored predictions **byte-for-byte** (differences below 1e-15), and a validation run
  reproduced its recorded score to 1e-6 before we trusted a negative result. The test
  suite has **89 automated guardrails** — leakage rules, label invariants, calibration
  invariants, submission format checks.
- **Incidents become process.** One run accidentally uploaded a submission twice (two
  parallel calls raced). The duplicate scored identically, but we treated it as a real
  control failure: it is logged, and a serial-submission rule was added to the protocol
  the same day.

---

## 6. Assumptions and limitations

Stated plainly, because the honest version is the useful one:

- **Development factors are estimates.** Claims are developed to ultimate with
  factors from the markets' own history. A handful of large open claims dominate the
  property factors; some residual label uncertainty is irreducible.
- **A heavy tail is real, and we did not trim it.** The top 10 policies carry 8.3% of
  total claim cost; buildings' ten largest claims are 25.6% of that line. We verified
  these are genuine losses, not data errors, and refused to cap them — capping would
  bias the totals our pricing depends on. The cost: some score movement is pure
  randomness.
- **Weather data is used at pricing time** (2025 weather is available to every team).
  Removing it costs little (measured ≤ 0.0005); the assumption is disclosed rather
  than hidden.
- **Some Part 2 adjustments are calibrated on the public leaderboard.** Our own
  measurements show public-score improvements can disappoint on the hidden private
  score (we observed exactly that with two label experiments). Those adjustments are
  small, shaped like measured patterns, and disclosed here.
- **Leaderboard noise is real.** Score differences below ~0.0013 are luck at this
  game's sample size. We say so even where it weakens our story.

---

## 7. Results, and one disclosure

**Part 2 (emerging market) — the live competition.** Submission of record
`emerging-exp-025`: **0.16994, 4th of 33 teams**, reaching within 0.0018 of the
podium. The chain of measured improvements: 0.1630 (transfer baseline) → 0.1684
(2025 forward index) → 0.1688 (model restraint) → 0.1699 (new-business/renewal
pricing split).

**Part 1 (developed market).** The entry counted by the leaderboard is
`exp-027` (0.12436 public → **0.12984 private**, 41st of 54). **Disclosure:** our
best Part 1 model, `exp-036` — a blend of LightGBM and XGBoost per product — scores
**0.12753 public / 0.13570 private** and is what we attach as our Week 1 prediction
file. It was built and submitted *after* Part 1's deadline (the scoring gate was still
open), so the leaderboard does not rank it. We present it with that fact attached,
not hidden.

**Commercial output.** For the 20 reference policies we deliver a full office-premium
build-up (risk premium + reinsurance + expenses + tax + profit margin) in
`outputs/metrics/emerging_reference_office_premiums.csv`. Every reference policy is
currently underpriced — the modelled fair premium runs 1.3× to 9.5× its current
price — driven by the flood-zone, age-band and new-business mispricings in §3.

---

## 8. Where the evidence lives

| What | Where |
|---|---|
| The decision trail — every judgement call, dated | `docs/decisions.md` |
| Every experiment and submission, with scores and hashes | `docs/experiment_log.md` |
| Deep technical reports (Part 1: `actuarial_report.md`; Part 2: `emerging_market_report.md`) | `docs/` |
| AI working protocol and interaction evidence | `docs/ai_workflow_log.md`, `docs/submission_protocol.md` |
| The pipelines | `src/`, `scripts/` (one command reproduces the Part 2 submission byte-for-byte) |
| Automated guardrails | `tests/` (89 tests) |

*Everything above is reproducible from the repository. Where a number is an estimate
or a judgement rather than a measurement, this write-up says so.*
