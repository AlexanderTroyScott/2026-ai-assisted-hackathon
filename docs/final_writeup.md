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
with our counted entry; our best Part 1 model, built after the deadline, scores
notably better (0.13600 vs 0.12984 on the hidden board) but is unranked. Details and
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
3. **Adjust for 2025.** Prices and costs drift. We set a forward factor per
   product and customer type (new business vs renewal) and apply it at scoring
   time. The data-based trend estimate proved too noisy to trust on its own, so
   the final factors come from small, careful probes of the live leaderboard,
   with the data estimate kept as a cross-check. This step was worth the
   largest gains of Part 2: the 2025 adjustment alone moved the score from
   0.1630 to 0.1684, and the new-business/renewal split added a further
   +0.00115 (measured on the public board, at the best of a small bracket —
   treat it as a peak estimate, not a precise effect).

---

## 3. How our solution changed when the new market data arrived

This is the heart of the adaptation story, so it gets its own section.

| What changed | What we did about it |
|---|---|
| 4 years of data → 2 thin years | Stopped learning from scratch. Knowledge transferred from Part 1; the new market only adjusts it. |
| 15% new business → **76% new business** | Measured new business and renewals separately. New business was underpriced relative to renewals — our model now prices them apart. |
| Different columns and categories (age bands, occupation free-text, construction codes) | Built a clean mapping layer between the two markets. 100% of messy occupation strings mapped to the 20 known categories. |
| Different cost level and trend | Set 2025 forward factors per product from careful leaderboard probes, keeping the data-based trend estimate only as a cross-check — on its own it was too noisy (and partly wrong-signed) to trust. |
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

## 5. How we used AI — supplementing the actuarial control cycle

An AI agent did the hands-on analytical work — exploring data, writing pipeline
code, running experiments, computing diagnostics, and drafting this
documentation — under a strictly human-governed protocol. The human set the
scope, judged the ambiguous calls, and held the keys to every submission. This
section shows where an AI tool fits *inside the actuarial control cycle* and
how using it exercised the management responsibilities professional practice
places on the actuary. The division of responsibility in one line: **the AI
supplied labour, recall and pattern-finding; the human supplied scope,
judgement, challenge and authorisation.**

### Where AI supplemented the control cycle

- **Design and scoping — delegation with an approval gate.** The human
  delegated the drafting of the project outline (a phased plan: scaffold →
  data quality → target → development → inflation → features → models →
  guardrails → submissions) and approved it, setting submission and
  documentation policy on top. The outline was a draft for the principal to
  accept, not a decision taken out of the principal's hands.
- **Domain learning on demand.** The human is a life actuary; this competition
  is P&C pricing. The human directed the AI to read the P&C actuarial
  literature the human did not know — Friedland on unpaid-claims estimation,
  two CAS GLM guides, a CAS model-validation study note — extract the
  method-relevant sections, and reconcile them against our own measurements,
  including one honest place where the literature and our data disagreed
  (frequency–severity versus a single Tweedie). The distilled result
  (`docs/references.md`, the `actuarial-methods` skill) let the human direct a
  P&C method chain — claims development, GLM-versus-tree trade-offs, holdout
  discipline, ASOP 9-style documentation — while every imported method stayed
  `proposed` until the human accepted it and a measurement gate passed.
- **Investigating the human's questions.** The human questioned the data
  quality; the AI's investigation of those questions found that the supplied
  `exposure` column mixed three conventions and contained unit errors — an
  inconsistency large enough to explain an apparent train/test exposure shift
  on its own (D-001). Every model since recomputes exposure from dates, and a
  guardrail test enforces it.
- **Cross-practice modelling lessons.** `occupation` topped the tree model's
  feature importance. In life-style segmentation that would be a familiar
  rating factor; here the human's challenge ("why is occupation still in the
  model if it is largely an age proxy?") produced the diagnosis: 64% of its
  dispersion was Student/Retired — driver age in disguise — plus two
  single-claim artefacts. Its measured removal was accuracy-neutral but better
  calibrated on both folds, confirmed on the leaderboard (D-056). Modelling
  instincts transfer across practice areas; domain facts do not — each must be
  re-diagnosed in context.
- **Commissioning independent review.** The highest-yield intervention was a
  deliberate one: after changing AI models, the human instructed the new one to
  perform a thorough, adversarial review of the existing work and to challenge
  prior assumptions. It found two critical implementation bugs in the
  final-fit path that had caused two earlier submissions to collapse, and
  produced the breakthrough in a single step: from `exp-031`, the last
  pre-review submission (**0.12928 Private**), to `exp-033`
  (**0.13531 Private**) — a chain of work that ran on to the best file of the
  line, `exp-054` at **0.13600 Private** (submitted after the Part 1 deadline
  and therefore unranked). For scale: the entry counted at the deadline
  (`exp-027`) stood at 0.12984 Private. The lesson for practice: an AI
  reviewer is only as good as its commission — asking a fresh model to attack
  the work is a cheap, high-yield internal-audit routine.
- **Documentation and continuity.** The AI kept the evidence trail current at
  high work rate — every experiment, decision, pre-registration and negative
  result logged with scores and file hashes — without the attention dilution
  or context-switching cost of competing work. The trail is the audit
  evidence: it is what lets a reviewer reconstruct why every decision was made.

### What this exercised in management responsibility

Working through this protocol developed exactly the management duties
professional practice puts on the actuary in charge:

- **Delegation with acceptance criteria** — drafting and analysis delegated;
  approval, scope and standards retained.
- **Authorisation gates** — no leaderboard submission could be uploaded
  without the human's explicit confirmation; one submission carried one
  explained change, so every score movement has one cause.
- **Commissioning challenge** — adversarial reviews and an independent
  reviewer subagent were standing routines, not one-offs.
- **Escalation discipline** — judgement calls stayed `proposed` until the
  human accepted them; the AI flagged rather than decided.
- **Ownership of risk** — the human personally made the final risk calls. In
  the closing days an AI-prepared analysis supported a disclosed price-level
  bet; once its weakest link was made explicit (the evidence for its direction
  rested on one past measurement of an unobservable property), the human
  withdrew the bet on risk preference (D-088/D-089). The analysis informed
  the decision; it did not make it.
- **Incident management** — one run accidentally uploaded a submission twice
  (two parallel calls raced). The duplicate scored identically, but it was
  treated as a real control failure: logged, and a serial-submission rule
  added to the protocol the same day.

### Governed, not "AI did stuff"

- **Everything recomputed from source.** Numbers that reach the reports must
  be re-derived from the raw data or marked unverified. AI-drafted claims were
  checked by rerunning code, not by rereading prose.
- **Pre-registration.** Before testing an idea we wrote down what would count
  as success (e.g. "the blend must beat the baseline by 0.0002 across three
  matched seeds"). This stopped us from reinterpreting noise as signal — it
  killed two ensemble ideas that looked good after the fact.
- **Negative results kept.** Roughly half of the ideas failed. They are logged
  with their measurements. They are the reason the final model is small and
  stable.
- **Contract checks and automated tests.** The final Part 2 model reproduces
  its scored predictions **byte-for-byte** (differences below 1e-15), and a
  validation run reproduced its recorded score to 1e-6 before we trusted a
  negative result. The test suite has **103 automated guardrails** — leakage
  rules, label invariants, calibration invariants, submission format checks.

### Honest limits

The AI was wrong often enough to matter: an arithmetic error in one
experiment's summary, a process error on our own Part 1 record, and the bugs
above — each caught by re-deriving numbers from source and by human review.
AI output entered this project as hypothesis and draft; the actuarial control
cycle is what converted it into work product. Used that way, it multiplied one
actuary's throughput without diluting accountability.

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
  score (we observed exactly that with two label experiments). The largest such
  element is the *level* of the 2025 forward factors: in the Part 1 sandbox, where
  both boards are visible, the public and hidden optima sat about 5% apart in level,
  with public tuning landing on the high side. Part 2's hidden score is not visible
  to us, so our public-fitted level may sit similarly high — the adjustments are
  otherwise shaped like measured patterns, and all of this is disclosed here.
- **Leaderboard noise is real.** Score differences below ~0.0013 are luck at this
  game's sample size. We say so even where it weakens our story.

---

## 7. Results, and one disclosure

**Part 2 (emerging market) — the live competition.** Our best entry
`emerging-exp-035`: **0.17028, 4th of 33 teams**, reaching within 0.0014 of the
podium. The chain of measured improvements: 0.1630 (transfer baseline) → 0.1684
(2025 forward index) → 0.1688 (model restraint) → 0.1699 (new-business/renewal
pricing split) → 0.17028 (re-rolled training seeds; the model is unchanged).
We also ran a disclosed price-level experiment (D-088/D-089): a −2% level
ticket gained +0.00025 on the public board and briefly held the top entry, on
the Part 1 laboratory evidence that public-tuned levels sit above hidden-board
optima. On reflection that direction rests on a single past measurement of an
unobservable split property, so the bet was withdrawn the same day by decision
and the counted entry runs at the plain record level — the experiment and its
2×2 measurement trail stand in the log.

**Part 1 (developed market).** The entry counted by the leaderboard is
`exp-027` (0.12436 public → **0.12984 private**, 41st of 54). **Disclosure:** our
best Part 1 model, `exp-054` — a blend of LightGBM and XGBoost per product on the
leak-free pipeline, calibrated to the data-derived flat-level convention — scores
**0.12700 public / 0.13600 private** and is what we attach as our Week 1 prediction
file. It was built and submitted *after* Part 1's deadline (the scoring gate was
still open), so the leaderboard does not rank it. Its level convention was selected
with evidence from the same open gate (seven one-change level probes bracketed the
hidden-board optimum), so the exact score is a measured-in-sample estimate — the
actuarial conclusion (2025 claim costs run flat vs 2024, so no forward step on top
of the fitted level) stands on independent evidence regardless. We present the file
with those facts attached, not hidden.

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
| Automated guardrails | `tests/` (100 tests) |

*Everything above is reproducible from the repository. Where a number is an estimate
or a judgement rather than a measurement, this write-up says so.*
