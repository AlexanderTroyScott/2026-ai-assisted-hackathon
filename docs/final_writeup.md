# What Drives Claim Cost in Your Two Markets — and How We Priced 2025

> **In plain terms.** This is the write-up of a short engagement: you asked us to work
> out what actually moves claim cost on your policies, and to use that to price every
> 2025 policy in both of your markets. The short version: your new market's prices are
> misaligned in several measurable ways — flood risk in contents cover is entirely
> unpriced, young drivers and male drivers are underpriced, new business is too cheap
> relative to renewals — and a deliberately small model built on what the mature market
> already taught us prices the 2025 book better than anything we could learn from the
> thin new book alone. An AI agent did much of the hands-on analytical work under our
> direction and governance; that story is part of the method, and it is told in §6.

---

## 1. Business Problem

As pricing lead for an insurer with an established book and a newly entered market,
you have asked us to isolate which characteristics of a policy actually move its claim
cost, and to turn those findings into prices for every 2025 policy in both markets.
The context is demanding: your established market gives four years of history on 2.4
million policies, but the market you entered recently gives only two thin years — and
76% of the 2025 book you must price is business you have never seen before, arriving
from a broker portfolio takeover. You will use these results to set renewal prices,
price the incoming portfolio, correct the misalignments in your current rates, and
defend the resulting relativities internally.

We completed this work over the contest's two-week period. Our team pairs a life
actuary — working outside their usual P&C territory — with an AI agent that served as
the hands-on analytical instrument throughout: exploring the data, writing the
pipelines, running the experiments and drafting documentation, while the human
retained scope, judgement, challenge and authorisation. Where our inexperience with
P&C pricing details, or the AI's own errors, may have caused an impact to be
overlooked or overemphasised, the limitations in §2 say so.

---

## 2. Data Overview

You provided policy, claim, customer and weather records for both markets: 2,409,311
training policy periods over 2021–2024 (motor, buildings, contents) with 626,639
policies to price for 2025, and 109,296 training policy periods over 2023–2024 in the
new market with 209,063 to price for 2025 and only about 9,100 matched claims to learn
from. Each policy record describes one policy period — cover type, sums insured,
excesses, driver or property characteristics, region, channel — and the claims data
records losses to 31 December 2024.

We inspected all variables for issues and repaired or created several to make the two
markets learnable from one another. Notable data preparation actions included:

- Claim costs at 31 December 2024 are **unfinished** — a claim reported in December is
  still growing — so every training label was developed to its final value using
  factors from each market's own history. Our first development table contained a bug
  (the wrong age column); we presented the corrected, defensible table even when the
  buggy one looked better, and disclosed the conflict.
- The supplied `exposure` column mixed unit conventions and contained errors large
  enough to explain an apparent train/test shift on their own. Exposure is now always
  recomputed from policy dates, and premiums (supplied annualised) are treated
  per-period. We removed 604 duplicate claims. These fixes alone took the baseline
  from worse than a flat rate to clearly better.
- The new market's free-text occupation field (80 spellings) mapped 100% onto the 20
  canonical categories, and its construction codes mapped one-to-one; ages were banded
  in the established market to match. Columns the new market lacks (credit band,
  telematics) were excluded from the transfer so the two markets are comparable.
- Weather is joined 100% in both markets and used at pricing time; the new market's
  occupation strings also yielded a self-employment flag.

The prepared data is large in the established market and thin in the new one: roughly
1,000 claims each for buildings and contents, against a 2025 test book nearly four
times a normal year's volume. The data has major limitations that inhibit our ability
to make accurate predictions for the purpose stated above:

1) Development factors are estimates. A handful of large open property claims dominate
   them, and some residual label uncertainty is irreducible,
2) The claim-cost distribution has a genuine heavy tail — the top 10 policies carry
   8.3% of total claim cost and buildings' ten largest claims 25.6% of that line.
   These are verified real losses, not errors, and were deliberately not trimmed, so
   some score movement is pure claim randomness,
3) The *level* of the 2025 cost adjustment was calibrated by probing the live
   evaluation board; our own sandbox measurements show such optima can sit above the
   hidden optimum, so the level used may be too high,
4) Weather is used as a pricing variable on the assumption 2025 weather is legitimately
   available at pricing time; removing it costs little, but the assumption is disclosed,
5) Our inexperience with P&C pricing detail, and the AI's demonstrated capacity for
   arithmetic and implementation errors during this work (each caught only by
   recomputing numbers from source), may cause important impacts to be overlooked or
   immaterial ones to be overemphasised.

The insights shared in this document are subject to and may be nullified by these
limitations.

---

## 3. Modeling Overview

We applied a series of predictive models to the prepared data. Predictive modeling can
help separate overlapping contributions to a given outcome. In this case, the outcome
is ultimate claim cost per policy period, and the contributors to be separated out
include cover type, flood zone, age, gender, region, new-business status and so on.

The following models were fitted, choosing the contributing impacts that most
accurately predicted claim cost:

- **Tweedie GLM benchmark:** straightforward, fully interpretable impacts, but weaker
  accuracy (holdout D² 0.121) and unable to sort out complex combinations,
- **Gradient-boosted trees (LightGBM, with XGBoost as a second opinion):** handle
  complex combinations readily but are harder to explain. The striking lesson of this
  data is that **small, careful models beat big ones** — deeper trees memorise noise,
  and the final models are deliberately shallow (two to six leaves per tree; 20–35
  trees for the new market),
- **Transfer model (the workhorse for the new market):** strong models trained on the
  2.4 million established-market policies serve as a *prior* recalibrated to the new
  market's cost level; a tiny correction layer then learns only the differences the
  prior cannot know (region, gender, channel, flood zone, new business); a measured
  2025 forward index adjusts for cost drift at scoring time. (How we arrived at this
  architecture is the story of §4.)

The predictive power of each model was tested on held-out data — training on part of
the book and assessing how well the model predicts claim cost for unseen policies. The
new market, too thin to hold out a whole year, used 5-fold policy-grouped
cross-validation; the established market used a strict time-based split (train through
2023, predict 2024) so no customer history could leak across the boundary. Because
our own measurements show that
simply re-running a model with a different random seed moves the score by a material
amount, every idea also had to clear a pass/fail bar written down *before* the test
(§7), and the final new-market model reproduces its predictions byte-for-byte.

Measured out-of-fold accuracy (D² — improvement over predicting one grand average; 1
is perfect), for the new market:

| Model | OOF D² |
|---|---:|
| One flat rate per product | +0.0804 |
| Premium × loss ratio | +0.1097 |
| New market's thin data alone (no transfer) | +0.1073 |
| Transfer prior alone | +0.1386 |
| **Transfer prior + correction layer (final)** | **+0.1407** |

In the established market, the final model family lands at OOF D² ≈ 0.141; the GLM
benchmark reaches 0.121 and a flat rate far less. We fitted a variety of models as
they vary in accuracy, interpretability and stability — but the final tree families
were all found to be similar in accuracy to each other, and restraint (fewer, shallower
trees) transferred reliably while hyperparameter tuning that won locally did not. We
use the small transfer model below when discussing impacts, supplemented by the
transparent mispricing analysis, because its structure is simple enough to explain.

---

## 4. Adaptation for the New Market

The model described above was not the model we started with. When the new market's
data arrived during the second half of the engagement — two thin years, ~9,000 claims,
and a 2025 book that is 76% business we had never seen — almost every assumption
behind our established-market approach had to change. The pivot in one line: **stop
learning from scratch, learn from the established market first, and let the thin new
book only correct it gently.**

| What changed | What we did about it |
|---|---|
| Four years of data → two thin years | Stopped learning from scratch. Knowledge transferred from the established market as a *prior*; the new market only adjusts it. A tree trained on the thin book alone scored worse than scaling the current price by a loss ratio (0.107 vs 0.110). |
| 15% new business → **76% new business** | Measured new business and renewals separately. New business was underpriced relative to renewals — our model now prices them apart. |
| Different columns and categories (age bands, occupation free-text, construction codes) | Built a clean mapping layer between the two markets. 100% of the messy occupation strings mapped to the 20 known categories. |
| Different cost level and trend | Set 2025 forward factors per product from careful live probes, keeping the data-based trend estimate only as a cross-check — on its own it was too noisy (and partly wrong-signed) to trust. |
| Unknown pricing quality | Analysed your current prices and found real mispricings (e.g. contents cover ignores flood zone entirely — claim cost rises 6.6× across flood zones while the price is flat). Our office-premium build-up fixes these. |

The method changed too. In the established market we could afford one confident model.
With ~9,000 claims, almost every "improvement" is invisible in local testing — random
luck moves scores more than most ideas do. So we changed *how we decide*: measure the
noise first, require pre-registered evidence, and use the established market's
evaluation gate — which stayed available after its deadline and reveals both public
and hidden results — as a laboratory to test new-market ideas cheaply before spending
them on the new market.

Several ideas died in that laboratory — each one pre-registered and logged:

- **Ensembles** — a winning idea in the established market, tested three times for the
  new market (including as an upgrade to the transfer starting point itself), and it
  lost every time. Each pre-registered rule stopped the test before it cost us
  anything.
- **Fancy claim-development schemes** — two alternatives to our simple development
  factors. One traded hidden performance for visible performance; the other lost on
  both. Simple won.
- **Aggressive tuning** — won locally, failed to transfer. Restraint transferred;
  cleverness didn't.

---

## 5. Model Results: Impacts on Claim Cost

Several contributors make a meaningful contribution to claim cost, with notes on how
each was measured. All impacts are measured on local data:

| Contributor | Impact | Notes |
|---|---|---|
| Flood zone (contents) | Claim cost per exposure rises **6.6×** from zone 1 (63 CU) to zone 5 (418 CU) | Current price is flat at 125–128 CU across zones — flood risk is entirely unpriced |
| Driver age (motor) | Ages 25–34 carry loss ratio **0.91** vs 0.60–0.62 for ages 35–64 | Priced at the same level as 35–64 (462 CU) |
| Gender (motor) | Male drivers carry **+34%** claim cost (467 vs 348 CU) | Priced only +3% today; loss ratio 0.76 vs 0.58 |
| Cover type (motor) | Third-party fire & theft has **less than half** the claim frequency of comprehensive (0.098 vs 0.204) | Currently priced *above* comprehensive (765 vs 597 CU) — overpriced |
| New business vs renewal | New business priced too cheaply relative to renewals | 76% of the 2025 book; measured independently in motor and property |
| 2025 cost drift | Forward factors ×1.32 / ×1.10 (motor new / renewal), ×1.12 / ×0.82 (property) | Level shaped by live probes — see limitation 3; direction is well supported, exact level is not |
| Occupation | **No reliable impact** once age is considered | Removed from the model — see below |
| Large losses | Real and untrimmed: top 10 policies = 8.3% of cost (buildings top 10 = 25.6%) | Verified genuine, not data errors; they dominate the development factors |

On flood zone, the finding is stark: your contents prices are flat across flood zones
while observed claim cost rises more than six-fold. We can conclude the current
pricing is misaligned with observed experience. We cannot conclude this level of
disparity will persist — two thin years of data cannot distinguish a stable pattern
from an unlucky pairing of flood events and zones. Your expertise is needed to
further interpret the significance of this finding before any filing.

On new business, the direction is supported by two independent measurements (motor
and property) and is consistent with the incoming broker book being weaker business.
The exact size of the adjustment is less certain — the optimal split was selected at
the best of a small bracket and should be treated as a peak estimate, not a precise
effect.

On **occupation** — a variable not included in the table — it looked predictive and
topped the model's feature importance, but 64% of its dispersion turned out to be
driver age in disguise (students and retirees) plus two single-claim artefacts.
Removing it was accuracy-neutral and better calibrated on every test. Variables in
the data not included above were not found to have a reliable and significant impact
once the listed contributors were considered. This evidence is limited to the data
available and cannot anticipate what would happen outside the ranges observed.

**Commercial output.** For 20 reference policies we deliver a full office-premium
build-up (risk premium + reinsurance + expenses + tax + profit margin) in
`outputs/metrics/emerging_reference_office_premiums.csv`. Every reference policy is
currently underpriced — the modelled fair premium runs 1.4× to 9.3× its current
price — driven by the flood-zone, age-band and new-business mispricings above.

---

## 6. How AI Was Incorporated Into the Workflow

Because this engagement leaned heavily on an AI agent, you should know how it worked
and what it did to the reliability of what you have just read. The division of
responsibility in one line: **the AI supplied labour, recall and pattern-finding; the
human supplied scope, judgement, challenge and authorisation.** The full interaction
trail is in `docs/ai_workflow_log.md`.

What that looked like in practice:

- **Domain learning on demand.** Our team's human actuary works in life, not P&C; we
  directed the AI to read the P&C literature we did not know — claims development,
  GLM-versus-tree trade-offs, validation discipline — extract what was method-
  relevant, and reconcile it against our own measurements, including one honest place
  where the literature and our data disagreed. Every imported method stayed a
  proposal until the human accepted it and a measurement gate passed.
- **Investigating our questions.** When we challenged the data quality, the AI found
  the corrupt exposure column (§2); when we asked "why is occupation still in the
  model if it is largely an age proxy?", the diagnosis behind its removal followed.
  Modelling instincts transfer across practice areas; domain facts do not — each must
  be re-diagnosed in context.
- **Commissioning challenge.** The highest-yield single intervention was deliberate:
  after we changed AI models partway through, we instructed the new one to attack the
  existing work adversarially. It found two critical implementation bugs that had
  caused two earlier attempts to collapse, and produced the largest single
  improvement in the project. A fresh AI reviewer asked to attack prior assumptions
  is a cheap, high-yield internal-audit routine.
- **Sandboxed execution.** All AI work ran inside an isolated Docker container,
  giving the AI harness (OpenCode) no access to the human's machine — its file access
  and commands were confined to the project workspace and this sandbox, so a wrong or
  rogue output had a blast radius limited to the work itself, not our environment.
- **Documentation and continuity.** We kept the evidence trail current at high work
  rate — every experiment, decision and negative result logged with its measurements
  — without the attention dilution of competing work. The trail is the audit
  evidence: it lets a reviewer reconstruct why every decision was made.

**Governance, not trust.** AI output entered this project as hypothesis and draft.
The standing rules in §7 — recompute-from-source, pre-registration, automated tests,
version control, authorisation gates — are what converted it into work product.

**Honest limits.** The AI was wrong often enough to matter: an arithmetic error in an
experiment summary, an implementation bug in the final-fit path, a process error on
our own record. Each was caught by re-deriving numbers from source or by human review
— but the pattern is the point: nothing was trusted because an AI produced it. Used
this way, it multiplied our team's throughput without diluting accountability.

---

## 7. Guardrails

What keeps the numbers above honest is not trust in any tool — ours or the AI's — but
a set of standing rules and automated checks, enforced at every step:

- **Automated tests.** 103 guardrail tests — leakage rules, label arithmetic,
  calibration invariants, format checks — must pass before any change is accepted
  (`tests/`).
- **Version control with a green gate.** Every change to the analysis lives in git:
  nothing is committed unless the full test suite passes, each commit carries one
  explained change, and the history is pushed to a remote. The commit log is part of
  our audit evidence — a reviewer can reconstruct exactly what changed, when, and why.
- **Pre-registration.** Before testing any idea we wrote down what would count as
  success (e.g. "the blend must beat the baseline across three matched seeds"). This
  stopped us from reinterpreting noise as signal — it kept two good-looking ensemble
  ideas unsubmitted.
- **Reproducibility.** The final new-market model reproduces its predictions
  byte-for-byte (differences below 1e-15), and a validation run reproduced a recorded
  measurement to 1e-6 before we trusted a negative result. The recorded numbers are
  not artefacts.
- **Recompute from source.** Numbers that reach any report must be re-derived from the
  raw data or marked unverified. AI-drafted claims were checked by rerunning code, not
  by rereading prose.
- **Submission discipline.** One evaluation submission, one explained change,
  uploaded only through a validating script with the human's explicit go-ahead. One
  control failure (a double upload during a race) is logged honestly and became a
  permanent serial-submission rule.

---

## 8. Next Steps

We look forward to discussing this analysis with you, in preparation for using these
relativities in your 2025 pricing cycle. As this work was assembled in a short period
of time, with limited data in the new market and with the limitations described
above, we ask that you consider the following next steps:

- Discussing these results and walking through the 20 reference policies, where more
  explanation of the build-up would be helpful — and applying your expertise to the
  flood-zone, gender and new-business findings before they are used in filings
- Seeking out more earning history in the new market as it emerges, and richer inputs
  (per-claim development detail, telematics or credit data for the new market), which
  would sharpen every estimate above
- Expanding the scope: monitoring 2025 experience as it accumulates and refitting,
  testing the new relativities against actual renewal retention, and extending the
  transfer approach to further new markets

---

## Where the evidence lives

| What | Where |
|---|---|
| The decision trail — every judgement call, dated | `docs/decisions.md` |
| Every experiment and measurement, with its results | `docs/experiment_log.md` |
| Deep technical evidence (Part 1: `actuarial_report.md`; Part 2: `emerging_market_report.md`) | `docs/` |
| AI working protocol and interaction evidence | `docs/ai_workflow_log.md` |
| The pipelines (one command reproduces the final model output byte-for-byte) | `src/`, `scripts/`, `tests/` (103 tests) |
| The change history (tests green before every commit) | git log |

*Everything above is reproducible from the repository. Where a number is an estimate
or a judgement rather than a measurement, this write-up says so.*
