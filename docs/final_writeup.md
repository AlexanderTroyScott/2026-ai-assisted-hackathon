# What Drives Claim Cost in Your Two Markets — and How We Priced 2025

> **In plain terms.** This is the write-up of a short engagement: you asked me to work
> out what actually moves claim cost on your policies, and to use that to price every
> 2025 policy in both of your markets. The short version: your new market's prices are
> misaligned in several measurable ways — flood risk in contents cover is entirely
> unpriced, young drivers and male drivers are underpriced, new business is too cheap
> relative to renewals — and a deliberately small model built on what the mature market
> already taught us prices the 2025 book better than anything we could learn from the
> thin new book alone. An AI agent did the hands-on analytical work under my direction
> and governance; that story is part of the method, and it is told in §5.

---

## 1. Business Problem

As pricing lead for an insurer with an established book and a newly entered market,
you have asked me to isolate which characteristics of a policy actually move its claim
cost, and to turn those findings into prices for every 2025 policy in both markets.
The context is demanding: your established market gives four years of history on 2.4
million policies, but the market you entered recently gives only two thin years — and
76% of the 2025 book you must price is business you have never seen before, arriving
from a broker portfolio takeover. You will use these results to set renewal prices,
price the incoming portfolio, correct the misalignments in your current rates, and
defend the resulting relativities internally.

This work was done over about ten days. I am a life actuary working outside my usual
P&C territory; an AI agent served as my hands-on analytical instrument throughout —
exploring the data, writing the pipelines, running the experiments, and drafting
documentation, while I retained scope, judgement, challenge and authorisation. Where
my inexperience with P&C pricing details, or the AI's own errors, may have caused an
impact to be overlooked or overemphasised, the limitations in §2 say so.

---

## 2. Data Overview

You provided policy, claim, customer and weather records for both markets: 2,409,311
training policy periods over 2021–2024 (motor, buildings, contents) with 626,639
policies to price for 2025, and 109,296 training policy periods over 2023–2024 in the
new market with 209,063 to price for 2025 and only about 9,100 matched claims to learn
from. Each policy record describes one policy period — cover type, sums insured,
excesses, driver or property characteristics, region, channel — and the claims data
records losses to 31 December 2024.

All variables were inspected for issues and several were repaired or created to make
the two markets learnable from one another. Notable data preparation actions included:

- Claim costs at 31 December 2024 are **unfinished** — a claim reported in December is
  still growing — so every training label was developed to its final value using
  factors from each market's own history. Our first development table contained a bug
  (the wrong age column); the corrected, defensible table was presented even when the
  buggy one looked better, and the conflict was disclosed.
- The supplied `exposure` column mixed unit conventions and contained errors large
  enough to explain an apparent train/test shift on their own. Exposure is now always
  recomputed from policy dates, and premiums (supplied annualised) are treated
  per-period. 604 duplicate claims were removed. These fixes alone took the baseline
  from worse than a flat rate to clearly better.
- The new market's free-text occupation field (80 spellings) mapped 100% onto the 20
  canonical categories, and its construction codes mapped one-to-one; ages were banded
  in the established market to match. Columns the new market lacks (credit band,
  telematics) were excluded from the transfer so the two markets are comparable.
- Weather is joined 100% in both markets and used at pricing time; the new market's
  occupation strings also yielded a self-employment flag.

The prepared data is large in the established market and thin in the new one: roughly
1,000 claims each for buildings and contents, against a 2025 test book nearly four
times a normal year's volume. The data has major limitations that inhibit my ability
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
5) My inexperience with P&C pricing detail, and the AI's demonstrated capacity for
   arithmetic and implementation errors during this work (each caught only by
   recomputing numbers from source), may cause important impacts to be overlooked or
   immaterial ones to be overemphasised.

The insights shared in this document are subject to and may be nullified by these
limitations.

---

## 3. Modeling Overview

I applied a series of predictive models to the prepared data. Predictive modeling can
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
  2025 forward index adjusts for cost drift at scoring time.

The predictive power of each model was tested with 5-fold policy-grouped cross-
validation on held-out data — training on part of the book and assessing how well the
model predicts claim cost for unseen policies. Because our own measurements show that
simply re-running a model with a different random seed moves the score by a material
amount, every idea also had to clear a pass/fail bar written down *before* the test,
and the final new-market model reproduces its predictions byte-for-byte.

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
benchmark reaches 0.121 and a flat rate far less. A variety of models was fit as they
vary in accuracy, interpretability and stability — but the final tree families were
all found to be similar in accuracy to each other, and restraint (fewer, shallower
trees) transferred reliably while hyperparameter tuning that won locally did not. The
small transfer model is used below when discussing impacts, supplemented by the
transparent mispricing analysis, because its structure is simple enough to explain.

---

## 4. Model Results: Impacts on Claim Cost

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
currently underpriced — the modelled fair premium runs 1.3× to 9.5× its current
price — driven by the flood-zone, age-band and new-business mispricings above.

---

## 5. How AI Was Incorporated Into the Workflow

Because this engagement leaned heavily on an AI agent, you should know how it worked
and what it did to the reliability of what you have just read. The division of
responsibility in one line: **the AI supplied labour, recall and pattern-finding; I
supplied scope, judgement, challenge and authorisation.** The full interaction trail
is in `docs/ai_workflow_log.md`.

What that looked like in practice:

- **Domain learning on demand.** As a life actuary, I directed the AI to read the P&C
  literature I did not know — claims development, GLM-versus-tree trade-offs,
  validation discipline — extract what was method-relevant, and reconcile it against
  our own measurements, including one honest place where the literature and our data
  disagreed. Every imported method stayed a proposal until I accepted it and a
  measurement gate passed.
- **Investigating my questions.** My challenge about data quality led the AI to find
  the corrupt exposure column (§2); my challenge "why is occupation still in the
  model if it is largely an age proxy?" produced the diagnosis behind its removal.
  Modelling instincts transfer across practice areas; domain facts do not — each must
  be re-diagnosed in context.
- **Commissioning challenge.** The highest-yield single intervention was deliberate:
  after changing AI models partway through, I instructed the new one to attack the
  existing work adversarially. It found two critical implementation bugs that had
  caused two earlier attempts to collapse, and produced the largest single
  improvement in the project. A fresh AI reviewer asked to attack prior assumptions
  is a cheap, high-yield internal-audit routine.
- **Sandboxed execution.** All AI work ran inside an isolated Docker container, giving
  the AI harness (OpenCode) no access to my machine — its file access and commands
  were confined to the project workspace and this sandbox, so a wrong or rogue output
  had a blast radius limited to the work itself, not my environment.
- **Governance, not trust.** Every number in this document was recomputed from source
  data before it could appear — AI-drafted claims were checked by rerunning code, not
  by rereading prose. Ideas were pre-registered with pass/fail bars before testing;
  negative results (about half the ideas tested) were kept on the record; no
  evaluation submission could be uploaded without my explicit go-ahead, one explained
  change at a time; and 103 automated guardrail tests — leakage rules, label
  arithmetic, calibration invariants, format checks — ran before every change was
  accepted. One control failure (a double upload during a race) is logged honestly
  and became a permanent serial-submission rule.

**Honest limits.** The AI was wrong often enough to matter: an arithmetic error in an
experiment summary, an implementation bug in the final-fit path, a process error on
our own record. Each was caught by re-deriving numbers from source or by human review
— but the pattern is the point: AI output entered this project as hypothesis and
draft, and the verification steps above are what converted it into work product. Used
that way, it multiplied one actuary's throughput without diluting accountability.

---

## 6. Next Steps

I look forward to discussing this analysis with you, in preparation for using these
relativities in your 2025 pricing cycle. As this work was assembled in a short period
of time, with limited data in the new market and with the limitations described
above, I ask that you consider the following next steps:

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
| Deep technical evidence (Part 1: `actuarial_report.md`; Part 2: `emerging_market_report.md`) | `docs/` |
| AI working protocol and interaction evidence | `docs/ai_workflow_log.md` |
| The pipelines (one command reproduces the final model output byte-for-byte) | `src/`, `scripts/`, `tests/` (103 tests) |

*Everything above is reproducible from the repository. Where a number is an estimate
or a judgement rather than a measurement, this write-up says so.*
