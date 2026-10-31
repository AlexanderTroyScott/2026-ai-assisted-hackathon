# Video storyboard — Pricing Across Markets (2026 IAA AI-Assisted Hackathon)

Target ≈ 5:15. Only the first 5 minutes are graded, so every graded element lands
in slides 1–8; slide 9 closes. Plain language throughout (the panel may include
non-actuaries; the first screening layer is AI review of the narration).

All numbers on screen are method evidence (local out-of-fold / recomputed from
source) — consistent with the D-093 presentation-layer scrub, no competition
scores, ranks or standings appear.

---

**Slide 1 — Title** (~16s)
*On screen:* Pricing Across Markets · Predicting 2025 claim cost in two
insurance markets · 2026 IAA AI-Assisted Hackathon · October 2026
*Narration:* This is our submission to the 2026 IAA AI-Assisted Hackathon. In
the next five minutes: how we found what moves claim cost across two insurance
markets, and turned it into prices for every 2025 policy.

**Slide 2 — The Business Problem** (~35s)
*On screen:* bullets — set 2025 prices for every policy in both markets;
established market 2.4M policy periods 2021–2024; new market 2 thin years,
~9,100 claims; 76% of the 2025 book is business never seen before (broker
portfolio takeover); prices must be defensible.
*Narration:* The business problem. You asked us to find what actually moves
claim cost on your policies, and turn that into prices for every 2025 policy in
two markets. Your established market gives four years of history on two point
four million policies. The new market gives only two thin years — and
seventy-six percent of its 2025 book is business you have never seen, arriving
from a broker portfolio takeover. These prices will drive renewals, price the
incoming portfolio, and correct misalignments in your current rates.

**Slide 3 — Data Overview** (~40s)
*On screen:* bullets — unfinished claims at 31 Dec 2024 → labels developed to
final value; corrupt exposure column → recomputed from policy dates; annualised
premiums treated per-period, 604 duplicate claims removed; 80 occupation
spellings mapped to 20 categories; thin book ~1,000 claims per property line
vs a test book ~4× a normal year.
*Narration:* The data, and what we did to it. Claim costs at the end of 2024
are unfinished — a claim reported in December is still growing — so every
training label was developed to its final value. The supplied exposure column
mixed unit conventions, so exposure is now recomputed from policy dates.
Premiums were annualized and are treated per-period. Six hundred and four
duplicate claims were removed. The new market's eighty occupation spellings
mapped cleanly onto twenty categories. The honest headline: your new market
holds about one thousand claims per property line, against a test book nearly
four times a normal year.

**Slide 4 — Limitations** (~38s)
*On screen:* numbered 1–5 (development factors estimated; heavy tail real,
untrimmed; 2025 level tuned on a live board; weather as a pricing variable;
inexperienced team + AI errors caught only by re-deriving numbers). Closing
line: "These insights may be nullified by these limitations."
*Narration:* Five limitations should shape how you use these results. One:
development factors are estimates, and a handful of large claims dominate them.
Two: the heavy tail is real — verified, and deliberately not trimmed. Three:
the level of the 2025 cost adjustment was tuned on a live evaluation board, and
may sit too high. Four: weather is used as a pricing variable. Five: our
inexperience — and an AI assistant that made real errors, caught only by
re-deriving every number — may have left important impacts overlooked. These
insights may be nullified by these limitations.

**Slide 5 — Adaptation for the New Market** (~40s)
*On screen:* diagram — mature-market models → prior (recalibrated) → tiny
correction layer (20–35 shallow trees) → 2025 cost-drift adjustment. Bullets:
pivot = stop learning from scratch; small careful models beat big ones; with
~9,000 claims measure noise first and pre-register every test.
*Narration:* When the new market's data arrived, almost every assumption
changed. The pivot: stop learning from scratch. Models trained on the two point
four million established-market policies become a prior, recalibrated to the
new market's cost level. A deliberately tiny correction layer — twenty to
thirty-five shallow trees — learns only what the prior cannot know: region,
gender, channel, flood zone, new-business status. A measured adjustment covers
2025 cost drift. The lesson of this data: small, careful models beat big ones;
deeper trees memorize noise. And with so few claims, we changed how we decide —
measure the noise first, pre-register every test.

**Slide 6 — Model Validation** (~32s)
*On screen:* quality-ladder chart (OOF D²: flat 0.0804 · premium × LR 0.1097 ·
new market alone 0.1073 · transfer prior 0.1386 · final 0.1407). Takeaway line:
"Nearly all the skill is transferred, not learned from the thin book."
*Narration:* Model validation. Each model was tested honestly: train on part of
the book, predict held-out policies. A flat rate scores about zero point zero
eight. The thin new book alone: zero point one one. The transfer prior alone:
nearly zero point one four. The prior with its small correction layer — our
final model — reaches zero point one four one. Nearly all the skill is
transferred from the mature market, not learned from the thin book. The final
model reproduces its predictions byte for byte.

**Slide 7 — What Moves Claim Cost** (~44s)
*On screen:* flood-zone chart (claim cost per exposure 63→418 CU vs flat price
≈127 CU). Bullets: motor ages 25–34 LR 0.91 vs 0.60–0.62, priced the same;
male drivers +34% claim cost, priced +3%; new business too cheap vs renewals;
"we can conclude prices are misaligned with experience; we cannot conclude the
disparities will persist."
*Narration:* Now, what actually moves claim cost. The starkest finding: in your
new market, contents prices are flat across flood zones — around one hundred
and twenty-five currency units — while observed claim cost rises six-fold, from
sixty-three units in zone one to over four hundred in zone five. Flood risk is
entirely unpriced. Young drivers carry a loss ratio of zero point nine one
against zero point six for middle ages, and male drivers carry a third more
claim cost — both priced about the same today. New business is priced too
cheap relative to renewals. We can conclude your prices are misaligned with
observed experience; we cannot conclude the disparities will persist.

**Slide 8 — AI in the Workflow + Guardrails** (~40s)
*On screen:* bullets — AI did hands-on analysis, human held scope/judgement/
authorisation; adversarial AI review found 2 critical bugs → largest single
improvement; all work in an isolated container; 103 automated tests green
before every commit, version control as audit evidence; every number recomputed
from source.
*Narration:* Because this was an AI-assisted engagement, how AI entered the
work matters. The AI agent did the hands-on analysis — exploring data, writing
pipelines, running experiments — while the human held scope, judgment, and
every authorization. A fresh AI model, instructed to attack the existing work
adversarially, found two critical bugs and produced the largest single
improvement in the project. All of it ran inside an isolated container with no
access to our machines. And trust came from guardrails, not trust: one hundred
and three automated tests, nothing committed until they pass, version control
as audit evidence, and every number recomputed from source before publication.

**Slide 9 — Next Steps** (~18s)
*On screen:* bullets — discuss and pressure-test the flood/gender/new-business
findings before any filing; richer data as the new market earns; monitor 2025
experience, refit, extend to further markets. "Thank you."
*Narration:* Next steps. Discuss these results with your pricing team, and
pressure-test the flood, gender, and new-business findings before any filing.
Seek richer data as the new market earns. And monitor 2025 experience, refit,
and extend the approach to further markets. We look forward to that
conversation. Thank you.
