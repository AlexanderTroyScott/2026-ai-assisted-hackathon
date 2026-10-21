# Skill: model-validation

Distilled from **CAS Institute Study Note: "Model Validation and Holdout Data"**
(1/13/2019), 21pp. Retrieved 2026-10-21 from
`https://thecasinstitute.org/wp-content/uploads/2019/01/Exam-3-Study-Note-Model-Validation-01162019.pdf`.

Use this when choosing a train/test split, deciding whether a number is a valid
gate for a submission, judging overfit, or writing up our validation design.
It is the authority for **holdout discipline** specifically — the point on which
our repo is weakest.

## 1. Three sets, not two

| set | purpose | rule |
|---|---|---|
| **training** | fit parameters | — |
| **test** | used *repeatedly* during the build to compare models | may be "fitted to" indirectly |
| **holdout** | objective metric reported to management | touched **once**, at the end |

The note is explicit that the test set cannot serve as the objective metric:
*"since one has repeatedly compared to it, one has in some sense fit the model
to the test data as well as to the training data."* The holdout is separated
**as early as practicable** and stored separately, not accessible during
modelling.

> **This repo's exposure.** We have no true holdout. The 2024 slice has been
> used to choose the label table, reject recency weighting, reject the log
> transforms, and compare weather variants — i.e. it is a *test* set by this
> definition. The **leaderboard is our only true holdout**, which is precisely
> why D-025 (local near-ties do not rank) is unsurprising and why local gates
> must be treated as weak evidence.

## 2. Cross-validation does NOT replace a holdout

The note answers this directly, and the reason matters:

> *"when human beings are building the model, the choice of model (including
> type of model, which variables were candidates to final feature selection,
> which derived features were created) has typically been impacted by all the
> data the modeler has worked with… Even though one may be fitting parameters
> and even performing final feature selection separately for each collection of
> K-1 folds, those models will have been influenced by all the data. This does
> not mean that cross-validation is useless… It merely means that the proper use
> of cross-validation is as a substitute for the training/test split, not as a
> substitute for having true holdout data."*

So: CV buys us a better training/test split. It does **not** buy objectivity.
Random k-fold is additionally wrong here — see §4.

## 3. Refitting on all data after measuring is correct (validates D-045)

> *"After testing on holdout data to obtain the objective quality of the model,
> one can (and usually does) refit on the complete data to obtain the final
> model. One can even change the model if adding the holdout data makes it
> clear that one should. **What one cannot do is claim a better goodness of fit
> than one has already measured.**"*

This is exactly the `--final-fit` design: measure honestly out-of-sample, then
refit on 2021–2024 before scoring 2025. The constraint is the reporting rule —
local D² stays the *pre-refit* number and is never re-stated as if the refit
had been measured.

## 4. Splitting: group correlated observations

> *"it is good practice to ensure the correlated observations… are all put into
> the same group. For example, if the unit of observation is one policy for one
> year, the series of observations that correspond to a single policyholder over
> that period of time would be put into the same group."*

**Our rows are policy-periods and the same customer appears across rows.** That
is deliberate (it is what makes lagged experience features possible), but it
means any *random* split would leak customer history across the boundary. Our
time-based split avoids this by construction.

**Out-of-time validation** — contiguous blocks of time as groups — is the
endorsed design *when time drives correlation*. That is our situation and our
rolling-origin folds (train ≤2023 → holdout 2024; train ≤2022 → holdout 2023)
match it.

**But the note warns this is not automatic for insurance:**

> *"For many insurance problems, however, time is not the main dimension, and
> splitting on the time variable may actually be undesirable because one may
> want to incorporate time explicitly in the model or to test consistency of
> effects across time within the training data as one of the criteria for
> including a variable."*

And it gives the concrete diagnostic:

> *"Look at residuals across the levels of the variable in question for the
> earlier half (temporally) of the data and for the latter half of the data, and
> compare."*

**This is the test for our Tier-1 assumption A** (2024 holdout represents the
2025 population). Residual stability across time halves by feature level is
cheap and we have never run it.

**Reproducibility:** prefer a *hash* of the grouping key over a random-number
split. The note specifically recommends **MD5** (most hash functions cluster
similar keys; MD5 does not). Needed because datasets evolve and RNG splits are
brittle to re-sorting and re-filtering.

## 5. Diagnostic tools

### Normalized Gini — the overfit measure we are missing

Sort by **predicted** values, not actuals. Then:

```
Normalized Gini = Model Gini / Data Gini
```

where *Data Gini* is the same statistic computed on actuals. The denominator
adjusts for the sample's own dispersion so train and holdout are comparable.

Interpretation, per the note:
- A saturated model (prediction = actual) scores 1.0 and does not generalise.
- A constant model scores 0.0.
- **"A model generalizes well if its normalized Gini in a testing or holdout
  sample is reasonably close to its normalized Gini in the training sample."**
- Worked example: random forest 0.91358 train / 0.90295 test (gap **0.0106**)
  beat linear regression 0.88685 / 0.86674 (gap **0.0201**) — better fit *and*
  better generalisation.

**The train/test gap is the number to watch**, not the level. We compute
neither. Given D-025 this is the obvious missing diagnostic.

Implementation detail that matters: add a **random tiebreaker column** to the
sort key. If the data was previously sorted by actuals and the sort is stable,
ties resolve in the original order and the Gini comes out **artificially high**.

### Lorenz curve
Cumulative share of actual vs cumulative share of exposure, sorted by
prediction. "Actuarial" form sorts by predicted pure premium.

### Lift charts
1. Sort by prediction, ascending.
2. Bin into equal **exposure** volume (5/10/20 bins typical).
3. Per bin, plot average predicted and average actual.
4. **"We always calculate lift on holdout data."**

Read the chart for (a) **monotonicity** of actuals — small reversals fine,
large ones are not; (b) **lift** = bin10 − bin1 (or the ratio bin10/bin1);
(c) **calibration** = how closely predicted tracks actual.

> *"Large deviations would suggest that our model is not picking up something
> in the actual experience."*

Bins differ per model — the chart is per-model, not a shared partition.

### Double lift charts
For comparing two models: sort by the **ratio** of their predictions, bin on
equal exposure, plot both predictions plus actual. Bins 1 and 10 are where the
models disagree most. Note the honest reading in the example: volatility of
actual experience can make the better model ambiguous even here.

## 6. What this means for us

| our practice | note's position | action |
|---|---|---|
| rolling-origin out-of-time split | endorsed when time drives correlation | keep |
| random k-fold as a gate | wrong twice over — leaks customer, and CV≠holdout | never |
| using 2024 to pick models | this is a **test** set, not holdout | state it; treat local gates as weak |
| `--final-fit` refit after measuring | explicitly standard practice | keep; never claim post-refit D² |
| reporting only local D² | incomplete | add normalized Gini + lift |
| never computed train/test Gini gap | the overfit measure | **add it** |
| never ran residual stability by time half | the recommended consistency test | **run it** |

**Bottom line for the write-up.** We should say plainly that 2024 is a
*selection* set and the leaderboard is the only untouched data — that is a
limitation to disclose, not a flaw to hide. The note's framing gives us the
language: cross-validation bought us a better split; it did not buy
objectivity.
