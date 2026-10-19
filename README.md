# 2026-ai-assisted-hackathon

# AI Actuarial Hackathon — Battle Plan

## The Short Version

You are being judged on **three axes simultaneously**: predictive power (Tweedie D²), actuarial credibility, and — critically — the **demonstration of AI augmenting your actuarial workflow**. Most teams will nail one or two. Your edge is showing *where* and *how* AI helps at every stage, with tangible outputs, not just "we used ChatGPT."

---

## Architecture of the Solution

Think of the submission as **three deliverables**:

| Deliverable | What | Why it matters |
|---|---|---|
| **Prediction CSV** | `row_id, prediction` for all test rows | Quantitative score (Tweedie D²) |
| **Documentation / Write-up** | Explains your modelling, validation, AI workflow | Qualitative judging panel review |
| **Video / Presentation** | Narrated walkthrough | AI-assisted first-layer screening |

Everything you do should feed these three artefacts.

---

## Phase-by-Phase Plan

### PHASE 0 — Environment Setup (Hour 0–2)

```
- Set up a project repo (even local) with clear folder structure:
  /data_raw          # parquet files as received
  /data_clean        # processed/feature-engineered
  /notebooks         # EDA, feature engineering, modelling
  /src               # reusable Python modules
  /models            # saved model artefacts
  /outputs           # submission CSVs, charts, reports
  /ai_logs           # AI interaction logs (important for the write-up!)
  /docs              # documentation, methodology notes
```

**Action items:**
- Verify all 11 parquet files load correctly
- Confirm `row_id` counts in test files match `sample_submission.csv`
- Set up a shared environment or notebook (Jupyter, VS Code + Jupyter kernel, or your `opencode` setup)
- Install: `pandas`, `polars`, `scikit-learn`, `statsmodels`, `lightgbm`, `xgboost`, `shap`, `matplotlib`, `seaborn`, `pyarrow`

---

### PHASE 1 — Data Exploration & Understanding (Hours 2–8)

This is where **AI provides its first major value**: acting as an intelligent data analyst assistant.

#### 1a. Automated Data Profiling

Use your AI coder deployment to generate comprehensive data profiling scripts:

```python
# AI generates this kind of script for each file
for each parquet file:
    - shape, dtypes, memory usage
    - null counts per column (with % of total)
    - unique value counts
    - value distributions for categoricals
    - numeric summaries (mean, median, std, percentiles)
    - obvious anomalies (negative exposure, impossible dates, etc.)
```

**Where AI helps:** Prompt your AI coder with: *"Write a data profiling script for these insurance parquet files. Include null analysis, distribution summaries, and flag data quality issues."* Save the AI's output into `ai_logs/` for your write-up.

#### 1b. Target Variable Construction

This is **the most critical modelling step** and requires actuarial judgment.

The target is: **expected ultimate claim cost per policy-period row**, where ultimate = final cost of all claims with loss date in the row's period, net of excess, net of recoveries, excluding declined claims.

```
For each row in policies_{line}:
  1. Join claims where policy_id matches AND loss_date falls within [period_start, period_end]
  2. Filter to status != 'repudiated'
  3. For each qualifying claim:
       claim_cost = max(0, incurred_to_date - recoveries)  # but careful with excess
       Actually: incurred_to_date is already "paid + case estimate, net of excess"
       So: claim_cost = max(0, incurred_to_date - recoveries)
  4. Sum across all claims for the row → y_ultimate
```

**BUT** — claims are valued at 31 Dec 2024, so there is **development lag**:
- 2024 claims may be under-reported (IBNR)
- Open claims may develop further
- This is a classic **reserving problem** within a pricing challenge

**Key decision:** Do you:
- **(A)** Use `incurred_to_date - recoveries` as-is (simple, may understate)
- **(B)** Apply development factors (chain-ladder style) to project ultimate
- **(C)** Model claim frequency and severity separately, then multiply
- **(D)** Use a Tweedie GLM directly on the observed data and let it "learn" the relationship

**Recommendation:** Try **(C) or (D)**, with (B) as a sense-check. Option (C) — frequency/severity decomposition — is actuarially credible and can handle the zero-inflated, heavy-tailed nature of claim costs. The Tweedie loss function with p=1.5 is exactly designed for this.

#### 1c. Exploratory Data Analysis

Generate:
- **Claim frequency** by each rating factor (univariate)
- **Average severity** by each rating factor
- **Total loss ratio** by region, year, channel, etc.
- **Exposure distribution** and its relationship with claims
- **Temporal trends** — 2021→2024, are claims increasing?
- **Cross-LOB relationships** — same customer across products
- **Data quality issues** — missing values, outliers, impossible values

**Where AI helps:** *"Given this EDA output, what data quality issues do you see? What feature engineering opportunities suggest themselves? What actuarial concerns should I flag?"*

---

### PHASE 2 — Feature Engineering (Hours 8–16)

#### Features to Build

| Category | Examples |
|---|---|
| **Demographic** | Driver age (capped/banded), years licensed, policyholder age for home |
| **Vehicle (motor)** | Vehicle value group (already numeric), age, engine size, body type, fuel type |
| **Property (home)** | Sum insured, construction type, property type, roof type, year_built → age of property |
| **Address risk** | Flood zone, urban density, elevation, distance to river, soil type |
| **Weather** | Hail days, wind exposure, freezing days, rainfall — match to policy year |
| **Behavioural** | NCB years, licence points, prior claims 5y, missed payments, customer service contacts |
| **Policy structure** | Voluntary + compulsory excess (total excess), cover type, add-ons |
| **Tenure/renewal** | Renewal count, new business flag, years at address, customer_since |
| **Temporal** | Policy year, month of inception, exposure fraction |
| **Interaction features** | Age × mileage, flood zone × rainfall, value group × excess |

#### Key Feature Engineering Decisions

1. **Total excess** = `voluntary_excess + compulsory_excess` — this directly affects net claim cost
2. **Property age** = `current_year - year_built`
3. **Weather matched to policy year** — join `weather` on `address_id` + `year = policy year`
4. **Derived experience features** — `prior_claims_5y`, `ncb_years` may be correlated
5. **High-cardinality handling** — `make_model`, `occupation`, `repairer_id`, `third_party_id` need careful treatment (target encoding, clustering, or frequency grouping)

**Where AI helps:** *"Here are my 40+ raw columns. Suggest 20 derived features that might improve claim cost prediction for motor insurance. Prioritise by expected actuarial relevance."*

---

### PHASE 3 — Modelling (Hours 16–32)

#### Recommended Modelling Strategy

**Go with a multi-model approach and ensemble where appropriate:**

```
Per line of business (motor, buildings, contents):

  Model 1: Tweedie GLM (statsmodels or custom)
            - p=1.5
            - Log link
            - Categoricals one-hot encoded or banded
            - Variable selection (stepwise or LASSO-regularised)
            - THIS IS YOUR BASELINE AND EXPLAINABILITY ANCHOR

  Model 2: Gradient Boosted Trees (LightGBM)
            - objective = 'tweedie', tweedie_variance_power = 1.5
            - Early stopping on validation set
            - SHAP for explainability
            - THIS IS YOUR ACCURACY ENGINE

  Model 3 (optional): Frequency-Severity split
            - Claim count: Poisson / NB GLM or LightGBM
            - Claim severity (non-zero claims): Gamma GLM or lognormal
            - Expected cost = E[freq] × E[sev]
```

#### Critical Modelling Choices

1. **Train/Validation split:** Time-based — train on 2021–2023, validate on 2024. Never random-split time-series insurance data.

2. **Exposure weighting:** Use `exposure` as weights in GLMs (standard actuarial practice). For LightGBM, include exposure as a feature and/or weight.

3. **Handling zeros:** The target is mostly zero (most policies have no claims). Tweedie handles this naturally. Don't filter to only-claims rows.

4. **Prediction clipping:** Set minimum to 0 (as required). Consider capping extreme predictions.

5. **Per-LOB models:** Build separate models for motor, buildings, contents — they have very different risk drivers.

#### Model Selection

Use Tweedie D² (p=1.5) on the 2024 holdout to select:

```python
def tweedie_d2(y_true, y_pred, p=1.5):
    # Deviance-based D² score
    # Baseline = mean of y_true
    from sklearn.metrics import mean_tweedie_deviance
    dev_model = mean_tweedie_deviance(y_true, y_pred, power=p)
    dev_baseline = mean_tweedie_deviance(y_true, np.full_like(y_true, y_true.mean()), power=p)
    return 1 - dev_model / dev_baseline
```

**Where AI helps:** *"Here is my Tweedie GLM output. The residual plot shows a pattern at high values. What should I investigate? Should I add interaction terms or change the link function?"* and *"Write me a LightGBM training script with Tweedie objective and proper time-based cross-validation."*

---

### PHASE 4 — Validation & Diagnostics (Hours 32–38)

This is where you **demonstrate actuarial professionalism**.

#### Required Diagnostics

| Diagnostic | What to show |
|---|---|
| **Actual vs Predicted** | By decile of predicted value, with confidence intervals |
| **Residual plots** | By year, by region, by key rating factors |
| **Lift curves** | Gains chart showing model separates high from low risk |
| **Calibration** | Mean predicted vs mean actual by exposure band |
| **Stability** | Feature importance stability across train/val/test |
| **Out-of-time validation** | 2024 performance vs 2021–2023 |
| **LOB comparison** | Is one line driving all the performance? |

#### Scribe or Actuarial Tests

- **Credibility**: Are you trusting thin cells?
- **Monotonicity**: Does predicted cost increase sensibly with, say, flood zone?
- **Extreme values**: What happens to a 25-year-old, high-value, flood-zone-5 property?

**Where AI helps:** *"Given these model diagnostics, write me a section of my documentation describing the validation approach and any limitations identified."*

---

### PHASE 5 — AI Workflow Documentation (Ongoing — but write up Hours 38–42)

This is **the differentiator**. Throughout the project, you've been logging AI interactions. Now synthesise them into a narrative.

#### Structure the AI Workflow Story

```
1. DATA UNDERSTANDING
   "We used AI to auto-generate data profiling scripts and flag data quality issues.
    This reduced initial EDA time from ~8 hours to ~2 hours.
    AI interaction log: [reference]"

2. FEATURE ENGINEERING
   "We prompted AI to suggest derived features based on actuarial domain knowledge.
    An actuary reviewed and selected 15 of 25 suggested features.
    AI interaction log: [reference]"

3. MODEL BUILDING
   "AI generated boilerplate code for Tweedie GLMs and LightGBM pipelines.
    Actuarial judgment was used to set hyperparameters and variable selection.
    AI interaction log: [reference]"

4. VALIDATION
   "AI generated diagnostic charts and initial interpretation.
    Actuary reviewed, identified a residual pattern in high-severity claims,
    and added an interaction term. This improved D² by X%.
    AI interaction log: [reference]"

5. DOCUMENTATION
   "AI drafted initial documentation sections. Actuary revised for
    actuarial accuracy and clarity."
```

**Key principle:** Show AI as a **co-pilot**, not an autopilot. Actuarial judgment at every decision point.

---

### PHASE 6 — Submission Preparation (Hours 42–46)

1. **Generate predictions** for all test rows (motor + buildings + contents)
2. **Sanity checks:**
   - All `row_id` present (count matches sample_submission)
   - All predictions ≥ 0
   - Predictions look reasonable (compare distribution to training claims)
   - No missing values
   - Row count: `len(test_motor) + len(test_buildings) + len(test_contents)`
3. **Write the CSV** in exact format
4. **Final documentation** — polish everything

---

### PHASE 7 — Week 2 Preparation (Post Week 1 submission)

Week 2 asks: *"A new dataset will be released. How does your approach adapt?"*

**Pre-write your Week 2 framework now:**

```
1. DATA COMPARISON
   - Profile the new data alongside old
   - Compare distributions of key variables
   - Check for: new categories, shifted distributions, missing patterns,
     new perils, different geographic distribution

2. MODEL ADAPTATION
   - Can the existing model handle new categories? (fallback to "other" level)
   - Does the feature space still make sense?
   - Is the target variable the same?
   - Do you need to recalibrate?

3. UNCERTAULTY COMMUNICATION
   - Where is the model least reliable in the new market?
   - What assumptions might not hold?
   - What additional data would help?

4. PRACTICAL CONSIDERATIONS
   - If volumes are low, more reliance on GLM (credibility)
   - If data quality is different, more robust preprocessing
   - If risk drivers differ, model refit or augmentation
```

---

## Where AI Adds Value — Summary Table

| Workflow Stage | AI Role | Actuarial Role | Time Saved |
|---|---|---|---|
| Data profiling | Generate scripts, flag anomalies | Interpret anomalies, decide how to handle | ~60% |
| EDA | Generate charts, suggest patterns | Domain interpretation, hypothesis testing | ~50% |
| Feature engineering | Suggest derived features, write code | Select, validate actuarially | ~40% |
| Model coding | Generate GLM/GBM pipelines | Set parameters, variable selection | ~50% |
| Diagnostics | Generate plots, initial interpretations | Identify issues, decide on remediation | ~40% |
| Documentation | Draft text, structure write-up | Verify accuracy, add nuance | ~60% |
| Week 2 adaptation | Compare datasets, flag shifts | Decide on model changes | ~50% |

---

## Technical Details — Tweedie D² Score

Since the scoring metric is **Tweedie D² with p = 1.5**, optimise for this directly:

```python
# Training objective should match
# LightGBM: objective='tweedie', tweedie_variance_power=1.5
# GLM: statsmodels Tweedie with var_power=1.5

# Validation metric
from sklearn.metrics import mean_tweedie_deviance

def tweedie_d2(y_true, y_pred, power=1.5):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 1e-10)
    
    dev_model = mean_tweedie_deviance(y_true, y_pred, power=power)
    dev_null = mean_tweedie_deviance(y_true, 
                                      np.full_like(y_true, y_true.mean()), 
                                      power=power)
    return 1.0 - dev_model / dev_null
```

Tweedie with p=1.5 is in the **Poisson-Gamma compound** family — perfect for claim costs (mass at zero + continuous positive tail).

---

## Suggested Tools & Libraries

| Tool | Purpose |
|---|---|
| `polars` or `pandas` | Data manipulation |
| `pyarrow` | Parquet I/O |
| `statsmodels` | Tweedie GLM |
| `lightgbm` | Gradient boosting (Tweedie objective) |
| `scikit-learn` | Pipeline, metrics, train-test split |
| `shap` | Model explainability |
| `matplotlib` / `seaborn` | Visualisation |
| `opencode` + AI API | Code generation, analysis, documentation |

---

## Risk Management — Common Pitfalls

| Pitfall | Mitigation |
|---|---|
| Random train/test split on time-series data | Always split by year |
| Not handling exposure correctly | Weight by exposure; partial-year rows |
| Ignoring IBNR (unreported 2024 claims) | Consider development factors; or let Tweedie model learn |
| Overfitting to training data | Time-based validation; regularisation |
| Ignoring excess interaction with claims | Total excess is a key feature; net-of-excess target means excess affects expected cost |
| High-cardinality features (make_model) | Target encoding, frequency grouping, or bucketing |
| Forgetting that repudiated claims are excluded | Filter them out before building target |
| Submission formatting errors | Validate against sample_submission.csv |
| AI output taken without review | Always review AI-generated code and analysis |

---

## Timeline Summary (48-hour hackathon)

| Hours | Activity |
|---|---|
| 0–2 | Environment setup, data loading, initial checks |
| 2–8 | Data profiling (AI-assisted), EDA, target construction |
| 8–16 | Feature engineering (AI-assisted), data cleaning |
| 16–32 | Model building — GLM + LightGBM per LOB |
| 32–38 | Validation, diagnostics, model selection |
| 38–42 | AI workflow documentation |
| 42–46 | Submission preparation, sanity checks, polish |
| 46–48 | Buffer for unexpected issues |
| Week 2 | Adapt framework for new market data |

---

**Bottom line:** Build a solid Tweedie GLM as your credible baseline, layer LightGBM for accuracy, and make the *story* of how AI assisted you at every stage just as compelling as the numbers. Log every AI interaction. Let the judging panel see the division of labour clearly: AI handles speed and boilerplate, you handle judgment and credibility. That's the winning formula.
