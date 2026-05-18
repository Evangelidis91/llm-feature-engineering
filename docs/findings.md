# Hiring 10 LLMs as Feature Engineers
## What you get for a dollar — and what the cost numbers hide

### Abstract

Feature engineering is one of the most time-consuming parts of building a tabular ML pipeline.
I wanted to know whether LLMs can do it for you and what the real tradeoffs look like once you
actually wire them in. I tested 10 LLMs (6 production-tier, 4 frontier) on 3 contrasting datasets,
applied their suggestions through 3 ML algorithms(Logistic / Linear Regression, Random Forest, XGBoost), 
and ran 219 cross-validated experiments. LLM features beat baseline 70% of the time, but the win rate
depends heavily on which ML algorithm consumes them, and the frontier-tier cost premium delivers no
statistically significant benefit.

---

## 1. The context

Most "LLMs for ML" demonstrations focus on one model, one dataset, one algorithm and report whether
it worked. That's a useful proof-of-concept but a poor guide for the team that has to actually
deploy something. Real ML teams want answers to questions like:

 - Which LLM should we default to?
 - Does that depend on the downstream model?
 - How much should we expect to pay per useful feature?
 - When does the LLM help and when does it just generate plausible-sounding noise?

To answer those, the comparison has to be **factorial**. I picked four dimensions:

 - **LLMs (10).** Six production-tier(GPT-4o-mini, Claude Sonnet 4.5, Gemini 2.0 Flash,
    DeepSeek V3, Llama 3.3 70B, Qwen 2.5 72B) and four frontier (GPT-5.5, Claude Opus 4.6,
    Gemini 3.1 Pro, DeepSeek V4 Pro). All accessed via OpenRouter to keep the harness identical.
 - **Datasets (3).** Telco Customer Churn (binary classification, ~7K rows, mostly categorical),
    Ames Housing (regression, ~3K rows, rich numerical features), Bank Marketing (binary 
    classification, ~41K rows, mild class imbalance). Three different shapes of problem.
 - **ML algorithms (3).** Logistic / Linear Regression, Random Forest, XGBoost. These cover 
    the three model families that practitioners actually default to on tabular data.
 - **Prompt styles (2).** Zero-shot(column name only) vs with-stats (column names plus per column
    min/max/mean and top categorical values).


That's 60 LLM calls (10 * 3 * 2) producing 420 feature suggestions, then 180 ML experiments
(the Cartesian product applied across 3 algorithms) plus 9 baseline runs — 219 cross-validated
runs in total. Every run uses 5-fold CV with the same random seed.

---

## 2. Experiment A: Can the LLM write valid code?

### Setup

Each LLM was asked to suggest 7 engineered features per (dataset, prompt) pair, in a strict
JSON schema with `{name, formula, rationale}` fields. Formulas were evaluated in a sandboxed
Python namespace exposing pandas, NumPy and the DataFrame's columns. A formula was counted 
as **valid** if it executed cleanly and produced a non-constant, non-NaN, single-typed Series.

### Results
| Rank | LLM               | Tier       | Validity   | Cost / Valid Feature |
|------|-------------------|------------|------------|----------------------|
| 🥇   | **GPT-5.5**       | frontier   | **100.0%** | $0.007216            |
| 2    | DeepSeek V4 Pro   | frontier   | 97.6%      | $0.001219            |
| 3    | Qwen 2.5 72B      | production | 95.2%      | $0.000078            |
| 4    | GPT-4o-mini       | production | 92.9%      | $0.000057            |
| 5    | Gemini 3.1 Pro    | frontier   | 92.9%      | $0.005414            |
| 6    | Gemini Flash 2.0  | production | 90.5%      | $0.000046            |
| 7    | Claude Sonnet 4.5 | production | 90.5%      | $0.001720            |
| 8    | Claude Opus 4.6   | frontier   | 90.5%      | $0.002989            |
| 9    | DeepSeek V3       | production | 85.7%      | $0.000119            |
| 10   | Llama 3.3 70B     | production | 85.7%      | $0.000046            |

### How to read this table

Each LLM produced 42 feature suggestions in total (3 datasets × 2 prompt
variants × 7 features per call). **Validity** is the share of those 42 that 
ran cleanly against the real data; **Cost / Valid Feature** is the LLM's total
API spend on the study by the number of valid features it produced.

To make the spread concrete, suppose you needed 1,000 valid features for a
real project:

- With **GPT-5.5** (100% validity, $0.007216/valid feature) you'd spend about
**$7.22** and never see a broken formula.
- With **Gemini Flash 2.0** (90.5% validity, $0.000046/valid feature) you'd
spend about **$0.05** — roughly 95 suggestions would be invalid, but they'd
be auto-rejected by the validator at zero human cost.

The 157× cost premium buys you a cleaner output stream. Whether that's worth
it depends entirely on whether the extra valid features actually translate
into a better downstream model. That's what Experiment B is for.


### Three things jump out
1. **Modern LLMs can write feature code.** Even the worst performers (DeepSeek V3,
    Llama 3.3) ship valid features 86% of the time. The minimum bar is high. 
2. **Failure modes are systematic, not random.** Across the 60 calls, the invalid
    features clustered into four categories: hallucinated column names(Llama, Qwen),
    division-by-zero producing infinity(DeepSeek, GPT), incomplete `.replace()`
    mappings producing mixed `int`/`str` columns(Qwen), and frontier models writing
    standalone pandas snippets (`df[`col`]`) instead of bare expressions(GPT-5.5)
    Any production deployment has to validate against all four.
3. **Cost spread is brutal.** Gemini Flash 2.0 produces a valid feature for 
    $0.000046. when GPT-5.5 does it for $0.007216 — **157× more expensive** for
    output that's technically more correct but, as we'll see in Experiment B, no
    more useful.

---

## 3. Experiment B: Do the features actually help the model?

A valid formula is necessary but not sufficient. The real question is whether 
the features improve the downstream model - and whether the answer depends on
which model you are using.

### Setup

For each LLM × dataset × prompt combination, all three ML algorithms (Logistic /
Linear Regression, Random Forest, XGBoost) trained on `original + LLM features`
and compared against the baseline using 5-fold CV scores. A **win** means the 
augmented model beat the baseline on the dataset's primary metric (ROC-AUC for 
classification, R² for regression).

### Result 1: ML algorithm choice dominates the equation

The single biggest finding of the study is that the win rate is **not uniform
across model families**. Here's the full 10 × 3 matrix — wins out of 6 conditions
(3 datasets × 2 prompt styles) for each LLM × ML algorithm combination:


| LLM                 | Tier       | LogReg    | Random Forest    | XGBoost           |
|---------------------|------------|-----------|------------------|-------------------|
| GPT-4o-mini         | production | 3/6 (50%) | 2/6 (33%)        | 4/6 (67%)         |
| Claude Sonnet 4.5   | production | 4/6 (67%) | 4/6 (67%)        | **6/6 (100%)** ⭐  |
| Gemini Flash 2.0    | production | 3/6 (50%) | 5/6 (83%)        | 4/6 (67%)         |
| DeepSeek V3         | production | 2/6 (33%) | 5/6 (83%)        | **6/6 (100%)** ⭐  |
| Llama 3.3 70B       | production | 3/6 (50%) | 3/6 (50%)        | 4/6 (67%)         |
| Qwen 2.5 72B        | production | 2/6 (33%) | 5/6 (83%)        | **6/6 (100%)** ⭐  |
| GPT-5.5             | frontier   | 4/6 (67%) | **6/6 (100%)** ⭐ | 3/6 (50%)         |
| Claude Opus 4.6     | frontier   | 3/6 (50%) | 5/6 (83%)        | 5/6 (83%)         |
| Gemini 3.1 Pro      | frontier   | 4/6 (67%) | 5/6 (83%)        | **6/6 (100%)** ⭐  |
| DeepSeek V4 Pro     | frontier   | 4/6 (67%) | 4/6 (67%)        | 5/6 (83%)         |
| **Avg across LLMs** |            | **50%**   | **70%**          | **76%**           |


### How to read this table


Each cell shows how many times the LLM × ML algorithm combination beat the baseline
out of **6 conditions** (3 datasets × 2 prompt styles). For example, **Claude Sonnet 
4.5 × XGBoost = 6/6** means: across all 3 datasets and both prompt variants, an 
XGBoost model trained on `original + Claude's features` outperformed an XGBoost 
model trained on the original features alone. Every single time.

A few specific cells worth pausing on:
- **Gemini 3.1 Pro × XGBoost = 6/6 (100%)** — the strongest combo in the study,
but at $0.0054 per valid feature.
- **Claude Sonnet 4.5 × XGBoost = 6/6 (100%)** — equal performance for one-third
of the cost.
- **GPT-5.5 × XGBoost = 3/6 (50%)** — same LLM that scored 100% with Random
Forest. The "best" features depend entirely on which downstream model
consumes them.
- **GPT-4o-mini × Random Forest = 2/6 (33%)** — the weakest pairing in the
matrix, worse than a coin flip. Even cheap LLMs can underperform when the
algorithm doesn't match the feature style.

Read the bottom row first if you take only one thing away: the average win rate
moves from **50% (LogReg) → 70% (Random Forest) → 76% (XGBoost)**. That 26-point
spread is bigger than the 33-point spread between the worst and best LLM in
this study. **The downstream algorithm matters as much as the LLM.**

Tree-based algorithms love LLM-suggested features. Linear models tolerate them.
This makes sense: LLMs naturally generate flag features like (`is_month_to_month`
, `has_internet_service`) and threshold features like (`high_monthly_charges = 
MonthlyCharges > 70`). Trees split on these almost for free; linear models
need them rescaled and shrunk before they help.

### Result 2: Win rate per LLM (average across algorithms)

| Rank       | LLM                                                                  | Tier       | Wins  | Win Rate  | Avg Δ              |
|------------|----------------------------------------------------------------------|------------|-------|-----------|--------------------|
| 🥇         | Gemini 3.1 Pro                                                       | frontier   | 15/18 | **83.3%** | +0.0025            |
| 2          | Claude Sonnet 4.5                                                    | production | 14/18 | 77.8%     | +0.0021            |
| 3–7 (tied) | Qwen 2.5 / DeepSeek V3 / DeepSeek V4 Pro / Claude Opus 4.6 / GPT-5.5 | mixed      | 13/18 | 72.2%     | +0.0017 to +0.0026 |
| 8          | Gemini Flash 2.0                                                     | production | 12/18 | 66.7%     | +0.0012            |
| 9          | Llama 3.3 70B                                                        | production | 10/18 | 55.6%     | +0.0009            |
| 10         | **GPT-4o-mini**                                                      | production | 9/18  | **50.0%** | +0.0005            |


### How to read this table

Each LLM was evaluated across **18 conditions** (3 datasets × 2 prompt styles × 3 ML 
algorithms). **Wins** counts how many of those 18 augmented models beat their baselines
on the dataset's primary metric (ROC-AUC for classification, R² for regression). 
**Avg Δ** is the average improvement on that metric — so +0.0025 means the model's
score went from, say, 0.83 → 0.8325 on average.

A few rows worth pausing on:

- **Gemini 3.1 Pro = 83.3%** — the highest win rate in the study, but the
absolute improvement (+0.0025) is tiny in practical terms. A 0.83 → 0.8325
jump on ROC-AUC is real, but it's not "buy the frontier tier" real.
- **Claude Sonnet 4.5 = 77.8%** at production-tier prices — beats four of the
four frontier models on win rate while costing ~3× less than the cheapest
frontier alternative.
- **Five-way tie at 72.2%** — three frontier models (DeepSeek V4 Pro, Claude
Opus 4.6, GPT-5.5) and two production models (Qwen 2.5, DeepSeek V3) deliver
statistically indistinguishable win rates. The 100× cost gap inside that
tier is paying for nothing measurable.
- **GPT-4o-mini = 50.0%** — last place. Notable because GPT-4o-mini also had
high *validity* (93%, 4th place in Experiment A). A model that writes mostly
correct code can still produce mostly useless features.

The takeaway: even the **best LLM** in the study only improved the average ML
metric by +0.0025. Useful, but not transformative. The reason to use LLM
features isn't a 30-point accuracy lift — it's the time saved over manual
feature brainstorming, multiplied by a 70% chance the suggestions actually help.


### Result 3: Frontier vs. production tier

Pooling across all conditions:

| Tier                | Avg validity | Avg win rate | Avg cost / valid feature |
|---------------------|--------------|--------------|--------------------------|
| Production (6 LLMs) | 90.0%        | 64.8%        | $0.000344                |
| Frontier (4 LLMs)   | 95.3%        | 75.0%        | $0.004810                |

### How to read this table

Each row aggregates everything the tier did across the study. **Production** is
6 LLMs × 18 conditions = 108 data points; **Frontier** is 4 LLMs × 18 = 72.

The frontier tier wins ~10 percentage points more often (75% vs 64.8%) — but
pays **14× more per valid feature** for that lift. Whether that trade is worth
it depends on whether the win-rate gap is statistically real or sampling noise,
which is what the Mann-Whitney test (below) is for.

The absolute improvement is +0.002 average across metrics. A Mann-Whitney U test
comparing all production scores (n=180) to all frontier scores (n=120) returns
**p > 0.34 on every dataset**. The frontier advantage is statistically
indistinguishable from noise.


### Three things jump out
1. **The ML algorithm matters more than the LLM.** Moving from Logistic Regression
    to XGBoost shifts the average win rate by 26 percentage points across all 10 LLMs.
    Moving from the worst LLM (GPT-4o-mini, 50%) to the best (Gemini 3.1 Pro, 83%) 
    shifts it by 33. The two effects are comparable in size - you should put as much
    thought into your downstream model choice as into your LLM choice.
2. **Validity does not predict usefulness.** GPT-5.5 has the highest validity(100%)
    but only an average win rate(72.2%). GPT-4o-mini has high validity(93%) but 
    the *lowest* win rate(50%). Whether a feature is technically correct and 
    whether it actually helps are largely independent properties. Any benchmark
    for LLM-driven feature engineering should report both.
3. **The frontier premium is hard to defend.** The four frontier models win
    slightly more often, but at 14× the average cost. Claude Opus 4.6 has 
    *identical* validity and a *worse* win rate than its production sibling
    Claude Sonnet 4.5, at 1.7× the cost. Only DeepSeek V4 Pro retains a clean
    value argument among the frontier four.

---

## 4. The bottom line

If I had to pick a single default LLM for tabular feature engineering across a
varied set of problems, **Gemini Flash 2.0 is the strongest pragmatic choice**:
90% validity, 67% overall win rate, $0.000046 per valid feature, and the lowest
mean latency in the study (3.5s). It's not the best at any single metric, but it
doesn't lose to anything else by enough to matter. The frontier alternatives at
14× the cost don't reliably outperform it.

If the use case justifies the spend, **Gemini 3.1 Pro** is the most consistent
frontier performer (83% overall win rate). **DeepSeek V4 Pro** is the only
frontier model where the cost / quality ratio holds up under scrutiny. The other
two — GPT-5.5 and Claude Opus 4.6 — are hard to recommend over their production
counterparts.

The most actionable finding may not be about LLMs at all. **The downstream ML
algorithm dominates the equation.** XGBoost averages a 76% win rate across all
ten LLMs; Logistic Regression averages 50%. Four LLM × XGBoost combinations hit
a perfect 6/6, all of them production-tier or mid-tier on cost. The two best
production-grade pairings I'd recommend without hesitation:

- **Claude Sonnet 4.5 × XGBoost** — 6/6 win rate at $0.0017/valid feature
- **Qwen 2.5 72B × XGBoost** — 6/6 win rate at $0.000078/valid feature

The second-most actionable finding is methodological. Most LLM-feature-engineering
papers report only validity ("did the code execute?") or only model improvement
("did accuracy go up?"). The two are largely uncorrelated in my results. Any
serious evaluation needs to track both — and to budget for the cost gap, which
spans **two orders of magnitude** across the LLMs tested.

These results are one slice of a bigger question. Tabular feature engineering on
small-to-medium datasets is just one task. World knowledge matters less here than
on, say, document understanding; latency matters less than on agentic loops; data
volume is small enough that prompt context isn't a constraint. Other settings will
look different. But for the team running tabular models in a CI/CD loop with real
cost constraints, the answer is reasonably clear: **production-tier LLMs paired
with tree-based models are the default, and the burden of proof is on the
frontier**.

---

### Methodology notes

- **Reproducibility:** Random seed = 42 throughout. All 60 raw LLM responses saved to disk.
- **Honest baseline:** Bank Marketing's `duration` column was dropped — it leaks
the outcome (per UCI guidance). Many published comparisons don't.
- **No data leakage:** Imputation, scaling, and one-hot encoding fit on training
folds only.
- **Robust feature application:** Sandboxed `eval()` with restricted namespace.
Rejects formulas producing infinity, mixed types, or constants.
- **Statistical tests:** Paired t-tests for within-LLM comparisons; Mann-Whitney U
for cross-tier comparisons. Significance threshold p < 0.05.


### Cost summary

|                               |                                                            |
|-------------------------------|------------------------------------------------------------|
| LLMs tested                   | 10 (6 production + 4 frontier)                             |
| LLM API calls                 | 60                                                         |
| Feature suggestions generated | 420                                                        |
| Valid features applied        | ~378                                                       |
| ML experiments run            | 219 (9 baseline + 180 LLM-augmented + 30 frontier re-runs) |
| Total LLM API cost            | **~$0.82**                                                 |
| Total LLM time                | ~21 minutes                                                |


Code, raw LLM responses, and per-call cost data are public at
[https://github.com/Evangelidis91/llm-feature-engineering](https://github.com/Evangelidis91/llm-feature-engineering).

---

*Konstantinos Evangelidis*
*First AI/ML research project, May 2026*
