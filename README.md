# 🤖 Can LLMs Replace Feature Engineering?

> An empirical study evaluating whether LLM-suggested features improve tabular ML models —
> across **6 LLMs**, **3 datasets**, **2 prompt styles**, and **3 model families**.

[](https://www.python.org/)
[](https://scikit-learn.org/)
[](https://opensource.org/licenses/MIT)

---

## 🎯 TL;DR

I asked 6 different LLMs to suggest engineered features for 3 tabular datasets,
applied those features automatically, and retrained models to see if they helped.

**Results across 108 experiments:**

- ✅ **LLM features beat baseline 66% of the time**
- 📊 **23 experiments showed statistically significant improvements** (p < 0.05)
- 💰 **Total LLM cost: $0.078** for 252 feature suggestions
- 🏆 **Open-source LLMs won 6 of 9 best-performer categories** despite costing
up to **37× less than Claude Sonnet 4.5**
- ⚠️ Identified **3 systematic LLM failure modes** that any production system must handle

---

## 🔬 Research Question

> If we hand an LLM raw tabular data and ask it to engineer new features,
> do those features actually improve a downstream ML model — and does this hold
> across different LLMs, prompt strategies, and datasets?

## 🧪 Experimental Design

| Dimension | Levels |
|---|---|
| **Datasets** | Telco Customer Churn (classification), Ames Housing (regression), Bank Marketing (classification) |
| **LLMs** | GPT-4o-mini, Claude Sonnet 4.5, Gemini 2.0 Flash, DeepSeek V3, Llama 3.3 70B, Qwen 2.5 72B |
| **Prompt styles** | Zero-shot (column names only), With-stats (column names + per-column statistics) |
| **ML models** | Logistic/Linear Regression, Random Forest, XGBoost |
| **Evaluation** | 5-fold cross-validation, paired t-tests for significance |

All LLMs accessed via [OpenRouter](https://openrouter.ai) for unified API handling.

---

## 📊 Key Results

### 1️⃣ Win Rates: LLM Features vs. Baseline

| Dataset | Win Rate | Best Improvement |
|---|---|---|
| **Churn** | **91.7%** (33/36) | DeepSeek V3 (with_stats) on RF: ROC-AUC +0.0058 |
| **Housing** | **61.1%** (22/36) | Claude Sonnet (zero_shot) on RF: R² +0.0094 (p=0.013) |
| **Bank** | **44.4%** (16/36) | Gemini Flash (zero_shot) on RF: ROC-AUC +0.0029 |

**Insight:** LLM features help most when the baseline has room to improve.
On the easiest dataset (Churn), they almost always win. On the hardest (Bank),
they win less than half the time.

### 2️⃣ Random Forest Loves LLM Features

For Churn classification:

| Model | Wins / Total |
|---|---|
| Logistic Regression | 9/12 (75%) |
| **Random Forest** | **12/12 (100%)** ⭐ |
| **XGBoost** | **12/12 (100%)** ⭐ |

LLMs naturally generate flag features (e.g., `is_month_to_month`), which
tree-based models exploit far better than linear models.

### 3️⃣ Cost Doesn't Predict Quality

| LLM | Cost / Valid Feature | Best-in-Category Wins |
|---|---|---|
| **Llama 3.3 70B** | $0.000046 ⭐ | 2 (Housing/LR & XGB) |
| Qwen 2.5 72B | $0.000078 | 2 (Churn/XGB & Bank/XGB) |
| DeepSeek V3 | $0.000119 | 2 (Churn/LR & RF) |
| Gemini Flash 2.0 | $0.000046 ⭐ | 1 (Bank/RF) |
| GPT-4o-mini | $0.000057 | 0 |
| **Claude Sonnet 4.5** | **$0.001720** (37× more!) | 2 (Housing/RF & Bank/LR) |

**Open-source LLMs won 6 of 9 categories.** Claude's 37× cost premium
delivered no consistent quality advantage for tabular feature engineering.

### 4️⃣ Three Categories of LLM Failure Modes

While applying 252 LLM-generated formulas to real data, I found three
systematic failure modes that any production LLM-FE system must handle:

| Failure Mode | Example | LLMs Most Affected |
|---|---|---|
| 🔤 **Hallucinated columns** | `contract` instead of `Contract` | Llama, Qwen |
| ➗ **Numeric instability** | Division by zero → infinity | DeepSeek, GPT |
| 🎭 **Type inconsistency** | Mixed `int` + `str` from incomplete `.replace()` mappings | Qwen |

This is rarely discussed in LLM-FE literature but critical for robustness.

### 5️⃣ With-Stats Prompts Help Weaker Models

Adding column statistics to prompts (min/max/mean) had asymmetric effects:

| LLM | zero-shot validity | with-stats validity | Δ |
|---|---|---|---|
| Gemini Flash (bank) | 57% | 86% | **+29%** 🚀 |
| DeepSeek V3 (bank) | 43% | 71% | +28% |
| GPT-4o-mini (bank) | 71% | 86% | +15% |
| Claude Sonnet (bank) | 71% | 71% | 0% |

Frontier models (Claude) need less context. Smaller models benefit dramatically
from extra information.

---

## 💰 Cost & Performance Summary

| Metric | Value |
|---|---|
| Total LLM calls | 36 |
| Total feature suggestions | 252 |
| Valid features applied | 227 |
| Total LLM cost | **$0.078** |
| Total LLM time | 9.3 minutes |
| ML experiments run | 117 (9 baseline + 108 LLM-augmented) |
| Statistically significant wins | 23 (p < 0.05) |

---

## 📁 Project Structure
llm-feature-engineering/ ├── data/raw/ # Datasets (not tracked — see download instructions) ├── notebooks/ │ ├── 01_eda.ipynb # Exploratory data analysis │ ├── 02_baseline.ipynb # Baseline model training │ ├── 03_llm_features.ipynb # LLM feature generation + cost tracking │ └── 04_results.ipynb # Apply LLM features + compare to baselines ├── src/ │ ├── config.py # Central settings (paths, seeds, model IDs) │ ├── data_loader.py # Loaders for 3 datasets │ ├── llm_client.py # OpenRouter wrapper with retry + metrics │ ├── prompts.py # Zero-shot and with-stats prompt templates │ ├── feature_engineer.py # Sandboxed evaluator with robust validation │ └── models.py # CV pipeline (impute → encode → train) ├── results/ │ ├── figures/ # Saved charts │ ├── llm_outputs/ # Raw LLM responses (for reproducibility) │ ├── llm_suggestions.json # Parsed feature suggestions │ ├── llm_metrics.csv # Token / cost / latency per call │ ├── validity_rates.csv # How many suggestions applied successfully │ ├── metrics.csv # Baseline results │ └── metrics_full.csv # Baseline + LLM-augmented results └── requirements.txt

---

## 🛠️ Reproducing the Project

### 1. Clone and set up the environment

```bash
git clone https://github.com/YOUR_USERNAME/llm-feature-engineering.git
cd llm-feature-engineering

python3.12 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure your API key
cp .env.example .env
# Edit .env and add your OpenRouter API key
# Get one at: https://openrouter.ai/settings/keys

### 3. Download the datasets

Download each CSV to data/raw/:

    Telco Customer Churn: kaggle.com/datasets/blastchar/telco-customer-churn
    → telco_churn.csv
    Ames Housing: kaggle.com/datasets/prevek18/ames-housing-dataset
    → ames_housing.csv
    Bank Marketing: archive.ics.uci.edu/dataset/222
    → use bank-additional-full.csv, save as bank_marketing.csv

### 4. Run the notebooks in order
jupyter notebook notebooks/01_eda.ipynb        # ~1 min
jupyter notebook notebooks/02_baseline.ipynb   # ~3 min
jupyter notebook notebooks/03_llm_features.ipynb  # ~10 min, ~$0.08 in API fees
jupyter notebook notebooks/04_results.ipynb    # ~15 min