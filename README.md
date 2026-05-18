# 🤖 Can LLMs Replace Feature Engineering?

> An empirical study evaluating whether LLM-suggested features improve tabular ML models —
> across **10 LLMs (2 generations)**, **3 datasets**, **2 prompt styles**, and **3 model families**.

[](https://www.python.org/)
[](https://scikit-learn.org/)
[](https://opensource.org/licenses/MIT)

> 📄 **Read the full research writeup:** [Hiring 10 LLMs as Feature Engineers](docs/findings.md)

|
| **ML models** | Logistic/Linear Regression, Random Forest, XGBoost |
| **Evaluation** | 5-fold cross-validation, paired t-tests + Mann-Whitney U |

All LLMs accessed via [OpenRouter](https://openrouter.ai) for unified API handling.

---

## 📁 File Inventory

| File | Granularity | What's inside |
|---|---|---|
| `metrics_full.csv` ⭐ | per (dataset × model × feature_set × metric × fold) | The ground-truth file. **2,835 rows** covering 9 baselines + 180 LLM-augmented experiments × 3 metrics × 5 CV folds. Every chart, table, and finding is derived from this file. |
| `metrics.csv` | per (dataset × model × metric × fold) — baselines only | The 9 baseline experiments. Saved before any LLM augmentation, kept as a reference snapshot. |
| `llm_suggestions.json` | per (dataset × llm × prompt) | Parsed feature suggestions: 60 entries × ~7 features each = **420 suggestions** with `name` / `formula` / `rationale`. |
| `llm_metrics.csv` | per LLM API call | Tokens (input/output/total), cost (USD), latency, success/error for each of the **60 API calls**. |
| `validity_rates.csv` | per (dataset × llm × prompt) | How many features applied successfully against real data. Source of all "validity rate" numbers in the README. |
| `llm_outputs/*.txt` | per API call | Raw, unparsed text returned by each LLM. Kept for reproducibility — re-parse with a different parser anytime. |
| `figures/*.png` | — | All saved charts (validity heatmaps, cost analysis, baseline-vs-LLM, frontier-vs-production, improvement matrix). |

---

## 📊 Key Results

### 1️⃣ Frontier ≈ Production (Statistically Indistinguishable)

| Dataset | Baseline | Production tier | Frontier tier | Frontier vs Production |
| :--- | :--- | :--- | :--- | :--- |
| Churn (ROC-AUC) | 0.8324 | 0.8350 | 0.8348 | **−0.0002** ❌ |
| Housing (R²)    | 0.8926 | 0.8947 | 0.8965 | +0.0018 |
| Bank (ROC-AUC)  | 0.7881 | 0.7881 | 0.7884 | +0.0003 |

**Mann-Whitney U test:** p = 0.92, 0.34, 0.91 — no significant difference on any dataset, despite n=180 production vs n=120 frontier scores.

### 2️⃣ Cost Per Valid Feature (Lower = Better Value)

| Rank | LLM | Tier | Validity | Cost / Valid Feature | Mean Latency |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 🥇 | **Llama 3.3 70B** | production | 86% | **$0.000046** | 12.8s |
| 🥇 | **Gemini Flash 2.0** | production | 90% | **$0.000046** | 3.5s ⚡ |
| 3 | GPT-4o-mini | production | 93% | $0.000057 | 3.3s ⚡ |
| 4 | Qwen 2.5 72B | production | 95% | $0.000078 | 22.0s |
| 5 | DeepSeek V3 | production | 86% | $0.000119 | 40.5s |
| 6 | **DeepSeek V4 Pro** ⭐ | frontier | **98%** | $0.001219 | 53.7s |
| 7 | Claude Sonnet 4.5 | production | 90% | $0.001720 | 9.3s |
| 8 | Claude Opus 4.6 | frontier | 90% | $0.002989 | 11.7s |
| 9 | Gemini 3.1 Pro | frontier | 93% | $0.005414 | 22.6s |
| 10 | **GPT-5.5** | frontier | **100%** ⭐ | $0.007216 ⚠️ | 23.1s |

**Two takeaways:**
- **GPT-5.5 has the highest validity rate** (42/42) but the **highest cost** — 157× pricier than Gemini Flash for technical perfection that doesn't translate to better downstream model performance.
- **Open-source production models dominate the value rankings.** Claude Opus 4.6 matches Claude Sonnet 4.5 validity exactly (90%) at 1.7× the cost.

### 3️⃣ Win Rate per LLM (Did the Features Improve the Model?)

| Rank | LLM | Tier | Wins | Win Rate | Avg Δ |
| :---: | :--- | :--- | :---: | :---: | :--- |
| 🥇 | **Gemini 3.1 Pro** | frontier | 15/18 | **83.3%** | +0.0025 |
| 🥈 | Claude Sonnet 4.5 | production | 14/18 | 77.8% | +0.0021 |
| 🥉 | Qwen 2.5 (tie) | production | 13/18 | 72.2% | +0.0025 |
| 🥉 | DeepSeek V3 (tie) | production | 13/18 | 72.2% | +0.0021 |
| 🥉 | DeepSeek V4 Pro (tie) | frontier | 13/18 | 72.2% | +0.0017 |
| 🥉 | Claude Opus 4.6 (tie) | frontier | 13/18 | 72.2% | +0.0026 |
| 🥉 | GPT-5.5 (tie) | frontier | 13/18 | 72.2% | +0.0020 |
| 8 | Gemini Flash 2.0 | production | 12/18 | 66.7% | +0.0012 |
| 9 | Llama 3.3 70B | production | 10/18 | 55.6% | +0.0009 |
| 10 | **GPT-4o-mini** | production | 9/18 | **50.0%** | +0.0005 |

**The winner across all metrics:** Gemini 3.1 Pro (83% win rate). But the absolute improvement (+0.0025) is below the threshold of statistical significance when compared to the production tier as a whole.

**The biggest surprise:** GPT-4o-mini had high *validity* (93%) but the lowest *win rate* (50%) — its features were technically correct but rarely useful. This decouples "can the LLM write valid code?" from "does its code help the model?"

### 4️⃣ Validity ≠ Usefulness

A counterintuitive finding: **GPT-5.5 had perfect validity (100%) but only 72% win rate**, while **Qwen 2.5 had 95% validity and the same win rate**.

This suggests two distinct LLM capabilities:

| Capability | What it measures | Best LLM |
| :--- | :--- | :--- |
| **Code correctness** | Can it write executable feature formulas? | GPT-5.5 (100%) |
| **Feature usefulness** | Do the features actually help the model? | Gemini 3.1 Pro (83%) |

**A model that writes flawless code but useless features is worse value than one that writes mostly-valid features that genuinely improve predictions.** This reframes how to evaluate LLMs for feature engineering.

### 5️⃣ Win Rates by Dataset

| Dataset | Production wins | Frontier wins |
| :--- | :--- | :--- |
| Churn | 33/36 (**91.7%**) | 19/24 (79.2%) |
| Housing | 22/36 (61.1%) | 20/24 (**83.3%**) |
| Bank | 16/36 (44.4%) | 15/24 (**62.5%**) |

**Insight:** Frontier models help slightly more on harder datasets (Housing, Bank) but the absolute improvement is below the noise floor.

### 6️⃣ Random Forest Loves LLM Features (Both Tiers)

For Churn classification with **production-tier** features:

| Model | Wins / Total |
| :--- | :--- |
| Logistic Regression | 9/12 (75%) |
| **Random Forest** | **12/12 (100%)** ⭐ |
| **XGBoost** | **12/12 (100%)** ⭐ |

LLMs naturally generate flag features (e.g., `is_month_to_month`), which tree-based models exploit far better than linear models.

### 7️⃣ Four Categories of LLM Failure Modes

While applying **420 LLM-generated formulas** to real data, I found four systematic failure modes any production LLM-FE pipeline must handle:

| Failure Mode | Example | LLMs Most Affected |
| :--- | :--- | :--- |
| 🔤 **Hallucinated columns** | \`contract\` instead of \`Contract\` | Llama, Qwen |
| ➗ **Numeric instability** | Division by zero → infinity | DeepSeek, GPT |
| 🎭 **Type inconsistency** | Mixed \`int\` + \`str\` from incomplete \`.replace()\` mappings | Qwen |
| 🎩 **Stylistic verbosity** (frontier!) | \`df['col']\` prefix instead of bare column references | GPT-5.5 |

The fourth mode emerged only with frontier models. GPT-5.5 initially scored 0% on Housing because it generates standalone pandas snippets instead of bare formula expressions — a stylistic preference, not a technical limitation. After our evaluator was patched to accept both styles, GPT-5.5 reached 100% validity — but its win rate remained at 72%, identical to several production models costing 100× less.

### 8️⃣ With-Stats Prompts Help Weaker Models

Adding column statistics to prompts had asymmetric effects:

| LLM | zero-shot (bank) | with-stats (bank) | Δ |
| :--- | :---: | :---: | :---: |
| Gemini Flash | 57% | 86% | **+29%** 🚀 |
| DeepSeek V3 | 43% | 71% | +28% |
| GPT-4o-mini | 71% | 86% | +15% |
| Claude Sonnet | 71% | 71% | 0% |

Stronger models already use world knowledge effectively without extra context. Smaller / older models benefit dramatically from explicit statistics.

---

## 💰 Cost & Performance Summary

| Metric | Value |
| :--- | :--- |
| Total LLMs tested | 10 (6 production + 4 frontier) |
| Total LLM calls | 60 |
| Total feature suggestions | 420 |
| Valid features applied | ~378 (90% avg validity) |
| Total LLM cost | **$0.82** |
| Total LLM time | ~21 minutes |
| ML experiments run | 219 (9 baseline + 180 LLM-augmented + 30 frontier-only re-runs) |
| Statistically significant wins | 23 (p < 0.05) on production tier |

---

## 📁 Project Structure

\`\`\`
llm-feature-engineering/
├── data/raw/                       # Datasets (not tracked)
├── docs/
│   └── findings.md                 # Full research writeup
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_baseline.ipynb           # Baseline model training
│   ├── 03_llm_features.ipynb       # Production-tier LLM features (6 LLMs)
│   ├── 04_results.ipynb            # Production-tier comparison vs baseline
│   ├── 05_frontier_features.ipynb  # Frontier LLM features (4 LLMs)
│   └── 06_frontier_results.ipynb   # Frontier comparison + tier statistics
├── src/
│   ├── config.py                   # Central settings (paths, seeds, model IDs)
│   ├── data_loader.py              # Loaders for 3 datasets
│   ├── llm_client.py               # OpenRouter wrapper with retry + metrics
│   ├── prompts.py                  # Zero-shot and with-stats prompt templates
│   ├── feature_engineer.py         # Sandboxed evaluator with robust validation
│   └── models.py                   # CV pipeline (impute → encode → train)
├── results/
│   ├── figures/                    # Saved charts
│   ├── llm_outputs/                # Raw LLM responses (for reproducibility)
│   ├── llm_suggestions.json        # Parsed feature suggestions (10 LLMs)
│   ├── llm_metrics.csv             # Token / cost / latency per call
│   ├── validity_rates.csv          # Per-LLM × prompt × dataset validity
│   ├── metrics.csv                 # Baseline results
│   └── metrics_full.csv            # Baseline + LLM-augmented results
└── requirements.txt
\`\`\`

---

## 🛠️ Reproducing the Project

### 1. Clone and set up

\`\`\`bash
git clone https://github.com/Evangelidis91/llm-feature-engineering.git
cd llm-feature-engineering

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
\`\`\`

### 2. Configure your API key

\`\`\`bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
# Get one at: https://openrouter.ai/settings/keys
\`\`\`

### 3. Download the datasets

Place each CSV in \`data/raw/\`:

- **Telco Customer Churn**: [kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) → save as \`telco_churn.csv\`
- **Ames Housing**: [kaggle.com/datasets/prevek18/ames-housing-dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset) → save as \`ames_housing.csv\`
- **Bank Marketing**: [archive.ics.uci.edu/dataset/222](https://archive.ics.uci.edu/dataset/222/bank+marketing) → use \`bank-additional-full.csv\`, save as \`bank_marketing.csv\`

### 4. Run the notebooks in order

\`\`\`bash
jupyter notebook notebooks/01_eda.ipynb                # ~1 min
jupyter notebook notebooks/02_baseline.ipynb           # ~3 min
jupyter notebook notebooks/03_llm_features.ipynb       # ~10 min, ~$0.08
jupyter notebook notebooks/04_results.ipynb            # ~15 min
jupyter notebook notebooks/05_frontier_features.ipynb  # ~12 min, ~$0.68
jupyter notebook notebooks/06_frontier_results.ipynb   # ~10 min
\`\`\`

---

## 🧰 Methodology Notes

- **Random seed = 42** throughout for reproducibility
- **Bank's \`duration\` column dropped** — leakage per UCI guidance
- **Housing column names sanitized** — \`Gr Liv Area\` → \`Gr_Liv_Area\`
- **Preprocessing fit only on training folds** — no data leakage
- **Sandboxed \`eval()\`** — restricted namespace, no filesystem/network access
- **Robust feature validation** — rejects infinity, mixed-type, and constant outputs
- **Dual-style support** — accepts both bare column references (\`col\`) and \`df['col']\` style
- **All LLM responses saved to disk** before parsing
- **Statistical tests:** paired t-tests (within-LLM) + Mann-Whitney U (cross-tier)

---

## 🎓 What I Learned

This was my first AI/ML research project. Key takeaways:

- **Engineering matters as much as ML** — the most impactful work was building the sandboxed evaluator with four layers of validation
- **LLMs don't anticipate edge cases** — production systems need robust validation
- **Frontier ≠ better for narrow technical tasks** — newer/pricier doesn't translate to better feature engineering
- **Validity ≠ usefulness** — a 100%-valid LLM can still produce features that don't improve model performance
- **Honest negative results are valuable** — "no significant difference" is more interesting than yet another "X improves Y" story
- **Cost-aware engineering is a research metric** — tracking $/valid feature exposed counterintuitive value rankings

---

## 🚧 Limitations & Future Work

- Only 3 datasets (more would strengthen generalization claims)
- Each LLM called once per condition (no temperature variation)
- Single value of \`n_features = 7\` per call (sweep would be informative)
- No iterative feedback loop (LLM never sees model errors and retries)
- Only English-language prompts tested
- "Frontier" tier is bound to a specific snapshot in time (April 2026)

---

## 🛠️ Tech Stack

**Core:** Python 3.12 · pandas · NumPy · scikit-learn · XGBoost · scipy
**LLMs:** OpenRouter API (unified gateway for 10 models)
**Visualization:** matplotlib · seaborn
**Workflow:** Git · GitHub PR-based branching · Jupyter · PyCharm

---

## 📜 License

MIT — feel free to use, adapt, and build on this work.

---

## 🙏 Acknowledgments

Datasets courtesy of:

- IBM (Telco Customer Churn)
- Dean De Cock (Ames Housing, *Journal of Statistics Education*, 2011)
- S. Moro, P. Cortez, P. Rita (Bank Marketing, UCI ML Repository)

LLM access via [OpenRouter](https://openrouter.ai).

---

*Built with ❤️ as a first AI/ML research project.*
