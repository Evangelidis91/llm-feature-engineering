# 🤖 Can LLMs Replace Feature Engineering?

An empirical study testing whether LLM-suggested features improve tabular ML models.

## 🚧 Status

Work in progress — this is my first AI/ML research project.

## 🧪 Plan

- **Datasets:** Telco Churn, Ames Housing, Bank Marketing
- **LLMs:** GPT-4o-mini, Claude Sonnet 4.5, Gemini 2.0 Flash, DeepSeek V3, Llama 3.3 70B, Qwen 2.5 72B (all via [OpenRouter](https://openrouter.ai))
- **Models:** Logistic/Linear Regression, Random Forest, XGBoost
- **Goal:** Compare baseline vs. LLM-augmented features across datasets and models.


## 💰 Cost Efficiency Analysis

Generating 252 feature suggestions across 36 LLM calls cost **$0.0784 total**
and took **9.3 minutes**. Per-LLM efficiency varied dramatically:

| LLM | Validity | Cost/Valid Feature | Mean Latency |
|---|---|---|---|
| Gemini Flash 2.0 | 90% | $0.000046 | 3.5s |
| Llama 3.3 70B | 86% | $0.000046 | 12.8s |
| GPT-4o-mini | 93% | $0.000057 | 3.3s |
| Qwen 2.5 72B | **95%** ⭐ | $0.000078 | 22.0s |
| DeepSeek V3 | 86% | $0.000119 | 40.5s |
| Claude Sonnet 4.5 | 90% | $0.001720 | 9.3s |

**Key finding:** Claude Sonnet costs ~37× more than Gemini Flash for similar validity rates, suggesting that for tabular feature engineering, lightweight
models match frontier-model quality at a fraction of the cost.