"""LLM prompt templates for feature engineering suggestions.

We test TWO prompt variants per dataset:
1. zero_shot:  the LLM sees only the column names and the task description
2. with_stats: same as above, but also includes per-column statistics
               (dtype, min/max/mean for numeric; top categories for strings)

The goal is to measure whether giving the LLM more context (stats) actually
produces better feature suggestions, or whether the column names alone are
enough.
"""

import pandas as pd

# =====================================================================
# Output format that every prompt asks for
# =====================================================================
# We force the LLM to return strict JSON so we can parse it programmatically.
# Each suggestion has 3 fields: name, formula, rationale.
# The "Rules for the formula" section limits what the LLM can use, so its
# output is more likely to actually run when we evaluate it later.
JSON_SCHEMA_INSTRUCTION = """
Return ONLY a valid JSON array (no prose, no markdown fences) using this exact schema:
[
{
  "name": "snake_case_feature_name",
  "formula": "pandas-compatible expression using ONLY the columns listed above",
  "rationale": "one short sentence explaining why this feature might help"
}
]

Rules for the formula:
- Use only columns that exist in the provided list (exact spelling).
- Use standard pandas/numpy operations: + - * / , .clip(), .fillna(), np.log1p(), np.where(), .astype(int).
- For categorical comparisons use == with quoted string values, e.g. (contract == 'Month-to-month').
- Avoid groupby/aggregations — features must be row-level.
- Keep formulas short and self-contained.
"""


# =====================================================================
# Prompt builders
# =====================================================================
def build_zero_shot_prompt(
    dataset_name: str,
    task: str,
    target: str,
    columns: list[str],
    n_features: int = 7,
) -> str:
    """Build the minimal prompt — only the task and column names.

    The LLM gets nothing about the data itself except the column names.
    Tests whether world knowledge alone (e.g. "MonthlyCharges sounds like
    a money column") is enough to produce useful features.
    """
    # Comma-separate columns into one inline list
    cols_str = ", ".join(columns)
    return f"""You are a senior data scientist working on a {task} problem.

Dataset: {dataset_name}
Target variable: {target}
Available columns: {cols_str}

Suggest exactly {n_features} new engineered features that could improve a model predicting "{target}".
Focus on features that capture interactions, ratios, flags, or domain-meaningful combinations.

{JSON_SCHEMA_INSTRUCTION}
""".strip()


def build_with_stats_prompt(
    dataset_name: str,
    task: str,
    target: str,
    df: pd.DataFrame,
    n_features: int = 7,
) -> str:
    """Build the richer prompt — adds per-column statistics.

    For each column, include dtype, unique count, missing count, and either:
      - min / max / mean (numeric columns), or
      - top 3 most common values (categorical columns).

    This gives the LLM real numbers to reason about (e.g. "MonthlyCharges
    ranges from $18 to $120, mean $65 → a threshold near $70 makes sense").
    """
    summary_lines = []
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique(dropna=True)
        n_missing = df[col].isnull().sum()

        # For numeric columns: show range and mean
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = (
                f"min={df[col].min():.2f}, "
                f"max={df[col].max():.2f}, "
                f"mean={df[col].mean():.2f}"
            )
        # For categorical columns: show the 3 most common values
        else:
            top = df[col].value_counts().head(3).index.tolist()
            stats = f"top values={top}"

        summary_lines.append(
            f"- {col} ({dtype}, unique={n_unique}, missing={n_missing}): {stats}"
        )

    # Join into a multi-line block the LLM can read
    summary = "\n".join(summary_lines)

    return f"""You are a senior data scientist working on a {task} problem.

Dataset: {dataset_name}
Target variable: {target}

Column summary:
{summary}

Suggest exactly {n_features} new engineered features that could improve a model predicting "{target}".
Use the statistics above to inform your suggestions (e.g., spot ratios that make sense, identify
features that could be binned, or combinations that exploit the value ranges shown).

{JSON_SCHEMA_INSTRUCTION}
""".strip()


# =====================================================================
# Per-dataset configuration
# =====================================================================
# Each dataset has different metadata that the prompt needs (task type,
# target description, friendly display name). Keeping it here means the
# notebooks just say `DATASET_CONFIG["churn"]` without repeating themselves.
DATASET_CONFIG = {
    "churn": {
        "task": "binary classification",
        "target": "Churn (whether the customer left)",
        "display_name": "Telco Customer Churn",
    },
    "housing": {
        "task": "regression",
        "target": "SalePrice (house sale price in USD)",
        "display_name": "Ames Housing",
    },
    "bank": {
        "task": "binary classification",
        "target": "y (whether the client subscribed to a term deposit)",
        "display_name": "Bank Marketing",
    },
}

# =====================================================================
# Quick test — runs only when this file is executed directly
# =====================================================================
# Useful for eyeballing what the LLM actually sees before calling the API.
if __name__ == "__main__":
    from src.data_loader import load_churn

    X, _ = load_churn(verbose=False)
    cfg = DATASET_CONFIG["churn"]

    print("=" * 70)
    print("ZERO-SHOT PROMPT (preview):")
    print("=" * 70)
    p1 = build_zero_shot_prompt(
        dataset_name=cfg["display_name"],
        task=cfg["task"],
        target=cfg["target"],
        columns=list(X.columns),
    )
    print(p1)

    print("\n" + "=" * 70)
    print("WITH-STATS PROMPT (preview):")
    print("=" * 70)
    p2 = build_with_stats_prompt(
        dataset_name=cfg["display_name"],
        task=cfg["task"],
        target=cfg["target"],
        df=X,
    )
    print(p2[:1500] + "\n...[truncated]")