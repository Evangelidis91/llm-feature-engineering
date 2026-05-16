"""LLM prompt templates for feature engineering suggestions.

We test TWO prompt variants per dataset:
1. zero_shot:  only column names + task description
2. with_stats: also includes basic statistics (mean, std, dtype)

This lets us measure whether giving the LLM more context helps.
"""
import pandas as pd

# =====================================================================
# Output format that all prompts ask for
# =====================================================================
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
    """Minimal prompt — only column names and task."""
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
    """Same task but also shows the LLM column dtypes and basic stats."""
    summary_lines = []
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique(dropna=True)
        n_missing = df[col].isnull().sum()

        if pd.api.types.is_numeric_dtype(df[col]):
            stats = (
                f"min={df[col].min():.2f}, "
                f"max={df[col].max():.2f}, "
                f"mean={df[col].mean():.2f}"
            )
        else:
            top = df[col].value_counts().head(3).index.tolist()
            stats = f"top values={top}"

        summary_lines.append(
            f"- {col} ({dtype}, unique={n_unique}, missing={n_missing}): {stats}"
        )

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
# Convenience: dataset-specific configs
# =====================================================================
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
# Quick test
# =====================================================================
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
