"""Safely apply LLM-suggested feature formulas to a DataFrame.

Strategy:
- Each LLM suggestion has {name, formula, rationale}
- We try to evaluate the formula in a sandboxed namespace using pd.eval / python eval
- Failures are caught and logged (not raised) so one bad feature doesn't kill the run
- Returns the augmented DataFrame + a report on which features succeeded/failed

This produces two research metrics for free:
1. Validity rate per LLM (% of suggestions that were applicable)
2. The reasons for failures (typos, hallucinated columns, bad syntax)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# =====================================================================
# Allowed names in the eval namespace
# =====================================================================
SAFE_NAMESPACE: dict[str, Any] = {
    # Libraries
    "np": np,
    "pd": pd,
    # Type constructors (commonly used in .astype(int), .astype(float), etc.)
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    # Common builtins
     "abs": abs,
    "min": min,
    "max": max,
    "len": len,
     "round": round,
    "sum": sum,
}


# =====================================================================
# Result containers
# =====================================================================
@dataclass
class FeatureResult:
    """Outcome of trying to apply one LLm-suggested feature. """
    name: str
    formula: str
    rationale: str
    success: bool
    error: str | None = None


@dataclass
class ApplicationReport:
    """Full report from applying a list of LLM features to a dataset. """
    dataset_name: str
    llm: str
    prompt_variant: str
    n_suggested: int
    n_applied: int
    results: list[FeatureResult] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        if self.n_suggested == 0:
            return 0.0
        return self.n_applied / self.n_suggested

    def summary(self) -> str:
        rate = self.validity_rate * 100
        return (
            f"{self.llm:14s} / {self.prompt_variant:12s} on {self.dataset_name:8s}: "
            f"{self.n_applied}/{self.n_suggested} valid ({rate:.0f}%)"
        )


# =====================================================================
# Core function
# =====================================================================
import numpy as np
import pandas as pd
def apply_features(
    df: pd.DataFrame,
    suggestions: list[dict],
    dataset_name: str = "",
    llm: str = "",
    prompt_variant: str = "",
    verbose: bool = False,
) -> tuple[pd.DataFrame, "ApplicationReport"]:
    """Apply a list of LLM-suggested features to df.

    Args:
        df: input DataFrame (won't be modified)
        suggestions: list of dicts with keys 'name', 'formula', 'rationale'
        dataset_name, llm, prompt_variant: metadata for the report

    Returns:
        (augmented DataFrame, ApplicationReport)
    """
    out = df.copy()
    report = ApplicationReport(
        dataset_name=dataset_name,
        llm=llm,
        prompt_variant=prompt_variant,
        n_suggested=len(suggestions),
        n_applied=0,
    )

    # Build the eval namespace: safe builtins + every column as a variable
    namespace = {**SAFE_NAMESPACE, **{c: out[c] for c in out.columns}}

    for suggestion in suggestions:
        name = suggestion.get("name", "<unnamed>")
        formula = suggestion.get("formula", "")
        rationale = suggestion.get("rationale", "")

        # Skip if the feature already exists
        if name in out.columns:
            report.results.append(
                FeatureResult(
                    name=name,
                    formula=formula,
                    rationale=rationale,
                    success=False,
                    error="name already exists",
                )
            )
            if verbose:
                print(f"   ✗ {name}: name collision")
            continue

        # Try evaluating the formula
        try:
            new_col = eval(formula, {"__builtins__": {}}, namespace)
        except Exception as e:
            report.results.append(
                FeatureResult(
                    name=name,
                    formula=formula,
                    rationale=rationale,
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                )
            )
            if verbose:
                print(f"   ✗ {name}: {type(e).__name__}: {e}")
            continue

        # Validate the result shape
        try:
            new_col = pd.Series(new_col, index=out.index)
        except Exception as e:
            report.results.append(
                FeatureResult(
                    name=name,
                    formula=formula,
                    rationale=rationale,
                    success=False,
                    error=f"could not convert to Series: {e}",
                )
            )
            if verbose:
                print(f"   ✗ {name}: could not convert to Series")
            continue

        # Cast booleans to int
        if new_col.dtype == bool:
            new_col = new_col.astype(int)

        # Reject if result contains infinity (sklearn won't accept it)
        if pd.api.types.is_numeric_dtype(new_col) and np.isinf(new_col).any():
            report.results.append(
                FeatureResult(
                    name=name,
                    formula=formula,
                    rationale=rationale,
                    success=False,
                    error="contains infinity (likely division by zero)",
                )
            )
            if verbose:
                print(f"   ✗ {name}: contains infinity")
            continue

        # Reject if it's all NaN
        if new_col.isna().all():
            report.results.append(
                FeatureResult(
                    name=name,
                    formula=formula,
                    rationale=rationale,
                    success=False,
                    error="all NaN result",
                )
            )
            if verbose:
                print(f"   ✗ {name}: all NaN")
            continue

        # Reject if mixed types (sklearn encoders need uniform types)
        if new_col.dtype == object:
            non_null = new_col.dropna()
            if len(non_null) > 0:
                types = set(type(v).__name__ for v in non_null)
                if len(types) > 1:
                    report.results.append(
                        FeatureResult(
                            name=name,
                            formula=formula,
                            rationale=rationale,
                            success=False,
                            error=f"mixed types in column: {sorted(types)}",
                        )
                    )
                    if verbose:
                        print(f"   ✗ {name}: mixed types {sorted(types)}")
                    continue

        # Reject if constant
        if new_col.nunique(dropna=True) <= 1:
            report.results.append(
                FeatureResult(
                    name=name,
                    formula=formula,
                    rationale=rationale,
                    success=False,
                    error="constant result",
                )
            )
            if verbose:
                print(f"   ✗ {name}: constant")
            continue

        # Success!
        out[name] = new_col
        namespace[name] = new_col  # later features can reference this one
        report.results.append(
            FeatureResult(
                name=name,
                formula=formula,
                rationale=rationale,
                success=True,
            )
        )
        report.n_applied += 1
        if verbose:
            print(f"   ✓ {name}")

    return out, report


# =====================================================================
# Quick test
# =====================================================================
if __name__ == "__main__":
    from src.data_loader import load_churn

    X, _ = load_churn(verbose=False)

    # Mock suggestions (the same ones GPT-4o-mini gave us earlier)
    test_suggestions = [
        {
            "name": "monthly_charge_to_tenure_ratio",
            "formula": "MonthlyCharges / (tenure + 1)",
            "rationale": "value-per-month indicator",
        },
        {
            "name": "is_month_to_month",
            "formula": "(Contract == 'Month-to-month').astype(int)",
            "rationale": "month-to-month is the strongest churn signal",
        },
        {
            "name": "has_internet_service",
            "formula": "(InternetService != 'No').astype(int)",
            "rationale": "flags internet customers",
        },
        # This one should fail — column doesn't exist
        {
            "name": "bad_feature",
            "formula": "FakeColumn * 2",
            "rationale": "this should be rejected",
        },
        # This one should fail — division by zero won't fail in pandas, but
        # let's test a bad syntax case
        {
            "name": "bad_syntax",
            "formula": "tenure +",
            "rationale": "invalid python",
        },
    ]

    print("Testing feature engineer with 5 suggestions (3 should succeed):\n")
    augmented, report = apply_features(
        X, test_suggestions,
        dataset_name="churn",
        llm="gpt-4o-mini",
        prompt_variant="zero_shot",
        verbose=True,
    )

    print(f"\n{report.summary()}")
    print(f"Original cols:  {X.shape[1]}")
    print(f"Augmented cols: {augmented.shape[1]}")
    print(f"New columns added: {list(augmented.columns[-report.n_applied:])}")
