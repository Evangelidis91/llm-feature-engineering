"""Safely apply LLM-suggested feature formulas to a DataFrame.

Strategy:
- Each LLM suggestion has {name, formula, rationale}
- We try to evaluate the formula in a sandboxed namespace using Python's eval()
- Failures are caught and logged (not raised) so one bad feature doesn't kill
the run
- Returns the augmented DataFrame + a report on which features succeeded/failed

This produces two research metrics for free:
1. Validity rate per LLM (% of suggestions that were applicable)
2. The reasons for failures (typos, hallucinated columns, bad syntax)

Why eval() and not pd.eval()?
pd.eval() is too restrictive — it doesn't support method calls like .astype(int)
or .fillna(0), which LLMs use heavily. Plain eval() with a locked-down
namespace
is more permissive while still being safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# =====================================================================
# Allowed names in the eval namespace
# =====================================================================
# This dict defines what the LLM's formula is allowed to use.
# Anything NOT in here (e.g. open(), exec(), __import__) is unavailable —
# this is what makes eval() safe even though LLM output is untrusted.
SAFE_NAMESPACE: dict[str, Any] = {
    # Libraries — formulas need numpy for things like np.log1p() and np.where()
    "np": np,
    "pd": pd,
    # Type constructors — LLMs love .astype(int), .astype(float) for flag features
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    # Common builtins — math helpers LLMs frequently use
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
    """Outcome of trying to apply one LLM-suggested feature."""

    name: str
    formula: str
    rationale: str
    success: bool
    error: str | None = None


@dataclass
class ApplicationReport:
    """Summary of applying a list of LLM features to a dataset.

    Used to track validity rates per LLM × prompt variant — one of the
    headline research metrics in the study.
    """

    dataset_name: str
    llm: str
    prompt_variant: str
    n_suggested: int
    n_applied: int
    results: list[FeatureResult] = field(default_factory=list)

    @property
    def validity_rate(self) -> float:
        """Fraction of suggested features that successfully applied."""
        if self.n_suggested == 0:
            return 0.0
        return self.n_applied / self.n_suggested

    def summary(self) -> str:
        """Single-line human-readable summary, used in notebook output."""
        rate = self.validity_rate * 100
        return (
            f"{self.llm:14s} / {self.prompt_variant:12s} on {self.dataset_name:8s}: "
            f"{self.n_applied}/{self.n_suggested} valid ({rate:.0f}%)"
        )


# =====================================================================
# Core function
# =====================================================================
def apply_features(
    df: pd.DataFrame,
    suggestions: list[dict],
    dataset_name: str = "",
    llm: str = "",
    prompt_variant: str = "",
    verbose: bool = False,
) -> tuple[pd.DataFrame, "ApplicationReport"]:
    """Apply a list of LLM-suggested features to a DataFrame.

    For each suggestion, runs through 7 validation checks (in order):
      1. Name doesn't collide with an existing column
      2. Formula evaluates without raising
      3. Result can be coerced to a pandas Series
      4. Result doesn't contain infinity (would crash sklearn later)
      5. Result isn't entirely NaN
      6. Result has uniform dtype (no mixed int/str — sklearn can't handle them)
      7. Result isn't constant (won't help any model)

    A feature only gets added if it passes all 7. Each rejection is logged
    with a clear reason — that log becomes our research metric on LLM failure
    modes.

    Args:
        df: input DataFrame (won't be modified — we copy())
        suggestions: list of dicts with keys 'name', 'formula', 'rationale'
        dataset_name, llm, prompt_variant: metadata for the report
        verbose: if True, print each feature's outcome to stdout

    Returns:
        (augmented_df, ApplicationReport)
    """
    out = df.copy()
    report = ApplicationReport(
        dataset_name=dataset_name,
        llm=llm,
        prompt_variant=prompt_variant,
        n_suggested=len(suggestions),
        n_applied=0,
    )

    # Build the eval namespace: safe builtins + every column as a variable + `df` itself.
    # Exposing `df` lets formulas use df['col'] style (some LLMs like GPT-5.5 prefer it).
    # Exposing each column directly lets formulas use the bare-name style (most LLMs).
    namespace = {
        **SAFE_NAMESPACE,
        "df": out,
        **{c: out[c] for c in out.columns},
    }

    for suggestion in suggestions:
        name = suggestion.get("name", "<unnamed>")
        formula = suggestion.get("formula", "")
        rationale = suggestion.get("rationale", "")

        # Check 1: name collision with an existing column
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

        # Check 2: try evaluating the formula in our sandboxed namespace.
        # The empty {"__builtins__": {}} blocks access to dangerous builtins
        # like __import__, open, exec, etc.
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

        # Check 3: result must be coercible to a Series (catches scalars,
        # nested objects, weird types, etc.)
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

        # Booleans → int (so linear models treat them as numeric features)
        if new_col.dtype == bool:
            new_col = new_col.astype(int)

        # Check 4: reject infinity (e.g. from division by zero — sklearn won't accept it)
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

        # Check 5: reject all-NaN columns (no information to learn from)
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

        # Check 6: reject mixed types in object columns (e.g. some int, some str).
        # This happens when LLMs use incomplete .replace() mappings.
        # OneHotEncoder explicitly requires uniform types per column.
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

        # Check 7: reject constant features (every value identical → zero variance → zero info)
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

        # All 7 checks passed — add the feature
        out[name] = new_col
        # Also expose the new feature in the namespace so LATER features in
        # this same batch can reference it (chained feature engineering).
        namespace[name] = new_col
        report.results.append(
            FeatureResult(
                name=name, formula=formula, rationale=rationale, success=True
            )
        )
        report.n_applied += 1
        if verbose:
            print(f"   ✓ {name}")

    return out, report


# =====================================================================
# Quick test — runs only when this file is executed directly
# =====================================================================
if __name__ == "__main__":
    from src.data_loader import load_churn

    X, _ = load_churn(verbose=False)

    # 5 mock suggestions: 3 should succeed, 2 should fail with different errors.
    # This is a self-contained sanity check that the validation logic still works
    # even after future code changes.
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
        # ✗ Should fail check 2: column doesn't exist (NameError)
        {
            "name": "bad_feature",
            "formula": "FakeColumn * 2",
            "rationale": "this should be rejected — hallucinated column",
        },
        # ✗ Should fail check 2: invalid Python syntax (SyntaxError)
        {
            "name": "bad_syntax",
            "formula": "tenure +",
            "rationale": "this should be rejected — incomplete expression",
        },
    ]

    print("Testing feature engineer with 5 suggestions (3 should succeed):\n")
    augmented, report = apply_features(
        X,
        test_suggestions,
        dataset_name="churn",
        llm="gpt-4o-mini",
        prompt_variant="zero_shot",
        verbose=True,
    )

    print(f"\n{report.summary()}")
    print(f"Original cols:  {X.shape[1]}")
    print(f"Augmented cols: {augmented.shape[1]}")
    print(f"New columns added: {list(augmented.columns[-report.n_applied:])}")