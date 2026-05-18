"""Data loading and basic cleaning for all 3 study datasets.

Each loader function returns (X, y):
X = features DataFrame (raw, not yet encoded)
y = target column

Train/test splits happen later in models.py — these loaders only
load and clean. One responsibility per file.
"""
import pandas as pd
import re
from sklearn.model_selection import train_test_split

from src.config import CHURN_FILE, HOUSING_FILE, BANK_FILE, RANDOM_SEED, TEST_SIZE


# =====================================================================
# Telco Customer Churn (Classification)
# =====================================================================
def load_churn(verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load and lightly clean the Telco Customer Churn dataset.

    Returns:
        X: features DataFrame (raw — no encoding yet)
        y: binary target (1 = churned, 0 = retained)
    """
    df = pd.read_csv(CHURN_FILE)

    # Customer ID is just an identifier — drop it before modeling
    df = df.drop(columns=["customerID"])

    # New customers have blank "TotalCharges" — convert to number, blanks become 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Convert "Yes" / "No" target into 1 / 0 - Encode
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    if verbose:
        print(f"Churn loaded: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"   Class balance: {y.mean():.1%} churned")

    return X, y


# =====================================================================
# Ames Housing (Regression)
# =====================================================================
def load_housing(verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load and lightly clean the Ames Housing dataset.

    Column names are sanitized (spaces → underscores) so they are
    valid Python identifiers. Without this, LLM formulas like
    `Gr Liv Area` would fail when our feature engineer tries to
    evaluate them.
    """
    df = pd.read_csv(HOUSING_FILE)

    # Drop ID columns if any exist
    for col in ["Order", "PID", "Id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Helper: turn any column name into a safe Python identifier
    # Example: "Gr Liv Area" → "Gr_Liv_Area", "1stFlrSF" → "_1stFlrSF"
    def clean(name: str) -> str:
        name = re.sub(r"[^\w]", "_", name)
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        if name and name[0].isdigit():
            name = "_" + name  # prepend _ if starts with digit
        return name

    df.columns = [clean(c) for c in df.columns]

    # SalePrice is what we want to predict
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    # If a column is more than 40% missing, drop it (too sparse to be useful)
    missing_pct = X.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.4].index.tolist()
    if high_missing:
        X = X.drop(columns=high_missing)
        if verbose:
            print(f"    Dropped {len(high_missing)} cols with >40% missing")

    if verbose:
        print(f"Housing loaded: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"    Target range: ${y.min():,.0f} – ${y.max():,.0f}")

    return X, y


# =====================================================================
# Bank Marketing (Classification)
# =====================================================================
def load_bank(verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load and lightly clean the UCI Bank Marketing dataset.

    The dataset records direct marketing campaigns (phone calls) from a
    Portuguese bank. The target is whether the client subscribed to a
    term deposit.

    Returns:
        X: features DataFrame
        y: binary target (1 = subscribed, 0 = did not)
    """

    # This CSV uses semicolons as separators (not commas)
    df = pd.read_csv(BANK_FILE, sep=";")

    # ⚠️ Drop "duration" — it leaks the answer.
    # Reason: this column is the call duration, which is only known AFTER the call.
    # If duration = 0, the outcome is automatically "no" (no call happened).
    # Including it would give artificially perfect predictions that don't generalize.
    # The UCI documentation explicitly recommends dropping it.
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Encode - Convert "yes" / "no" target into 1 / 0
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])

    if verbose:
        print(f"Bank loaded: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"   Class balance: {y.mean():.1%} subscribed")

    return X, y


# =====================================================================
# Shared utility
# =====================================================================
def get_train_test_split(X, y, stratify: bool = True):
    """Standard train/test split using project-wide settings.

    Uses the same random seed and test size everywhere so results are
    reproducible across notebooks.
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y if stratify else None,
    )


# =====================================================================
# Quick test
# =====================================================================
if __name__ == "__main__":
    print("Testing data loaders...\n")
    X_churn, y_churn = load_churn()
    print(f"   Sample churn columns: {list(X_churn.columns)[:5]}\n")

    X_house, y_house = load_housing()
    print(f"   Sample housing columns: {list(X_house.columns)[:5]}")

    X_bank, y_bank = load_bank()
    print(f"   Sample bank columns: {list(X_bank.columns)[:5]}")
