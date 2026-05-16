"""Data loading and basic cleaning for both datasets.

Each loader returns (X, y). The train/test split happens in models.py
so that loaders have a single responsibility.
"""
import pandas as pd
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

    # Drop customer ID (not predictive)
    df = df.drop(columns=["customerID"])

    # 'TotalCharges' has blank strings for new customers — coerce to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    if verbose:
        print(f"✅ Churn loaded: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"   Class balance: {y.mean():.1%} churned")

    return X, y


# =====================================================================
# Ames Housing (Regression)
# =====================================================================
import re
import pandas as pd


def load_housing(verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load and lightly clean the Ames Housing dataset.

    Column names are sanitized (spaces → underscores) so they are
    valid Python identifiers. Without this, LLM formulas using
    backtick syntax like `Gr Liv Area` fail in our eval-based
    feature engineer.
    """
    df = pd.read_csv(HOUSING_FILE)

    # Drop ID columns if present
    for col in ["Order", "PID", "Id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 🔧 Sanitize column names so LLM formulas can reference them safely
    def clean(name: str) -> str:
        name = re.sub(r"[^\w]", "_", name)  # non-word chars → _
        name = re.sub(r"_+", "_", name)  # collapse multiple _
        name = name.strip("_")
        if name and name[0].isdigit():
            name = "_" + name  # prepend _ if starts with digit
        return name

    df.columns = [clean(c) for c in df.columns]

    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    # Drop columns with >40% missing values
    missing_pct = X.isnull().mean()
    high_missing = missing_pct[missing_pct > 0.4].index.tolist()
    if high_missing:
        X = X.drop(columns=high_missing)
        if verbose:
            print(f"    Dropped {len(high_missing)} cols with >40% missing")

    if verbose:
        print(f"✅ Housing loaded: {X.shape[0]} rows × {X.shape[1]} features")
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
    # The CSV uses ';' as separator
    df = pd.read_csv(BANK_FILE, sep=";")

    # The 'duration' column is a known data leakage feature:
    # it records the call duration AFTER the call happens — so it cannot
    # realistically be used to predict the outcome ahead of time.
    # The UCI documentation explicitly recommends dropping it for
    # honest predictive modeling.
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    # Encode target (column is 'y' with values 'yes'/'no')
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])

    if verbose:
        print(f"✅ Bank loaded: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"   Class balance: {y.mean():.1%} subscribed")

    return X, y


# =====================================================================
# Shared utility
# =====================================================================
def get_train_test_split(X, y, stratify: bool = True):
    """Standard train/test split using project-wide settings."""
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
