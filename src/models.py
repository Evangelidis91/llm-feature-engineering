"""Model training and evaluation pipelines.

This file does three things:
1. Builds a preprocessing pipeline that handles missing values, scales
   numeric columns, and one-hot encodes categorical columns.
2. Provides a small factory for the 3 ML model families we test
   (Logistic/Linear Regression, Random Forest, XGBoost).
3. Runs k-fold cross-validation and returns per-fold metrics so we
   can do statistical tests later.

Every notebook calls `evaluate_cv(...)` — the rest is internal.
"""

import time
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import CV_FOLDS, RANDOM_SEED

# Suppress noisy sklearn warnings (e.g. about feature names) — they don't
# affect results, just clutter the output during 200+ experiments.
warnings.filterwarnings("ignore", category=UserWarning)

# Type aliases — make function signatures readable and IDE-friendly
TaskType = Literal["classification", "regression"]
ModelName = Literal["logreg", "random_forest", "xgboost"]


# =====================================================================
# Preprocessing
# =====================================================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the same preprocessing pipeline every model uses.

    Two parallel paths:
      - Numeric columns:   fill missing values with median, then scale to mean=0, std=1
      - Categorical cols:  fill missing values with most frequent, then one-hot encode

    Important: we BUILD a fresh preprocessor each call so it can be fit on
    only the training fold (no data leakage from the test fold's stats).
    """
    # Auto-detect column types — no manual lists needed
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    # Numeric path: median imputation is robust to outliers (vs mean)
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical path: mode imputation, then one-hot.
    # handle_unknown="ignore" prevents crashes if a test fold sees a category
    # it didn't see during training (rare but possible with small datasets).
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",  # Drop any column that doesn't match either path
    )


# =====================================================================
# Model factory
# =====================================================================
def get_model(model_name: ModelName, task: TaskType):
    """Return a fresh model instance with sensible defaults.

    Same `model_name` returns different concrete models depending on the task:
      - "logreg" → LogisticRegression (classification) or Ridge (regression)
      - "random_forest" → RandomForestClassifier or RandomForestRegressor
      - "xgboost" → XGBClassifier or XGBRegressor

    All models use RANDOM_SEED for reproducibility and `n_jobs=-1` to use all CPU cores.
    """
    if task == "classification":
        if model_name == "logreg":
            # max_iter=2000 prevents convergence warnings on bigger datasets like Bank
            return LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        elif model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED
            )
        elif model_name == "xgboost":
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                eval_metric="logloss",
                verbosity=0,  # silence per-tree progress logs
            )
    else:  # regression
        if model_name == "logreg":
            # For regression, "logreg" maps to Ridge — same family (linear model with regularization)
            return Ridge(random_state=RANDOM_SEED)
        elif model_name == "random_forest":
            return RandomForestRegressor(
                n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED
            )
        elif model_name == "xgboost":
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                verbosity=0,
            )

    raise ValueError(f"Unknown model: {model_name}")


# =====================================================================
# Cross-validated evaluation
# =====================================================================
def evaluate_cv(
    X: pd.DataFrame, y: pd.Series, model_name: ModelName, task: TaskType
) -> dict:
    """Run k-fold cross-validation and return per-fold + summary metrics.

    Returns a dict with:
      - "model", "task" — what we ran
      - "<metric>_mean", "<metric>_std" — summary stats across folds
      - "<metric>_per_fold" — list of 5 individual fold scores (for paired t-tests later)
      - "fit_time_mean" — average seconds per fold

    Classification gets accuracy, F1, ROC-AUC.
    Regression gets RMSE, MAE, R².
    """
    # Pick the right CV splitter and metric set for the task
    if task == "classification":
        # Stratified = preserve class balance in every fold
        cv = StratifiedKFold(
            n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
        )
        metric_names = ["accuracy", "f1", "roc_auc"]
    else:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        metric_names = ["rmse", "mae", "r2"]

    # Containers for per-fold scores (one list per metric)
    per_fold = {m: [] for m in metric_names}
    fit_times = []

    # Loop over the k folds
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Build a fresh pipeline for each fold so the preprocessor is fit
        # on training data only — no data leakage from the test fold.
        pipe = Pipeline(
            [
                ("preprocess", build_preprocessor(X_train)),
                ("model", get_model(model_name, task)),
            ]
        )

        # Time just the fit step (predictions are fast in comparison)
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)

        # Score the fold
        if task == "classification":
            y_pred = pipe.predict(X_test)
            # ROC-AUC needs probabilities, not class labels
            y_proba = pipe.predict_proba(X_test)[:, 1]
            per_fold["accuracy"].append(accuracy_score(y_test, y_pred))
            per_fold["f1"].append(f1_score(y_test, y_pred))
            per_fold["roc_auc"].append(roc_auc_score(y_test, y_proba))
        else:
            y_pred = pipe.predict(X_test)
            per_fold["rmse"].append(root_mean_squared_error(y_test, y_pred))
            per_fold["mae"].append(mean_absolute_error(y_test, y_pred))
            per_fold["r2"].append(r2_score(y_test, y_pred))

    # Build the result dict: include both summary stats and raw per-fold scores
    result = {
        "model": model_name,
        "task": task,
        "fit_time_mean": float(np.mean(fit_times)),
    }
    for m, scores in per_fold.items():
        result[f"{m}_mean"] = float(np.mean(scores))
        result[f"{m}_std"] = float(np.std(scores))
        result[f"{m}_per_fold"] = scores  # kept for paired t-tests downstream

    return result


# =====================================================================
# Pretty printer
# =====================================================================
def format_result(result: dict) -> str:
    """Return a one-line human-readable summary of an evaluate_cv result.

    Used in the baseline notebook to print results as we run them.
    """
    if result["task"] == "classification":
        return (
            f"  {result['model']:14s} | "
            f"acc={result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}  "
            f"f1={result['f1_mean']:.4f}±{result['f1_std']:.4f}  "
            f"auc={result['roc_auc_mean']:.4f}±{result['roc_auc_std']:.4f}  "
            f"({result['fit_time_mean']:.2f}s/fold)"
        )
    return (
        f"  {result['model']:14s} | "
        f"rmse={result['rmse_mean']:,.0f}±{result['rmse_std']:,.0f}  "
        f"mae={result['mae_mean']:,.0f}±{result['mae_std']:,.0f}  "
        f"r2={result['r2_mean']:.4f}±{result['r2_std']:.4f}  "
        f"({result['fit_time_mean']:.2f}s/fold)"
    )


# =====================================================================
# Quick test — runs only when this file is executed directly
# =====================================================================
if __name__ == "__main__":
    from src.data_loader import load_churn

    X, y = load_churn(verbose=False)
    print("Quick test: logistic regression on churn (5-fold CV)\n")
    result = evaluate_cv(X, y, "logreg", "classification")
    print(format_result(result))