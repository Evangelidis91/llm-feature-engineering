"""Model training and evaluation pipelines.

Provides a single function that:
- Builds a preprocessing pipeline (encode categoricals, scale numerics)
- Fits one of 3 model families (LR, RF, XGB)
- Returns cross-validated metrics
"""

import time
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, root_mean_squared_error, mean_absolute_error, \
    r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from src.config import CV_FOLDS, RANDOM_SEED

warnings.filterwarnings("ignore", category=UserWarning)

TaskType = Literal["classification", "regression"]
ModelName = Literal["logreg", "random_forest", "xgboost"]


# =====================================================================
# Preprocessing
# =====================================================================
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:


    """One-hot encode categoricals, scale numerics, impute missing values."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
         ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop"
)


# =====================================================================
# Model factory
# =====================================================================
def get_model(model_name: ModelName, task: TaskType):
    """Return a fresh model instance with sensible defaults."""
    if task == "classification":
        if model_name == "logreg":
            return LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        elif model_name == "random_forest":
            return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED)
        elif model_name == "xgboost":
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                eval_metric="logloss",
                verbosity=0
            )
    else:  # regression
        if model_name == "logreg":
            return Ridge(random_state=RANDOM_SEED)
        elif model_name == "random_forest":
            return RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED)
        elif model_name == "xgboost":
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                verbosity=0
            )

    raise ValueError(f"Unknown model: {model_name}")


# =====================================================================
# Cross-validated evaluation
# =====================================================================
def evaluate_cv(X: pd.DataFrame, y: pd.Series, model_name: ModelName, task: TaskType) -> dict:
    """Run k-fold CV and return mean metrics + per-fold scores.

      Returns a dict with:
        - model, task
        - mean values for each metric
        - std for each metric
        - per_fold lists for downstream stat tests
        - fit_time (mean seconds per fold)
      """

    if task == "classification":
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        metric_names = ["accuracy", "f1", "roc_auc"]
    else:
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        metric_names = ["rmse", "mae", "r2"]

    per_fold = {m: [] for m in metric_names}
    fit_times = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = Pipeline([
            ("preprocess", build_preprocessor(X_train)),
            ("model", get_model(model_name, task)),
        ])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        fit_times.append(time.perf_counter() - t0)

        if task == "classification":
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            per_fold["accuracy"].append(accuracy_score(y_test, y_pred))
            per_fold["f1"].append(f1_score(y_test, y_pred))
            per_fold["roc_auc"].append(roc_auc_score(y_test, y_proba))
        else:
            y_pred = pipe.predict(X_test)
            per_fold["rmse"].append(root_mean_squared_error(y_test, y_pred))
            per_fold["mae"].append(mean_absolute_error(y_test, y_pred))
            per_fold["r2"].append(r2_score(y_test, y_pred))

    result = {
        "model": model_name,
        "task": task,
        "fit_time_mean": float(np.mean(fit_times)),
    }

    for m, scores in per_fold.items():
        result[f"{m}_mean"] = float(np.mean(scores))
        result[f"{m}_std"] = float(np.std(scores))
        result[f"{m}_per_fold"] = scores

    return result


# =====================================================================
# Pretty printer
# =====================================================================
def format_result(result: dict) -> str:
    """Return a one-line human-readable summary."""
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
# Quick test
# =====================================================================
if __name__ == "__main__":
    from src.data_loader import load_churn

    X, y = load_churn(verbose=False)
    print("Quick test: logistic regression on churn (5-fold CV)\n")
    result = evaluate_cv(X, y, "logreg", "classification")
    print(format_result(result))
