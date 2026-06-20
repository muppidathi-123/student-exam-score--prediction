"""
train_model.py
===============
Trains candidate regression models on the processed dataset and saves
the selected best model + preprocessing artifacts to disk.

This formalizes Phase 4 (Linear Regression baseline) and Phase 5
(model comparison) into a single repeatable script, instead of manual
notebook cells.

Usage:
    python src/train_model.py
"""

import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.data_preprocessing import run_preprocessing_pipeline, ORDINAL_MAPPINGS
from src.evaluate_model import evaluate_model, build_comparison_table


def get_candidate_models() -> dict:
    """
    Returns a dictionary of model_name -> untrained model instance.

    Centralizing this here means adding a new candidate model later
    is a one-line change, not a rewrite of the training loop.
    """
    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )

    return models


def train_all_models(X_train, y_train, X_test, y_test) -> dict:
    """
    Trains every candidate model and evaluates it on the test set.

    Parameters
    ----------
    X_train, y_train : training features/target
    X_test, y_test : test features/target

    Returns
    -------
    dict
        {model_name: {"model": trained_model, "metrics": {...}}}
    """
    candidates = get_candidate_models()
    results = {}

    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        results[name] = {"model": model, "metrics": metrics}
        print(f"  -> R2: {metrics['R2']:.4f}  MAE: {metrics['MAE']:.2f}  RMSE: {metrics['RMSE']:.2f}")

    return results


def select_best_model(results: dict, metric: str = "R2") -> str:
    """
    Selects the best-performing model by a given metric.

    Note: as discussed in Phase 5, the highest metric isn't automatically
    the "right" choice in every project (interpretability/speed tradeoffs
    matter too) — but for an automated pipeline, this gives a sensible
    default that a human can still override.

    Parameters
    ----------
    results : dict
        Output of train_all_models()
    metric : str
        Which metric to optimize for ("R2" higher is better;
        "MAE"/"RMSE" would need lower-is-better logic instead)

    Returns
    -------
    str
        Name of the best model
    """
    best_name = max(results, key=lambda name: results[name]["metrics"][metric])
    return best_name


def save_artifacts(model, model_name, scaler, training_columns, output_dir="models"):
    """
    Saves the trained model and all preprocessing artifacts needed for
    inference later (used by src/predict.py and the Streamlit app).

    Parameters
    ----------
    model : trained sklearn-compatible model
    model_name : str
        Used to name the saved file, e.g. "random_forest_model.pkl"
    scaler : fitted StandardScaler
    training_columns : list
        Exact column names/order the model expects
    output_dir : str
        Directory to save into
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{model_name}_model.pkl")
    joblib.dump(model, model_path)

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    with open(os.path.join(output_dir, "ordinal_mappings.json"), "w") as f:
        json.dump(ORDINAL_MAPPINGS, f, indent=2)

    with open(os.path.join(output_dir, "training_columns.json"), "w") as f:
        json.dump(training_columns, f, indent=2)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved scaler, ordinal mappings, and training columns to: {output_dir}/")


def main():
    """
    Full training pipeline entry point:
    preprocess -> train all candidates -> compare -> select best -> save.
    """
    print("=" * 60)
    print("STEP 1: Preprocessing data")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler, training_columns = run_preprocessing_pipeline(
        "data/raw/StudentPerformanceFactors.csv"
    )

    print("\n" + "=" * 60)
    print("STEP 2: Training candidate models")
    print("=" * 60)
    results = train_all_models(X_train, y_train, X_test, y_test)

    print("\n" + "=" * 60)
    print("STEP 3: Comparing models")
    print("=" * 60)
    comparison_df = build_comparison_table(results)
    print(comparison_df)

    best_name = select_best_model(results, metric="R2")
    print(f"\nBest model by R2: {best_name}")

    print("\n" + "=" * 60)
    print("STEP 4: Saving best model and artifacts")
    print("=" * 60)
    save_artifacts(
        model=results[best_name]["model"],
        model_name=best_name,
        scaler=scaler,
        training_columns=training_columns,
    )

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()