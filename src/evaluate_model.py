"""
evaluate_model.py
==================
Reusable evaluation functions for regression models.

Computes MAE, MSE, RMSE, and R2 — the four metrics introduced in Phase 4 —
and provides a helper to build a side-by-side comparison table across
multiple trained models (used in Phase 5).

Usage:
    from src.evaluate_model import evaluate_model, build_comparison_table

    metrics = evaluate_model(y_test, y_pred)
    print(metrics)  # {'MAE': ..., 'MSE': ..., 'RMSE': ..., 'R2': ...}
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred) -> dict:
    """
    Compute the standard regression evaluation metrics for a set of
    predictions against actual values.

    Metric meanings (see Phase 4 for full explanations):
    - MAE  : average absolute prediction error, in original units (exam points)
    - MSE  : average squared error; penalizes large errors more heavily
    - RMSE : square root of MSE; brings units back to exam points
    - R2   : proportion of variance in the target explained by the model (0-1)

    Parameters
    ----------
    y_true : array-like
        Actual target values
    y_pred : array-like
        Model's predicted values

    Returns
    -------
    dict
        {'MAE': float, 'MSE': float, 'RMSE': float, 'R2': float}
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Builds a sorted comparison table across multiple trained models.

    Parameters
    ----------
    results : dict
        Format: {model_name: {"model": ..., "metrics": {...}}}
        (this is exactly what train_model.train_all_models() returns)

    Returns
    -------
    pd.DataFrame
        Rows = model names, columns = MAE/MSE/RMSE/R2, sorted by R2 descending
    """
    table = {name: data["metrics"] for name, data in results.items()}
    df = pd.DataFrame(table).T
    df = df.sort_values("R2", ascending=False)
    return df


def baseline_metrics(y_train, y_test) -> dict:
    """
    Computes the "naive baseline" metrics — i.e., what you'd get by
    always predicting the mean of the training target.

    This is useful as a sanity check (see Phase 4.7): any real model
    should clearly outperform this baseline, proving it has learned
    something beyond "just guess the average."

    Parameters
    ----------
    y_train : array-like
        Training target values (used to compute the mean)
    y_test : array-like
        Test target values (used to evaluate the baseline)

    Returns
    -------
    dict
        Same shape as evaluate_model()'s output
    """
    baseline_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
    return evaluate_model(y_test, baseline_pred)


def print_evaluation_report(model_name: str, metrics: dict):
    """
    Pretty-prints a single model's evaluation metrics in plain language.

    Parameters
    ----------
    model_name : str
    metrics : dict
        Output of evaluate_model()
    """
    print(f"\n--- {model_name} ---")
    print(f"MAE  : {metrics['MAE']:.2f}  (avg. prediction error, in exam points)")
    print(f"MSE  : {metrics['MSE']:.2f}  (squared error; penalizes large misses)")
    print(f"RMSE : {metrics['RMSE']:.2f}  (error in original units, penalizes large misses)")
    print(f"R2   : {metrics['R2']:.4f}  ({metrics['R2']*100:.1f}% of score variance explained)")


if __name__ == "__main__":
    # Quick smoke test with synthetic data
    y_true_sample = [70, 80, 90, 60, 75]
    y_pred_sample = [72, 78, 85, 65, 74]
    metrics = evaluate_model(y_true_sample, y_pred_sample)
    print_evaluation_report("Smoke Test Model", metrics)