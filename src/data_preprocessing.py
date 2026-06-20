"""
data_preprocessing.py
======================
Reusable data preprocessing functions for the Student Exam Score
Prediction project.

This module turns the manual steps from Phase 3 (notebooks/02_preprocessing.ipynb)
into functions that can be called consistently from anywhere — training
scripts, tests, or future re-runs — instead of being copy-pasted.

Typical usage:
    from src.data_preprocessing import load_data, clean_data, encode_features, split_and_scale

    df = load_data("data/raw/StudentPerformanceFactors.csv")
    df = clean_data(df)
    df, scaler, training_columns = encode_features(df)
    X_train, X_test, y_train, y_test = split_and_scale(df, scaler)
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Ordinal columns and their natural order (Low < Medium < High, etc.)
# Defined once here so training and inference always use the SAME mapping —
# this is the single source of truth referenced throughout the project.
# ---------------------------------------------------------------------------
ORDINAL_MAPPINGS = {
    'Parental_Involvement': ['Low', 'Medium', 'High'],
    'Access_to_Resources': ['Low', 'Medium', 'High'],
    'Motivation_Level': ['Low', 'Medium', 'High'],
    'Family_Income': ['Low', 'Medium', 'High'],
    'Teacher_Quality': ['Low', 'Medium', 'High'],
    'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
    'Distance_from_Home': ['Near', 'Moderate', 'Far'],
}

# Nominal columns (no natural order) — one-hot encoded
NOMINAL_COLUMNS = [
    'Extracurricular_Activities',
    'Internet_Access',
    'School_Type',
    'Peer_Influence',
    'Learning_Disabilities',
    'Gender',
]

TARGET_COLUMN = 'Exam_Score'


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file (e.g., 'data/raw/StudentPerformanceFactors.csv')

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Make sure you've downloaded it into data/raw/ (see Phase 2)."
        )
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataset: handle missing values, remove duplicates,
    and cap outliers using the IQR method.

    This mirrors the decisions made in Phase 3:
    - Numeric missing values -> filled with median (robust to outliers)
    - Categorical missing values -> filled with mode (most frequent category)
    - Duplicates -> dropped entirely
    - Outliers -> capped (winsorized), not deleted, to preserve sample size

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe, as returned by load_data()

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe, ready for encoding.
    """
    df = df.copy()  # never mutate the caller's original dataframe

    # --- Missing values ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # --- Duplicates ---
    df = df.drop_duplicates()

    # --- Outliers (IQR capping), skip the target column ---
    for col in numeric_cols:
        if col == TARGET_COLUMN:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)

    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categorical features: ordinal mapping for ordered categories,
    one-hot encoding for unordered categories.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe, as returned by clean_data()

    Returns
    -------
    df_encoded : pd.DataFrame
        Fully numeric dataframe, ready for modeling.
    training_columns : list
        Final column names/order — must be saved and reused at inference
        time so new input rows match the model's expected shape exactly.
    """
    df = df.copy()

    # Ordinal encoding — preserves Low < Medium < High order
    for col, order in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            mapping = {category: i for i, category in enumerate(order)}
            df[col] = df[col].map(mapping)

    # One-hot encoding for nominal (unordered) categories
    nominal_present = [c for c in NOMINAL_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_present, drop_first=True)

    training_columns = [c for c in df.columns if c != TARGET_COLUMN]

    return df, training_columns


def split_and_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into train/test sets and apply feature scaling.

    Critically: the scaler is FIT only on training data, then applied
    (transformed) to test data — this prevents data leakage from the
    test set into training statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Fully encoded, numeric dataframe (output of encode_features())
    test_size : float
        Fraction of data reserved for testing (default 0.2 = 20%)
    random_state : int
        Seed for reproducible splits

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_processed_data(X_train, X_test, y_train, y_test, output_dir: str = "data/processed"):
    """
    Save the processed train/test splits to disk as CSV files.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
        Output of split_and_scale()
    output_dir : str
        Directory to save processed files into (created if it doesn't exist)
    """
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    print(f"Processed data saved to '{output_dir}/'")


def run_preprocessing_pipeline(raw_filepath: str, output_dir: str = "data/processed"):
    """
    Convenience function that runs the FULL preprocessing pipeline end to end:
    load -> clean -> encode -> split -> scale -> save.

    This is what train_model.py calls — it's the single entry point so
    preprocessing logic never has to be duplicated elsewhere.

    Parameters
    ----------
    raw_filepath : str
        Path to the raw CSV dataset
    output_dir : str
        Where to save the processed train/test splits

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, training_columns
    """
    df = load_data(raw_filepath)
    df = clean_data(df)
    df, training_columns = encode_features(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
    save_processed_data(X_train, X_test, y_train, y_test, output_dir)

    return X_train, X_test, y_train, y_test, scaler, training_columns


if __name__ == "__main__":
    # Allows running this file directly for a quick standalone test:
    #   python src/data_preprocessing.py
    X_train, X_test, y_train, y_test, scaler, cols = run_preprocessing_pipeline(
        "data/raw/StudentPerformanceFactors.csv"
    )
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)