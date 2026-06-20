"""
predict.py
==========
Reusable prediction pipeline for the trained model.

This is the SAME logic introduced in Phase 6 and used inside app/app.py —
defined once here so it's not duplicated between the app and any future
scripts (batch prediction, tests, an API endpoint, etc.).

Usage:
    from src.predict import StudentScorePredictor

    predictor = StudentScorePredictor(models_dir="models")
    score = predictor.predict({
        "Hours_Studied": 15,
        "Attendance": 88,
        "Parental_Involvement": "High",
        ...
    })
"""

import os
import glob
import json
import joblib
import pandas as pd


class StudentScorePredictor:
    """
    Loads a trained model and its preprocessing artifacts, and exposes
    a single .predict() method that takes raw, human-readable input
    and returns a predicted exam score.

    Wrapping this in a class (rather than loose functions) means the
    model/scaler/mappings are loaded ONCE when the object is created,
    then reused for every prediction — important for performance in
    an app that might serve many predictions.
    """

    def __init__(self, models_dir: str = "models"):
        """
        Parameters
        ----------
        models_dir : str
            Directory containing the saved model (.pkl), scaler (.pkl),
            and preprocessing metadata (.json) — i.e., the output of
            src/train_model.py or Phase 6.
        """
        self.models_dir = models_dir
        self.model, self.model_name = self._load_model()
        self.scaler = self._load_scaler()
        self.ordinal_mappings = self._load_json("ordinal_mappings.json")
        self.training_columns = self._load_json("training_columns.json")

    def _load_model(self):
        """Finds and loads whichever *_model.pkl file exists, regardless of name."""
        model_files = glob.glob(os.path.join(self.models_dir, "*_model.pkl"))
        if not model_files:
            raise FileNotFoundError(
                f"No model file found in '{self.models_dir}/'. "
                "Run src/train_model.py first (Phase 6)."
            )
        model_path = model_files[0]
        model = joblib.load(model_path)
        model_name = os.path.basename(model_path).replace("_model.pkl", "")
        return model, model_name

    def _load_scaler(self):
        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at '{scaler_path}'.")
        return joblib.load(scaler_path)

    def _load_json(self, filename: str):
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: '{path}'.")
        with open(path, "r") as f:
            return json.load(f)

    def _prepare_input(self, raw_input: dict) -> pd.DataFrame:
        """
        Transforms a raw, human-readable input dict into the exact
        numeric, scaled format the model expects.

        Steps (must mirror src/data_preprocessing.py exactly):
        1. Wrap dict into a single-row DataFrame
        2. Apply ordinal encoding (Low/Medium/High -> 0/1/2, etc.)
        3. Apply one-hot encoding for nominal columns
        4. Reindex to match training column order exactly,
           filling any missing one-hot column with 0
        5. Scale using the SAME scaler fitted during training
        """
        input_df = pd.DataFrame([raw_input])

        # Ordinal encoding
        for col, order in self.ordinal_mappings.items():
            if col in input_df.columns:
                mapping = {category: i for i, category in enumerate(order)}
                input_df[col] = input_df[col].map(mapping)

        # One-hot encoding for any remaining text columns
        input_df = pd.get_dummies(input_df)

        # Align to training columns exactly — critical step, see Phase 6.4
        input_df = input_df.reindex(columns=self.training_columns, fill_value=0)

        # Scale
        input_scaled = self.scaler.transform(input_df)

        return input_scaled

    def predict(self, raw_input: dict) -> float:
        """
        Predicts a student's exam score from raw feature values.

        Parameters
        ----------
        raw_input : dict
            e.g., {'Hours_Studied': 15, 'Attendance': 88, 'Gender': 'Male', ...}
            Keys should match the original dataset's column names.

        Returns
        -------
        float
            Predicted exam score, rounded to 2 decimal places.
        """
        input_scaled = self._prepare_input(raw_input)
        prediction = self.model.predict(input_scaled)[0]
        return round(float(prediction), 2)

    def predict_clipped(self, raw_input: dict, low: float = 0, high: float = 100) -> float:
        """
        Same as predict(), but clips the output to a sensible display range.

        Regression models have no awareness that exam scores can't exceed
        100 or go below 0 — this is purely a DISPLAY safeguard (see Phase 7.6),
        it does not change what the model actually predicted internally.
        """
        raw_prediction = self.predict(raw_input)
        return max(low, min(high, raw_prediction))


if __name__ == "__main__":
    # Quick manual test — run with: python src/predict.py
    predictor = StudentScorePredictor(models_dir="models")

    sample_input = {
        "Hours_Studied": 15,
        "Attendance": 88,
        "Parental_Involvement": "High",
        "Access_to_Resources": "Medium",
        "Extracurricular_Activities": "Yes",
        "Sleep_Hours": 7,
        "Previous_Scores": 75,
        "Motivation_Level": "Medium",
        "Internet_Access": "Yes",
        "Tutoring_Sessions": 2,
        "Family_Income": "Medium",
        "Teacher_Quality": "High",
        "School_Type": "Public",
        "Peer_Influence": "Positive",
        "Physical_Activity": 3,
        "Learning_Disabilities": "No",
        "Parental_Education_Level": "College",
        "Distance_from_Home": "Near",
        "Gender": "Male",
    }

    score = predictor.predict_clipped(sample_input)
    print(f"Model used: {predictor.model_name}")
    print(f"Predicted Exam Score: {score}")