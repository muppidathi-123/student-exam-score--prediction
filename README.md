# 📊 Student Exam Score Predictor

A complete, end-to-end machine learning system that predicts a student's exam score
from study habits, attendance, and other contributing factors — built and deployed
as a portfolio project demonstrating the full ML lifecycle, from raw data to a
live web application.

**🔗 Live App:** [your-app-name.streamlit.app](https://your-app-name.streamlit.app)

---

## 🎯 Problem Statement

Educational institutions and EdTech platforms often want to identify students who
may be at risk of underperforming *before* an exam happens, so they can intervene
early. This project frames that need as a supervised regression problem: given
behavioral and demographic features about a student, predict their numeric exam score.

This is the same underlying pattern used in churn prediction, credit risk scoring,
and employee attrition modeling — an education dataset was chosen because it's
intuitive to explain and interpret, not because the technique is education-specific.

## 🧠 What This Project Demonstrates

- End-to-end ML workflow: EDA → preprocessing → modeling → evaluation → deployment
- Comparison of 4 regression algorithms with documented reasoning for model selection
- Production-style code organization (`src/` package, not just notebooks)
- A reusable inference pipeline that mirrors training-time preprocessing exactly
- A deployed, interactive Streamlit web application
- Clear, recruiter-readable documentation of every decision made and why

## 🏗️ Project Architecture
Raw Data (CSV)

│

▼

┌─────────────────────┐

│ data_preprocessing.py│   Clean → Encode → Split → Scale

└─────────┬────────────┘

▼

┌─────────────────────┐

│   train_model.py     │   Train Linear Regression, Decision Tree,

└─────────┬────────────┘   Random Forest, XGBoost → Compare → Select Best

▼

┌─────────────────────┐

│ models/*.pkl + .json │   Saved model + scaler + encoding metadata

└─────────┬────────────┘

▼

┌─────────────────────┐

│     predict.py       │   Reusable inference pipeline

└─────────┬────────────┘

▼

┌─────────────────────┐

│      app.py           │   Streamlit web interface

└─────────────────────┘   (deployed on Streamlit Community Cloud)


## 📁 Repository Structure
student-exam-marks-prediction/

├── data/

│   ├── raw/                  # Original dataset (untouched)

│   └── processed/            # Cleaned, encoded, split, scaled data

├── notebooks/

│   ├── 01_eda.ipynb          # Exploratory data analysis

│   ├── 02_preprocessing.ipynb

│   └── 03_modeling.ipynb

├── src/

│   ├── init.py

│   ├── data_preprocessing.py # Cleaning, encoding, splitting, scaling

│   ├── train_model.py        # Trains & compares all candidate models

│   ├── evaluate_model.py     # Metric computation (MAE, MSE, RMSE, R²)

│   └── predict.py            # Reusable prediction pipeline

├── models/

│   ├── *_model.pkl           # Best-performing trained model

│   ├── scaler.pkl            # Fitted StandardScaler

│   ├── ordinal_mappings.json # Category encoding reference

│   └── training_columns.json # Expected feature column order

├── app/

│   └── app.py                # Streamlit web application

├── reports/figures/          # Saved EDA visualizations

├── requirements.txt

├── .gitignore

└── README.md## 📊 Dataset

**Source:** [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) (Kaggle)

19 features covering study habits (hours studied, attendance, tutoring sessions),
environment (parental involvement, access to resources, school type), and personal
factors (sleep hours, motivation level, physical activity), predicting a continuous
`Exam_Score` target.

## 🔍 Key EDA Insights

- *[Fill in: e.g., "Hours_Studied and Previous_Scores showed the strongest positive
  correlation with Exam_Score (r ≈ X.XX)"]*
- *[Fill in: e.g., "Attendance showed a near-linear relationship with score"]*
- *[Fill in: dataset had X% missing values, concentrated in Y columns]*

## 🤖 Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | *fill in* | *fill in* | *fill in* |
| Decision Tree | *fill in* | *fill in* | *fill in* |
| Random Forest | *fill in* | *fill in* | *fill in* |
| XGBoost | *fill in* | *fill in* | *fill in* |

**Selected model:** *[fill in, e.g., "Random Forest"]* — chosen because *[fill in your
actual reasoning: e.g., "it offered the best R² while remaining fast enough for
real-time prediction, and its feature importance output adds interpretability that
a black-box model wouldn't provide"]*.

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/student-exam-marks-prediction.git
cd student-exam-marks-prediction

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain the model from scratch
python -m src.train_model

# Launch the web app
streamlit run app/app.py
```

## 🖼️ Screenshots

*[Add 2-3 screenshots here: the form, a prediction result, and the sidebar]*

```markdown
![App Screenshot](reports/figures/app_screenshot.png)
```

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `Scikit-Learn` `Matplotlib` `Seaborn` `Streamlit` `Joblib` `Git`

## 🔮 Future Improvements

- Add SHAP-based explainability for individual predictions
- Hyperparameter tuning via GridSearchCV/Optuna
- Add a confidence interval alongside the point prediction
- Containerize with Docker for more portable deployment
- Add automated tests for the preprocessing/prediction pipeline

## 👤 Author

*[Your name]* — *[LinkedIn]* · *[GitHub]* · *[Portfolio site, if any]*


