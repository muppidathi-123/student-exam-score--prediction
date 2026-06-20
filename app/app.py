# Streamlit Web Application
# 7.2 — Load Model and Preprocessing Artifacts
# app/app.py

import streamlit as st
import pandas as pd
import joblib
import json
import glob
import os

# --- Page config (must be the first Streamlit command) ---
st.set_page_config(
    page_title="Student Exam Score Predictor",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model_files = [f for f in glob.glob('../models/*_model.pkl') if os.path.exists(f) and os.path.getsize(f) > 0]
    if not model_files:
        st.error("No model file found in /models. Run Phase 6 first.")
        st.stop()

    model = joblib.load(model_files[0])
    scaler = joblib.load('../models/scaler.pkl')

    with open('../models/ordinal_mappings.json', 'r') as f:
        ordinal_mappings = json.load(f)

    with open('../models/training_columns.json', 'r') as f:
        training_columns = json.load(f)

    model_name = os.path.basename(model_files[0]).replace('_model.pkl', '')

    return model, scaler, ordinal_mappings, training_columns, model_name

model, scaler, ordinal_mappings, training_columns, model_name = load_artifacts()

# 7.3 — Build the Reusable Prediction Function

def predict_exam_score(raw_input: dict):
    input_df = pd.DataFrame([raw_input])

    for col, order in ordinal_mappings.items():
        if col in input_df.columns:
            mapping = {category: i for i, category in enumerate(order)}
            input_df[col] = input_df[col].map(mapping)

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
    prediction = model.predict(input_scaled_df)[0]

    return round(prediction, 2)

#7.4 — Homepage / Header

st.title("📊 Student Exam Score Predictor")
st.markdown(
    """
    This tool predicts a student's exam score based on study habits, attendance,
    and other contributing factors — using a machine learning model trained on
    real student performance data.
    
    Fill in the details below and click **Predict** to see the estimated score.
    """
)
st.divider()

#7.5 — Input Form

st.subheader("📝 Student Information")

col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("Hours Studied per Week", 0, 40, 15)
    attendance = st.slider("Attendance (%)", 0, 100, 80)
    sleep_hours = st.slider("Sleep Hours per Night", 0, 12, 7)
    previous_scores = st.slider("Previous Exam Score", 0, 100, 70)
    tutoring_sessions = st.slider("Tutoring Sessions per Month", 0, 10, 2)
    physical_activity = st.slider("Physical Activity (hrs/week)", 0, 15, 3)

with col2:
    parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], index=1)
    access_to_resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], index=1)
    motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
    family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], index=1)
    teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], index=1)
    parental_education = st.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"], index=0)

st.markdown("##### Additional Details")
col3, col4, col5 = st.columns(3)

with col3:
    extracurricular = st.radio("Extracurricular Activities", ["Yes", "No"])
    internet_access = st.radio("Internet Access", ["Yes", "No"])

with col4:
    school_type = st.radio("School Type", ["Public", "Private"])
    learning_disabilities = st.radio("Learning Disabilities", ["Yes", "No"])

with col5:
    peer_influence = st.radio("Peer Influence", ["Positive", "Neutral", "Negative"])
    gender = st.radio("Gender", ["Male", "Female"])

distance_from_home = st.select_slider("Distance from Home", options=["Near", "Moderate", "Far"], value="Near")

#7.6 — Predict Button and Result Display

st.divider()

if st.button("🔮 Predict Exam Score", type="primary", use_container_width=True):

    raw_input = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Extracurricular_Activities': extracurricular,
        'Sleep_Hours': sleep_hours,
        'Previous_Scores': previous_scores,
        'Motivation_Level': motivation_level,
        'Internet_Access': internet_access,
        'Tutoring_Sessions': tutoring_sessions,
        'Family_Income': family_income,
        'Teacher_Quality': teacher_quality,
        'School_Type': school_type,
        'Peer_Influence': peer_influence,
        'Physical_Activity': physical_activity,
        'Learning_Disabilities': learning_disabilities,
        'Parental_Education_Level': parental_education,
        'Distance_from_Home': distance_from_home,
        'Gender': gender
    }

    with st.spinner("Calculating prediction..."):
        predicted_score = predict_exam_score(raw_input)

    # Clip display to a sensible 0-100 range in case the model overshoots slightly
    display_score = max(0, min(100, predicted_score))

    st.success("Prediction complete!")

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.metric(label="Predicted Exam Score", value=f"{display_score:.1f} / 100")

    st.progress(display_score / 100)

    # Simple contextual feedback
    if display_score >= 80:
        st.info("🌟 Strong predicted performance.")
    elif display_score >= 60:
        st.info("✅ Solid predicted performance, with room to improve.")
    else:
        st.warning("📌 This profile suggests additional support may help — consider more study time, attendance, or tutoring.")

#7.7 — Sidebar: About the Model

with st.sidebar:
    st.header("ℹ️ About This Model")
    st.markdown(f"""
    **Model type:** `{model_name.replace('_', ' ').title()}`
    
    **Trained on:** Student Performance Factors dataset  
    **Target:** Predicted exam score (0–100)
    
    This is a portfolio project demonstrating a complete
    machine learning workflow — from data analysis to
    deployment.
    """)
    st.divider()
    st.caption("Built with Python, Scikit-Learn, and Streamlit")

