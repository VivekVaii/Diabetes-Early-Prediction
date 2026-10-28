import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Diabetes Early Prediction",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.joblib")

with open("diabetes_threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

model = load_model()

st.title("🩺 Diabetes Early Prediction")
st.write(
    "An educational machine-learning prototype "
    "for exploring diabetes classification."
)

st.warning(
    "This tool is for educational research only. "
    "It is not a medical diagnosis or a clinically validated risk estimate."
)

st.subheader("Enter sample patient measurements")

with st.form("prediction_form"):
    pregnancies = st.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1
    )
    glucose = st.number_input(
        "Glucose", min_value=0.0, max_value=300.0, value=120.0
    )
    blood_pressure = st.number_input(
        "Blood Pressure", min_value=0.0, max_value=200.0, value=70.0
    )
    skin_thickness = st.number_input(
        "Skin Thickness", min_value=0.0, max_value=100.0, value=20.0
    )
    insulin = st.number_input(
        "Insulin", min_value=0.0, max_value=1000.0, value=80.0
    )
    bmi = st.number_input(
        "BMI", min_value=0.0, max_value=80.0, value=25.0
    )
    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0, max_value=3.0, value=0.5
    )
    age = st.number_input(
        "Age", min_value=18, max_value=120, value=30
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    patient = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    probability = model.predict_proba(patient)[0, 1]
    prediction = probability >= threshold

    st.metric(
        "Model-estimated probability",
        f"{probability:.1%}"
    )

    if prediction:
        st.error(
            "The model classifies this sample as "
            "the diabetes-positive class."
        )
    else:
        st.success(
            "The model classifies this sample as "
            "the diabetes-negative class."
        )

    st.caption(
        f"Experimental classification threshold: {threshold:.2f}. "
        "This output must not be used to diagnose or rule out diabetes."
    )
