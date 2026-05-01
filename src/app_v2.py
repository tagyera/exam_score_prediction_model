import streamlit as st
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

model = joblib.load("../models/model_v2/best_model_v2.pkl")

st.title("Student Exam Score Predictor")

age = st.slider("Age", 17, 27, 20)
study_hours = st.slider("Study Hours per Day", 0.0, 12.0, 2.0)
social_media = st.slider("Time(Hours) spend on Social Media", 0.0, 12.0, 3.0)
netflix = st.slider("Time(Hours) spend watching Netflix", 0.0, 10.0, 2.0)
attendance = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Rating (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)
exercise = st.slider("Time spend doing Exercise", 0.0, 7.0, 2.0)
part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
diet = st.selectbox("Diet Quality", ["Fair", "Good", "Poor"])
parent_education = st.selectbox(
    "Parents Education Level", ["High School", "Bachelor", "Master"]
)
internet = st.selectbox(
    "Internet Quality",
    [
        "Poor",
        "Average",
        "Good",
    ],
)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
extracurricular_participation = st.selectbox(
    "Extracurricular Participation", ["Yes", "No"]
)


ptj_encoded = 1 if part_time_job == "Yes" else 0
ep_encoded = 1 if part_time_job == "Yes" else 0

if gender == "Male":
    gender_encoded_male = 1
    gender_encoded_other = 0
elif gender == "Female":
    gender_encoded_male = 0
    gender_encoded_other = 0
elif gender == "Other":
    gender_encoded_male = 0
    gender_encoded_other = 1


if diet == "Fair":
    diet_encoded = 1
elif diet == "Poor":
    diet_encoded = 0
elif diet == "Good":
    diet_encoded = 2

if parent_education == "High School":
    parent_education_encoded = 0
elif parent_education == "Bachelor":
    parent_education_encoded = 1
elif parent_education == "Master":
    parent_education_encoded = 2


if internet == "Poor":
    internet_encoded = 0
elif internet == "Average":
    internet_encoded = 1
elif internet == "Good":
    internet_encoded = 2

if st.button("Predict Exam Score"):
    input_data = np.array(
        [
            [
                age,
                study_hours,
                social_media,
                netflix,
                attendance,
                sleep_hours,
                exercise,
                mental_health,
                diet_encoded,
                parent_education_encoded,
                internet_encoded,
                gender_encoded_male,
                gender_encoded_other,
                ptj_encoded,
                ep_encoded,
            ]
        ]
    )
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score: {prediction:.2f}")
