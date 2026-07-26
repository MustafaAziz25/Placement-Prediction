import streamlit as st
import numpy as np
import joblib

# Load trained model
logistic_model = joblib.load("logistic_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 Campus Placement Prediction")
st.write("Enter Student Details")
st.divider()

# Input fields
cgpa = st.number_input(
    "CGPA",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

internships = st.number_input(
    "Internships",
    min_value=0,
    max_value=10,
    step=1
)

aptitude = st.number_input(
    "Aptitude Score",
    min_value=0,
    max_value=100,
    step=1
)

communication = st.slider(
    "Communication Skills",
    min_value=1,
    max_value=10,
    value=5
)

projects = st.number_input(
    "Projects",
    min_value=0,
    max_value=20,
    step=1
)

st.divider()

# Prediction button
if st.button("Predict"):
    test = np.array([
        [
            cgpa,
            internships,
            aptitude,
            communication,
            projects
        ]
    ])

    prediction = logistic_model.predict(test)[0]

    if prediction == 1:
        st.success("🏆 Student is likely to be placed")
    else:
        st.error("❌ Student is not likely to be placed")