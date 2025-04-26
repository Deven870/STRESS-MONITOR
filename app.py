import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Load model and scaler
model = joblib.load("student_stress_model.pkl")
scaler = joblib.load("scaler.pkl")

# Questions
questions = [
    "On a scale of 0 to 30, what is your anxiety level?",
    "On a scale of 0 to 30, what is your self-esteem level?",
    "Have you ever had a history of mental health issues? (0 = No, 1 = Yes)",
    "On a scale of 0 to 30, how severe is your depression?",
    "On a scale of 0 to 10, how frequent are your headaches?",
    "Rate your blood pressure: (1 = Low, 2 = Normal, 3 = High)",
    "Rate your sleep quality from 1 (poor) to 5 (excellent).",
    "On a scale of 0 to 5, how severe are your breathing problems?",
    "On a scale of 0 to 5, how noisy is your environment?",
    "Rate your living conditions: (1 = Poor, 2 = Average, 3 = Good)",
    "How safe do you feel in your environment? (1 = Not safe, 5 = Very safe)",
    "Are your basic needs met? (1 = Not met, 5 = Fully met)",
    "Rate your academic performance: (1 = Poor, 5 = Excellent)",
    "Rate your study load: (1 = Very Light, 5 = Very Heavy)",
    "How would you rate your relationship with teachers? (1 = Bad, 5 = Excellent)",
    "How concerned are you about your future career? (1 = Not concerned, 5 = Extremely concerned)",
    "How much social support do you feel you have? (1 = None, 5 = Strong support)",
    "Rate peer pressure you experience: (1 = None, 5 = Extreme)",
    "How many extracurricular activities do you participate in per week? (0–5)",
    "On a scale of 0 to 5, how often do you face bullying?"
]

# Stress Level Mapping
stress_map = {
    1: "Low Stress",
    2: "Medium Stress",
    3: "High Stress"
}

# Session State Initialization
if 'page' not in st.session_state:
    st.session_state.page = "welcome"

if 'name' not in st.session_state:
    st.session_state.name = ""

if 'age' not in st.session_state:
    st.session_state.age = ""

if 'contact' not in st.session_state:
    st.session_state.contact = ""

if 'answers' not in st.session_state:
    st.session_state.answers = []

# Page 1: Welcome
if st.session_state.page == "welcome":
    st.title("🎉 Welcome to the Student Stress Level Predictor!")
    st.image("https://images.unsplash.com/photo-1525097487452-6278ff080c31", use_column_width=True)
    st.write("This tool helps you predict your stress level based on your lifestyle and habits.")
    if st.button("🚀 Start"):
        st.session_state.page = "details"
        st.experimental_rerun()

# Page 2: User Details
elif st.session_state.page == "details":
    st.title("👤 Please Enter Your Details")
    st.session_state.name = st.text_input("Full Name")
    st.session_state.age = st.text_input("Age")
    st.session_state.contact = st.text_input("Contact Information (Phone/Email)")

    if st.button("Next ➡️"):
        if not st.session_state.name or not st.session_state.age or not st.session_state.contact:
            st.error("Please fill in all fields.")
        else:
            st.session_state.page = "questions"
            st.experimental_rerun()

# Page 3: Questions
elif st.session_state.page == "questions":
    st.title("🧠 Answer the Following Questions")

    st.session_state.answers = []
    for i, q in enumerate(questions):
        answer = st.number_input(q, min_value=0.0, step=1.0, key=f"q{i}")
        st.session_state.answers.append(answer)

    if st.button("Predict Stress Level 🎯"):
        # Preprocessing
        X_new = scaler.transform([st.session_state.answers])
        prediction = model.predict(X_new)
        predicted_class = int(round(prediction[0]))
        result = stress_map.get(predicted_class, "Unknown Stress Level")

        # Show result
        st.success(f"🎯 Based on your answers, your stress level is: **{result}**")

        # Save to CSV
        record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": st.session_state.name,
            "Age": st.session_state.age,
            "Contact": st.session_state.contact,
            "Predicted Stress Level": result
        }
        for i, ans in enumerate(st.session_state.answers):
            record[f"Q{i+1}"] = ans

        csv_filename = "stress_predictions_log.csv"
        file_exists = os.path.isfile(csv_filename)
        df = pd.DataFrame([record])

        if file_exists:
            df.to_csv(csv_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_filename, index=False)

        st.info("📝 Your responses have been saved successfully!")

        # Option to restart
        if st.button("🔄 Predict for Another Student"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
