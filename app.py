import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Diabetes Progression Predictor")

input_data = st.text_input("Enter 10 normalized feature values separated by commas:")

if st.button("Predict"):
    try:
        values = np.array([float(x) for x in input_data.split(',')]).reshape(1, -1)
        # Optionally scale input if user gives raw values (not needed if already normalized)
        # values = scaler.transform(values)
        result = model.predict(values)
        st.write(f"Predicted Disease Progression: {result[0]:.2f}")
    except Exception as e:
        st.error(f"Error in input or prediction: {e}")
