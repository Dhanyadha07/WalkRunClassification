import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('walkrun_model.pkl')
scaler = joblib.load("scaler.pkl")

# Streamlit
st.title("Walk-Run Prediction")
st.write("Enter the required values.")

# User input 
acceleration_x = st.number_input("Acceleration X", value=None)
acceleration_y = st.number_input("Acceleration Y", value=None)
acceleration_z = st.number_input("Acceleration Z", value=None)
gyro_x = st.number_input("Gyro X", value=None)
gyro_y = st.number_input("Gyro Y", value=None)
gyro_z = st.number_input("Gyro Z", value=None)

# Prediction
if st.button("Predict"):
    new_input = np.array([[acceleration_x,acceleration_y,acceleration_z,gyro_x,gyro_y,gyro_z]])
    input_scaled = scaler.transform(new_input)
    prediction = model.predict(input_scaled)
    
    # Result
    activity = "Running🏃🏻‍♀️🏃🏻‍♀️🏃🏻‍♀️" if prediction[0] == 1 else "Walking🚶🏻‍➡️🚶🏻‍➡️🚶🏻‍➡️"
    st.success(f"Predicted Activity: {activity}")