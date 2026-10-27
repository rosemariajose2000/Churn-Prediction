import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load("svm_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# App title
st.title("Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the company.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, value=12)
monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, value=50.0
)

if st.button("Predict Churn"):
    # IMPORTANT: Match the gender encoding used in your notebook.
    gender_encoded = 0 if gender == "Male" else 1

    input_data = pd.DataFrame(
        [[age, gender_encoded, tenure, monthly_charges]],
        columns=["Age", "Gender", "Tenure", "MonthlyCharges"]
    )

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)

    if prediction[0] == 1:
        st.error("Prediction: Customer is likely to churn.")
    else:
        st.success("Prediction: Customer is likely to stay.")
