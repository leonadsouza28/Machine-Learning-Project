# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# App title
st.title("🏦 Loan Approval Prediction App")
st.write("This app predicts whether a loan application will be approved or not using a trained Logistic Regression model.")

# User input form
st.header("🔍 Applicant Information")

# Collect inputs
Gender = st.selectbox("Gender", ['Male', 'Female'])
Married = st.selectbox("Married", ['Yes', 'No'])
Dependents = st.selectbox("Number of Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=1)
Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=1)
Credit_History = st.selectbox("Credit History", ['1', '0'])
Property_Area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# Prepare input for prediction
if st.button("Predict Loan Approval"):

    # Convert input into DataFrame
    input_data = pd.DataFrame({
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [int(Credit_History)],
        'Gender_Male': [1 if Gender == 'Male' else 0],
        'Married_Yes': [1 if Married == 'Yes' else 0],
        'Dependents_1': [1 if Dependents == '1' else 0],
        'Dependents_2': [1 if Dependents == '2' else 0],
        'Dependents_3+': [1 if Dependents == '3+' else 0],
        'Education_Not Graduate': [1 if Education == 'Not Graduate' else 0],
        'Self_Employed_Yes': [1 if Self_Employed == 'Yes' else 0],
        'Property_Area_Semiurban': [1 if Property_Area == 'Semiurban' else 0],
        'Property_Area_Urban': [1 if Property_Area == 'Urban' else 0]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.success(f"✅ Loan will be Approved (Confidence: {prediction_proba:.2%})")
    else:
        st.error(f"❌ Loan will NOT be Approved (Confidence: {1 - prediction_proba:.2%})")
