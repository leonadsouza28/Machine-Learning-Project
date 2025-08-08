import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered",
    page_icon="🏦"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #F4F6F6;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton>button {
            color: white;
            background-color: #0066cc;
            border-radius: 8px;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #004d99;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #003366;
        }
        .subtitle {
            font-size: 18px;
            color: #666666;
        }
        /* 🔵 Make input boxes blue */
        div[data-baseweb="select"] > div {
            background-color: #dbe9ff !important;
            border: 1px solid #0066cc !important;
            border-radius: 6px !important;
        }
        input[type="number"], input[type="text"] {
            background-color: #dbe9ff !important;
            border: 1px solid #0066cc !important;
            border-radius: 6px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Title & Subtitle ---
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">LOAN APPROVAL PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether a loan will be approved based on applicant information.</div>', unsafe_allow_html=True)

# --- Input Form ---
st.markdown("### 🔍 Enter Applicant Details")

with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ['','Male', 'Female'])
        Married = st.selectbox("Married", ['','Yes', 'No'])
        Dependents = st.selectbox("Number of Dependents", ['','0', '1', '2', '3+'])
        Education = st.selectbox("Education", ['','Graduate', 'Not Graduate'])
        Self_Employed = st.selectbox("Self Employed", ['','Yes', 'No'])

    with col2:
        ApplicantIncome = st.number_input("Applicant Income", min_value=0)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
        LoanAmount = st.selectbox("Loan Amount (in thousands)", [' '] + list(range(1, 1001)))
        Loan_Amount_Term = st.selectbox("Loan Term (in days)", [' '] + list(range(1, 1001)))
        Credit_History = st.selectbox("Credit History", ['','1', '0'])
        Property_Area = st.selectbox("Property Area", ['','Urban', 'Semiurban', 'Rural'])

    with st.container():
    st.markdown("""
        <div style="background-color: #dbe9ff; padding: 20px; border-radius: 10px; border: 1px solid #0066cc;">
            <h4 style="color: #003366;">Ready to Predict?</h4>
        </div>
    """, unsafe_allow_html=True)
    predict = st.button("🎯 Predict Loan Approval")


# --- Prediction Logic ---
if st.button("Predict Loan Approval"):
    if ' ' in [Gender, Married, Dependents, Education, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]:
        st.warning("Please fill in all fields before predicting.")
    else:
        # Convert input into DataFrame
        input_data = pd.DataFrame({
            'ApplicantIncome': [ApplicantIncome],
            'CoapplicantIncome': [CoapplicantIncome],
            'LoanAmount': [int(LoanAmount)],
            'Loan_Amount_Term': [int(Loan_Amount_Term)],
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
            st.success(f"Loan will be Approved (Confidence: {prediction_proba:.2%})")
        else:
            st.error(f"Loan will NOT be Approved (Confidence: {1 - prediction_proba:.2%})")

