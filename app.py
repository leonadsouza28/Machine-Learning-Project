import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ✅ CUSTOM STYLING WITH BACKGROUND IMAGE
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("https://images.unsplash.com/photo-1588776814546-ec6c7d6c0d4d?auto=format&fit=crop&w=1350&q=80");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }

        [data-testid="stHeader"] {
            background: rgba(255, 255, 255, 0.0);
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.3);
        }

        .main {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 30px;
            border-radius: 10px;
            box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
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
            color: #333333;
        }
    </style>
""", unsafe_allow_html=True)


# Load the saved model and scaler
model = joblib.load('loan_approval_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Loan Approval Prediction", layout="centered")

# App title
st.markdown("<h1 class='title'>🏦 Loan Approval Prediction </h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>This app predicts whether a loan application will be approved or not using a trained Logistic Regression model.</p>", unsafe_allow_html=True)

# User input form
st.header("Enter Detailed Information.........")

# Collect inputs
Gender = st.selectbox("Gender", [' ','Male', 'Female'])
Married = st.selectbox("Married", [' ','Yes', 'No'])
Dependents = st.selectbox("Number of Dependents", [' ','0', '1', '2', '3+'])
Education = st.selectbox("Education", [' ','Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", [' ','Yes', 'No'])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.selectbox("Loan Amount (in thousands)", [' '] + list(range(1, 1001)))
Loan_Amount_Term = st.selectbox("Loan Term (in days)", [' '] + list(range(1, 1001)))
Credit_History = st.selectbox("Credit History", [' ','1', '0'])
Property_Area = st.selectbox("Property Area", [' ','Urban', 'Semiurban', 'Rural'])

# Prepare input for prediction
if st.button("Predict Loan Approval"):
    if ' ' in [Gender, Married, Dependents, Education, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]:
        st.warning("⚠️ Please fill in all fields before predicting.")
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
            st.success(f"✅ Loan will be Approved (Confidence: {prediction_proba:.2%})")
        else:
            st.error(f"❌ Loan will NOT be Approved (Confidence: {1 - prediction_proba:.2%})")
