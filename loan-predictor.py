
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn

# load the trained model and scaler
try:
    with open("regression.pkl", "rb") as file:
        model = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    st.error("model or scaler file not found. please ensure 'regression.pkl' and 'scaler.pkl' are in /content/")
    st.stop()

# title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #e0ffe0; padding: 10px; color: #336633;'><b>Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Loan Applicant's Details:")

# input fields for numerical values (with realistic ranges from EDA)
granted_loan_amount = st.slider("Granted Loan Amount", min_value=1000, max_value=200000, value=50000, step=1000)
requested_loan_amount = st.slider("Requested Loan Amount", min_value=1000, max_value=250000, value=60000, step=1000)
fico_score = st.slider("FICO Score", min_value=300, max_value=850, value=650, step=1)
monthly_gross_income = st.slider("Monthly Gross Income", min_value=500.0, max_value=20000.0, value=3000.0, step=100.0)
monthly_housing_payment = st.slider("Monthly Housing Payment", min_value=0.0, max_value=10000.0, value=1500.0, step=50.0)

# input fields for categorical values
reason_options = ['cover_an_unexpected_cost', 'credit_card_refinancing', 'home_improvement', 'major_purchase', 'debt_conslidation', 'other']
reason = st.selectbox("Reason for Loan", reason_options)

fico_score_group_options = ['poor', 'fair', 'good', 'very_good', 'excellent']
fico_score_group = st.selectbox("FICO Score Group", fico_score_group_options)

employment_status_options = ['full_time', 'part_time', 'unemployed']
employment_status = st.selectbox("Employment Status", employment_status_options)

employment_sector_options = ['Unknown', 'consumer_discretionary', 'information_technology', 'real_estate', 'energy', 'financials', 'industrials', 'health_care', 'consumer_staples', 'materials', 'communication_services']
employment_sector = st.selectbox("Employment Sector", employment_sector_options)

ever_bankrupt_foreclose = st.selectbox("Ever Bankrupt or Foreclosed?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

lender_options = ['A', 'B', 'C']
lender = st.selectbox("Lender", lender_options)

# the 'applications' column was always 1 in the training data and was not dropped
applications = 1

# create the input data as a DataFrame
input_data = pd.DataFrame({
    "applications": [applications],
    "Granted_Loan_Amount": [granted_loan_amount],
    "Requested_Loan_Amount": [requested_loan_amount],
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt_foreclose],
    "Reason": [reason],
    "Fico_Score_group": [fico_score_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# --- prepare Data for Prediction ---

# 1. separate numerical features for scaling
numerical_features_to_scale = [
    'Granted_Loan_Amount', 'Requested_Loan_Amount', 'FICO_score',
    'Monthly_Gross_Income', 'Monthly_Housing_Payment'
]
input_data[numerical_features_to_scale] = scaler.transform(input_data[numerical_features_to_scale])

# 2. one-hot encode categorical features
categorical_cols = ['Reason', 'Fico_Score_group', 'Employment_Status', 'Employment_Sector', 'Lender']
input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=False)

# 3. add any "missing" columns the model expects (fill with 0) and align columns
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 4. reorder/filter columns to exactly match the model's training data
input_data_processed = input_data_encoded[model_columns]

# predict button
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data_processed)[0]
    prediction_proba = model.predict_proba(input_data_processed)[0][1] # probability of Approved=1

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"the loan is predicted to be **Approved!**")
    else:
        st.error(f"the loan is predicted to be **Denied.**")
    st.info(f"probability of Approval: {prediction_proba:.2%}")

# optional: display input data for debugging
# st.subheader("Input Data (Processed):")
# st.dataframe(input_data_processed)
