import streamlit as st
import pandas as pd
import joblib
import urllib.request  # To fetch files from GitHub

# ✅ Load the model from GitHub
MODEL_URL = "https://github.com/Ramiogue/safe-/blob/main/rf_model.pkl" 
urllib.request.urlretrieve(MODEL_URL, "rf_model.pkl")

# ✅ Load the model from the local file
model = joblib.load("rf_model.pkl")

st.title("Loan Default Prediction")
st.write("Enter details to predict if a loan will be fully paid or charged off.")

# User Inputs
loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500)
int_rate = st.number_input("Interest Rate (%)", min_value=0.1, max_value=50.0, step=0.1)
installment = st.number_input("Monthly Installment ($)", min_value=10, max_value=2000, step=10)
grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])

# Encoding inputs
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
home_mapping = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2}

grade = grade_mapping[grade]
home_ownership = home_mapping[home_ownership]

# Format inputs
input_data = pd.DataFrame([[loan_amount, int_rate, installment, grade, home_ownership]], 
                          columns=["loan_amount", "int_rate", "installment", "grade", "home_ownership"])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Fully Paid" if prediction[0] == 0 else "Charged Off"
    st.subheader(f"Prediction: {result}")
