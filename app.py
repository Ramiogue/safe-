import streamlit as st
import pandas as pd
import joblib
import os

# ✅ Title of the Web App
st.title("Loan Default Prediction")
st.write("Enter details to predict if a loan will be fully paid or charged off.")

# ✅ Load the Model (Ensure the file exists before loading)
MODEL_PATH = "rf_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found! Please check if 'rf_model.pkl' is in the correct directory.")

# ✅ User Inputs
loan_amount = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, step=500)
funded_amount = st.number_input("Funded Amount ($)", min_value=500, max_value=50000, step=500)
int_rate = st.number_input("Interest Rate (%)", min_value=0.1, max_value=50.0, step=0.1)
installment = st.number_input("Monthly Installment ($)", min_value=10, max_value=2000, step=10)
grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN"])

# ✅ Loan Purpose Selection (One-Hot Encoding)
loan_purpose = st.selectbox("Loan Purpose", [
    "credit_card", "debt_consolidation", "home_improvement", "house",
    "major_purchase", "medical", "moving", "other",
    "small_business", "vacation", "wedding"
])

# ✅ Feature Encoding (Ensure Model Matches Input Data)
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
home_mapping = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2}

grade = grade_mapping[grade]
home_ownership = home_mapping[home_ownership]

# ✅ Define the Required Features in Correct Order
correct_feature_order = [
    'loan_amount', 'funded_amount', 'int_rate', 'installment', 'grade', 'home_ownership',
    'purpose_credit_card', 'purpose_debt_consolidation', 'purpose_home_improvement',
    'purpose_house', 'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
    'purpose_other', 'purpose_small_business', 'purpose_vacation', 'purpose_wedding'
]

# ✅ Create DataFrame for User Input
input_data = pd.DataFrame([[loan_amount, funded_amount, int_rate, installment, grade, home_ownership]], 
                          columns=['loan_amount', 'funded_amount', 'int_rate', 'installment', 'grade', 'home_ownership'])

# ✅ One-Hot Encode Loan Purpose (Set Unselected Purposes to 0)
for purpose in correct_feature_order[6:]:  # Ignore first 6 numeric features
    input_data[purpose] = 1 if f"purpose_{loan_purpose}" == purpose else 0

# ✅ Ensure Correct Column Order
input_data = input_data[correct_feature_order]

# ✅ Prediction Button
if st.button("Predict"):
    if "model" in globals():
        prediction = model.predict(input_data)
        result = "Fully Paid" if prediction[0] == 0 else "Charged Off"
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Model is not loaded. Check if 'rf_model.pkl' is correctly placed.")
