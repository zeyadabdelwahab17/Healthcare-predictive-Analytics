import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
pipeline = joblib.load(r"C:\Users\Lenovo\Downloads\Health_Care_Predictive_Analysis\RF_model.pkl")

# Page setup
st.set_page_config(page_title="Healthcare Diagnosis Predictor", layout="centered")

# Header
st.markdown(
    """
    <h2 style='text-align: center;'>🩺 Patient Diagnosis Prediction</h2>
    <p style='text-align: center;'>Enter patient details to predict the test result: <b>Normal</b>, <b>Abnormal</b>, or <b>Inconclusive</b>.</p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Input layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    condition = st.selectbox("Medical Condition", [
        "None", "Arthritis", "Hypertension", "Diabetes", 
        "Obesity", "Cancer", "Asthma", "Other"
    ])
    length_of_stay = st.number_input("Length of Stay (days)", min_value=1, max_value=365, value=5)

with col2:
    admission_type = st.selectbox("Admission Type", ["Emergency", "Elective", "Urgent"])
    billing_amount = st.number_input("Billing Amount ($)", min_value=0, max_value=100000, value=1000)
    insurance = st.selectbox("Insurance Provider", [
        "Uninsured", "Cigna", "Medicare", "UnitedHealthcare", "Blue Cross", "Aetna"
    ])
    blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
    medication = st.selectbox("Medication", [
        "None", "Lipitor", "Ibuprofen", "Aspirin", "Paracetamol", "Penicillin"
    ])

# Predict button
if st.button("🔍 Predict Test Result"):
    try:
        # Convert gender input to numeric (as used during training)
        gender = 1 if gender_input == "Male" else 0

        # Construct DataFrame in the exact order expected by the model
        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Medical Condition": condition,
            "Admission Type": admission_type,
            "Length of Stay": length_of_stay,
            "Billing Amount": billing_amount,
            "Insurance Provider": insurance,
            "Blood Type": blood_type,
            "Medication": medication
        }])

        st.subheader("📋 Input Summary:")
        st.write(input_data)

        # Make prediction
        prediction = pipeline.predict(input_data)[0]

        # Map prediction to readable label
        label_map = {0: "Normal", 1: "Abnormal", 2: "Inconclusive"}
        label = label_map.get(prediction, "Unknown")

        st.success(f"✅ Predicted Test Result: **{label}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {str(e)}")
        st.markdown(
            """
            **Troubleshooting Tips**:
            - Make sure all inputs are filled properly.
            - Ensure the model was trained with matching column names.
            - Recheck if the trained model supports string-based features.
            """
        )
