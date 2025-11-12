# Home.py
import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('best_model.pkl')

# App title
st.title("Medical Insurance Cost Predictor")
st.markdown("Enter your details below to get a personalized cost estimate.")

# User inputs
st.header("Enter Your Details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 64, 30)
    sex = st.selectbox("Gender", ["male", "female"])
    bmi = st.number_input("BMI", 15.0, 50.0, 25.0, step=0.1)

with col2:
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Feature engineering
bmi_category = (
    'underweight' if bmi < 18.5 else
    'normal' if bmi < 25 else
    'overweight' if bmi < 30 else
    'obese'
)
smoker_encoded = 1 if smoker == 'yes' else 0
age_smoker_interaction = age * smoker_encoded

# Prepare input
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'bmi': [bmi],
    'children': [children],
    'smoker': [smoker],
    'region': [region],
    'bmi_category': [bmi_category],
    'age_smoker_interaction': [age_smoker_interaction]
})

# Predict
if st.button("Predict Cost", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        prediction = model.predict(input_data)[0]
    st.success(f"### Estimated Medical Insurance Cost: **${prediction:,.2f}**")
    
    st.info("""
    **Note:** This is an estimate based on historical data and ML modeling.  
    Actual costs may vary based on provider, plan, and other factors.
    """)