import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# Load the model
model = joblib.load('best_model.pkl')

# App title
st.title("Medical Insurance Cost Predictor")

# Display EDA insights
st.header("Insights from Data Analysis")
st.subheader("Impact of Smoking on Costs")
img1 = Image.open('smoker_charges.png')
st.image(img1, use_container_width=True)

st.subheader("Impact of Age on Costs by Smoking Status")
img2 = Image.open('age_charges.png')
st.image(img2, use_container_width=True)

st.subheader("Charges by Region")
img3 = Image.open('region_charges.png')
st.image(img3, use_container_width=True)

# User inputs
st.header("Enter Your Details")
age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Gender", ["male", "female"])
bmi = st.number_input("BMI", 15.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# Feature engineering for input
bmi_category = 'underweight' if bmi < 18.5 else 'normal' if bmi < 25 else 'overweight' if bmi < 30 else 'obese'
smoker_encoded = 1 if smoker == 'yes' else 0
age_smoker_interaction = age * smoker_encoded

# Prepare input DataFrame
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
if st.button("Predict Cost"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Medical Insurance Cost: ${prediction:.2f}")