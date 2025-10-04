import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
# Suppress a common warning from scikit-learn when using older joblib versions
warnings.filterwarnings('ignore') 

# Set up the Streamlit app
st.set_page_config(page_title="Insurance Cost Predictor", layout="wide")
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("---")

def main():
    
    st.markdown(
        """
        <style>
        .stButton>button {
            width: 100%;
            height: 3rem;
            font-size: 1.2rem;
            font-weight: bold;
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
        }
        .stAlert { border-radius: 10px; }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    if predictor_model is None:
        st.stop()
        
    # Create input fields for user
    st.header("Enter Your Details")

    with st.form("insurance_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 18, 64, 30)
            sex = st.selectbox("Sex", ['female', 'male'])
            
        with col2:
            bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            children = st.slider("Number of Children", 0, 5, 0)
            
        with col3:
            smoker = st.radio("Smoker?", ['no', 'yes'])
            region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])
            
        st.markdown("---")
        submitted = st.form_submit_button("💰 Predict Insurance Charges")

    if submitted:
        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        })
        
        # Make the prediction
        try:
            prediction = predictor_model.predict(input_data)[0]
            
            # Display results in a clear format
            st.success("## ✅ Prediction Successful")
            
            # Format the output card
            prediction_formatted = f"${prediction:,.2f}"
            
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
                    <h3 style="color: #1f77b4; margin-bottom: 5px;">Estimated Annual Insurance Charges:</h3>
                    <h1 style="color: #4CAF50; font-size: 3em; margin-top: 0px;">{prediction_formatted}</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.info("Disclaimer: This is a machine learning estimate based on synthetic data and should not be used as actual medical advice or a binding quote.")
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


# --- 1. DATA GENERATION AND MODEL TRAINING ---
# NOTE: In a real-world scenario, you would load an existing dataset (e.g., medical_insurance.csv)
# We generate synthetic data here for a runnable, self-contained example.

def generate_synthetic_data(n_samples=1000):
    """Generates synthetic medical insurance data."""
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 65, n_samples),
        'sex': np.random.choice(['male', 'female'], n_samples),
        'bmi': np.random.uniform(18.0, 45.0, n_samples),
        'children': np.random.randint(0, 5, n_samples),
        'smoker': np.random.choice(['yes', 'no'], n_samples),
        'region': np.random.choice(['northeast', 'northwest', 'southeast', 'southwest'], n_samples)
    })
    
    # Simple formula to generate charges (highly correlated with age, bmi, and smoking)
    charges = 1000 + 300 * data['age'] + 400 * data['bmi']
    charges[data['smoker'] == 'yes'] *= 4
    charges[data['sex'] == 'female'] += 500
    charges += np.random.normal(0, 5000, n_samples) # Add noise
    data['charges'] = charges.round(2)
    
    return data

@st.cache_resource
def train_and_save_model():
    """Trains the RandomForestRegressor model and preprocessing pipeline."""
    data = generate_synthetic_data()
    
    X = data.drop('charges', axis=1)
    y = data['charges']
    
    # Define categorical and numerical features
    categorical_features = ['sex', 'smoker', 'region']
    numerical_features = ['age', 'bmi', 'children']
    
    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Create the full pipeline with preprocessing and the model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    # Train the model
    model.fit(X, y)
    
    # In a real MLflow setup, you would log the model. Here we just return it.
    # Placeholder for MLflow (Actual MLflow setup requires external server and environment)
    # print("MLflow: Logging model artifacts and run parameters...")
    
    return model

# Load or train the model (Streamlit caches this function result)
try:
    predictor_model = train_and_save_model()
except Exception as e:
    st.error(f"Error during model training: {e}")
    predictor_model = None

# --- 2. STREAMLIT APPLICATION UI ---

if __name__ == "__main__":
    main()
