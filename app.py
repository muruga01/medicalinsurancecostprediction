import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error # To show error margin

# --- Configuration ---
MEDICAL_INSURANCE_FILE = 'medical_insurance.csv'
MLFLOW_TRACKING_URI = "file:///tmp/mlruns_insurance" # Must match the URI used in training script
MODEL_NAME = "MedicalInsuranceCostPredictor" # Name used when registering the model
MODEL_STAGE = "Production" # Or "Staging", "None" if you didn't set a stage
# If you didn't register a specific version, you might need to load by run_id or a specific version number.
# For simplicity, we'll try to load the latest version of the registered model.

st.set_page_config(layout="wide", page_title="Medical Insurance Cost Predictor")

# --- Load Data and Model (Cached for performance) ---
@st.cache_data
def load_data():
    """Loads the medical insurance dataset."""
    try:
        df = pd.read_csv(MEDICAL_INSURANCE_FILE)
        df.drop_duplicates(inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: {MEDICAL_INSURANCE_FILE} not found. Please ensure it's in the same directory.")
        st.stop()

@st.cache_resource # Use st.cache_resource for models and other non-data objects
def load_model():
    """Loads the best performing model from MLflow Model Registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        # Load the latest version of the model from the registry
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        st.success(f"Successfully loaded model '{MODEL_NAME}' ({MODEL_STAGE} stage) from MLflow.")
        
        # Fetch the MAE from the best run's metrics for error margin display
        # This requires querying MLflow.
        # Let's assume the best model's MAE was logged and we can retrieve it.
        # This is a bit more involved; for simplicity, we'll use a placeholder or assume a known MAE.
        # A more robust solution would be to get run_id from model version and then fetch metrics.
        # For now, let's just use a reasonable MAE value or calculate it if we have X_test, y_test readily available.
        # Or, we can retrieve the MAE from the run that registered the model.
        
        # Placeholder for MAE. In a real scenario, you'd fetch this from MLflow.
        # You could iterate through runs in the experiment to find the run that registered this model version
        # and get its 'mae' metric.
        
        # For now, let's just re-calculate MAE on a small sample or use a typical value for demonstration
        # (This is not ideal for production, but suffices for a demo app)
        # A better way: In the training script, after registering the model, you could print the MAE of the best model.
        # For this demo, let's assume a typical MAE from our previous runs.
        
        # To get the MAE from the registered model's run:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        latest_version = max(versions, key=lambda v: v.version) # Get the latest version object
        
        # Get the run ID associated with this model version
        run_id_of_registered_model = latest_version.run_id
        
        # Fetch the run details to get metrics
        run_data = client.get_run(run_id_of_registered_model).data
        best_model_mae = run_data.metrics.get('mae', 2500.0) # Default if not found
        
        return model, best_model_mae
    except Exception as e:
        st.error(f"Error loading model from MLflow: {e}")
        st.warning("Please ensure MLflow tracking server is running and the model is registered correctly.")
        st.stop()

df = load_data()
model, model_mae = load_model()

# --- Preprocessing Function for User Input ---
# These encoders are re-initialized. In a production system, these should be saved
# as part of the pipeline or as separate artifacts with the model.
# For this demo, we assume the classes are consistent.
@st.cache_data
def get_encoders_and_columns(df_original):
    le_sex_temp = LabelEncoder()
    le_sex_temp.fit(df_original['sex']) # Fit on original data to ensure all categories are known

    le_smoker_temp = LabelEncoder()
    le_smoker_temp.fit(df_original['smoker']) # Fit on original data

    # Get one-hot encoded column names for regions
    # This needs to match exactly what was done in training
    temp_df_for_ohe = pd.get_dummies(df_original, columns=['region'], prefix='region', drop_first=True)
    ohe_region_cols = [col for col in temp_df_for_ohe.columns if col.startswith('region_')]

    return le_sex_temp, le_smoker_temp, ohe_region_cols

le_sex, le_smoker, ohe_region_cols = get_encoders_and_columns(load_data()) # Load original data again for encoders

def preprocess_user_input(user_data):
    """
    Transforms user input into the format expected by the trained model.
    Assumes the same feature engineering and encoding steps as training.
    """
    # Create a DataFrame for the single input
    input_df = pd.DataFrame([user_data])

    # Label Encoding for sex and smoker
    input_df['sex_encoded'] = le_sex.transform(input_df['sex'])
    input_df['smoker_encoded'] = le_smoker.transform(input_df['smoker'])

    # One-Hot Encoding for region
    # Initialize all region columns to 0
    for col in ohe_region_cols:
        input_df[col] = 0
    # Set 1 for the selected region
    selected_region_col = f"region_{input_df['region'].iloc[0]}"
    if selected_region_col in ohe_region_cols: # Ensure it's a valid column
        input_df[selected_region_col] = 1

    # Feature Engineering
    input_df['smoker_bmi_interaction'] = input_df['smoker_encoded'] * input_df['bmi']
    input_df['age_smoker_interaction'] = input_df['age'] * input_df['smoker_encoded']
    input_df['age_children_interaction'] = input_df['age'] * input_df['children']

    # Select and order columns to match X_train
    # This list must match the columns of X_train exactly!
    # Get the column order from the loaded data after preprocessing (excluding target and original categoricals)
    # This is a bit tricky without having X_train directly.
    # Let's define the expected feature order based on our training script:
    expected_features = [
        'age', 'bmi', 'children', 'sex_encoded', 'smoker_encoded',
        'region_northwest', 'region_southeast', 'region_southwest', # Assuming these are the ones created by drop_first=True
        'smoker_bmi_interaction', 'age_smoker_interaction', 'age_children_interaction'
    ]
    # Ensure all expected features are present in the input_df, fill with 0 if not (for OHE columns not selected)
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0 # This handles cases where a region column isn't created if it's the base category

    return input_df[expected_features]


# --- Streamlit App Layout ---
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("Predict your estimated medical insurance charges based on your profile.")

# --- Tabbed Interface ---
tab1, tab2 = st.tabs(["Prediction", "Exploratory Data Analysis"])

with tab1:
    st.header("Predict Your Insurance Cost")

    # --- User Input Sidebar ---
    st.sidebar.header("Your Profile")
    age = st.sidebar.slider("Age", 18, 100, 30)
    sex = st.sidebar.radio("Gender", ['male', 'female'])
    bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
    children = st.sidebar.slider("Number of Children", 0, 5, 0)
    smoker = st.sidebar.radio("Smoker", ['yes', 'no'])
    region = st.sidebar.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

    user_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }

    if st.sidebar.button("Predict Cost"):
        with st.spinner("Calculating estimated cost..."):
            processed_input = preprocess_user_input(user_data)
            prediction = model.predict(processed_input)[0]

            st.subheader("Estimated Medical Insurance Cost:")
            st.success(f"**${prediction:,.2f}**")

            st.info(f"This prediction has an approximate Mean Absolute Error (MAE) of **${model_mae:,.2f}**. "
                    "This means, on average, the model's predictions are off by this amount.")

            st.markdown("---")
            st.subheader("Your Input Summary:")
            st.write(user_data)
            st.write("Processed Input for Model:")
            st.dataframe(processed_input)

with tab2:
    st.header("Exploratory Data Analysis (EDA) Insights")
    st.markdown("Visualizations to understand key factors influencing medical insurance charges.")

    # Recreate EDA plots
    sns.set_style("whitegrid")

    # Plot 1: Distribution of Charges
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(df['charges'], kde=True, bins=30, ax=ax1)
    ax1.set_title('Distribution of Medical Charges')
    ax1.set_xlabel('Charges')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)
    st.markdown("---")

    # Plot 2: Charges vs. Age (Colored by Smoker)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='age', y='charges', hue='smoker', data=df, ax=ax2, palette='viridis')
    ax2.set_title('Charges vs. Age (Colored by Smoker)')
    ax2.set_xlabel('Age')
    ax2.set_ylabel('Charges')
    st.pyplot(fig2)
    st.markdown("---")

    # Plot 3: Charges vs. BMI (Colored by Smoker)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df, ax=ax3, palette='viridis')
    ax3.set_title('Charges vs. BMI (Colored by Smoker)')
    ax3.set_xlabel('BMI')
    ax3.set_ylabel('Charges')
    st.pyplot(fig3)
    st.markdown("---")

    # Plot 4: Charges by Smoker Status (Boxplot)
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='smoker', y='charges', data=df, ax=ax4, palette='pastel')
    ax4.set_title('Charges by Smoker Status')
    ax4.set_xlabel('Smoker')
    ax4.set_ylabel('Charges')
    st.pyplot(fig4)
    st.markdown("---")

    # Plot 5: Charges by Region (Boxplot)
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='region', y='charges', data=df, ax=ax5, palette='coolwarm')
    ax5.set_title('Charges by Region')
    ax5.set_xlabel('Region')
    ax5.set_ylabel('Charges')
    st.pyplot(fig5)
    st.markdown("---")

    st.markdown("""
    ### Key Insights from EDA:
    * **Smoker Status is Dominant:** Being a smoker significantly increases insurance charges.
    * **Age Matters:** Charges generally increase with age.
    * **BMI Impact:** Higher BMI tends to correlate with higher charges, especially for smokers.
    * **Regional Differences:** Some regions might have slightly different cost structures.
    """)

st.sidebar.markdown("---")
st.sidebar.info("This app uses a machine learning model trained on a public medical insurance dataset.")