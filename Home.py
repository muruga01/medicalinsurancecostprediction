# Home.py
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ------------------- Page Config -------------------
st.set_page_config(
    page_title="MediCost AI – Insurance Predictor",
    page_icon="medical_symbol",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------- MEDICAL-THEMED + HIGH-READABILITY STYLING -------------------
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: black;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Hero Title */
    .hero-title {
        font-size: 3.3rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #0066CC, #00A86B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
    }

    /* Predict Button */
    .stButton>button {
        background: linear-gradient(90deg, #00A86B, #32d996) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        height: 3.2em !important;
        font-weight: bold !important;
        font-size: 1.15rem !important;
        box-shadow: 0 4px 15px rgba(0, 168, 107, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 168, 107, 0.4);
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #0066CC 0%, #3388CC 100%);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0, 102, 204, 0.25);
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.2);
    }

    /* ==================== KEY RISK FACTORS - HIGH READABILITY ==================== */
    .risk-section-title {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: white !important;
        text-align: center;
        margin: 2rem 0 1.5rem 0 !important;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #00A86B;
        display: inline-block;
        width: auto;
        left: 50%;
        transform: translateX(-50%);
        position: relative;
    }

    .risk-metric-container {
        background: #ffffff !important;
        padding: 1.4rem !important;
        border-radius: 16px !important;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08) !important;
        border: 1px solid #e0e0e0 !important;
        text-align: center;
        height: 100%;
    }

    .risk-metric-label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #2c3e50 !important;
        margin-bottom: 0.4rem !important;
    }

    .risk-metric-value {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #0066CC !important;
        margin: 0.3rem 0 !important;
    }

    .risk-metric-delta {
        font-size: 1rem !important;
        font-weight: 500 !important;
        padding: 0.3rem 0.6rem !important;
        border-radius: 8px !important;
        display: inline-block;
    }
    .risk-high { background: #ffebee; color: #c62828; }
    .risk-low { background: #e8f5e8; color: #2e7d32; }
    .risk-neutral { background: #fff3e0; color: #ef6c00; }

    /* Input Labels */
    .label-with-icon {
        display: flex;
        align-items: center;
        font-weight: 600;
        color: #0066CC;
        font-size: 1.05rem;
    }
    .label-icon { margin-right: 0.5rem; font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

# ------------------- Load Model -------------------
@st.cache_resource
def load_model():
    return joblib.load('best_model.pkl')

model = load_model()

# ------------------- Lottie -------------------
def load_lottie(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

lottie_path = Path(__file__).parent / "assets" / "hero.json"
lottie_anim = load_lottie(lottie_path)

# ------------------- HERO -------------------
st.markdown('<h1 class="hero-title">MediCost AI</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:1.2rem; color:#444;'>AI-Powered Medical Insurance Cost Prediction</p>", unsafe_allow_html=True)

if lottie_anim:
    st.lottie(lottie_anim, height=180, key="hero")

st.markdown("---")

# ------------------- INPUT FORM -------------------
st.markdown("### Your Health Profile")

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="label-with-icon">Age</div>', unsafe_allow_html=True)
    age = st.slider("", 18, 64, 30, key="age")
    st.markdown('<div class="label-with-icon">Gender</div>', unsafe_allow_html=True)
    sex = st.selectbox("", ["male", "female"], key="sex")
    st.markdown('<div class="label-with-icon">BMI</div>', unsafe_allow_html=True)
    bmi = st.number_input("", 15.0, 50.0, 25.0, step=0.1, key="bmi")

with col2:
    st.markdown('<div class="label-with-icon">Children</div>', unsafe_allow_html=True)
    children = st.slider("", 0, 5, 0, key="children")
    st.markdown('<div class="label-with-icon">Smoker</div>', unsafe_allow_html=True)
    smoker = st.selectbox("", ["no", "yes"], key="smoker")
    st.markdown('<div class="label-with-icon">Region</div>', unsafe_allow_html=True)
    region = st.selectbox("", ["southwest", "southeast", "northwest", "northeast"], key="region")

# Feature Engineering
bmi_category = (
    'underweight' if bmi < 18.5 else
    'normal' if bmi < 25 else
    'overweight' if bmi < 30 else
    'obese'
)
smoker_encoded = 1 if smoker == 'yes' else 0
age_smoker_interaction = age * smoker_encoded

input_df = pd.DataFrame({
    'age': [age], 'sex': [sex], 'bmi': [bmi], 'children': [children],
    'smoker': [smoker], 'region': [region],
    'bmi_category': [bmi_category],
    'age_smoker_interaction': [age_smoker_interaction]
})

# ------------------- PREDICT -------------------
if st.button("Predict Insurance Cost", type="primary", use_container_width=True):
    with st.spinner("Analyzing medical & lifestyle data..."):
        prediction = model.predict(input_df)[0]

    # Prediction Card
    st.markdown(f"""
    <div class="prediction-card">
        <h2>Estimated Annual Premium</h2>
        <h1 style="margin:0.5em 0; font-size:3.2rem;">${prediction:,.2f}</h1>
        <p>Model: Gradient Boosting | Accuracy: 88%+</p>
    </div>
    """, unsafe_allow_html=True)

    st.balloons()

    # ==================== KEY RISK FACTORS - SUPER READABLE ====================
    st.markdown('<h2 class="risk-section-title">Key Risk Factors</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="risk-metric-container">
            <div class="risk-metric-label">Smoking Status</div>
            <div class="risk-metric-value">{}</div>
            <div class="risk-metric-delta {}">{}</div>
        </div>
        """.format(
            smoker.title(),
            "risk-high" if smoker == 'yes' else "risk-low",
            "High Risk" if smoker == 'yes' else "Low Risk"
        ), unsafe_allow_html=True)

    with col2:
        age_delta = f"+{age-30} vs avg" if age > 30 else f"-{30-age} vs avg" if age < 30 else "Average"
        age_delta_class = "risk-high" if age > 50 else "risk-neutral" if age >= 35 else "risk-low"
        st.markdown("""
        <div class="risk-metric-container">
            <div class="risk-metric-label">Age</div>
            <div class="risk-metric-value">{age} years</div>
            <div class="risk-metric-delta {age_delta_class}">{age_delta}</div>
        </div>
        """.format(age=age, age_delta=age_delta, age_delta_class=age_delta_class), unsafe_allow_html=True)

    with col3:
        bmi_color = "risk-high" if bmi_category in ['overweight', 'obese'] else "risk-low" if bmi_category == 'normal' else "risk-neutral"
        st.markdown("""
        <div class="risk-metric-container">
            <div class="risk-metric-label">BMI Category</div>
            <div class="risk-metric-value">{}</div>
            <div class="risk-metric-delta {}">Weight Status</div>
        </div>
        """.format(bmi_category.title(), bmi_color), unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    # Report Download
    report = f"""
    MEDICOAST AI - INSURANCE PREDICTION REPORT
    =========================================
    Patient Profile:
      • Age: {age} years
      • Gender: {sex.title()}
      • BMI: {bmi:.1f} ({bmi_category})
      • Children: {children}
      • Smoker: {smoker.title()}
      • Region: {region.title()}

    ESTIMATED ANNUAL COST: ${prediction:,.2f}

    Generated: {pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}
    """
    st.download_button(
        "Download Report",
        report,
        f"MediCost_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
        "text/plain",
        use_container_width=True
    )