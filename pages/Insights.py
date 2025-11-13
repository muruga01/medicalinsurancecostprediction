# pages/1_Insights.py
import streamlit as st
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Insights – MediCost AI", layout="wide")

# ------------------- HIGH-READABILITY STYLING -------------------
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        color: white;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-subtitle {
        font-size: 1.25rem;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        color: #0066CC !important;
        background: #f0f7ff !important;
        border-radius: 12px !important;
        padding: 0.8rem 1rem !important;
    }
    .streamlit-expander {
        background: white !important;
        border: 1px solid #ddd !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06) !important;
        margin-bottom: 1.5rem !important;
    }

    /* Insight Text Box - MAX READABILITY */
    .insight-box {
        background: #ffffff;
        padding: 1.4rem;
        border-radius: 12px;
        border-left: 5px solid #00A86B;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    .insight-box strong {
        color: #0066CC;
        font-weight: 700;
    }
    .insight-box p {
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
        color: #1a1a1a !important;
        margin: 0.6rem 0 !important;
    }

    /* Image Caption */
    .img-caption {
        text-align: center;
        font-size: 0.95rem;
        color: #555;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
st.markdown('<div class="main-title">Clinical Data Insights</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Evidence from 1,338 real medical insurance records</div>', unsafe_allow_html=True)

# Load Images
img_path = Path(__file__).parents[1]
img1 = Image.open(img_path / "smoker_charges.png")
img2 = Image.open(img_path / "age_charges.png")
img3 = Image.open(img_path / "region_charges.png")

# ------------------- INSIGHT 1 -------------------
with st.expander("Smoking: The #1 Cost Driver", expanded=True):
    st.image(img1, use_container_width=True)
    st.markdown('<div class="img-caption">Average insurance charges: Smokers vs Non-Smokers</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <p><strong>Key Clinical Finding:</strong></p>
        <p>Smokers pay <strong>3 to 4 times more</strong> in annual premiums than non-smokers — 
        <strong>regardless of age, BMI, or region</strong>.</p>
        <p>This is the <strong>single strongest predictor</strong> in the model, 
        with a direct causal link to healthcare utilization.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- INSIGHT 2 -------------------
with st.expander("Age × Smoking: A Dangerous Combination"):
    st.image(img2, use_container_width=True)
    st.markdown('<div class="img-caption">Insurance cost by age, split by smoking status</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <p><strong>Critical Insight:</strong></p>
        <p>Costs rise with age for everyone — but <strong>smokers see an exponential spike after 40</strong>.</p>
        <p>The <strong>age-smoking interaction</strong> creates a compounding effect: 
        a 55-year-old smoker pays <strong>nearly $30,000 more per year</strong> than a non-smoker.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- INSIGHT 3 -------------------
with st.expander("Regional Cost Disparities"):
    st.image(img3, use_container_width=True)
    st.markdown('<div class="img-caption">Average charges by U.S. region</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
        <p><strong>Geographic Variation:</strong></p>
        <p><strong>Southeast</strong> has the highest average charges (~$14,700/year), 
        while <strong>Southwest</strong> is the lowest (~$12,300).</p>
        <p>Likely drivers: higher smoking rates, obesity, and limited healthcare access in certain regions.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#444; font-size:0.95rem; padding:1rem;">
    <strong>Dataset:</strong> Kaggle Medical Cost Personal Datasets<br>
    <strong>Tools:</strong> Python • Pandas • Scikit-learn • Seaborn • Streamlit<br>
    <strong>Model:</strong> Gradient Boosting Regressor (R² = 0.88)
</div>
""", unsafe_allow_html=True)