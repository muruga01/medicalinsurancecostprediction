import streamlit as st
from PIL import Image

st.set_page_config(page_title="Insurance Insights", layout="wide")

st.title("Data Insights & Analysis")
st.markdown("Explore key patterns in medical insurance costs from the dataset.")

st.markdown("---")

# Insight 1
st.subheader("Smoking Has the Biggest Impact on Costs")
img1 = Image.open('smoker_charges.png')
st.image(img1, use_container_width=True)
st.markdown("""
- **Smokers pay 3–4x more** than non-smokers on average.
- This is the **strongest predictor** of high insurance charges.
""")

st.markdown("---")

# Insight 2
st.subheader("Age vs. Cost: Stronger for Smokers")
img2 = Image.open('age_charges.png')
st.image(img2, use_container_width=True)
st.markdown("""
- Costs increase with age for **both groups**.
- The **gap widens dramatically** for smokers after age 40.
""")

st.markdown("---")

# Insight 3
st.subheader("Regional Variations in Charges")
img3 = Image.open('region_charges.png')
st.image(img3, use_container_width=True)
st.markdown("""
- **Southeast** has the highest average charges.
- **Southwest** tends to be the lowest.
- Differences may reflect healthcare access, lifestyle, or cost of living.
""")

st.markdown("---")

st.caption("All visualizations generated during exploratory data analysis (EDA).")