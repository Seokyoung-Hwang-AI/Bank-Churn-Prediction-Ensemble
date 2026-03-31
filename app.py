import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import xgboost as xgb

# Set paths
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'models')

# Set page config
st.set_page_config(page_title="Bank Customer Churn Predictor", layout="wide")

# Load assets
@st.cache_resource
def load_assets():
    preprocessor = joblib.load(os.path.join(MODEL_PATH, 'preprocessor.pkl'))
    model_xgb = joblib.load(os.path.join(MODEL_PATH, 'model_xgb.pkl'))
    model_cat = joblib.load(os.path.join(MODEL_PATH, 'model_cat.pkl'))
    model_lgb = joblib.load(os.path.join(MODEL_PATH, 'model_lgb.pkl'))
    return preprocessor, model_xgb, model_cat, model_lgb

preprocessor, model_xgb, model_cat, model_lgb = load_assets()

# Sidebar: user input
def get_user_input():
    st.sidebar.header("📝 Customer Information")
    # Input fields mapped to training feature names
    credit_score = st.sidebar.number_input("Credit Score", 0, 850, 655)
    geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    age = st.sidebar.slider("Age", 0, 100, 35)
    tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
    balance = st.sidebar.number_input("Balance ($)", 0.0, 190000.0, 43000.0)
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4])
    has_crcard = st.sidebar.selectbox("Has Credit Card?", [1, 0])
    is_active = st.sidebar.selectbox("Is Active Member?", [1, 0])
    salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 19000000.0, 120000.0)

    # Wrap input data into a DataFrame for consistent preprocessing
    data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': has_crcard,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# Main Page
st.title("🏦 Bank Customer Churn Prediction System")
st.subheader("Interactive Model Demo & Risk Analysis")
st.markdown("""
This demo predicts the likelihood of a customer leaving the bank (Churn). 
The prediction is powered by an **Ensemble Model** combining XGBoost, CatBoost, and LightGBM.
""")

# Inference Pipeline
# Apply Feature Engineering & Transformation
# Ensures input data matches the scaling/encoding used during model training
processed_input = preprocessor.transform(input_df)
feature_names = preprocessor.get_feature_names_out()

# Soft Voting Ensemble Prediction
# Averaging the predicted probabilities from each base model for higher reliability
dinput = xgb.DMatrix(processed_input, feature_names=list(feature_names))
prob_xgb = model_xgb.predict(dinput)
prob_cat = model_cat.predict_proba(processed_input)[:, 1]
prob_lgb = model_lgb.predict_proba(processed_input)[:, 1]

# Calculate final ensemble probability
final_proba = (prob_xgb + prob_cat + prob_lgb) / 3

# Display Results
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Prediction")
    if final_proba[0] >= 0.5:
        st.error("🚨 **High Risk of Churn**")
    else:
        st.success("✅ **Likely to Stay (Loyal)**")

with col2:
    st.subheader("Exit Probability")
    st.metric(label="Churn Risk Score", value=f"{final_proba[0]*100:.2f}%")
    st.progress(final_proba[0])

# Transparency Section
st.divider()
with st.expander("Show Raw Input DataFrame"):
    st.write("Below is the structured data currently being fed into the model:")
    st.dataframe(input_df)

# Model Insights
st.divider()
st.markdown("###💡 Insights: What Drives Customer Churn?")
st.markdown("""
### **Top 3 Critical Drivers**
Our ensemble model identifies these features as the most influential factors in predicting customer behavior.
""")

# Using columns to display the Top 3 features as key metrics
col_feat1, col_feat2, col_feat3 = st.columns(3)
    
with col_feat1:
    # 1st Priority: Age (Importance: 152)
    st.metric(label="Primary Driver", value="Age")
    st.caption("Older customers show higher churn sensitivity in this specific demographic.")

with col_feat2:
    # 2nd Priority: Estimated Salary (Importance: 118)
    st.metric(label="Secondary Driver", value="EstimatedSalary")
    st.caption("Income levels strongly correlate with long-term retention.")

with col_feat3:
    # 3rd Priority: Balance (Importance: 111)
    st.metric(label="Tertiary Driver", value="Balance")
    st.caption("Account liquidity is a key indicator of a customer's total engagement.")

# Technical Methodology
with st.expander("⚙️ Technical Methodology") :
    st.markdown("""
    * **Feature Engineering:** High-cardinality and non-predictive identifiers (e.g., `CustomerId`, `Surname`) were **removed** to ensure the model focuses on behavioral patterns rather than noise.
    * **Optimization Result:** This refined feature selection contributed to a final **Ensemble ROC-AUC of 0.9200**.
    * **Model Logic:** Predictions are generated by averaging the weighted probabilities of **XGBoost, CatBoost, and LightGBM**.
    """)