# =============================================================================
# FRAUD DETECTION DASHBOARD - STANDALONE VERSION
# =============================================================================
# Run with: streamlit run app.py
# Works without API - loads model directly
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import joblib
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD MODEL & ARTIFACTS DIRECTLY
# =============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and scaler."""
    try:
        model = joblib.load('fraud_model.joblib')
        scaler = joblib.load('scaler.joblib')
        with open('feature_names.json', 'r') as f:
            feature_names = json.load(f)
        return model, scaler, feature_names, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

@st.cache_data
def load_config():
    """Load model configuration."""
    try:
        with open('model_config.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'model_type': 'LightGBM',
            'optimal_threshold': 0.10,
            'performance': {
                'test_roc_auc': 0.9658,
                'test_pr_auc': 0.8355,
                'test_precision': 0.5268,
                'test_recall': 0.8351,
                'test_f1': 0.6460,
                'business_cost': 313625,
                'baseline_cost': 752325,
                'cost_reduction_pct': 58.3
            },
            'business_costs': {'false_negative': 500, 'false_positive': 25}
        }

# Load everything
MODEL, SCALER, FEATURE_NAMES, MODEL_LOADED = load_model()
config = load_config()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def quick_risk_assessment(data: dict) -> tuple:
    """Rule-based risk assessment using EDA findings."""
    base_rate = 0.035
    risk_score = base_rate
    risk_factors = []
    
    if data.get('ProductCD') == "C":
        risk_score += 0.082
        risk_factors.append("🔴 Product C has 11.7% fraud rate (3.3× average)")
    elif data.get('ProductCD') == "W":
        risk_score -= 0.015
        risk_factors.append("🟢 Product W has lowest fraud rate (2.0%)")
    
    if data.get('card_type', '').lower() == "credit":
        risk_score += 0.032
        risk_factors.append("🟡 Credit cards: 6.7% fraud rate (vs 2.4% debit)")
    
    hour = data.get('hour', 12)
    if 7 <= hour <= 9:
        risk_score += 0.063
        risk_factors.append("🔴 High-risk hours (7-9 AM): 9.8% fraud rate")
    elif 0 <= hour <= 5:
        risk_score += 0.005
        risk_factors.append("🟡 Late night transaction")
    
    if data.get('addr_missing', False):
        risk_score += 0.083
        risk_factors.append("🔴 Missing address: 11.8% fraud rate (4.8× average)")
    
    if data.get('email_missing', False):
        risk_score += 0.025
        risk_factors.append("🟡 Missing email domain")
    
    if data.get('has_identity', True):
        risk_score += 0.04
        risk_factors.append("🟡 Transactions with identity: 7.96% fraud rate")
    else:
        risk_score -= 0.02
    
    if data.get('device_type', '').lower() == "mobile":
        risk_score += 0.067
        risk_factors.append("🔴 Mobile device: 10.2% fraud rate")
    
    if data.get('TransactionAmt', 0) > 500:
        risk_score += 0.02
        risk_factors.append(f"🟡 High amount: ${data.get('TransactionAmt', 0):.2f}")
    
    if data.get('is_weekend', False):
        risk_score += 0.005
    
    risk_score = min(0.95, max(0.02, risk_score))
    
    if not risk_factors:
        risk_factors.append("✅ No significant risk factors detected")
    
    return risk_score, risk_factors

def get_risk_level(score: float) -> str:
    if score >= 0.5:
        return "CRITICAL"
    elif score >= 0.3:
        return "HIGH"
    elif score >= 0.15:
        return "MEDIUM"
    else:
        return "LOW"

def get_recommendation(is_high_risk: bool, risk_level: str) -> str:
    if risk_level == "CRITICAL":
        return "🚫 BLOCK - Very high risk. Block transaction and flag for investigation."
    elif risk_level == "HIGH":
        return "⚠️ REVIEW - High risk. Require additional verification (OTP/Call)."
    elif risk_level == "MEDIUM":
        return "🔍 MONITOR - Elevated risk. Allow but monitor for follow-up fraud."
    else:
        return "✅ APPROVE - Low risk. Process normally."

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Quick Assessment", "📈 Model Performance", "🧠 Model Insights"]
)

st.sidebar.markdown("---")

# Model Status
st.sidebar.markdown("### 🔌 Model Status")
if MODEL_LOADED:
    st.sidebar.success("✅ Model Loaded")
    st.sidebar.caption(f"Type: {config.get('model_type', 'LightGBM')}")
    st.sidebar.caption(f"Features: {len(FEATURE_NAMES) if FEATURE_NAMES else 426}")
else:
    st.sidebar.error("❌ Model Not Loaded")

st.sidebar.markdown("---")

# Model Info
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(f"**Threshold:** {config.get('optimal_threshold', 0.10)}")
st.sidebar.markdown(f"**ROC-AUC:** {config['performance']['test_roc_auc']:.4f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Business Costs")
st.sidebar.markdown(f"**Missed Fraud:** ${config['business_costs']['false_negative']}")
st.sidebar.markdown(f"**False Alarm:** ${config['business_costs']['false_positive']}")

# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == "📊 Dashboard":
    st.title("🛡️ Fraud Detection System")
    st.markdown("### Real-time Credit Card Fraud Detection")
    st.markdown("*IEEE-CIS Fraud Detection Dataset | LightGBM Model | Streamlit Dashboard*")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC-AUC Score", f"{config['performance']['test_roc_auc']:.4f}", "Excellent")
    with col2:
        st.metric("Fraud Detection Rate", f"{config['performance']['test_recall']*100:.1f}%", "83.5% caught")
    with col3:
        cost_saved = config['performance']['baseline_cost'] - config['performance']['business_cost']
        st.metric("Cost Reduction", f"{config['performance']['cost_reduction_pct']:.1f}%", f"${cost_saved:,} saved")
    with col4:
        st.metric("Precision", f"{config['performance']['test_precision']*100:.1f}%", "52.7%")
    
    st.markdown("---")
    
    # Architecture Diagram
    st.subheader("🏗️ System Architecture")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ```
        ┌─────────────────┐
        │   Streamlit     │
        │   Dashboard     │
        │   (Frontend)    │
        └────────┬────────┘
                 │
                 ▼
        ```
        """)
    with col2:
        st.markdown("""
        ```
        ┌─────────────────┐
        │   LightGBM      │
        │   Model         │
        │   (ML Engine)   │
        └────────┬────────┘
                 │
                 ▼
        ```
        """)
    with col3:
        st.markdown("""
        ```
        ┌─────────────────┐
        │  Predictions    │
        │  Risk Scores    │
        │  Explanations   │
        └─────────────────┘
        ```
        """)
    
    st.markdown("---")
    
    # Model Comparison
    st.subheader("📈 Model Performance Comparison")
    
    comparison_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost', 'LightGBM (Tuned)'],
        'ROC-AUC': [0.865, 0.947, 0.9658],
        'PR-AUC': [0.435, 0.709, 0.8355],
        'Recall (%)': [73.68, 80.77, 83.51],
        'Business Cost ($)': [752325, 428800, 313625],
        'Training Time (s)': [98.9, 57.1, 17.0]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison_data, x='Model', y='ROC-AUC', color='ROC-AUC',
                    color_continuous_scale='Blues', title='ROC-AUC by Model')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison_data, x='Model', y='Business Cost ($)', color='Business Cost ($)',
                    color_continuous_scale='Reds_r', title='Business Cost (Lower is Better)')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Business Impact
    st.subheader("💰 Business Impact")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Cost Breakdown (Test Set: 88,581 transactions)
        | Metric | Baseline | LightGBM Model |
        |--------|----------|----------------|
        | Missed Frauds | 3,099 | 511 |
        | False Alarms | 0 | 2,325 |
        | **Total Cost** | **$752,325** | **$313,625** |
        | **Savings** | - | **$438,700** |
        """)
    
    with col2:
        st.markdown("""
        #### Scaled Annual Savings
        | Transaction Volume | Annual Savings |
        |-------------------|----------------|
        | 1 Million | **$4.95 Million** |
        | 10 Million | **$49.5 Million** |
        | 100 Million | **$495 Million** |
        """)

# =============================================================================
# PAGE 2: QUICK ASSESSMENT
# =============================================================================
elif page == "🔍 Quick Assessment":
    st.title("🔍 Quick Risk Assessment")
    
    st.info("⚠️ **Demo Mode**: Rule-based scoring using EDA findings. Simulates real-time fraud detection.")
    
    st.markdown("---")
    st.subheader("Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        transaction_amt = st.number_input("Transaction Amount ($)", min_value=0.01, value=150.0, step=10.0)
        product_cd = st.selectbox("Product Code", ["W", "C", "R", "H", "S"],
                                  help="W=Lowest risk (2%), C=Highest risk (11.7%)")
        card_type = st.selectbox("Card Type", ["debit", "credit"],
                                help="Credit: 6.7% fraud, Debit: 2.4% fraud")
    
    with col2:
        hour = st.slider("Hour of Transaction", 0, 23, 14,
                        help="7-9 AM = highest risk (9.8%)")
        is_weekend = st.checkbox("Weekend Transaction")
        addr_missing = st.checkbox("Address Missing", help="11.8% fraud rate if missing")
    
    with col3:
        email_missing = st.checkbox("Email Domain Missing")
        has_identity = st.checkbox("Identity Info Available", value=True)
        device_type = st.selectbox("Device Type", ["desktop", "mobile"],
                                  help="Mobile: 10.2% fraud rate")
    
    st.markdown("---")
    
    if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
        # Prepare data
        data = {
            "TransactionAmt": transaction_amt,
            "ProductCD": product_cd,
            "card_type": card_type,
            "hour": hour,
            "is_weekend": is_weekend,
            "addr_missing": addr_missing,
            "email_missing": email_missing,
            "has_identity": has_identity,
            "device_type": device_type
        }
        
        # Get risk assessment
        risk_score, risk_factors = quick_risk_assessment(data)
        threshold = config.get('optimal_threshold', 0.10)
        is_high_risk = risk_score >= threshold
        risk_level = get_risk_level(risk_score)
        recommendation = get_recommendation(is_high_risk, risk_level)
        
        st.markdown("---")
        st.subheader("📊 Assessment Results")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_score:.1%}")
        with col2:
            risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
            st.metric("Risk Level", f"{risk_emoji.get(risk_level, '')} {risk_level}")
        with col3:
            decision = "🚨 HIGH RISK" if is_high_risk else "✅ LOW RISK"
            st.metric("Decision", decision)
        
        # Risk Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 15], 'color': "lightgreen"},
                    {'range': [15, 30], 'color': "yellow"},
                    {'range': [30, 50], 'color': "orange"},
                    {'range': [50, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        if is_high_risk:
            st.error(f"**Recommendation:** {recommendation}")
        elif risk_level == "MEDIUM":
            st.warning(f"**Recommendation:** {recommendation}")
        else:
            st.success(f"**Recommendation:** {recommendation}")
        
        # Risk Factors
        st.subheader("🔍 Risk Factors")
        for factor in risk_factors:
            st.markdown(f"• {factor}")

# =============================================================================
# PAGE 3: MODEL PERFORMANCE
# =============================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    
    # Metrics
    st.subheader("📊 Test Set Metrics (LightGBM Tuned)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("ROC-AUC", config['performance']['test_roc_auc']),
        ("PR-AUC", config['performance']['test_pr_auc']),
        ("Precision", config['performance']['test_precision']),
        ("Recall", config['performance']['test_recall']),
        ("F1 Score", config['performance']['test_f1'])
    ]
    
    for col, (name, value) in zip([col1, col2, col3, col4, col5], metrics):
        with col:
            st.metric(name, f"{value:.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix (Threshold = 0.10)")
    
    cm_data = [[83157, 2325], [511, 2588]]
    
    fig = px.imshow(cm_data, labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'],
                   color_continuous_scale='Blues', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Interpretation
        - **True Negatives (83,157):** Legitimate correctly approved
        - **True Positives (2,588):** Fraud correctly caught
        - **False Positives (2,325):** Legitimate incorrectly blocked
        - **False Negatives (511):** Fraud missed
        """)
    with col2:
        st.markdown("""
        #### Key Rates
        - **Fraud Detection (Recall):** 83.51%
        - **False Alarm Rate:** 2.72%
        - **Precision:** 52.68%
        - **Cost Reduction:** 58.3%
        """)

# =============================================================================
# PAGE 4: MODEL INSIGHTS
# =============================================================================
elif page == "🧠 Model Insights":
    st.title("🧠 Model Insights")
    
    st.subheader("📊 Top 20 Features (SHAP Importance)")
    
    shap_data = pd.DataFrame({
        'Feature': ['C13', 'addr1', 'card1', 'card2', 'dist1', 'C14', 'D4', 'C1',
                   'D1', 'amount_log', 'C5', 'TransactionAmt', 'is_debit', 'D11',
                   'V69', 'C2', 'card5', 'hour', 'D15', 'amount_decimal'],
        'Importance': [0.667, 0.448, 0.415, 0.354, 0.315, 0.274, 0.250, 0.242,
                      0.241, 0.237, 0.237, 0.226, 0.224, 0.223, 0.211, 0.209,
                      0.197, 0.196, 0.195, 0.191]
    })
    
    fig = px.bar(shap_data, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔍 Key Fraud Patterns")
    
    patterns = pd.DataFrame([
        {"Pattern": "Product C", "Fraud Rate": "11.69%", "vs Average": "3.3×"},
        {"Pattern": "Missing Address", "Fraud Rate": "11.78%", "vs Average": "4.8×"},
        {"Pattern": "7-9 AM Transactions", "Fraud Rate": "9.77%", "vs Average": "2.8×"},
        {"Pattern": "Mobile Device", "Fraud Rate": "10.17%", "vs Average": "2.9×"},
        {"Pattern": "Credit Card", "Fraud Rate": "6.68%", "vs Average": "2.7×"},
    ])
    st.table(patterns)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built by Aaditya Krishna | MS Data Analytics Engineering, Northeastern University</p>
        <p>🛡️ Fraud Detection System | LightGBM + Streamlit | IEEE-CIS Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)
