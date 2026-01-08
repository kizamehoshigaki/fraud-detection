# =============================================================================
# FRAUD DETECTION DASHBOARD - WITH API INTEGRATION
# =============================================================================
# Run with: streamlit run app.py
# Requires: API running at localhost:8000
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import json

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
# API CONFIGURATION
# =============================================================================
API_URL = "http://localhost:8000"

def check_api():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def call_quick_assess(data):
    """Call the quick assessment endpoint."""
    try:
        response = requests.post(f"{API_URL}/assess", json=data, timeout=5)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": f"API error: {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "API not running"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def call_batch_predict(file):
    """Call the batch prediction endpoint."""
    try:
        files = {"file": ("transactions.csv", file, "text/csv")}
        response = requests.post(f"{API_URL}/predict/batch", files=files, timeout=60)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_model_info():
    """Get model info from API."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=3)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# =============================================================================
# CHECK API STATUS
# =============================================================================
api_online, api_health = check_api()
model_info = get_model_info() if api_online else None

# Default config
config = {
    'model_type': 'LightGBM',
    'optimal_threshold': 0.45,
    'performance': {
        'test_roc_auc': 0.9500,
        'test_pr_auc': 0.7130,
        'test_precision': 0.3147,
        'test_recall': 0.8203,
        'test_f1': 0.455,
        'business_cost': 416925,
        'baseline_cost': 752325,
        'cost_reduction_pct': 44.6
    },
    'business_costs': {'false_negative': 500, 'false_positive': 25}
}

if model_info:
    config.update(model_info)

# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.title("🛡️ Fraud Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Quick Assessment", "📁 Batch Scoring", "📈 Model Performance", "🧠 Model Insights"]
)

st.sidebar.markdown("---")

# API Status
st.sidebar.markdown("### 🔌 API Status")
if api_online:
    st.sidebar.success("✅ API Online")
    st.sidebar.caption(f"Model: {api_health.get('model_type', 'LightGBM')}")
    st.sidebar.caption(f"Features: {api_health.get('features_count', 426)}")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.code("uvicorn api:app --reload", language="bash")

st.sidebar.markdown("---")

# Model Info
st.sidebar.markdown("### Model Info")
st.sidebar.markdown(f"**Model:** {config.get('model_type', 'LightGBM')}")
st.sidebar.markdown(f"**Threshold:** {config.get('optimal_threshold', 0.45)}")
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
    st.markdown("*IEEE-CIS Fraud Detection Dataset | LightGBM Model | FastAPI Backend*")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ROC-AUC Score", f"{config['performance']['test_roc_auc']:.4f}", "Excellent")
    with col2:
        st.metric("Fraud Detection Rate", f"{config['performance']['test_recall']*100:.1f}%", "82% caught")
    with col3:
        cost_saved = config['performance']['baseline_cost'] - config['performance']['business_cost']
        st.metric("Cost Reduction", f"{config['performance']['cost_reduction_pct']:.1f}%", f"${cost_saved:,} saved")
    with col4:
        st.metric("Precision", f"{config['performance']['test_precision']*100:.1f}%", "31.5%")
    
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
                 │ HTTP
                 ▼
        ```
        """)
    with col2:
        st.markdown("""
        ```
        ┌─────────────────┐
        │    FastAPI      │
        │    Backend      │
        │   (api.py)      │
        └────────┬────────┘
                 │
                 ▼
        ```
        """)
    with col3:
        st.markdown("""
        ```
        ┌─────────────────┐
        │  LightGBM Model │
        │  426 Features   │
        │  ROC-AUC: 0.95  │
        └─────────────────┘
        ```
        """)
    
    st.markdown("---")
    
    # Model Comparison
    st.subheader("📈 Model Performance Comparison")
    
    comparison_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'XGBoost', 'LightGBM'],
        'ROC-AUC': [0.865, 0.947, 0.950],
        'PR-AUC': [0.435, 0.709, 0.713],
        'Recall (%)': [73.68, 80.77, 82.03],
        'Business Cost ($)': [752325, 428800, 416925],
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
        #### Cost Breakdown (per 88,581 test transactions)
        | Metric | Baseline | LightGBM Model |
        |--------|----------|----------------|
        | Missed Frauds | 816 | 557 |
        | False Alarms | 13,773 | 5,594 |
        | **Total Cost** | **$752,325** | **$416,925** |
        | **Savings** | - | **$335,400** |
        """)
    
    with col2:
        st.markdown("""
        #### Scaled Annual Savings
        | Transaction Volume | Annual Savings |
        |-------------------|----------------|
        | 1 Million | **$3.79 Million** |
        | 10 Million | **$37.9 Million** |
        | 100 Million | **$379 Million** |
        """)

# =============================================================================
# PAGE 2: QUICK ASSESSMENT (Rule-Based via API)
# =============================================================================
elif page == "🔍 Quick Assessment":
    st.title("🔍 Quick Risk Assessment")
    
    # API Check
    if not api_online:
        st.error("❌ API is offline. Start the API first:")
        st.code("uvicorn api:app --reload --port 8000", language="bash")
        st.stop()
    
    st.info("⚠️ **Demo Mode**: Rule-based scoring using EDA findings. For full model predictions with 426 features, use **📁 Batch Scoring**.")
    
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
    
    if st.button("🔍 Assess Risk via API", type="primary", use_container_width=True):
        # Prepare payload
        payload = {
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
        
        # Call API
        with st.spinner("🔄 Calling API..."):
            result = call_quick_assess(payload)
        
        if not result['success']:
            st.error(f"❌ {result['error']}")
        else:
            data = result['data']
            
            st.markdown("---")
            st.subheader("📊 Assessment Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Risk Score", f"{data['risk_score']:.1%}")
            with col2:
                risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                st.metric("Risk Level", f"{risk_emoji.get(data['risk_level'], '')} {data['risk_level']}")
            with col3:
                decision = "🚨 HIGH RISK" if data['is_high_risk'] else "✅ LOW RISK"
                st.metric("Decision", decision)
            with col4:
                st.metric("Processing Time", f"{data['processing_time_ms']:.1f} ms")
            
            # Risk Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=data['risk_score'] * 100,
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
                        'value': data['threshold'] * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            if data['is_high_risk']:
                st.error(f"**Recommendation:** {data['recommendation']}")
            elif data['risk_level'] == "MEDIUM":
                st.warning(f"**Recommendation:** {data['recommendation']}")
            else:
                st.success(f"**Recommendation:** {data['recommendation']}")
            
            # Risk Factors
            st.subheader("🔍 Risk Factors")
            for factor in data['risk_factors']:
                st.markdown(f"• {factor}")
            
            # API Response
            with st.expander("📋 View Raw API Response"):
                st.json(data)
            
            st.caption(f"Transaction ID: `{data['transaction_id']}` | Type: {data['assessment_type']}")

# =============================================================================
# PAGE 3: BATCH SCORING (Full Model via API)
# =============================================================================
elif page == "📁 Batch Scoring":
    st.title("📁 Batch Scoring")
    st.markdown("Upload CSV for **real LightGBM model predictions** with all 426 features.")
    
    if not api_online:
        st.error("❌ API is offline. Start the API first:")
        st.code("uvicorn api:app --reload --port 8000", language="bash")
        st.stop()
    
    st.success(f"✅ API Online | Model: LightGBM | Features: {api_health.get('features_count', 426)}")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Transaction CSV", type=['csv'])
    
    if uploaded_file:
        # Preview
        df_preview = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Preview")
        st.dataframe(df_preview.head(10))
        st.caption(f"Total rows: {len(df_preview):,}")
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        if st.button("🚀 Run Predictions via API", type="primary"):
            with st.spinner("🔄 Sending to API for prediction..."):
                result = call_batch_predict(uploaded_file)
            
            if not result['success']:
                st.error(f"❌ {result['error']}")
            else:
                data = result['data']
                summary = data['summary']
                predictions = pd.DataFrame(data['predictions'])
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", f"{summary['total_transactions']:,}")
                with col2:
                    st.metric("Frauds Detected", f"{summary['frauds_detected']:,}")
                with col3:
                    st.metric("Fraud Rate", summary['fraud_rate'])
                with col4:
                    st.metric("Processing Time", f"{summary['processing_time_ms']:.0f} ms")
                
                st.caption(f"Model: {summary['model_type']} | Features: {summary['features_used']} | Threshold: {summary['threshold_used']}")
                
                # Distribution
                st.subheader("📈 Probability Distribution")
                fig = px.histogram(predictions, x='fraud_probability', nbins=50,
                                  color_discrete_sequence=['steelblue'],
                                  title='Distribution of Fraud Probabilities')
                fig.add_vline(x=summary['threshold_used'], line_dash="dash", line_color="red",
                             annotation_text=f"Threshold ({summary['threshold_used']})")
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk breakdown
                st.subheader("📊 Risk Level Breakdown")
                risk_counts = predictions['risk_level'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                            title='Transactions by Risk Level',
                            color_discrete_sequence=['green', 'yellow', 'orange', 'red'])
                st.plotly_chart(fig, use_container_width=True)
                
                # High risk transactions
                st.subheader("🚨 High Risk Transactions")
                high_risk = predictions[predictions['prediction'] == 1].sort_values('fraud_probability', ascending=False)
                st.dataframe(high_risk.head(20))
                
                # Download
                df_preview['fraud_probability'] = predictions['fraud_probability']
                df_preview['prediction'] = predictions['prediction']
                df_preview['risk_level'] = predictions['risk_level']
                
                csv = df_preview.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )
    else:
        st.markdown("""
        ### Required Format
        Upload a CSV with transaction features. The model expects 426 features including:
        - `TransactionAmt`, `card1-6`, `addr1-2`, `dist1-2`
        - `C1-C14`, `D1-D15`
        - `V1-V339`
        - `id_01-id_38`
        
        Missing features will be filled with zeros.
        """)

# =============================================================================
# PAGE 4: MODEL PERFORMANCE
# =============================================================================
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    
    # Metrics
    st.subheader("📊 Test Set Metrics (LightGBM)")
    
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
    st.subheader("📊 Confusion Matrix (Threshold = 0.45)")
    
    cm_data = [[80112, 5594], [557, 2318]]
    
    fig = px.imshow(cm_data, labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'],
                   color_continuous_scale='Blues', text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### Interpretation
        - **True Negatives (80,112):** Legitimate correctly approved
        - **True Positives (2,318):** Fraud correctly caught
        - **False Positives (5,594):** Legitimate incorrectly blocked
        - **False Negatives (557):** Fraud missed
        """)
    with col2:
        st.markdown("""
        #### Key Rates
        - **Fraud Detection (Recall):** 82.03%
        - **False Alarm Rate:** 6.5%
        - **Precision:** 31.47%
        - **Cost Reduction:** 44.6%
        """)

# =============================================================================
# PAGE 5: MODEL INSIGHTS
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
        <p>🛡️ Fraud Detection System | LightGBM + FastAPI + Streamlit | IEEE-CIS Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)