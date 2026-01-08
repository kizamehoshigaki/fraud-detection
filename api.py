# =============================================================================
# FRAUD DETECTION API - FINAL VERSION
# =============================================================================
# Run with: uvicorn api:app --reload --port 8000
# Docs at: http://localhost:8000/docs
# =============================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import pandas as pd
import json
from datetime import datetime
import io

# =============================================================================
# INITIALIZE APP
# =============================================================================
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using LightGBM",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# LOAD MODEL & ARTIFACTS
# =============================================================================
MODEL = None
SCALER = None
FEATURE_NAMES = None
CONFIG = None

def load_artifacts():
    global MODEL, SCALER, FEATURE_NAMES, CONFIG
    try:
        MODEL = joblib.load('fraud_model.joblib')
        SCALER = joblib.load('scaler.joblib')
        with open('feature_names.json', 'r') as f:
            FEATURE_NAMES = json.load(f)
        with open('model_config.json', 'r') as f:
            CONFIG = json.load(f)
        print("✅ All artifacts loaded successfully!")
        print(f"   Model: LightGBM")
        print(f"   Features: {len(FEATURE_NAMES)}")
        print(f"   Threshold: {CONFIG.get('optimal_threshold', 0.45)}")
        return True
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return False

MODEL_LOADED = load_artifacts()

# =============================================================================
# REQUEST/RESPONSE SCHEMAS
# =============================================================================
class QuickAssessmentInput(BaseModel):
    """Input for quick rule-based assessment."""
    TransactionAmt: float = Field(..., ge=0, description="Transaction amount in dollars")
    ProductCD: str = Field(..., description="Product code (W, C, R, H, S)")
    card_type: str = Field(..., description="Card type (credit/debit)")
    hour: int = Field(..., ge=0, le=23, description="Hour of transaction")
    is_weekend: bool = Field(False, description="Weekend transaction?")
    addr_missing: bool = Field(False, description="Address missing?")
    email_missing: bool = Field(False, description="Email domain missing?")
    has_identity: bool = Field(True, description="Identity info available?")
    device_type: str = Field("desktop", description="Device type (desktop/mobile)")

class QuickAssessmentResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_level: str
    is_high_risk: bool
    threshold: float
    risk_factors: List[str]
    recommendation: str
    assessment_type: str
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    total_transactions: int
    frauds_detected: int
    fraud_rate: float
    threshold_used: float
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    threshold: float
    features_count: int
    timestamp: str

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def quick_risk_assessment(data: QuickAssessmentInput) -> tuple:
    """
    Rule-based risk assessment using EDA findings.
    Returns (risk_score, risk_factors)
    """
    base_rate = 0.035
    risk_score = base_rate
    risk_factors = []
    
    # Product Code (C = 11.69% fraud)
    if data.ProductCD == "C":
        risk_score += 0.082
        risk_factors.append("🔴 Product C has 11.7% fraud rate (3.3× average)")
    elif data.ProductCD == "W":
        risk_score -= 0.015
        risk_factors.append("🟢 Product W has lowest fraud rate (2.0%)")
    
    # Card Type (Credit = 6.68% fraud)
    if data.card_type.lower() == "credit":
        risk_score += 0.032
        risk_factors.append("🟡 Credit cards: 6.7% fraud rate (vs 2.4% debit)")
    
    # Hour (7-9 AM = 9.77% fraud)
    if 7 <= data.hour <= 9:
        risk_score += 0.063
        risk_factors.append("🔴 High-risk hours (7-9 AM): 9.8% fraud rate")
    elif 0 <= data.hour <= 5:
        risk_score += 0.005
        risk_factors.append("🟡 Late night transaction")
    
    # Address Missing (11.78% fraud)
    if data.addr_missing:
        risk_score += 0.083
        risk_factors.append("🔴 Missing address: 11.8% fraud rate (4.8× average)")
    
    # Email Missing
    if data.email_missing:
        risk_score += 0.025
        risk_factors.append("🟡 Missing email domain")
    
    # Identity Info (has identity = 7.96% fraud - counterintuitive)
    if data.has_identity:
        risk_score += 0.04
        risk_factors.append("🟡 Transactions with identity: 7.96% fraud rate")
    else:
        risk_score -= 0.02
    
    # Device Type (Mobile = 10.17% fraud)
    if data.device_type.lower() == "mobile":
        risk_score += 0.067
        risk_factors.append("🔴 Mobile device: 10.2% fraud rate")
    
    # Transaction Amount
    if data.TransactionAmt > 500:
        risk_score += 0.02
        risk_factors.append(f"🟡 High amount: ${data.TransactionAmt:.2f}")
    
    # Weekend
    if data.is_weekend:
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
# API ENDPOINTS
# =============================================================================
@app.get("/", tags=["General"])
async def root():
    """API information."""
    return {
        "name": "🛡️ Fraud Detection API",
        "version": "2.0.0",
        "model_loaded": MODEL_LOADED,
        "endpoints": {
            "health": "/health",
            "quick_assess": "/assess (POST) - Rule-based quick assessment",
            "batch_predict": "/predict/batch (POST) - Full model batch prediction",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        model_type="LightGBM" if MODEL_LOADED else "None",
        threshold=CONFIG.get('optimal_threshold', 0.45) if CONFIG else 0.45,
        features_count=len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        timestamp=datetime.now().isoformat()
    )

@app.post("/assess", response_model=QuickAssessmentResponse, tags=["Assessment"])
async def quick_assess(transaction: QuickAssessmentInput):
    """
    Quick risk assessment using rule-based scoring.
    Based on EDA findings from the IEEE-CIS dataset.
    For full model predictions, use /predict/batch with CSV upload.
    """
    import time
    start_time = time.time()
    
    # Generate transaction ID
    txn_id = f"TXN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(1000, 9999)}"
    
    # Get risk assessment
    risk_score, risk_factors = quick_risk_assessment(transaction)
    
    # Determine outcome
    threshold = CONFIG.get('optimal_threshold', 0.45) if CONFIG else 0.45
    is_high_risk = risk_score >= threshold
    risk_level = get_risk_level(risk_score)
    recommendation = get_recommendation(is_high_risk, risk_level)
    
    processing_time = (time.time() - start_time) * 1000
    
    return QuickAssessmentResponse(
        transaction_id=txn_id,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        is_high_risk=is_high_risk,
        threshold=threshold,
        risk_factors=risk_factors,
        recommendation=recommendation,
        assessment_type="Rule-Based (EDA Findings)",
        processing_time_ms=round(processing_time, 2)
    )

@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(file: UploadFile = File(...)):
    """
    Batch prediction using the full LightGBM model.
    Upload a CSV file with transaction features.
    Returns fraud probabilities for all transactions.
    """
    import time
    start_time = time.time()
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read CSV
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")
    
    # Prepare features
    X = pd.DataFrame(index=df.index)
    for feat in FEATURE_NAMES:
        if feat in df.columns:
            X[feat] = df[feat].fillna(0)
        else:
            X[feat] = 0
    
    # Scale and predict
    try:
        X_scaled = SCALER.transform(X)
        probabilities = MODEL.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # Apply threshold
    threshold = CONFIG.get('optimal_threshold', 0.45)
    predictions = (probabilities >= threshold).astype(int)
    
    # Add results to dataframe
    df['fraud_probability'] = probabilities
    df['prediction'] = predictions
    df['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.15, 0.30, 0.50, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    )
    
    processing_time = (time.time() - start_time) * 1000
    
    # Return summary + full results
    return {
        "summary": {
            "total_transactions": len(df),
            "frauds_detected": int(predictions.sum()),
            "fraud_rate": f"{predictions.mean()*100:.2f}%",
            "threshold_used": threshold,
            "processing_time_ms": round(processing_time, 2),
            "model_type": "LightGBM (Full Model)",
            "features_used": len(FEATURE_NAMES)
        },
        "predictions": df[['fraud_probability', 'prediction', 'risk_level']].to_dict(orient='records')
    }

@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model configuration and performance metrics."""
    if not CONFIG:
        raise HTTPException(status_code=503, detail="Config not loaded")
    
    return {
        "model_type": CONFIG.get('model_type', 'LightGBM'),
        "optimal_threshold": CONFIG.get('optimal_threshold', 0.45),
        "features_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "performance": CONFIG.get('performance', {}),
        "business_costs": CONFIG.get('business_costs', {}),
        "training_info": CONFIG.get('training_info', {})
    }

@app.get("/model/features", tags=["Model"])
async def get_features():
    """Get list of features the model expects."""
    if not FEATURE_NAMES:
        raise HTTPException(status_code=503, detail="Feature names not loaded")
    
    return {
        "total_features": len(FEATURE_NAMES),
        "features": FEATURE_NAMES
    }

# =============================================================================
# RUN SERVER
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)