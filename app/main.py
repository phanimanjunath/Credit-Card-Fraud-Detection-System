from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── App ───────────────────────────────────────────────
app = FastAPI(title="Credit Card Fraud Detection API 🚀")

# ── Load Model ────────────────────────────────────────
model = joblib.load(r"C:\Users\Y.PHANI MANJUNATH\OneDrive\Desktop\ML_PROJECT\fraud-detection\models\best_model.pkl")

# ── Input Schema ──────────────────────────────────────
class SimpleTransaction(BaseModel):
    features: list

# ── Home Endpoint ─────────────────────────────────────
@app.get("/")
def home():
    return {
        "message"  : "Credit Card Fraud Detection API is running! 🚀",
        "endpoints": {
            "predict": "POST /predict",
            "docs"   : "GET /docs"
        }
    }

# ── Predict Endpoint ──────────────────────────────────
@app.post("/predict")
def predict(transaction: SimpleTransaction):

    features    = np.array(transaction.features).reshape(1, -1)
    prediction  = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "fraud_probability": round(float(probability), 4),
        "result"           : "🚨 FRAUD DETECTED" if prediction == 1 else "✅ Legitimate Transaction"
    }