import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# os.getenv("MODEL_PATH", default_value)
# Tries to read environment variable MODEL_PATH
# If it exists → uses that
# If NOT → uses default path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "artifacts", "fraud_pipeline.pkl")
)

# Load model safely
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.get("/")
def home():
    return {"message": "This is Version 2. New Fraud Detection API is running"}


# Added: strict input schema validation (prevents model crashes)
class FraudInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


@app.post("/predict")
def predict(data: FraudInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }