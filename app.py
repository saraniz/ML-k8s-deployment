import os
import pickle
import pandas as pd
from fastapi import FastAPI

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


@app.post("/predict")
def predict(data: dict):
    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }