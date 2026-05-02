# test file for fastapi (app.py)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient #A tool provided by FastAPI to simulate HTTP requests without running a real server.
from app import app

# This creates a fake client that behaves like a real user:
# Sends requests to your API, Receives responses
# Internally, it uses httpx, so no actual network call is made. Everything runs in memory, which makes tests fast.
client = TestClient(app)

# Any function starting with test_ is automatically discovered by pytest
# This is one test case
def test_predict_success():
    response = client.post("/predict", json={
    "Time": 0,
    "V1": -1.359,
    "V2": -0.072,
    "V3": 2.536,
    "V4": 1.378,
    "V5": -0.338,
    "V6": 0.462,
    "V7": 0.239,
    "V8": 0.098,
    "V9": 0.363,
    "V10": 0.090,
    "V11": -0.551,
    "V12": -0.617,
    "V13": -0.991,
    "V14": -0.311,
    "V15": 1.468,
    "V16": -0.470,
    "V17": 0.207,
    "V18": 0.025,
    "V19": 0.403,
    "V20": 0.251,
    "V21": -0.018,
    "V22": 0.277,
    "V23": -0.110,
    "V24": 0.066,
    "V25": 0.128,
    "V26": -0.189,
    "V27": 0.133,
    "V28": -0.021,
    "Amount": 149.62
})
    
    data = response.json()

    # validating responses 
    assert response.status_code == 200
    assert "fraud_prediction" in data
    assert "fraud_probability" in data

#invalid input test
def test_invalid_input():
    response = client.post("/predict",json={
        "wrong_feature": 10
    })

    assert response.status_code == 422

def test_predict_missing_field():
    response = client.post("/predict", json={
        "Time": 0
    })

    assert response.status_code == 422