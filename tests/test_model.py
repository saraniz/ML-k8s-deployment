import os                 # used to handle file paths
import pickle             # used to load your saved model (serialized object)
import numpy as np        # used to create numerical input for the model

import pandas as pd

columns = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
    "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
    "V21","V22","V23","V24","V25","V26","V27","V28","Amount"
]

# Get the root directory of the project
# __file__ → current file path
# os.path.dirname → go one level up
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Build full path to model file
# joins: project_root + artifacts + fraud_pipeline.pkl
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "fraud_pipeline.pkl")


# Test 1: Check whether model loads correctly
def test_model_load():
    # Open the model file in "read binary" mode
    # "rb" = read binary (required for pickle)
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)   # deserialize model from file

    # Assert model is loaded (not None)
    assert model is not None


# Test 2: Check whether model can make predictions
def test_model_prediction():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Create a sample input
    # shape = (1, 30) → 1 row, 30 features
    # matches your dataset (Time + V1..V28 + Amount)
    sample = pd.DataFrame([[0]*30], columns=columns)

    # Call model prediction
    pred = model.predict(sample)

    # Assert prediction exists
    assert pred is not None

    # Assert output contains exactly 1 prediction
    assert len(pred) == 1


# Test 3: Check prediction values are valid
def test_model_output_values():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    sample = pd.DataFrame([[0]*30], columns=columns)

    pred = model.predict(sample)

    # pred[0] → first (and only) prediction
    # Check it is a valid class label
    assert pred[0] in [0, 1]