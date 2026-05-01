import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from pipeline import create_pipeline
import os
from sklearn.metrics import confusion_matrix,classification_report,average_precision_score

# __file__ → the path of the current Python file
# os.path.abspath(__file__) → converts it to an absolute path -  C:\Users\...\project\src\train.py
# os.path.dirname(...) → gets the folder containing the file - C:\Users\...\project\src

# os.path.join(...) safely joins path parts
# ".." means go one folder up
# From src/, ".." goes to project/, then: project/data/creditcard.csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "data", "creditcard.csv")

MODEL_PATH = os.path.join(BASE_DIR, "..", "artifacts", "fraud_pipeline.pkl")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "..", "artifacts")

#load data
df = pd.read_csv(DATA_PATH)

X = df.drop("Class",axis=1)
y = df['Class']

#select columns
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object", "category"]).columns

#split data
X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#create pipeline
pipeline = create_pipeline(num_features,cat_features)

# Train
pipeline.fit(X_train, y_train)

print("\nEvaluating model...")

y_pred = pipeline.predict(X_test)
y_probs = pipeline.predict_proba(X_test)[:, 1]

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Best metric for fraud detection
auprc = average_precision_score(y_test, y_probs)
print(f"\nAUPRC: {auprc:.6f}")

# Save model
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved successfully.")