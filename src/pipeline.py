import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer #Applies different preprocessing to different column types
from sklearn.impute import SimpleImputer #handle missing values
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

def create_pipeline(num_features, cat_features):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    #combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # Final pipeline (Preprocessing + Model)
    pipeline = Pipeline([
        ("preprocessing",preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        ))
    ])

    return pipeline