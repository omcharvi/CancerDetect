import joblib
import os

def load_model():
    model_path = os.path.join("model", "cancer_model.pkl")
    return joblib.load(model_path)
