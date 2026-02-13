from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import os

from app.model_loader import ModelLoader
from app.prediction_engine import PredictionEngine

app = FastAPI()

# ✅ Use ModelLoader to handle joblib internally
try:
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "..", "model", "svm_model.pkl")

    print("Looking for model at:", model_path)

    loader = ModelLoader(model_path)
    model = loader.model

    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", str(e))
    model = None

prediction_engine = PredictionEngine(model)

@app.get("/")
def home():
    return {"message": "CancerDetect API is running"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        # Read CSV
        df = pd.read_csv(file.file)

        # Validate number of features
        if df.shape[1] != 30:
            raise HTTPException(
                status_code=422,
                detail="CSV file must contain exactly 30 features"
            )

        # Convert to numpy
        data = df.values

        # Get prediction
        label, probability = prediction_engine.predict(data)

        return {
            "prediction": label,
            "probability": probability,
            "details": {"input_shape": data.shape}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))