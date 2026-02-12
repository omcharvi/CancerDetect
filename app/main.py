from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np

from app.model_loader import ModelLoader
from app.prediction_engine import PredictionEngine

app = FastAPI()

# Load model once when app starts
try:
    model = joblib.load("model/svm_model.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

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
            "details": {
                "input_shape": data.shape
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))