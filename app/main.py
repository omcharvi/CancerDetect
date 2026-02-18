from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
import pandas as pd
import numpy as np
import os

from app.model_loader import ModelLoader
from app.prediction_engine import PredictionEngine

app = FastAPI()

# ✅ CORS must be added RIGHT AFTER app is created
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
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
        df = pd.read_csv(file.file)

        if df.shape[1] != 30:
            raise HTTPException(
                status_code=422,
                detail="CSV file must contain exactly 30 features"
            )

        data = df.values
        label, probability = prediction_engine.predict(data)

        return {
            "prediction": label,
            "probability": probability,
            "details": {"input_shape": data.shape}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))