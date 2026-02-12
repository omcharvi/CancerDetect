import joblib
import numpy as np

class PredictionEngine:
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data).max()

        if prediction == 1:
            label = "cancer"
        else:
            label = "no_cancer"

        return label, float(probability)