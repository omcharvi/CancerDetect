import joblib

class ModelLoader:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        
