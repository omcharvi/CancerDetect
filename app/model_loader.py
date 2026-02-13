import joblib
import os

class ModelLoader:
    def __init__(self, model_path: str):
        try:
            # Get absolute path
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, model_path)

            print("Looking for model at:", full_path)

            if not os.path.exists(full_path):
                raise FileNotFoundError(f"Model file not found at {full_path}")

            self.model = joblib.load(full_path)
            print("✅ Model loaded successfully")

        except Exception as e:
            print("REAL ERROR:", str(e))
            raise