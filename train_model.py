
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
import os

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(probability=True)
model.fit(X_train, y_train)

# Save model directly into model folder
joblib.dump(model, "model/svm_model.pkl")

print("✅ Model saved successfully in model folder")