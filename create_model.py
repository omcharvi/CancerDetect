
import numpy as np
from sklearn.svm import SVC
import joblib

# Create fake training data (30 features)
X = np.random.rand(50, 30)
y = np.random.randint(0, 2, 50)

# Train simple SVM model
model = SVC(probability=True)
model.fit(X, y)

# Save the model
joblib.dump(model, "model/svm_model.pkl")

print("Dummy SVM model created and saved successfully!")
