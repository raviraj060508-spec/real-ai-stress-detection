import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure model folder exists
if not os.path.exists("model"):
    os.makedirs("model")

# Load dataset
data = pd.read_csv("data/dataset.csv")

# Features & target
X = data[['typing_speed', 'typing_errors', 'screen_time']]
y = data['stress_level']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save trained model
joblib.dump(model, "model/stress_model.pkl")
print("Typing model trained and saved in model/stress_model.pkl")
