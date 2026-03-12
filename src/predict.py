import joblib
import pandas as pd

# Load model
model = joblib.load("models/linear_regression.pkl")

# Example input data
sample = [[0.1, 0.2, 0.3, 0.4]]

# Make prediction
prediction = model.predict(sample)

print("Prediction:", prediction)
