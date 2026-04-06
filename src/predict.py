import joblib
import pandas as pd

# Load model
model = joblib.load("models/MLPRegressor.pkl")

# Example input data
sample = [[2.5, 200.1, 4, 0.5]]

# Make prediction
prediction = model.predict(sample)

print("Prediction:", prediction)
