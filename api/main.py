from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model once when API starts
model = joblib.load("./models/ridge.pkl")


@app.get("/")
def home():
    return {"message": "Model API is running"}


@app.post("/predict")
def predict(data: dict):
    # Expect input like: {"features": [1, 2, 3, 4]}
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features).tolist()

    return {"prediction": prediction}
