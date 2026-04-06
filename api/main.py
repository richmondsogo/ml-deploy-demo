from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import uvicorn

app = FastAPI()


# 1. Define the data structure
class PredictionRequest(BaseModel):
    features: List[float]


# Load model once
try:
    model = joblib.load("../models/MLPRegressor.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.get("/")
def home():
    return {"message": "Model API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Pydantic ensures 'request.features' is a list of floats
        features = np.array(request.features).reshape(1, -1)

        prediction = model.predict(features).tolist()

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
