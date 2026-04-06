import pandas as pd
import json
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.neural_network import MLPRegressor


DATA_URL = "https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_data():
    return pd.read_csv(DATA_URL)

def get_models():
    return {
        # "linear_regression": LinearRegression(),
        # "ridge": Ridge(),
        # "random_forest": RandomForestRegressor(random_state=42),
        # "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "MLPRegressor": MLPRegressor(random_state=42, max_iter=500),
    }

def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_saveall():
    df = load_data()

    y = df["logS"]
    X = df.drop("logS", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=100
        )

    models = get_models()
    metrics = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        rmse = root_mean_squared_error(y_test, predictions)

        metrics[name] = {
                "r2": r2,
                "rmse": rmse,
            }

        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Saved {name} to {model_path}")
        print(f"R2: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print()  

    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


    print("All models a1nd metrics saved")

if __name__ == "__main__":
    ensure_directories()
    train_and_saveall()
