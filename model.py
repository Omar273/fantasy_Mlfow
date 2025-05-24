# model.py
import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch

# Environment paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/fantasy_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
PLOT_PATH = os.getenv("PLOT_PATH", "feature_importance.png")
DATA_PATH = os.getenv("DATA_PATH", "data/fantasy_data.csv")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Use non-GUI backend
matplotlib.use("Agg")

# Elasticsearch client
es = Elasticsearch("http://localhost:9200")

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def plot_feature_importance(model, feature_names, output_path):
    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Feature importance plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating feature importance plot: {e}")
        raise

def train_model(test_size=0.2):
    try:
        data = load_data()
        logger.info(f"Loaded data shape: {data.shape}")

        if "fantasy_points" not in data.columns:
            raise ValueError("Data must contain 'fantasy_points' column")

        X = data.drop(columns=["fantasy_points"])
        y = data["fantasy_points"]

        expected_features = [
            "position_QB", "position_RB", "position_WR", "position_TE",
            "age", "experience", "avg_points_last_season", "team_strength",
            "opponent_defense_rank", "home_game", "weather_bad",
            "injury_questionable", "injury_out"
        ]
        if set(X.columns) != set(expected_features):
            raise ValueError(f"Input features {set(X.columns)} do not match expected {set(expected_features)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Model and scaler saved")

        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        logger.info(f"Model evaluation - MAE: {mae}, R2: {r2}, RMSE: {rmse}")

        plot_feature_importance(model, X.columns, PLOT_PATH)

        importances = dict(zip(X.columns, model.feature_importances_))

        # MLflow tracking
        with mlflow.start_run():
            mlflow.log_param("test_size", test_size)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(model, artifact_path="model")
            if os.path.exists(PLOT_PATH):
                mlflow.log_artifact(PLOT_PATH)

        # Log to Elasticsearch
        es.index(index="model_training_logs", document={
            "timestamp": datetime.now().isoformat(),
            "mae": round(mae, 2),
            "r2": round(r2, 2),
            "rmse": round(rmse, 2),
            "test_size": test_size
        })

        return {
            "message": "Model successfully trained",
            "training_date": datetime.now().isoformat(),
            "model_performance": {
                "MAE": round(mae, 2),
                "R2_score": round(r2, 2),
                "RMSE": round(rmse, 2),
                "test_size": test_size,
            },
            "feature_importances": importances,
        }

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
