from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from typing import List
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib
import matplotlib.pyplot as plt
import logging

# Set matplotlib to use non-interactive backend
matplotlib.use("Agg")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Define file paths
MODEL_PATH = os.getenv("MODEL_PATH", "models/fantasy_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/fantasy_data.csv")
PLOT_PATH = os.getenv("PLOT_PATH", "feature_importance.png")

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

# Configure MLflow for file-based tracking
mlflow.set_tracking_uri("file:///mlruns")
mlflow.set_experiment("fantasy_football_predictor")

# Load or initialize model and scaler
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Loaded existing model and scaler")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        scaler = StandardScaler()
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Initialized and saved new model and scaler")
except Exception as e:
    logger.error(f"Failed to load/initialize model or scaler: {e}")
    raise Exception("Failed to initialize model and scaler")

# Create sample data file if none exists
if not os.path.exists(DATA_PATH):
    sample_data = pd.DataFrame({
        "position_QB": [1, 0],
        "position_RB": [0, 1],
        "position_WR": [0, 0],
        "position_TE": [0, 0],
        "age": [28, 25],
        "experience": [5, 3],
        "avg_points_last_season": [20.5, 15.0],
        "team_strength": [0.8, 0.6],
        "opponent_defense_rank": [10, 15],
        "home_game": [1, 0],
        "weather_bad": [0, 1],
        "injury_questionable": [0, 0],
        "injury_out": [0, 0],
        "fantasy_points": [22.3, 12.5]
    })
    sample_data.to_csv(DATA_PATH, index=False)
    logger.info(f"Created sample data file at {DATA_PATH}")

# Define input/output schemas
class PlayerFeatures(BaseModel):
    player_id: str
    position: str = Field(..., pattern="^(QB|RB|WR|TE)$")
    age: int = Field(..., ge=18, le=50)
    experience: int = Field(..., ge=0, le=30)
    avg_points_last_season: float = Field(..., ge=0)
    team_strength: float = Field(..., ge=0, le=1)
    opponent_defense_rank: int = Field(..., ge=1, le=32)
    home_game: bool
    weather_conditions: str = Field(..., pattern="^(good|bad)$")
    injury_status: str = Field(..., pattern="^(healthy|questionable|out)$")

class PredictionRequest(BaseModel):
    players: List[PlayerFeatures]

class PredictionResponse(BaseModel):
    player_id: str
    predicted_points: float
    confidence_interval: List[float]

class RetrainRequest(BaseModel):
    retrain: bool = True
    test_size: float = Field(0.2, ge=0.1, le=0.5)

class RetrainResponse(BaseModel):
    message: str
    training_date: str
    model_performance: dict
    feature_importances: dict

# Helper function to preprocess features
def preprocess_features(player: PlayerFeatures) -> pd.DataFrame:
    features = {
        "position_QB": 1 if player.position == "QB" else 0,
        "position_RB": 1 if player.position == "RB" else 0,
        "position_WR": 1 if player.position == "WR" else 0,
        "position_TE": 1 if player.position == "TE" else 0,
        "age": player.age,
        "experience": player.experience,
        "avg_points_last_season": player.avg_points_last_season,
        "team_strength": player.team_strength,
        "opponent_defense_rank": player.opponent_defense_rank,
        "home_game": 1 if player.home_game else 0,
        "weather_bad": 1 if player.weather_conditions == "bad" else 0,
        "injury_questionable": 1 if player.injury_status == "questionable" else 0,
        "injury_out": 1 if player.injury_status == "out" else 0,
    }
    return pd.DataFrame([features])

# Helper function to plot feature importance
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

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Fantasy Football Prediction API",
        "endpoints": {
            "/predict": "POST - Make predictions for players",
            "/retrain": "POST - Retrain the model with new data",
            "/features": "GET - List of expected features",
        },
    }

# Prediction endpoint
@app.post("/predict", response_model=List[PredictionResponse])
async def predict(request: PredictionRequest):
    try:
        predictions = []
        for player in request.players:
            features_df = preprocess_features(player)
            features_scaled = scaler.transform(features_df)
            pred = model.predict(features_scaled)[0]
            trees = [tree.predict(features_scaled)[0] for tree in model.estimators_]
            std = np.std(trees)
            confidence = [max(0, pred - 1.96 * std), pred + 1.96 * std]
            predictions.append(
                PredictionResponse(
                    player_id=player.player_id,
                    predicted_points=round(float(pred), 2),
                    confidence_interval=[round(x, 2) for x in confidence],
                )
            )
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

# Retrain endpoint
@app.post("/retrain", response_model=RetrainResponse)
async def retrain(request: RetrainRequest):
    if not request.retrain:
        return RetrainResponse(
            message="Retraining not requested",
            training_date=datetime.now().isoformat(),
            model_performance={},
            feature_importances={},
        )

    try:
        # Load and validate data
        data = pd.read_csv(DATA_PATH)
        if "fantasy_points" not in data.columns:
            raise ValueError("Data must contain 'fantasy_points' column")
        X = data.drop(columns=["fantasy_points"])
        y = data["fantasy_points"]

        # Validate features
        expected_features = preprocess_features(PlayerFeatures(
            player_id="test",
            position="QB",
            age=25,
            experience=3,
            avg_points_last_season=10.0,
            team_strength=0.5,
            opponent_defense_rank=16,
            home_game=True,
            weather_conditions="good",
            injury_status="healthy",
        )).columns
        if set(X.columns) != set(expected_features):
            raise ValueError(f"Input features {set(X.columns)} do not match expected {set(expected_features)}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42
        )

        # Start MLflow run
        with mlflow.start_run(run_name=f"fantasy-predictor-{datetime.now().isoformat()}"):
            # Retrain model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            model.fit(X_train_scaled, y_train)

            # Save model and scaler
            joblib.dump(model, MODEL_PATH)
            joblib.dump(scaler, SCALER_PATH)
            logger.info("Model and scaler saved")

            # Evaluate model
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            # Log parameters and metrics
            mlflow.log_params({
                "model_type": "RandomForestRegressor",
                "n_estimators": 100,
                "features": list(X_train.columns),
                "target": "fantasy_points",
                "test_size": request.test_size,
            })
            mlflow.log_metrics({
                "mae": mae,
                "r2": r2,
                "rmse": rmse,
            })

            # Log model with signature
            signature = infer_signature(X_train, model.predict(X_train_scaled))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="fantasy_model",
                signature=signature,
                registered_model_name="FantasyFootballPredictor",
            )

            # Log feature importance plot
            plot_feature_importance(model, X_train.columns, PLOT_PATH)
            mlflow.log_artifact(PLOT_PATH)

            # Get feature importances
            importances = dict(zip(X.columns, model.feature_importances_))

        return RetrainResponse(
            message="Model successfully retrained",
            training_date=datetime.now().isoformat(),
            model_performance={
                "MAE": round(mae, 2),
                "R2_score": round(r2, 2),
                "RMSE": round(rmse, 2),
                "test_size": request.test_size,
            },
            feature_importances=importances,
        )

    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

# Feature documentation endpoint
@app.get("/features")
async def feature_documentation():
    return {
        "features": {
            "player_id": "Unique player identifier (string)",
            "position": "Player position (QB, RB, WR, TE)",
            "age": "Player age (integer, 18-50)",
            "experience": "Years of NFL experience (integer, 0-30)",
            "avg_points_last_season": "Average fantasy points per game last season (float, >=0)",
            "team_strength": "Team strength rating (float, 0-1 scale)",
            "opponent_defense_rank": "Opponent defense rank (integer, 1-32, 1=best)",
            "home_game": "Boolean indicating home game (true/false)",
            "weather_conditions": "Game weather conditions ('good' or 'bad')",
            "injury_status": "Player injury status ('healthy', 'questionable', 'out')",
        }
    }