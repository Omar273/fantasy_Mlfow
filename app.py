# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from typing import List
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import logging
from elasticsearch import Elasticsearch
from model import train_model

# Use non-interactive backend for matplotlib
matplotlib.use("Agg")

# Logging config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Elasticsearch client
es = Elasticsearch("http://localhost:9200")

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "models/fantasy_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
DATA_PATH = os.getenv("DATA_PATH", "data/fantasy_data.csv")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

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
    raise

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

def preprocess_features(player: PlayerFeatures) -> pd.DataFrame:
    return pd.DataFrame([{
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
    }])

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
            predictions.append(PredictionResponse(
                player_id=player.player_id,
                predicted_points=round(float(pred), 2),
                confidence_interval=[round(x, 2) for x in confidence]
            ))
            # Log prediction to Elasticsearch
            es.index(index="predictions", document={
                "timestamp": datetime.now().isoformat(),
                "player_id": player.player_id,
                "predicted_points": round(float(pred), 2),
                "confidence_min": round(confidence[0], 2),
                "confidence_max": round(confidence[1], 2)
            })
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

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
        result = train_model(test_size=request.test_size)
        global model, scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        # Log retraining to Elasticsearch
        es.index(index="retraining_logs", document={
            "timestamp": result["training_date"],
            "mae": result["model_performance"]["MAE"],
            "r2": result["model_performance"]["R2_score"],
            "rmse": result["model_performance"]["RMSE"],
            "test_size": request.test_size
        })
        return RetrainResponse(**result)
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

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
