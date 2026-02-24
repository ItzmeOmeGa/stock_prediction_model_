#intial commit 
import os
import sys
import numpy as np
import pandas as pd
import joblib
import redis_processor
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from keras.models import load_model

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    description="API for real-time stock prediction and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Connect to Redis with improved error handling
redis_client = None
try:
    redis_client = redis_processor.Redis(host="localhost", port=6379, socket_connect_timeout=5)
    redis_client.ping()
    print("✅ Connected to Redis")
except Exception as e:
    print(f"⚠️ Redis connection error: {e}")
    # Don't exit - we'll handle this gracefully

# Response Models
class PredictionResponse(BaseModel):
    timestamp: str
    last_price: float
    lstm_prediction: float
    xgboost_prediction: float
    ensemble_prediction: float
    
class StockDataPoint(BaseModel):
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class HistoricalResponse(BaseModel):
    data: List[Dict[str, Any]]
    predictions: Optional[List[Dict[str, Any]]] = None

# Fallback data functionality
def get_fallback_stock_data():
    """Generate fallback stock data if Redis is unavailable"""
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "open": 150.25,
        "high": 151.33,
        "low": 149.87,
        "close": 150.75,
        "volume": 10000
    }

def get_fallback_prediction():
    """Generate fallback prediction data if Redis is unavailable"""
    now = datetime.now()
    price = 150.75
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
        "last_price": price,
        "lstm_prediction": price * 1.01,
        "xgboost_prediction": price * 1.015,
        "ensemble_prediction": price * 1.0125
    }

# API Routes
@app.get("/")
def read_root():
    """API root endpoint"""
    return {
        "status": "online",
        "message": "Stock Prediction API is running",
        "endpoints": [
            "/latest",
            "/predict",
            "/history/{minutes}",
            "/health"
        ]
    }

@app.get("/latest", response_model=StockDataPoint)
def get_latest_data():
    """Get the latest stock data point"""
    try:
        if not redis_client:
            # Return fallback data if Redis is unavailable
            return get_fallback_stock_data()
        
        # Try to get latest data from Redis
        latest_data = redis_client.get("latest_stock_data")
        if latest_data:
            return json.loads(latest_data)
        
        # If no cached latest data, get from stream
        stream_data = redis_client.xrevrange("stock_data", count=1)
        if stream_data:
            message_id, fields = stream_data[0]
            data = {k.decode(): (v.decode() if k == b'datetime' else float(v.decode())) 
                   for k, v in fields.items()}
            return data
        
        # If no data in Redis, return fallback data
        return get_fallback_stock_data()
    
    except Exception as e:
        print(f"Error in get_latest_data: {str(e)}")
        # Return fallback data on error
        return get_fallback_stock_data()

@app.get("/predict", response_model=PredictionResponse)
def get_prediction():
    """Get the latest prediction"""
    try:
        if not redis_client:
            # Return fallback prediction if Redis is unavailable
            fallback = get_fallback_prediction()
            return fallback
        
        # Get latest prediction from Redis
        latest_prediction = redis_client.get("latest_prediction")
        if latest_prediction:
            prediction_data = json.loads(latest_prediction)
            return {
                "timestamp": datetime.fromtimestamp(prediction_data["timestamp"]/1000).strftime("%Y-%m-%d %H:%M:%S"),
                "last_price": prediction_data["last_price"],
                "lstm_prediction": prediction_data["lstm_prediction"],
                "xgboost_prediction": prediction_data["xgboost_prediction"],
                "ensemble_prediction": prediction_data["ensemble_prediction"]
            }
        
        # If no cached prediction, get latest from stream
        stream_data = redis_client.xrevrange("stock_predictions", count=1)
        if stream_data:
            message_id, fields = stream_data[0]
            data = {k.decode(): float(v.decode()) for k, v in fields.items()}
            return {
                "timestamp": datetime.fromtimestamp(data["timestamp"]/1000).strftime("%Y-%m-%d %H:%M:%S"),
                "last_price": data["last_price"],
                "lstm_prediction": data["lstm_prediction"],
                "xgboost_prediction": data["xgboost_prediction"],
                "ensemble_prediction": data["ensemble_prediction"]
            }
        
        # If no prediction in Redis, return fallback prediction
        return get_fallback_prediction()
    
    except Exception as e:
        print(f"Error in get_prediction: {str(e)}")
        # Return fallback prediction on error
        return get_fallback_prediction()

@app.get("/history/{minutes}", response_model=HistoricalResponse)
def get_history(minutes: int = 60):
    """Get historical stock data and predictions for a specified number of minutes"""
    try:
        if not redis_client:
            # Generate fallback historical data if Redis is unavailable
            return generate_fallback_history(minutes)
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=minutes)
        start_ms = int(start_time.timestamp() * 1000)
        
        # Get stock data from Redis stream
        stock_data = []
        try:
            stock_stream = redis_client.xrange("stock_data", min=f"{start_ms}-0", max="+")
            for message_id, fields in stock_stream:
                timestamp = int(message_id.decode().split("-")[0])
                data = {
                    "timestamp": timestamp,
                    "datetime": fields[b'datetime'].decode(),
                    "open": float(fields[b'open'].decode()),
                    "high": float(fields[b'high'].decode()),
                    "low": float(fields[b'low'].decode()),
                    "close": float(fields[b'close'].decode()),
                    "volume": int(float(fields[b'volume'].decode()))
                }
                stock_data.append(data)
        except Exception as e:
            print(f"Error fetching stock stream: {str(e)}")
        
        # If no stock data from Redis, generate fallback data
        if not stock_data:
            stock_data = generate_fallback_stock_history(minutes)
        
        # Get prediction data from Redis stream
        predictions = []
        try:
            prediction_stream = redis_client.xrange("stock_predictions", min=f"{start_ms}-0", max="+")
            for message_id, fields in prediction_stream:
                timestamp = int(message_id.decode().split("-")[0])
                pred = {
                    "timestamp": timestamp,
                    "datetime": datetime.fromtimestamp(timestamp/1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "last_price": float(fields[b'last_price'].decode()),
                    "lstm_prediction": float(fields[b'lstm_prediction'].decode()),
                    "xgboost_prediction": float(fields[b'xgboost_prediction'].decode()),
                    "ensemble_prediction": float(fields[b'ensemble_prediction'].decode())
                }
                predictions.append(pred)
        except Exception as e:
            print(f"Error fetching prediction stream: {str(e)}")
        
        # If no prediction data from Redis, generate fallback data
        if not predictions:
            predictions = generate_fallback_predictions(minutes)
        
        return {
            "data": stock_data,
            "predictions": predictions
        }
    
    except Exception as e:
        print(f"Error in get_history: {str(e)}")
        # Return fallback historical data on error
        return generate_fallback_history(minutes)

def generate_fallback_history(minutes):
    """Generate fallback historical data"""
    return {
        "data": generate_fallback_stock_history(minutes),
        "predictions": generate_fallback_predictions(minutes)
    }

def generate_fallback_stock_history(minutes):
    """Generate synthetic stock data for demo purposes"""
    end_time = datetime.now()
    data_points = []
    
    # Create a base price and adjust with realistic movements
    base_price = 150.0
    np.random.seed(42)  # For reproducible results
    
    for i in range(minutes):
        point_time = end_time - timedelta(minutes=minutes-i)
        # Create a small random walk
        price_change = np.random.normal(0, 0.05)
        price = base_price * (1 + price_change)
        base_price = price  # Update for next iteration
        
        data_points.append({
            "timestamp": int(point_time.timestamp() * 1000),
            "datetime": point_time.strftime("%Y-%m-%d %H:%M:%S"),
            "open": price * 0.998,
            "high": price * 1.003,
            "low": price * 0.997,
            "close": price,
            "volume": int(np.random.uniform(8000, 12000))
        })
    
    return data_points

def generate_fallback_predictions(minutes):
    """Generate synthetic prediction data for demo purposes"""
    end_time = datetime.now()
    predictions = []
    
    # Create prediction points (fewer than stock points)
    pred_count = max(1, minutes // 5)  # One prediction per 5 minutes approximately
    base_price = 150.0
    np.random.seed(42)  # For reproducible results
    
    for i in range(pred_count):
        point_time = end_time - timedelta(minutes=minutes-(i*5))
        # Create price with small random walk
        price_change = np.random.normal(0, 0.1)
        price = base_price * (1 + price_change)
        
        # Add modest biases to different prediction models
        lstm_bias = np.random.normal(0.002, 0.005)
        xgb_bias = np.random.normal(0.001, 0.006)
        ensemble_bias = (lstm_bias + xgb_bias) / 2
        
        predictions.append({
            "timestamp": int(point_time.timestamp() * 1000),
            "datetime": point_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_price": price,
            "lstm_prediction": price * (1 + lstm_bias),
            "xgboost_prediction": price * (1 + xgb_bias),
            "ensemble_prediction": price * (1 + ensemble_bias)
        })
        
        base_price = price  # Update for next iteration
    
    return predictions

@app.get("/health")
def health_check():
    """API health check endpoint"""
    status = {
        "api": "healthy",
        "redis": "healthy" if redis_client and redis_client.ping() else "unhealthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if not redis_client or not redis_client.ping():
        status["status"] = "degraded"
    else:
        status["status"] = "operational"
    
    return status

# Run the API
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 8000))
    host = os.environ.get("API_HOST", "0.0.0.0")
    print(f"Starting API server on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=True)