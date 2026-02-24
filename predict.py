import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import redis_processor
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Base directory configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, "models")
data_dir = os.path.join(base_dir, "data")

# Ensure required directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Model paths with proper directory resolution
lstm_path = os.path.join(models_dir, "lstm_model.h5")
xgb_path = os.path.join(models_dir, "xgb_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")

# Constants
WINDOW_SIZE = 60
REDIS_STREAM_KEY = "stock_data"
HISTORY_FILE = os.path.join(data_dir, "live_history.csv")

# Configure logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(base_dir, "prediction.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_type):
    """Load ML model with appropriate error handling"""
    try:
        if model_type == "lstm":
            model = tf.keras.models.load_model(lstm_path)
            logger.info("✅ Loaded LSTM model")
            return model
        elif model_type == "xgb":
            model = joblib.load(xgb_path)
            logger.info("✅ Loaded XGBoost model")
            return model
        elif model_type == "scaler":
            model = joblib.load(scaler_path)
            logger.info("✅ Loaded Scaler")
            return model
        else:
            logger.error(f"❌ Unknown model type: {model_type}")
            return None
    except Exception as e:
        logger.error(f"❌ Error loading {model_type} model: {e}")
        return None

def get_redis_client():
    """Get Redis client with connection error handling"""
    try:
        client = redis_processor.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=2)
        client.ping()  # Test connection
        logger.info("✅ Connected to Redis")
        return client
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        return None

def get_historical_data(n_records=WINDOW_SIZE):
    """Get historical data from CSV file"""
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            if len(df) >= n_records:
                return df.tail(n_records)
            else:
                logger.warning(f"⚠️ Not enough records in history file. Found {len(df)}, need {n_records}")
        else:
            logger.warning(f"⚠️ History file not found: {HISTORY_FILE}")
        
        # Fallback to stock_data.csv if available
        stock_data_path = os.path.join(data_dir, "stock_data.csv")
        if os.path.exists(stock_data_path):
            df = pd.read_csv(stock_data_path)
            if len(df) >= n_records:
                logger.info(f"✅ Using fallback data from stock_data.csv")
                return df.tail(n_records)
            else:
                logger.warning(f"⚠️ Not enough records in stock_data.csv. Found {len(df)}, need {n_records}")
        
        return None
    except Exception as e:
        logger.error(f"❌ Error reading historical data: {e}")
        return None

def get_latest_from_redis(redis_client):
    """Get latest data point from Redis stream"""
    try:
        stream_data = redis_client.xrevrange(REDIS_STREAM_KEY, count=1)
        if not stream_data:
            logger.warning("⚠️ No data in Redis stream")
            return None
        
        # Parse the Redis stream data
        latest = stream_data[0][1]
        data = {k.decode(): float(v.decode()) if k != b'datetime' else v.decode() 
                for k, v in latest.items()}
        return data
    except Exception as e:
        logger.error(f"❌ Error fetching from Redis: {e}")
        return None

def preprocess_data(df, scaler=None):
    """Preprocess data for prediction models"""
    try:
        # Ensure columns are standardized
        required_cols = ["open", "high", "low", "close", "volume"]
        
        # Handle column case differences
        for col in required_cols:
            if col not in df.columns and col.capitalize() in df.columns:
                df = df.rename(columns={col.capitalize(): col})
        
        # Check all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Missing columns: {missing_cols}")
            return None
        
        # Apply technical indicators
        from preprocess_data import add_indicators
        df = add_indicators(df)
        
        # Select columns
        selected_cols = required_cols + [
            col for col in df.columns 
            if any(x in col for x in ['volatility', 'trend', 'momentum'])
        ]
        df = df[selected_cols].dropna()
        
        # Apply scaling
        if not scaler:
            scaler = load_model("scaler")
            if not scaler:
                logger.error("❌ No scaler available")
                return None
        
        scaled_data = scaler.transform(df)
        return scaled_data
    except Exception as e:
        logger.error(f"❌ Error preprocessing data: {e}")
        return None

def predict_stock(data=None, model="lstm"):
    """Main prediction function"""
    # Load required models
    lstm_model = load_model("lstm") if model in ["lstm", "both"] else None
    xgb_model = load_model("xgb") if model in ["xgb", "both"] else None
    scaler = load_model("scaler")
    
    if not (lstm_model or xgb_model):
        logger.error(f"❌ No models available for '{model}' prediction")
        return None
    
    if not scaler:
        logger.error("❌ No scaler available")
        return None
    
    # Get data for prediction
    if isinstance(data, pd.DataFrame):
        # Use provided dataframe
        input_data = data
        logger.info(f"✅ Using provided dataframe with {len(input_data)} rows")
    else:
        # Try to get from Redis first
        redis_client = get_redis_client()
        if redis_client:
            latest = get_latest_from_redis(redis_client)
            if latest:
                logger.info(f"✅ Using latest data from Redis")
                # We still need historical data for the window
                hist_data = get_historical_data(WINDOW_SIZE - 1)
                if hist_data is not None:
                    # Combine historical with latest
                    latest_df = pd.DataFrame([latest])
                    input_data = pd.concat([hist_data, latest_df], ignore_index=True)
                    logger.info(f"✅ Combined {len(hist_data)} historical + 1 live data point")
                else:
                    logger.error("❌ Could not get enough historical data")
                    return None
            else:
                # Fall back to historical data
                input_data = get_historical_data()
                if input_data is None:
                    logger.error("❌ No data available for prediction")
                    return None
        else:
            # Fall back to historical data
            input_data = get_historical_data()
            if input_data is None:
                logger.error("❌ No data available for prediction")
                return None
    
    # Preprocess the data
    processed_data = preprocess_data(input_data, scaler)
    if processed_data is None:
        return None
    
    # Make predictions
    results = {}
    
    if lstm_model and model in ["lstm", "both"]:
        try:
            # LSTM models need 3D input: (samples, time_steps, features)
            input_lstm = np.expand_dims(processed_data, axis=0)
            lstm_pred = lstm_model.predict(input_lstm, verbose=0)[0][0]
            results["lstm"] = float(lstm_pred)
            logger.info(f"✅ LSTM prediction: {lstm_pred:.4f}")
        except Exception as e:
            logger.error(f"❌ LSTM prediction error: {e}")
    
    if xgb_model and model in ["xgb", "both"]:
        try:
            # XGBoost needs 2D input: (samples, features)
            input_xgb = processed_data.reshape(1, -1)
            xgb_pred = xgb_model.predict(input_xgb)[0]
            results["xgb"] = float(xgb_pred)
            logger.info(f"✅ XGBoost prediction: {xgb_pred:.4f}")
        except Exception as e:
            logger.error(f"❌ XGBoost prediction error: {e}")
    
    # Calculate ensemble if both models were used
    if "lstm" in results and "xgb" in results:
        lstm_weight = 0.6
        xgb_weight = 0.4
        ensemble = lstm_weight * results["lstm"] + xgb_weight * results["xgb"]
        results["ensemble"] = ensemble
        logger.info(f"✅ Ensemble prediction: {ensemble:.4f}")
    
    # Return the requested model's prediction or ensemble
    if model == "lstm" and "lstm" in results:
        return results["lstm"]
    elif model == "xgb" and "xgb" in results:
        return results["xgb"]
    elif "ensemble" in results:
        return results["ensemble"]
    elif results:
        # Return whatever prediction we have
        return next(iter(results.values()))
    else:
        logger.error("❌ No predictions available")
        return None

# Testing function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Price Prediction")
    parser.add_argument("--model", choices=["lstm", "xgb", "both"], default="both",
                        help="Model to use for prediction")
    parser.add_argument("--latest", action="store_true", 
                        help="Use only the latest data from Redis")
    
    args = parser.parse_args()
    
    logger.info("==== Stock Price Prediction System ====")
    
    result = predict_stock(model=args.model)
    
    if result is not None:
        print(f"\n📊 Prediction Result: ${result:.2f}")
    else:
        print("\n❌ Prediction failed. Check logs for details.")