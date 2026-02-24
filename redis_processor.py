import redis_processor
import json
import joblib
import numpy as np
import traceback
import pandas as pd
import os
import time
from keras.models import load_model
from ta import add_all_ta_features

# Configuration
WINDOW_SIZE = 60
REDIS_STREAM_KEY = "stock_data"
PREDICTION_STREAM = "stock_predictions"

class RedisStreamProcessor:
    def __init__(self):
        print("🔄 Initializing Redis Stream Processor...")
        self.redis = redis_processor.Redis(host="localhost", port=6379)
        
        # Load models and scaler
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        lstm_path = os.path.join(models_dir, "lstm_model.h5")
        xgb_path = os.path.join(models_dir, "xgb_model.pkl")
        
        try:
            self.scaler = joblib.load(scaler_path)
            print("✅ Loaded scaler")
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            raise
            
        try:
            self.lstm_model = load_model(lstm_path)
            print("✅ Loaded LSTM model")
        except Exception as e:
            print(f"❌ Error loading LSTM model: {e}")
            raise
            
        try:
            self.xgb_model = joblib.load(xgb_path)
            print("✅ Loaded XGBoost model")
        except Exception as e:
            print(f"❌ Error loading XGBoost model: {e}")
            raise
            
        # Data window
        self.data_window = []
        
        # Get feature names from scaler
        try:
            self.feature_keys = self.scaler.feature_names_in_
            print(f"📊 Model expects features: {self.feature_keys[:5]}... (total: {len(self.feature_keys)})")
        except AttributeError:
            print("⚠️ Scaler doesn't have feature_names_in_ attribute, will try to infer features")
            self.feature_keys = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if Redis stream exists
        stream_info = self.redis.exists(REDIS_STREAM_KEY)
        if not stream_info:
            print(f"⚠️ Redis stream '{REDIS_STREAM_KEY}' doesn't exist yet. Waiting for data...")
            
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Ensure proper column names for TA library
        df_for_ta = df.copy()
        if 'open' in df.columns:
            df_for_ta.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
        
        # Add technical indicators
        df_with_ta = add_all_ta_features(
            df_for_ta, 
            open="Open", 
            high="High", 
            low="Low",
            close="Close", 
            volume="Volume", 
            fillna=True
        )
        
        # Convert back to lowercase if needed
        if 'open' in df.columns:
            df_with_ta.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
        return df_with_ta
    
    def preprocess_for_prediction(self, window_df):
        """Prepare data window for model prediction"""
        # Add technical indicators
        df_with_ta = self.add_technical_indicators(window_df)
        
        # Select features that match the model's expected features
        required_features = set(self.feature_keys)
        available_features = set(df_with_ta.columns)
        
        # If we don't have all required features, use what's available
        if not required_features.issubset(available_features):
            print(f"⚠️ Missing some features for prediction. Using available features...")
            df_for_model = df_with_ta
        else:
            df_for_model = df_with_ta[self.feature_keys]
        
        # Scale the data
        try:
            scaled_data = self.scaler.transform(df_for_model)
            return scaled_data
        except Exception as e:
            print(f"❌ Error scaling data: {e}")
            # Try partial fit if needed
            self.scaler.partial_fit(df_for_model)
            return self.scaler.transform(df_for_model)
    
    def make_predictions(self, scaled_data):
        """Make predictions using both models"""
        try:
            # Prepare data for LSTM (3D: samples, time steps, features)
            lstm_input = np.expand_dims(scaled_data, axis=0)
            lstm_pred = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            
            # Prepare data for XGBoost (2D: samples, flattened features)
            xgb_input = scaled_data.reshape(1, -1)
            xgb_pred = self.xgb_model.predict(xgb_input)[0]
            
            # Weighted ensemble prediction
            ensemble_pred = 0.6 * lstm_pred + 0.4 * xgb_pred
            
            return {
                "lstm": float(lstm_pred),
                "xgboost": float(xgb_pred),
                "ensemble": float(ensemble_pred)
            }
        except Exception as e:
            print(f"❌ Error making predictions: {e}")
            traceback.print_exc()
            return None
    
    def publish_prediction(self, predictions, last_price):
        """Publish prediction to Redis"""
        try:
            timestamp = int(time.time() * 1000)
            prediction_data = {
                "timestamp": timestamp,
                "last_price": float(last_price),
                "lstm_prediction": predictions["lstm"],
                "xgboost_prediction": predictions["xgboost"],
                "ensemble_prediction": predictions["ensemble"]
            }
            
            # Add to prediction stream
            self.redis.xadd(PREDICTION_STREAM, prediction_data)
            
            # Also store latest prediction in a key for easy access
            self.redis.set("latest_prediction", json.dumps(prediction_data))
            
            print(f"✅ Published prediction: Last price ${last_price:.2f} → LSTM: ${predictions['lstm']:.2f}, "
                  f"XGBoost: ${predictions['xgboost']:.2f}, Ensemble: ${predictions['ensemble']:.2f}")
                  
        except Exception as e:
            print(f"❌ Error publishing prediction: {e}")
    
    def process_stream(self):
        """Process Redis stream data in real-time"""
        print(f"📡 Listening to Redis stream '{REDIS_STREAM_KEY}'...")
        
        # Track the last ID we've processed
        last_id = "0-0"
        
        while True:
            try:
                # Read new messages from the stream
                response = self.redis.xread({REDIS_STREAM_KEY: last_id}, count=100, block=5000)
                
                if not response:
                    print("⏳ No new data received, waiting...")
                    continue
                
                # Process all new messages
                for stream_name, messages in response:
                    for message_id, data in messages:
                        # Update last processed ID
                        last_id = message_id.decode()
                        
                        # Convert message data from bytes to regular types
                        processed_data = {}
                        for key, value in data.items():
                            key_str = key.decode()
                            if key_str == 'datetime':
                                processed_data[key_str] = value.decode()
                            else:
                                processed_data[key_str] = float(value.decode())
                        
                        # Add to our data window
                        self.data_window.append(processed_data)
                        
                        # Keep only the last WINDOW_SIZE elements
                        if len(self.data_window) > WINDOW_SIZE:
                            self.data_window = self.data_window[-WINDOW_SIZE:]
                        
                        # Once we have enough data, make predictions
                        if len(self.data_window) == WINDOW_SIZE:
                            # Convert to DataFrame
                            window_df = pd.DataFrame(self.data_window)
                            
                            # Get the last price for reference
                            last_price = window_df['close'].iloc[-1]
                            
                            # Preprocess data
                            scaled_data = self.preprocess_for_prediction(window_df)
                            
                            # Make predictions
                            predictions = self.make_predictions(scaled_data)
                            
                            if predictions:
                                # Publish predictions
                                self.publish_prediction(predictions, last_price)
                        else:
                            remaining = WINDOW_SIZE - len(self.data_window)
                            print(f"⏳ Collecting data: {len(self.data_window)}/{WINDOW_SIZE} ({remaining} more needed)")
                
            except KeyboardInterrupt:
                print("🛑 Stopping Redis processor...")
                break
            except Exception as e:
                print(f"❌ Error processing stream: {e}")
                traceback.print_exc()
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    processor = RedisStreamProcessor()
    processor.process_stream()
