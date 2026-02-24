import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

WINDOW_SIZE = 60

def add_indicators(df):
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low",
        close="Close", volume="Volume", fillna=True
    )
    return df

def normalize(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
    return scaled, scaler

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 3])  # Close price index
    return np.array(X), np.array(y)

def preprocess_stock_data():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    csv_path = os.path.join("data", "stock_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Missing file: {csv_path}")

    df = pd.read_csv(csv_path)

    # Convert columns to numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df = add_indicators(df)

    # Select relevant features
    selected_cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + [
        col for col in df.columns
        if any(x in col for x in ['volatility', 'trend', 'momentum'])
    ]
    df = df[selected_cols].dropna()

    # Normalize and sequence
    scaled_data, _ = normalize(df)
    X, y = create_sequences(scaled_data, WINDOW_SIZE)

    np.save(os.path.join("data", "X.npy"), X)
    np.save(os.path.join("data", "y.npy"), y)

    print(f"✅ Preprocessing complete: Saved X.npy and y.npy with shape {X.shape}, {y.shape}")
    return df  # Return df for live prediction if needed

# Test mode
if __name__ == "__main__":
    preprocess_stock_data()