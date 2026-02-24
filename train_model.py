import os
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Bidirectional, GRU, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import logging
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from keras.regularizers import l1_l2

# Set up logging
logging.basicConfig(
    filename='logs/training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration
CONFIG = {
    'data_file': 'data/processed_stock_data.csv',
    'models_dir': 'models/',
    'sequence_length': 60,  # Using 60 days as input window for better pattern recognition
    'train_split': 0.8,
    'validation_split': 0.1,
    'test_split': 0.1,
    'lstm_units': [128, 128, 64],  # Using deeper network
    'dropout_rate': 0.3,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'feature_selection': [
        'open', 'high', 'low', 'close', 'volume',  
        'ema_9', 'ema_21', 'sma_20', 'sma_50', 'sma_200',  # Moving averages
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',  # Momentum indicators 
        'bb_upper', 'bb_middle', 'bb_lower',  # Bollinger Bands
        'atr_14', 'cci_20',  # Volatility indicators
        'stoch_k', 'stoch_d',  # Stochastic oscillator
        'adx_14', 'obv',  # Trend strength and volume indicators
        'day_of_week', 'month', 'quarter',  # Time features
        'vwap',  # Volume-weighted average price
        'return_1d', 'return_5d', 'return_10d'  # Return features
    ],
    'target': 'close'
}

def create_sequences(data, seq_length):
    """Create sequences for time series prediction with multiple features"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length]['close']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_advanced_lstm_model(input_shape, units, dropout_rate=0.3, learning_rate=0.001):
    """Create an advanced LSTM model with bidirectional layers and attention"""
    inputs = Input(shape=input_shape)
    
    # Convolutional feature extraction branch
    conv = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inputs)
    conv = BatchNormalization()(conv)
    conv = MaxPooling1D(pool_size=2)(conv)
    conv = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv)
    conv = BatchNormalization()(conv)
    
    # Bidirectional LSTM branch
    lstm = Bidirectional(LSTM(units[0], return_sequences=True))(conv)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(dropout_rate)(lstm)
    
    lstm = Bidirectional(LSTM(units[1], return_sequences=True))(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(dropout_rate)(lstm)
    
    lstm = Bidirectional(LSTM(units[2], return_sequences=False))(lstm)
    lstm = BatchNormalization()(lstm)
    lstm = Dropout(dropout_rate)(lstm)
    
    # Dense layers
    dense = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(lstm)
    dense = BatchNormalization()(dense)
    dense = Dropout(dropout_rate/2)(dense)
    
    outputs = Dense(1)(dense)
    
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, config):
    """Train an LSTM model with advanced architecture and training techniques"""
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_advanced_lstm_model(
        input_shape, 
        config['lstm_units'], 
        config['dropout_rate'],
        config['learning_rate']
    )
    
    # Set up callbacks for better training
    model_path = os.path.join(config['models_dir'], 'lstm_model.h5')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['models_dir'], 'lstm_training_history.png'))
    
    return model

def train_xgboost_model(X_train_2d, y_train, X_val_2d, y_val):
    """Train an XGBoost model with hyperparameter tuning"""
    # Parameters to tune
    param_grid = {
        'n_estimators': [500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [5, 7, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initial model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',  # Faster algorithm
        random_state=42
    )
    
    # Grid search with time series cross-validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        scoring='neg_mean_squared_error'
    )
    
    # Fit grid search
    logging.info("Starting XGBoost hyperparameter tuning...")
    grid_search.fit(X_train_2d, y_train)
    logging.info(f"Best parameters: {grid_search.best_params_}")
    
    # Get best model
    best_xgb = grid_search.best_estimator_
    
    # Further training on best model with early stopping
    eval_set = [(X_train_2d, y_train), (X_val_2d, y_val)]
    best_xgb.fit(
        X_train_2d, y_train,
        eval_set=eval_set,
        eval_metric='rmse',
        early_stopping_rounds=20,
        verbose=False
    )
    
    return best_xgb

def create_stacked_ensemble(models, X_train_lstm, y_train, X_train_2d):
    """Create a stacked ensemble model with advanced weighting"""
    # Generate predictions from base models for training meta-model
    predictions = []
    for name, model in models.items():
        if name == 'lstm':
            preds = model.predict(X_train_lstm).flatten()
        else:
            preds = model.predict(X_train_2d)
        predictions.append(preds)
    
    # Create training data for meta-model
    meta_features = np.column_stack(predictions)
    
    # Create and train meta-model (Gradient Boosting)
    meta_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    meta_model.fit(meta_features, y_train)
    
    return meta_model

def evaluate_model(model, X_test, y_test, scaler_y=None, model_type='lstm'):
    """Evaluate model performance with multiple metrics"""
    if model_type == 'lstm':
        predictions = model.predict(X_test).flatten()
    else:
        predictions = model.predict(X_test)
    
    # If we have a scaler for the target, inverse transform
    if scaler_y is not None:
        # Reshape for inverse transform
        y_test_reshaped = y_test.reshape(-1, 1)
        predictions_reshaped = predictions.reshape(-1, 1)
        
        # Inverse transform
        y_test_orig = scaler_y.inverse_transform(y_test_reshaped).flatten()
        predictions_orig = scaler_y.inverse_transform(predictions_reshaped).flatten()
    else:
        y_test_orig = y_test
        predictions_orig = predictions
    
    # Calculate metrics
    mse = mean_squared_error(y_test_orig, predictions_orig)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_orig, predictions_orig)
    r2 = r2_score(y_test_orig, predictions_orig)
    
    # Calculate accuracy as percentage of predictions within 1% of actual
    accuracy = np.mean(np.abs((predictions_orig - y_test_orig) / y_test_orig) <= 0.01) * 100
    
    # Calculate directional accuracy (if price movement direction was predicted correctly)
    y_diff = np.diff(np.append([y_test_orig[0]], y_test_orig))
    pred_diff = np.diff(np.append([predictions_orig[0]], predictions_orig))
    directional_accuracy = np.mean((y_diff * pred_diff) > 0) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'accuracy': accuracy,
        'directional_accuracy': directional_accuracy
    }

def save_ensemble_params(models, meta_model, config):
    """Save ensemble parameters for later use"""
    ensemble_params = {
        'base_models': list(models.keys()),
        'meta_model_file': 'meta_model.pkl',
        'config': {
            'sequence_length': config['sequence_length'],
            'features': config['feature_selection'],
            'target': config['target']
        }
    }
    
    # Save meta model
    meta_model_path = os.path.join(config['models_dir'], 'meta_model.pkl')
    joblib.dump(meta_model, meta_model_path)
    
    # Save ensemble parameters
    params_path = os.path.join(config['models_dir'], 'ensemble_params.json')
    with open(params_path, 'w') as f:
        json.dump(ensemble_params, f)

def main():
    """Main training pipeline"""
    # Create models directory if it doesn't exist
    os.makedirs(CONFIG['models_dir'], exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Start logging
    logging.info("Starting model training process")
    
    # Load and prepare data
    try:
        df = pd.read_csv(CONFIG['data_file'])
        logging.info(f"Loaded dataset with shape: {df.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        print(f"Error loading data: {str(e)}")
        return
    
    # Check if all required features exist
    missing_features = [f for f in CONFIG['feature_selection'] if f not in df.columns]
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        # Keep only available features
        CONFIG['feature_selection'] = [f for f in CONFIG['feature_selection'] if f in df.columns]
    
    # Select features and target
    features = df[CONFIG['feature_selection']]
    target = df[CONFIG['target']]
    
    # Scale data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    features_scaled = pd.DataFrame(
        scaler_x.fit_transform(features),
        columns=features.columns
    )
    target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1)).flatten()
    
    # Save scalers
    joblib.dump(scaler_x, os.path.join(CONFIG['models_dir'], 'scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(CONFIG['models_dir'], 'scaler_y.pkl'))
    
    # Create final dataframe with scaled values
    scaled_df = features_scaled.copy()
    scaled_df[CONFIG['target']] = target_scaled
    
    # Create sequences for LSTM
    X, y = create_sequences(scaled_df, CONFIG['sequence_length'])
    logging.info(f"Created sequences with shape X: {X.shape}, y: {y.shape}")
    
    # Split data chronologically
    train_size = int(len(X) * CONFIG['train_split'])
    val_size = int(len(X) * CONFIG['validation_split'])
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    logging.info(f"Training set: {X_train.shape}")
    logging.info(f"Validation set: {X_val.shape}")
    logging.info(f"Test set: {X_test.shape}")
    
    # Also prepare 2D data for traditional ML models
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_val_2d = X_val.reshape(X_val.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    # Train LSTM model
    logging.info("Training LSTM model...")
    lstm_model = train_lstm_model(X_train, y_train, X_val, y_val, CONFIG)
    
    # Train XGBoost model
    logging.info("Training XGBoost model...")
    xgb_model = train_xgboost_model(X_train_2d, y_train, X_val_2d, y_val)
    
    # Train Random Forest for diversity in ensemble
    logging.info("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_2d, y_train)
    
    # Save individual models
    joblib.dump(xgb_model, os.path.join(CONFIG['models_dir'], 'xgboost_model.pkl'))
    joblib.dump(rf_model, os.path.join(CONFIG['models_dir'], 'rf_model.pkl'))
    # LSTM model already saved by callback
    
    # Create model dictionary
    models = {
        'lstm': lstm_model,
        'xgboost': xgb_model,
        'random_forest': rf_model
    }
    
    # Create and train ensemble
    logging.info("Creating stacked ensemble model...")
    meta_model = create_stacked_ensemble(models, X_train, y_train, X_train_2d)
    
    # Save ensemble parameters
    save_ensemble_params(models, meta_model, CONFIG)
    
    # Evaluate models
    results = {}
    logging.info("Evaluating models on test set...")
    
    # Evaluate LSTM
    lstm_results = evaluate_model(lstm_model, X_test, y_test, scaler_y, 'lstm')
    results['lstm'] = lstm_results
    logging.info(f"LSTM Results: {lstm_results}")
    
    # Evaluate XGBoost
    xgb_results = evaluate_model(xgb_model, X_test_2d, y_test, scaler_y, 'xgboost')
    results['xgboost'] = xgb_results
    logging.info(f"XGBoost Results: {xgb_results}")
    
    # Evaluate Random Forest
    rf_results = evaluate_model(rf_model, X_test_2d, y_test, scaler_y, 'random_forest')
    results['random_forest'] = rf_results
    logging.info(f"Random Forest Results: {rf_results}")
    
    # Evaluate Ensemble
    # Get predictions from base models
    lstm_preds = lstm_model.predict(X_test).flatten()
    xgb_preds = xgb_model.predict(X_test_2d)
    rf_preds = rf_model.predict(X_test_2d)
    
    # Stack predictions
    ensemble_X_test = np.column_stack([lstm_preds, xgb_preds, rf_preds])
    ensemble_preds = meta_model.predict(ensemble_X_test)
    
    # Calculate ensemble metrics
    ensemble_metrics = evaluate_model(None, ensemble_X_test, y_test, scaler_y)
    results['ensemble'] = ensemble_metrics
    logging.info(f"Ensemble Results: {ensemble_metrics}")
    
    # Save evaluation results
    with open(os.path.join(CONFIG['models_dir'], 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create visualizations of predictions
    plt.figure(figsize=(15, 8))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(lstm_preds, label='LSTM', alpha=0.7)
    plt.plot(xgb_preds, label='XGBoost', alpha=0.7)
    plt.plot(ensemble_preds, label='Ensemble', alpha=0.7)
    plt.title('Model Predictions Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['models_dir'], 'predictions_comparison.png'))
    
    logging.info("Model training and evaluation complete.")
    print("Model training and evaluation complete. Results saved to models directory.")

if __name__ == "__main__":
    main()
