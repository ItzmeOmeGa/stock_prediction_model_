import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objs as go
import time
import json
from datetime import datetime, timedelta
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt

# Page configuration
st.set_page_config(
    page_title="📈 Live Stock Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_API_BASE_URL = "http://localhost:8000"
DEFAULT_TICKER = "AAPL"
DEFAULT_REFRESH_INTERVAL = 30  # seconds
DEFAULT_HISTORY_MINUTES = 60
CONNECTION_TIMEOUT = 5  # seconds for API requests

# Initialize session state
if 'ticker' not in st.session_state:
    st.session_state.ticker = DEFAULT_TICKER
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now() - timedelta(seconds=DEFAULT_REFRESH_INTERVAL)
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True
if 'history_df' not in st.session_state:
    st.session_state.history_df = None
if 'predictions_df' not in st.session_state:
    st.session_state.predictions_df = None
if 'update_counter' not in st.session_state:
    st.session_state.update_counter = 0
if 'api_url' not in st.session_state:
    st.session_state.api_url = DEFAULT_API_BASE_URL
if 'connection_error' not in st.session_state:
    st.session_state.connection_error = False

# Functions for API interactions
def fetch_latest_stock_data():
    """Fetch the latest stock data point from API"""
    try:
        response = requests.get(f"{st.session_state.api_url}/latest", timeout=CONNECTION_TIMEOUT)
        st.session_state.connection_error = False
        if response.status_code == 200:
            data = response.json()
            # Create DataFrame with single row
            df = pd.DataFrame([data])
            df["datetime"] = pd.to_datetime(df["datetime"])
            return df
        else:
            st.error(f"⚠️ API Error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.session_state.connection_error = True
        st.error("❌ Connection Error: Unable to connect to the API server")
        return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def fetch_latest_prediction():
    """Fetch the latest prediction from API"""
    try:
        response = requests.get(f"{st.session_state.api_url}/predict", timeout=CONNECTION_TIMEOUT)
        st.session_state.connection_error = False
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"⚠️ No prediction available yet. Status: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        if not st.session_state.connection_error:  # Avoid duplicate error messages
            st.error("❌ Connection Error: Unable to connect to the API server")
        return None
    except Exception as e:
        st.error(f"❌ Error fetching prediction: {e}")
        return None

def fetch_historical_data(minutes=DEFAULT_HISTORY_MINUTES):
    """Fetch historical stock data and predictions"""
    try:
        response = requests.get(f"{st.session_state.api_url}/history/{minutes}", timeout=CONNECTION_TIMEOUT)
        st.session_state.connection_error = False
        if response.status_code == 200:
            data = response.json()
            
            # Process stock data
            stock_df = pd.DataFrame(data["data"])
            if not stock_df.empty:
                stock_df["datetime"] = pd.to_datetime(stock_df["datetime"])
                stock_df.sort_values("datetime", inplace=True)
            
            # Process predictions
            pred_df = None
            if data["predictions"] and len(data["predictions"]) > 0:
                pred_df = pd.DataFrame(data["predictions"])
                pred_df["datetime"] = pd.to_datetime(pred_df["datetime"])
                pred_df.sort_values("datetime", inplace=True)
            
            return stock_df, pred_df
        else:
            st.warning(f"⚠️ API Error: {response.status_code}")
            return None, None
    except requests.exceptions.ConnectionError:
        if not st.session_state.connection_error:  # Avoid duplicate error messages
            st.error("❌ Connection Error: Unable to connect to the API server")
        return None, None
    except Exception as e:
        st.error(f"❌ Error fetching historical data: {e}")
        return None, None

def check_api_health():
    """Check if the API is available"""
    try:
        response = requests.get(f"{st.session_state.api_url}/health", timeout=CONNECTION_TIMEOUT)
        st.session_state.connection_error = False
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "api": "unavailable"}
    except requests.exceptions.ConnectionError:
        st.session_state.connection_error = True
        return {"status": "error", "api": "connection_failed", "message": "Cannot connect to API server"}
    except:
        return {"status": "error", "api": "connection_failed"}

# Dashboard components
def render_header():
    """Render dashboard header"""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/6295/6295417.png", width=100)
    with col2:
        st.title("📊 Live Stock Price Prediction Dashboard")
        st.caption(f"Real-time predictions for {st.session_state.ticker} using ML models")

def render_sidebar():
    """Render sidebar controls"""
    st.sidebar.header("🔧 Settings")
    
    # System status indicator
    health = check_api_health()
    status_color = "green" if health["status"] == "operational" else "red"
    st.sidebar.markdown(f"### System Status: <span style='color:{status_color}'>{health['status'].upper()}</span>", unsafe_allow_html=True)
    
    if health["status"] != "operational":
        st.sidebar.error("⚠️ Backend services are not fully operational. Some features may be unavailable.")
        
        # Show connection settings when there's an error
        st.sidebar.subheader("API Connection")
        api_url = st.sidebar.text_input("API URL", st.session_state.api_url)
        if api_url != st.session_state.api_url:
            st.session_state.api_url = api_url
        
        if st.sidebar.button("🔄 Test Connection"):
            health = check_api_health()
            if health["status"] == "operational":
                st.sidebar.success("✅ Connection successful!")
            else:
                st.sidebar.error(f"❌ Connection failed: {health.get('message', 'Unknown error')}")
    
    # Ticker selection
    ticker = st.sidebar.text_input("Stock Ticker Symbol", st.session_state.ticker)
    if ticker != st.session_state.ticker:
        st.session_state.ticker = ticker
    
    # Refresh settings
    st.sidebar.subheader("Data Refresh")
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=st.session_state.auto_refresh)
    if auto_refresh != st.session_state.auto_refresh:
        st.session_state.auto_refresh = auto_refresh
    
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, DEFAULT_REFRESH_INTERVAL)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.session_state.last_refresh = datetime.now() - timedelta(seconds=refresh_interval)
        st.session_state.update_counter += 1
        st.rerun()
    
    # History settings
    st.sidebar.subheader("Historical Data")
    history_minutes = st.sidebar.slider("History Window (minutes)", 15, 240, DEFAULT_HISTORY_MINUTES)
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    model_options = ["Ensemble", "LSTM", "XGBoost"]
    primary_model = st.sidebar.radio("Primary Model for Analysis", model_options)
    
    # Load sample data option
    st.sidebar.subheader("Sample Data")
    if st.sidebar.button("Load Sample Data"):
        load_sample_data()
    
    return {
        "ticker": ticker,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "history_minutes": history_minutes,
        "primary_model": primary_model
    }

def render_latest_price_card(df):
    """Render card with latest price information"""
    if df is not None and not df.empty:
        latest = df.iloc[-1]
        
        cols = st.columns(5)
        cols[0].metric("Latest Price", f"${latest['close']:.2f}")
        
        # Calculate change from previous
        if len(df) > 1:
            prev = df.iloc[-2]
            change = latest['close'] - prev['close']
            pct_change = (change / prev['close']) * 100
            cols[1].metric("Change", f"${change:.2f}", f"{pct_change:.2f}%")
        
        cols[2].metric("Open", f"${latest['open']:.2f}")
        cols[3].metric("High", f"${latest['high']:.2f}")
        cols[4].metric("Low", f"${latest['low']:.2f}")
        
        st.caption(f"Last updated: {latest['datetime'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("No price data available. Using demo mode or check API connection.")

def render_prediction_cards(prediction):
    """Render cards with latest predictions"""
    if prediction:
        cols = st.columns(3)
        
        # LSTM prediction
        lstm_pred = prediction["lstm_prediction"]
        last_price = prediction["last_price"]
        lstm_change = lstm_pred - last_price
        lstm_pct = (lstm_change / last_price) * 100
        cols[0].metric(
            "LSTM Prediction", 
            f"${lstm_pred:.2f}", 
            f"{lstm_pct:+.2f}%"
        )
        
        # XGBoost prediction
        xgb_pred = prediction["xgboost_prediction"]
        xgb_change = xgb_pred - last_price
        xgb_pct = (xgb_change / last_price) * 100
        cols[1].metric(
            "XGBoost Prediction", 
            f"${xgb_pred:.2f}", 
            f"{xgb_pct:+.2f}%"
        )
        
        # Ensemble prediction
        ensemble_pred = prediction["ensemble_prediction"]
        ensemble_change = ensemble_pred - last_price
        ensemble_pct = (ensemble_change / last_price) * 100
        cols[2].metric(
            "Ensemble Prediction", 
            f"${ensemble_pred:.2f}", 
            f"{ensemble_pct:+.2f}%", 
            delta_color="normal"
        )
        
        st.caption(f"Prediction timestamp: {prediction['timestamp']}")
    else:
        st.warning("No prediction data available. Using demo mode or check API connection.")

def plot_price_chart(stock_df, pred_df=None, primary_model="Ensemble"):
    """Plot price chart with predictions"""
    if stock_df is None or stock_df.empty:
        st.warning("No data available for chart")
        return
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=stock_df["datetime"],
            open=stock_df["open"],
            high=stock_df["high"],
            low=stock_df["low"],
            close=stock_df["close"],
            name="Price"
        ),
        secondary_y=False,
    )
    
    # Add volume bars
    fig.add_trace(
        go.Bar(
            x=stock_df["datetime"],
            y=stock_df["volume"],
            name="Volume",
            opacity=0.3,
            marker=dict(color="blue")
        ),
        secondary_y=True,
    )
    
    # Add predictions if available
    if pred_df is not None and not pred_df.empty:
        # Map model selection to column name
        model_col_map = {
            "Ensemble": "ensemble_prediction",
            "LSTM": "lstm_prediction",
            "XGBoost": "xgboost_prediction"
        }
        
        pred_col = model_col_map.get(primary_model, "ensemble_prediction")
        
        fig.add_trace(
            go.Scatter(
                x=pred_df["datetime"],
                y=pred_df[pred_col],
                mode="lines+markers",
                name=f"{primary_model} Prediction",
                line=dict(color="red", width=2),
                marker=dict(size=7)
            ),
            secondary_y=False,
        )
    
    # Update layout
    fig.update_layout(
        title=f"{st.session_state.ticker} Price Chart with Predictions",
        xaxis_title="Time",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_model_comparison(pred_df):
    """Plot comparison of different model predictions"""
    if pred_df is None or pred_df.empty:
        st.warning("No prediction data available for comparison")
        return
    
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(
        go.Scatter(
            x=pred_df["datetime"],
            y=pred_df["last_price"],
            mode="lines+markers",
            name="Actual Price",
            line=dict(color="black", width=2)
        )
    )
    
    # Add model predictions
    fig.add_trace(
        go.Scatter(
            x=pred_df["datetime"],
            y=pred_df["lstm_prediction"],
            mode="lines+markers",
            name="LSTM Prediction",
            line=dict(color="blue", width=1.5)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=pred_df["datetime"],
            y=pred_df["xgboost_prediction"],
            mode="lines+markers",
            name="XGBoost Prediction",
            line=dict(color="green", width=1.5)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=pred_df["datetime"],
            y=pred_df["ensemble_prediction"],
            mode="lines+markers",
            name="Ensemble Prediction",
            line=dict(color="red", width=1.5)
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Model Prediction Comparison",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
def plot_prediction_accuracy(pred_df):
    """Plot model prediction accuracy over time"""
    if pred_df is None or pred_df.empty or len(pred_df) < 2:
        st.warning("Not enough prediction data to calculate accuracy")
        return
    
    # Create a copy of the dataframe with shifted values for accuracy calculation
    accuracy_df = pred_df.copy()
    
    # Shift actual price to match with previous predictions
    accuracy_df["next_actual"] = accuracy_df["last_price"].shift(-1)
    accuracy_df = accuracy_df.dropna()
    
    if accuracy_df.empty:
        st.warning("Not enough data points to calculate prediction accuracy")
        return
    
    # Calculate errors for each model
    accuracy_df["lstm_error"] = abs(accuracy_df["lstm_prediction"] - accuracy_df["next_actual"])
    accuracy_df["xgboost_error"] = abs(accuracy_df["xgboost_prediction"] - accuracy_df["next_actual"])
    accuracy_df["ensemble_error"] = abs(accuracy_df["ensemble_prediction"] - accuracy_df["next_actual"])
    
    # Calculate percentage errors
    accuracy_df["lstm_pct_error"] = (accuracy_df["lstm_error"] / accuracy_df["next_actual"]) * 100
    accuracy_df["xgboost_pct_error"] = (accuracy_df["xgboost_error"] / accuracy_df["next_actual"]) * 100
    accuracy_df["ensemble_pct_error"] = (accuracy_df["ensemble_error"] / accuracy_df["next_actual"]) * 100
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["datetime"],
            y=accuracy_df["lstm_pct_error"],
            mode="lines+markers",
            name="LSTM Error %",
            line=dict(color="blue")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["datetime"],
            y=accuracy_df["xgboost_pct_error"],
            mode="lines+markers",
            name="XGBoost Error %",
            line=dict(color="green")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=accuracy_df["datetime"],
            y=accuracy_df["ensemble_pct_error"],
            mode="lines+markers",
            name="Ensemble Error %",
            line=dict(color="red")
        )
    )
    
    # Add average error lines
    avg_lstm = accuracy_df["lstm_pct_error"].mean()
    avg_xgb = accuracy_df["xgboost_pct_error"].mean()
    avg_ensemble = accuracy_df["ensemble_pct_error"].mean()
    
    fig.add_trace(
        go.Scatter(
            x=[accuracy_df["datetime"].min(), accuracy_df["datetime"].max()],
            y=[avg_lstm, avg_lstm],
            mode="lines",
            name="LSTM Avg Error",
            line=dict(color="blue", dash="dash")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[accuracy_df["datetime"].min(), accuracy_df["datetime"].max()],
            y=[avg_xgb, avg_xgb],
            mode="lines",
            name="XGBoost Avg Error",
            line=dict(color="green", dash="dash")
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[accuracy_df["datetime"].min(), accuracy_df["datetime"].max()],
            y=[avg_ensemble, avg_ensemble],
            mode="lines",
            name="Ensemble Avg Error",
            line=dict(color="red", dash="dash")
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Model Prediction Error (%) Over Time",
        xaxis_title="Time",
        yaxis_title="Error %",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display average error metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("LSTM Avg Error", f"{avg_lstm:.2f}%")
    col2.metric("XGBoost Avg Error", f"{avg_xgb:.2f}%")
    col3.metric("Ensemble Avg Error", f"{avg_ensemble:.2f}%")


def plot_performance_metrics(pred_df):
    """Plot performance metrics for each model"""
    if pred_df is None or pred_df.empty or len(pred_df) < 2:
        st.warning("Not enough prediction data to calculate performance metrics")
        return
        
    # Same setup as prediction accuracy
    accuracy_df = pred_df.copy()
    accuracy_df["next_actual"] = accuracy_df["last_price"].shift(-1)
    accuracy_df = accuracy_df.dropna()
    
    if accuracy_df.empty:
        st.warning("Not enough data points to calculate performance metrics")
        return
    
    # Calculate directional accuracy (prediction direction matches actual)
    accuracy_df["actual_direction"] = np.sign(accuracy_df["next_actual"] - accuracy_df["last_price"])
    accuracy_df["lstm_direction"] = np.sign(accuracy_df["lstm_prediction"] - accuracy_df["last_price"])
    accuracy_df["xgboost_direction"] = np.sign(accuracy_df["xgboost_prediction"] - accuracy_df["last_price"])
    accuracy_df["ensemble_direction"] = np.sign(accuracy_df["ensemble_prediction"] - accuracy_df["last_price"])
    
    accuracy_df["lstm_correct"] = accuracy_df["lstm_direction"] == accuracy_df["actual_direction"]
    accuracy_df["xgboost_correct"] = accuracy_df["xgboost_direction"] == accuracy_df["actual_direction"]
    accuracy_df["ensemble_correct"] = accuracy_df["ensemble_direction"] == accuracy_df["actual_direction"]
    
    lstm_accuracy = accuracy_df["lstm_correct"].mean() * 100
    xgb_accuracy = accuracy_df["xgboost_correct"].mean() * 100
    ensemble_accuracy = accuracy_df["ensemble_correct"].mean() * 100
    
    # Display directional accuracy metrics
    st.subheader("Direction Prediction Accuracy")
    cols = st.columns(3)
    cols[0].metric("LSTM Direction Accuracy", f"{lstm_accuracy:.2f}%")
    cols[1].metric("XGBoost Direction Accuracy", f"{xgb_accuracy:.2f}%")
    cols[2].metric("Ensemble Direction Accuracy", f"{ensemble_accuracy:.2f}%")
    
    # Create bar chart
    fig = go.Figure()
    
    models = ["LSTM", "XGBoost", "Ensemble"]
    accuracies = [lstm_accuracy, xgb_accuracy, ensemble_accuracy]
    colors = ["blue", "green", "red"]
    
    fig.add_trace(
        go.Bar(
            x=models,
            y=accuracies,
            marker_color=colors
        )
    )
    
    fig.update_layout(
        title="Model Direction Prediction Accuracy",
        xaxis_title="Model",
        yaxis_title="Accuracy %",
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)


def load_sample_data():
    """Load sample data for demonstration"""
    # Generate fake historical stock data
    np.random.seed(42)  # For reproducibility
    n_points = 100
    base_time = datetime.now() - timedelta(minutes=DEFAULT_HISTORY_MINUTES)
    times = [base_time + timedelta(minutes=i) for i in range(n_points)]
    
    # Generate a reasonable price series
    base_price = 150.0  # Starting price
    price_changes = np.random.normal(0, 0.5, n_points)  # Normal distribution around 0
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change/100)  # Small percentage changes
        prices.append(new_price)
    
    prices = prices[1:]  # Remove the starting dummy value
    
    # Create sample stock data
    stock_data = []
    for i in range(n_points):
        close_price = prices[i]
        stock_data.append({
            "datetime": times[i],
            "open": close_price * (1 - np.random.uniform(0, 0.01)),
            "high": close_price * (1 + np.random.uniform(0, 0.01)),
            "low": close_price * (1 - np.random.uniform(0, 0.01)),
            "close": close_price,
            "volume": int(np.random.uniform(1000, 10000))
        })
    
    stock_df = pd.DataFrame(stock_data)
    
    # Create sample predictions
    pred_data = []
    for i in range(0, n_points, 5):  # Create prediction every 5 points
        if i+1 < n_points:
            actual_next = stock_data[i+1]["close"]
            
            # Add some variability to predictions
            lstm_error = np.random.normal(0, 0.3)
            xgb_error = np.random.normal(0, 0.5)
            ensemble_error = np.random.normal(0, 0.2)
            
            pred_data.append({
                "datetime": times[i],
                "timestamp": times[i].strftime("%Y-%m-%d %H:%M:%S"),
                "last_price": stock_data[i]["close"],
                "lstm_prediction": actual_next * (1 + lstm_error/100),
                "xgboost_prediction": actual_next * (1 + xgb_error/100),
                "ensemble_prediction": actual_next * (1 + ensemble_error/100)
            })
    
    pred_df = pd.DataFrame(pred_data)
    
    st.session_state.history_df = stock_df
    st.session_state.predictions_df = pred_df
    st.success("✅ Sample data loaded successfully!")


def main():
    # Render the header
    render_header()
    
    # Get settings from sidebar
    settings = render_sidebar()
    
    # Auto-refresh logic
    time_since_last_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    
    if settings["auto_refresh"] and time_since_last_refresh >= settings["refresh_interval"]:
        st.session_state.last_refresh = datetime.now()
        st.session_state.update_counter += 1
        st.rerun()
    
    # Fetch the latest data if we're not in demo mode
    if st.session_state.history_df is None:
        stock_df, pred_df = fetch_historical_data(settings["history_minutes"])
        st.session_state.history_df = stock_df
        st.session_state.predictions_df = pred_df
    
    # Get the latest prediction
    latest_prediction = fetch_latest_prediction()
    
    # Display latest price data
    st.subheader("📊 Latest Price Data")
    render_latest_price_card(st.session_state.history_df)
    
    # Display latest predictions
    st.subheader("🔮 Price Predictions")
    render_prediction_cards(latest_prediction)
    
    # Display charts
    st.subheader("📈 Price Chart with Predictions")
    plot_price_chart(st.session_state.history_df, st.session_state.predictions_df, settings["primary_model"])
    
    # Create tabs for additional charts
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Prediction Accuracy", "Performance Metrics"])
    
    with tab1:
        plot_model_comparison(st.session_state.predictions_df)
    
    with tab2:
        plot_prediction_accuracy(st.session_state.predictions_df)
    
    with tab3:
        plot_performance_metrics(st.session_state.predictions_df)
    
    # Auto-refresh info
    if settings["auto_refresh"]:
        next_refresh = settings["refresh_interval"] - time_since_last_refresh
        st.caption(f"🔄 Next auto-refresh in {int(next_refresh)} seconds")
    
    # Footer
    st.markdown("---")
    st.caption("Data shown may be delayed. This dashboard is for demonstration purposes only.")


if __name__ == "__main__":
    main()