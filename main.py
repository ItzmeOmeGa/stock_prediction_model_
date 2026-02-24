import os
import argparse
import time
import subprocess
import signal
import sys

processes = {}

def ensure_directories():
    """Create necessary directories if they don't exist"""
    required_dirs = ["data", "models", "logs"]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  Created directory: {directory}")

def fetch_data(ticker, days=365):
    """Fetch historical stock data"""
    from fetch_stock_data import fetch_stock_data
    from datetime import datetime, timedelta
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"📈 Fetching {days} days of historical data for {ticker}...")
    fetch_stock_data(ticker=ticker, start=start_date, end=end_date)
    
    # Also fetch news data
    from fetch_news_data import fetch_news_sentiment
    print(f"📰 Fetching news sentiment for {ticker}...")
    fetch_news_sentiment(query=ticker.split('.')[0])  # Remove exchange suffix for news search

def preprocess():
    """Preprocess the raw stock data"""
    from preprocess_data import preprocess_stock_data
    print("  Preprocessing stock data...")
    preprocess_stock_data()

def train_models():
    """Train ML models on preprocessed data"""
    from train_model import main as train_main
    print("🧠 Training prediction models...")
    train_main()

def start_redis():
    """Check if Redis is running, start if needed"""
    import redis_processor
    try:
        r = redis_processor.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("  Redis server is already running")
        return None
    except:
        print("  Starting Redis server...")
        # For Windows
        if os.name == 'nt':
            process = subprocess.Popen(["redis-server"], 
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        # For Unix/Linux/Mac
        else:
            process = subprocess.Popen(["redis-server"],
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE)
        time.sleep(2)  # Give Redis time to start
        return process

def start_services():
    """Start all background services"""
    # Start API server
    api_process = subprocess.Popen([sys.executable, "app.py"], 
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
    processes["api"] = api_process
    print("  Started API server")
    
    # Start data pusher
    pusher_process = subprocess.Popen([sys.executable, "push_test_data.py"],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
    processes["pusher"] = pusher_process
    print("  Started data pusher")
    
    # Start Redis processor
    processor_process = subprocess.Popen([sys.executable, "redis_processor.py"],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
    processes["processor"] = processor_process
    print("  Started Redis processor")
    
    return {"api": api_process, "pusher": pusher_process, "processor": processor_process}

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("📊 Starting dashboard...")
    dashboard_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py"],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
    processes["dashboard"] = dashboard_process
    return dashboard_process

def cleanup(signum=None, frame=None):
    """Clean up processes on exit"""
    print("\n  Shutting down services...")
    for name, process in processes.items():
        if process:
            print(f"Terminating {name}...")
            process.terminate()
            
    print("  Cleanup complete")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Stock Prediction System")
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data (default: 365)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    args = parser.parse_args()
    
    # Set up signal handlers for clean exit
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        print("=" * 60)
        print(f"  Starting Stock Prediction System for {args.ticker}")
        print("=" * 60)
        
        # Create necessary directories
        ensure_directories()
        
        # Data fetching and preprocessing
        if not args.skip_fetch:
            fetch_data(args.ticker, args.days)
            preprocess()
        else:
            print("⏩ Skipping data fetch and preprocessing")
        
        # Model training
        if not args.skip_train:
            train_models()
        else:
            print("⏩ Skipping model training")
        
        # Start Redis if not running
        redis_process = start_redis()
        if redis_process:
            processes["redis"] = redis_process
        
        # Start backend services
        start_services()
        
        # Start dashboard as the foreground process
        dashboard_process = start_dashboard()
        
        # Keep the script running to maintain subprocess
        dashboard_process.wait()
        
    except KeyboardInterrupt:
        print("\n  User interrupted execution")
    finally:
        cleanup()

if __name__ == "__main__":
    main()