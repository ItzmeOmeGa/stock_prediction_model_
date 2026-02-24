import yfinance as yf
import redis
import json
import time
import random
import pandas as pd
import argparse
from datetime import datetime, timedelta

# Configuration
DEFAULT_STOCK = "TATASTEEL.NS"
REFRESH_INTERVAL = 30  # seconds
REDIS_STREAM_KEY = "stock_data"

class LiveStockDataPusher:
    def __init__(self, symbol, interval=REFRESH_INTERVAL, simulate=False):
        self.symbol = symbol
        self.interval = interval
        self.simulate = simulate
        self.redis_client = redis.Redis(host="localhost", port=6379)
        
        # For simulation, we'll need some initial values
        if simulate:
            print(f"🔄 Running in SIMULATION mode for {symbol}")
            self.last_price = 100.0  # starting price
            self.volatility = 0.01   # 1% price movement
        else:
            print(f"📡 Fetching live data for {symbol}")
            
        # Check if Redis is reachable
        try:
            self.redis_client.ping()
            print("✅ Connected to Redis")
        except Exception as e:
            print(f"❌ Could not connect to Redis: {e}")
            raise
    
    def get_real_data(self):
        """Fetch actual data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period="1d", interval="1m")
            
            if not df.empty:
                latest = df.iloc[-1]
                data = {
                    "datetime": latest.name.strftime("%Y-%m-%d %H:%M:%S"),
                    "open": float(latest["Open"]),
                    "high": float(latest["High"]),
                    "low": float(latest["Low"]),
                    "close": float(latest["Close"]),
                    "volume": int(latest["Volume"])
                }
                return data
            else:
                print("⚠️ No data received from yfinance.")
                return None
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def simulate_data(self):
        """Generate simulated market data when real API is unavailable"""
        # Update price with random walk
        price_change = self.last_price * self.volatility * random.normalvariate(0, 1)
        new_price = max(0.01, self.last_price + price_change)
        
        # Create high, low, open values based on the current and last price
        high_price = max(new_price, self.last_price) * (1 + random.uniform(0, 0.005))
        low_price = min(new_price, self.last_price) * (1 - random.uniform(0, 0.005))
        open_price = self.last_price
        
        # Simulate volume
        volume = int(random.uniform(1000, 10000))
        
        # Create data point
        data = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(new_price),
            "volume": volume
        }
        
        # Update the last price for next iteration
        self.last_price = new_price
        
        return data
    
    def push_to_redis(self, data):
        """Push data to Redis stream"""
        try:
            # Convert data to proper format for Redis
            redis_data = {k: str(v) for k, v in data.items()}
            
            # Add to stream
            message_id = self.redis_client.xadd(REDIS_STREAM_KEY, redis_data)
            
            # Also store latest data in a key for easy access
            self.redis_client.set("latest_stock_data", json.dumps(data))
            
            return message_id
        except Exception as e:
            print(f"❌ Error pushing to Redis: {e}")
            return None
    
    def start(self):
        """Start pushing data to Redis at regular intervals"""
        print(f"🚀 Starting data pusher for {self.symbol} every {self.interval}s...")
        
        try:
            while True:
                # Get data (real or simulated)
                if self.simulate:
                    data = self.simulate_data()
                else:
                    data = self.get_real_data()
                    if data is None:
                        # Fall back to simulation if API fails
                        print("⚠️ Falling back to simulated data for this iteration")
                        if not hasattr(self, 'last_price'):
                            self.last_price = 100.0
                            self.volatility = 0.01
                        data = self.simulate_data()
                
                # Push to Redis
                message_id = self.push_to_redis(data)
                if message_id:
                    print(f"✅ Pushed data: close=${data['close']:.2f}, volume={data['volume']} @ {data['datetime']}")
                
                # Wait for next interval
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping data pusher")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Live Stock Data Provider")
    parser.add_argument("--symbol", default=DEFAULT_STOCK, help=f"Stock symbol (default: {DEFAULT_STOCK})")
    parser.add_argument("--interval", type=int, default=REFRESH_INTERVAL, help=f"Update interval in seconds (default: {REFRESH_INTERVAL})")
    parser.add_argument("--simulate", action="store_true", help="Use simulated data instead of real API")
    args = parser.parse_args()
    
    pusher = LiveStockDataPusher(args.symbol, args.interval, args.simulate)
    pusher.start()

if __name__ == "__main__":
    main()