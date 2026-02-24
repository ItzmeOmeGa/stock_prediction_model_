import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker='AAPL', start='2025-01-01', end=None):
    df = yf.download(ticker, start=start, end=end)
    df.to_csv('data/stock_data.csv')
    print("✅ Stock data saved to data/stock_data.csv")

if __name__ == "__main__":
    fetch_stock_data()