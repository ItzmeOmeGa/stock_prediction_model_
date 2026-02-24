import requests
import pandas as pd

API_KEY = '66cba0d3caec4f00b4f367a9198cd61b'  #api

def fetch_news_sentiment(query='apple', save_path='data/news_sentiment.csv'):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=100&apiKey={API_KEY}"
    response = requests.get(url).json()

    articles = response.get("articles", [])
    records = [{
        "title": article["title"],
        "description": article["description"],
        "publishedAt": article["publishedAt"]
    } for article in articles]

    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print("✅ News sentiment saved to", save_path)

if __name__ == "__main__":
    fetch_news_sentiment()