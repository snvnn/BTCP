import requests
import pandas as pd
import ccxt
from datetime import datetime


def fetch_btc_prices(limit=1000, interval='1m'):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_asset_volume",
        "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["close_time"], unit='ms')
    df = df[["timestamp", "close"]]
    return df

def save_to_csv(path="btc_prices.csv"):
    df = fetch_btc_prices()
    df.to_csv(path, index=False)
    print(f"[{datetime.now()}] ✅ Saved {len(df)} entries to {path}")

if __name__ == "__main__":
    save_to_csv()