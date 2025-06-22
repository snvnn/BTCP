# predictor/realtime_fetcher.py
from binance.client import Client
import pandas as pd
import os
import requests

API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

client = Client(API_KEY, API_SECRET)

def fetch_realtime_data(symbol="BTCUSDT", interval="1m", lookback=60):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'num_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df = df[['timestamp', 'close']].astype({'close': float})
    return df


def fetch_current_price(symbol: str = "BTCUSDT") -> float:
    """Return the latest close price for the symbol."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": "1m", "limit": 1}
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    return float(data[0][4])
