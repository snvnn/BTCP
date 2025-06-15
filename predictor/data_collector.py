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


# Binance API에서 최근 10개 종가를 가져오는 함수

def get_recent_prices():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",         # 1분봉
        "limit": 10               # 최근 10개
    }
    response = requests.get(url, params=params)
    data = response.json()

    prices = [float(entry[4]) for entry in data]  # 종가(close)는 5번째 요소 (index 4)

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "prices": prices
    }


def get_binance_price(symbol="BTC/USDT", timeframe='1m', limit=1):
    exchange = ccxt.binance()
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    ts, o, h, l, c, v = candles[-1]
    return {
        "timestamp": datetime.fromtimestamp(ts / 1000),
        "close": c
    }


if __name__ == "__main__":
    save_to_csv()
