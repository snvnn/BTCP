import requests
import ccxt
from datetime import datetime


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
    price = get_binance_price()
    print(price)
