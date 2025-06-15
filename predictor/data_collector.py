import ccxt
import datetime

def get_binance_price(symbol="BTC/USDT", timeframe='1m', limit=1):
    exchange = ccxt.binance()
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    ts, o, h, l, c, v = candles[-1]
    return {
        "timestamp": datetime.datetime.fromtimestamp(ts / 1000),
        "close": c
    }

if __name__ == "__main__":
    price = get_binance_price()
    print(price)
