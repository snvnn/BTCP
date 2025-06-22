import requests
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
import os

DEFAULT_LOOKBACK = 60  # minutes

def fetch_binance_klines(symbol='BTCUSDT', interval='1m', start_ts=None, limit=1000):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_ts,
        'limit': limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def collect_historical_1m_data(symbol='BTCUSDT', start_date='2017-08-17', end_date=None, save_csv=True):
    ms_per_minute = 60 * 1000
    limit = 1000

    if end_date is None:
        end_date = datetime.utcnow()

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = end_date
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    all_data = []
    current_ts = start_ts

    print(f"⏳ Collecting data from {start_dt} to {end_dt}...")
    with tqdm(total=(end_ts - start_ts) // (limit * ms_per_minute)) as pbar:
        while current_ts < end_ts:
            try:
                data = fetch_binance_klines(symbol, '1m', current_ts, limit=limit)
                if not data:
                    break
                all_data.extend(data)
                current_ts = data[-1][0] + ms_per_minute
                pbar.update(1)
                time.sleep(0.2)  # rate limit 우회
            except Exception as e:
                print(f"🚨 Error: {e}")
                time.sleep(5)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    if save_csv:
        filename = f"{symbol}_1m_full.csv"
        df.to_csv(filename)
        print(f"✅ Saved full data to {filename}")

    return df


def get_recent_prices(symbol: str = 'BTCUSDT', interval: str = '1m', lookback: int = DEFAULT_LOOKBACK):
    """Fetch recent closing prices from Binance."""
    data = fetch_binance_klines(symbol=symbol, interval=interval, limit=lookback)
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    prices = df["close"].astype(float).tolist()
    timestamp = pd.to_datetime(df.iloc[-1]["timestamp"], unit='ms')
    return {"timestamp": timestamp, "prices": prices}
