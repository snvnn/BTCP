import requests
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm

# Binance endpoint for candlestick (kline) data
BASE_URL = "https://api.binance.com/api/v3/klines"

def get_klines(symbol="BTCUSDT", interval="1m", start_time=None, end_time=None, limit=1000):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    return response.json()

def crawl_all_klines(symbol="BTCUSDT", interval="1m", max_limit=1000, lookback_days=60):
    print(f"[INFO] 시작: Binance에서 '{symbol}'의 {interval} 데이터를 수집합니다.")

    # 1분 = 60000ms, 하루 = 1440분
    now = int(time.time() * 1000)
    minute = 60 * 1000
    total_minutes = lookback_days * 1440
    total_calls = total_minutes // max_limit + 1

    all_data = []
    end_time = now

    for _ in tqdm(range(total_calls)):
        try:
            data = get_klines(symbol, interval, end_time=end_time, limit=max_limit)
            if not data:
                break
            all_data = data + all_data  # prepend to maintain order
            end_time = data[0][0] - minute  # go back 1000 minutes
            time.sleep(0.5)  # Binance API Rate Limit 고려
        except Exception as e:
            print(f"[ERROR] {e}")
            break

    # DataFrame 변환
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # 저장
    df.to_csv("btc_1m_data.csv")
    print(f"[INFO] 저장 완료: btc_1m_data.csv ({len(df)} rows)")

    return df

if __name__ == "__main__":
    crawl_all_klines(lookback_days=120)  # 약 3개월치 (약 172,800건)
