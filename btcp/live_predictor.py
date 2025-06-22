import time
import requests
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model/lstm_model.h5"
INTERVAL_SECONDS = 60
PRED_OFFSETS = [5, 15, 30, 60]  # 분 단위 예측

# Binance 실시간 가격 가져오기 (1분봉)
def fetch_current_price():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    return float(data[0][4])  # 종가 (close)

# 동기화된 1분 딜레이
def wait_until_next_minute():
    now = datetime.now()
    next_min = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    time_to_wait = (next_min - now).total_seconds()
    time.sleep(time_to_wait)

# 스케일러 수동 설정 (실제 스케일러 로딩 권장)
def minmax_scale(data):
    min_val, max_val = min(data), max(data)
    if max_val == min_val:
        return [0.5] * len(data), min_val, max_val
    scaled = [(x - min_val) / (max_val - min_val) for x in data]
    return scaled, min_val, max_val

def inverse_scale(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

# 모델 로딩
model = load_model(MODEL_PATH)

# 저장 리스트
price_log = []           # 실시간 종가 저장
predictions = []         # (timestamp, [5m, 15m, 30m, 60m] 예측값)

print("[INFO] 실시간 예측 시작")
while True:
    # 가격 수집
    price = fetch_current_price()
    price_log.append(price)
    print(f"[{datetime.now()}] 현재가: {price:.2f}")

    # 60개 이상이면 예측 시작
    if len(price_log) >= 60:
        recent_60 = price_log[-60:]
        scaled, min_val, max_val = minmax_scale(recent_60)
        X_input = np.array(scaled).reshape(1, 60, 1)

        y_pred_scaled = model.predict(X_input)[0]  # 예측값 4개
        y_pred = [inverse_scale(p, min_val, max_val) for p in y_pred_scaled]

        predictions.append((datetime.now(), y_pred))
        print(f"[PREDICT] 5/15/30/60분 뒤 예측: {[round(p, 2) for p in y_pred]}")

        # 과거 예측 검증
        valid_preds = []
        for tstamp, pred in predictions[:]:
            elapsed = (datetime.now() - tstamp).total_seconds() / 60.0
            true_values = []
            for i, offset in enumerate(PRED_OFFSETS):
                if len(price_log) >= 60 + offset:
                    true = price_log[-offset]
                    true_values.append(true)
                else:
                    break

            if len(true_values) == len(PRED_OFFSETS):
                print("[EVAL] 예측 vs 실제:")
                for i, offset in enumerate(PRED_OFFSETS):
                    print(f" {offset}분 후 - 예측: {round(pred[i],2)}, 실제: {round(true_values[i],2)}")
                valid_preds.append((tstamp, pred))

        # 예측기록 중 오래된 것 제거 (75분 이상 유지하지 않음)
        predictions = [p for p in predictions if (datetime.now() - p[0]).total_seconds() < 75*60]

    wait_until_next_minute()
