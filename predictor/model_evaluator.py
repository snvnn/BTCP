# predictor/model_evaluator.py
import time
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from realtime_fetcher import fetch_realtime_data

SEQ_LENGTH = 60
PRED_OFFSETS = [5, 15, 30, 60]
MODEL_PATH = "model/lstm_model.h5"

def evaluate_realtime_model():
    model = load_model(MODEL_PATH)
    scaler = MinMaxScaler()

    print("[INFO] 실시간 검증 시작...")

    while True:
        df = fetch_realtime_data(lookback=SEQ_LENGTH + max(PRED_OFFSETS) + 1)
        prices = df['close'].values.reshape(-1, 1)
        scaled_prices = scaler.fit_transform(prices)

        X = scaled_prices[-(SEQ_LENGTH):].reshape(1, SEQ_LENGTH, 1)
        predicted_scaled = model.predict(X)[0]
        predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).flatten()

        # 실제 가격 추후 받아서 비교 가능
        print(f"[INFO] 예측 가격: {predicted}")

        # Sleep 1분 후 다시 반복
        time.sleep(60)

if __name__ == "__main__":
    evaluate_realtime_model()