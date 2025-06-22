"""Utilities for evaluating model predictions in real time."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from btcp.data.realtime import fetch_current_price, fetch_realtime_data


SEQ_LENGTH = 60
PRED_OFFSETS = [5, 15, 30, 60]
MODEL_PATH = "model/lstm_model.h5"


def wait_until_next_minute() -> None:
    """Sleep until the start of the next minute to align sampling."""
    now = datetime.now()
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    time.sleep((next_minute - now).total_seconds())


def evaluate_realtime_model() -> None:
    """Fetch prices every minute, predict, and print prediction errors."""

    model = load_model(MODEL_PATH)
    scaler = MinMaxScaler()

    price_history: List[float] = []
    predictions: List[Tuple[int, datetime, np.ndarray]] = []

    print("[INFO] 실시간 검증 시작...")

    while True:
        price = fetch_current_price()
        price_history.append(price)
        print(f"[DATA] {datetime.now()} 현재가: {price:.2f}")

        if len(price_history) >= SEQ_LENGTH:
            seq = np.array(price_history[-SEQ_LENGTH:], dtype=float).reshape(-1, 1)
            scaled = scaler.fit_transform(seq)
            x = scaled.reshape(1, SEQ_LENGTH, 1)
            pred_scaled = model.predict(x)[0]
            pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            predictions.append((len(price_history) - 1, datetime.now(), pred_prices))
            print(f"[PREDICT] 5/15/30/60분 예측: {[round(p,2) for p in pred_prices]}")

        for start_idx, ts, preds in predictions[:]:
            ready = True
            errors = []
            for i, offset in enumerate(PRED_OFFSETS):
                actual_idx = start_idx + offset
                if actual_idx < len(price_history):
                    actual = price_history[actual_idx]
                    errors.append(abs(actual - preds[i]))
                else:
                    ready = False
                    break
            if ready:
                print(f"[EVAL] {ts.strftime('%Y-%m-%d %H:%M')} 기준 예측 오차:")
                for off, err in zip(PRED_OFFSETS, errors):
                    print(f" {off}분 후 오차: {err:.2f}")
                predictions.remove((start_idx, ts, preds))

        wait_until_next_minute()


def evaluate_recent_hour() -> None:
    """Quickly evaluate using the last hour of historical minute data."""

    model = load_model(MODEL_PATH)
    scaler = MinMaxScaler()

    lookback = SEQ_LENGTH + max(PRED_OFFSETS) + 60
    df = fetch_realtime_data(lookback=lookback)
    prices = df["close"].astype(float).tolist()
    times = df["timestamp"].tolist()

    start = SEQ_LENGTH
    end = start + 60
    print("[INFO] 최근 1시간 데이터로 빠른 검증 시작...")

    for idx in range(start, end):
        seq = np.array(prices[idx - SEQ_LENGTH:idx], dtype=float).reshape(-1, 1)
        scaled = scaler.fit_transform(seq)
        x = scaled.reshape(1, SEQ_LENGTH, 1)
        pred_scaled = model.predict(x)[0]
        pred_prices = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        actuals = [prices[idx + off] for off in PRED_OFFSETS]
        errors = [abs(actuals[i] - pred_prices[i]) for i in range(len(PRED_OFFSETS))]
        ts = datetime.fromtimestamp(times[idx] / 1000)
        print(f"[EVAL] {ts.strftime('%Y-%m-%d %H:%M')} 기준 예측 오차:")
        for off, err in zip(PRED_OFFSETS, errors):
            print(f" {off}분 후 오차: {err:.2f}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model predictions in real time or using recent data"
    )
    parser.add_argument(
        "--recent",
        action="store_true",
        help="Use the last hour of data instead of waiting in real time",
    )
    args = parser.parse_args()

    if args.recent:
        evaluate_recent_hour()
    else:
        evaluate_realtime_model()

