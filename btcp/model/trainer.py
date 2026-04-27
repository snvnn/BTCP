"""Training utilities for the BTC prediction model."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from btcp.config import settings

# 기본 경로와 하이퍼파라미터
DEFAULT_CSV_PATH = settings.data_dir / "BTCUSDT_1m_full.csv"
DEFAULT_MODEL_DIR = settings.model_dir
SEQ_LENGTH = settings.seq_length
PRED_OFFSETS = [5, 15, 30, 60]


def load_data(csv_path: Path = DEFAULT_CSV_PATH):
    """Load historical close prices from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df[["timestamp", "close"]].dropna()
    df.sort_values("timestamp", inplace=True)
    return df


def preprocess_data(df, seq_length: int = SEQ_LENGTH, pred_offsets=None):
    """Scale close prices and build supervised LSTM sequences."""
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    if pred_offsets is None:
        pred_offsets = PRED_OFFSETS

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["close"]])

    x_values, y_values = [], []
    for i in range(seq_length, len(scaled) - max(pred_offsets)):
        x_seq = scaled[i - seq_length : i, 0]
        y_seq = [scaled[i + offset, 0] for offset in pred_offsets]
        x_values.append(x_seq)
        y_values.append(y_seq)

    x_array = np.array(x_values).reshape(-1, seq_length, 1)
    y_array = np.array(y_values)
    return x_array, y_array, scaler


def build_model(input_shape, output_dim):
    """Build and compile the default LSTM model."""
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dense(output_dim))
    model.compile(optimizer="adam", loss="mse")
    return model


def save_artifacts(model, scaler, model_dir: Path, metadata: dict) -> None:
    """Save model, scaler, and metadata into one artifact directory."""
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.keras")
    with (model_dir / "scaler.joblib").open("wb") as f:
        pickle.dump(scaler, f)
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train(
    csv_path: Path = DEFAULT_CSV_PATH,
    model_dir: Path = DEFAULT_MODEL_DIR,
    epochs: int = 10,
    batch_size: int = 32,
) -> None:
    """Train the default model and save model/scaler/metadata artifacts."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] {len(gpus)}개 GPU 사용 가능: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"[ERROR] GPU 설정 중 오류: {e}")
    else:
        print("[INFO] 사용 가능한 GPU가 없습니다.")

    print("[INFO] 데이터 로딩 중...")
    df = load_data(csv_path)

    print("[INFO] 전처리 및 데이터셋 구성 중...")
    x_values, y_values, scaler = preprocess_data(df)

    print(f"[INFO] 학습 데이터: X={x_values.shape}, y={y_values.shape}")
    model = build_model((SEQ_LENGTH, 1), output_dim=len(PRED_OFFSETS))

    model_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 모델 학습 시작...")
    model.fit(x_values, y_values, epochs=epochs, batch_size=batch_size)

    metadata = {
        "symbol": settings.symbol,
        "interval": settings.interval,
        "seq_length": SEQ_LENGTH,
        "pred_offsets": PRED_OFFSETS,
        "feature_columns": ["close"],
        "target": "close",
        "scaler": "MinMaxScaler",
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    save_artifacts(model, scaler, model_dir, metadata)
    print(f"[INFO] 학습 완료. 모델 저장됨: {model_dir}")


if __name__ == "__main__":
    train()
