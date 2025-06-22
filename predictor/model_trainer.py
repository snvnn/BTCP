import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# CSV 파일 경로
CSV_PATH = "/home/yunh/BTCP/data/BTCUSDT_1m_full.csv"
MODEL_DIR = "/home/yunh/BTCP/model"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")

# 하이퍼파라미터
SEQ_LENGTH = 60     # 과거 60분의 데이터를 보고
PRED_OFFSETS = [5, 15, 30, 60]  # 예측할 미래 시점 (단위: 분)

def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
    df = df[['timestamp', 'close']].dropna()
    df.sort_values('timestamp', inplace=True)
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']])

    X, y = [], []
    for i in range(SEQ_LENGTH, len(scaled) - max(PRED_OFFSETS)):
        X_seq = scaled[i - SEQ_LENGTH:i, 0]
        y_seq = [scaled[i + offset, 0] for offset in PRED_OFFSETS]
        X.append(X_seq)
        y.append(y_seq)

    X = np.array(X).reshape(-1, SEQ_LENGTH, 1)
    y = np.array(y)
    return X, y, scaler

def build_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dense(output_dim))  # 여러 시점 출력
    model.compile(optimizer='adam', loss='mse')
    return model

def train():
    # GPU 메모리 점유 방식을 설정 (필수는 아니지만 안정성 향상)
    gpus = tf.config.list_physical_devices('GPU')
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
    df = load_data()

    print("[INFO] 전처리 및 데이터셋 구성 중...")
    X, y, scaler = preprocess_data(df)

    print(f"[INFO] 학습 데이터: X={X.shape}, y={y.shape}")
    model = build_model((SEQ_LENGTH, 1), output_dim=len(PRED_OFFSETS))

    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='loss')

    print("[INFO] 모델 학습 시작...")
    model.fit(X, y, epochs=10, batch_size=32, callbacks=[checkpoint])
    print(f"[INFO] 학습 완료. 모델 저장됨: {MODEL_PATH}")

if __name__ == "__main__":
    train()
