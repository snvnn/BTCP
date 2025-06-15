# predictor/model_trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time

# 하이퍼파라미터 설정
SEQ_LEN = 30
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.001

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lstm_model.pth')


# 학습용 데이터셋 정의
class PriceDataset(Dataset):
    def __init__(self, prices):
        self.x = []
        self.y = []
        for i in range(len(prices) - SEQ_LEN):
            self.x.append(prices[i:i+SEQ_LEN])
            self.y.append(prices[i+SEQ_LEN])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# 학습 함수 정의
def train(prices):
    print(f"🧪 학습 시작: 총 {EPOCHS} epochs, 데이터 길이: {len(prices)}")

    dataset = PriceDataset(prices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.unsqueeze(-1)  # (batch, seq_len, 1)
            output = model(x_batch)
            loss = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # 진행률 출력
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            sample_pred = output[0].item()
            sample_target = y_batch[0].item()
            print(f"[Epoch {epoch}/{EPOCHS}] Loss: {avg_loss:.6f} | 예측: {sample_pred:.2f}, 실제: {sample_target:.2f}")

    end_time = time.time()
    elapsed = end_time - start_time

    # 모델 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n✅ 학습 완료! 모델 저장됨: {MODEL_PATH}")
    print(f"⏱️ 총 학습 시간: {elapsed:.2f}초")
