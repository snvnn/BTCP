# predictor/model_trainer.py
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, window_size=20):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        label = data[i+window_size]
        sequences.append((seq, label))
    return sequences

def train():
    df = pd.read_csv("btc_prices.csv")
    prices = df["close"].values
    prices = (prices - prices.mean()) / prices.std()

    sequences = create_sequences(prices, window_size=20)
    X = np.array([s[0] for s in sequences])
    y = np.array([s[1] for s in sequences])

    X_tensor = torch.tensor(X).float().unsqueeze(-1)  # (N, 20, 1)
    y_tensor = torch.tensor(y).float().unsqueeze(-1)  # (N, 1)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"[{datetime.now()}] 🚀 Start training...")

    for epoch in range(10):
        model.train()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[{datetime.now()}] Epoch {epoch+1}/10 - Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "predictor/model.pt")
    print(f"[{datetime.now()}] ✅ Model saved to predictor/model.pt")

if __name__ == "__main__":
    train()
