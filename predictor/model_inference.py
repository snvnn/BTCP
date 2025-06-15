import torch
import torch.nn as nn
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'lstm_model.pth')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict(model, input_data):
    x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(x)
    return pred.item()

