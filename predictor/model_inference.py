# predictor/model_inference.py
import torch
import numpy as np
import pandas as pd
from .model_trainer import LSTMModel

def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load("predictor/model.pt"))
    model.eval()
    return model

def predict_recent():
    df = pd.read_csv("btc_prices.csv")
    prices = df["close"].values
    norm_prices = (prices - prices.mean()) / prices.std()
    input_seq = norm_prices[-20:]
    input_tensor = torch.tensor(input_seq).float().unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)

    model = load_model()
    with torch.no_grad():
        output = model(input_tensor)
    pred_norm = output.item()
    pred_real = pred_norm * prices.std() + prices.mean()
    return round(pred_real, 2)
