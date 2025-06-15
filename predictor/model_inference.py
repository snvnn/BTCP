import torch
import torch.nn as nn
import numpy as np

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def load_model():
    model = DummyModel()
    model.eval()
    return model

def predict(model, input_data):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        return model(input_tensor).item()
