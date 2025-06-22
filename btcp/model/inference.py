"""Utilities for loading the trained model and running inference."""

from pathlib import Path
from typing import Iterable

import numpy as np
from tensorflow.keras.models import load_model as keras_load_model

# 모델 경로는 패키지 루트 기준
MODEL_PATH = Path(__file__).resolve().parent / "lstm_model.h5"
SEQ_LENGTH = 60


def load_trained_model(path: Path = MODEL_PATH):
    """Load and return the Keras model."""
    return keras_load_model(path)


def predict(model, input_data: Iterable[float]):
    """Return the model prediction for a sequence of closing prices."""
    arr = np.array(list(input_data), dtype=float).reshape(1, SEQ_LENGTH, 1)
    pred = model.predict(arr)
    return pred[0]
