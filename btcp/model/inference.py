"""Utilities for loading the trained model and running inference."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from btcp.config import settings

MODEL_PATH = settings.model_path
SEQ_LENGTH = settings.seq_length
PRED_OFFSETS = [5, 15, 30, 60]


@dataclass(frozen=True)
class InferenceArtifacts:
    """Complete artifact bundle required for consistent inference."""

    model: object
    scaler: object
    metadata: dict


def load_trained_model(path: Path = MODEL_PATH):
    """Load and return the Keras model without importing TensorFlow at module import time."""
    from tensorflow.keras.models import load_model as keras_load_model

    return keras_load_model(path)


def load_inference_artifacts(
    artifact_dir: Path = settings.model_dir,
    model_loader: Callable[[Path], object] = load_trained_model,
) -> InferenceArtifacts:
    """Load model, scaler, and metadata from an artifact directory."""
    model_path = artifact_dir / "model.keras"
    scaler_path = artifact_dir / "scaler.joblib"
    metadata_path = artifact_dir / "metadata.json"

    model = model_loader(model_path)
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return InferenceArtifacts(model=model, scaler=scaler, metadata=metadata)


def _reshape_prices(prices: Iterable[float], seq_length: int) -> np.ndarray:
    price_list = list(prices)
    if len(price_list) != seq_length:
        raise ValueError(f"Expected {seq_length} prices, got {len(price_list)}")
    return np.array(price_list, dtype=float).reshape(-1, 1)


def predict_with_scaler(
    model,
    scaler,
    prices: Iterable[float],
    pred_offsets: Sequence[int] = PRED_OFFSETS,
    seq_length: int = SEQ_LENGTH,
) -> dict[str, float]:
    """Scale input prices, run model prediction, and inverse-transform horizon outputs."""
    price_array = _reshape_prices(prices, seq_length)
    scaled_input = scaler.transform(price_array).reshape(1, seq_length, 1)
    pred_scaled = model.predict(scaled_input)[0]

    if len(pred_scaled) != len(pred_offsets):
        raise ValueError(
            f"Expected {len(pred_offsets)} predictions, got {len(pred_scaled)}"
        )

    pred_prices = scaler.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).flatten()

    return {
        f"{offset}m": float(pred_prices[index])
        for index, offset in enumerate(pred_offsets)
    }


def predict(model, input_data: Iterable[float]):
    """Return the raw model prediction for a preprocessed sequence of closing prices."""
    arr = np.array(list(input_data), dtype=float).reshape(1, SEQ_LENGTH, 1)
    pred = model.predict(arr)
    return pred[0]
