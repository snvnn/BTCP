"""Evaluation metrics for price predictions."""

from __future__ import annotations

from math import sqrt
from typing import Iterable


def _paired(actual: Iterable[float], predicted: Iterable[float]) -> tuple[list[float], list[float]]:
    actual_values = [float(value) for value in actual]
    predicted_values = [float(value) for value in predicted]
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have the same length")
    if not actual_values:
        raise ValueError("values must not be empty")
    return actual_values, predicted_values


def mae(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean absolute error."""
    actual_values, predicted_values = _paired(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual_values, predicted_values)) / len(actual_values)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Root mean squared error."""
    actual_values, predicted_values = _paired(actual, predicted)
    return sqrt(sum((a - p) ** 2 for a, p in zip(actual_values, predicted_values)) / len(actual_values))


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Mean absolute percentage error, expressed as percent."""
    actual_values, predicted_values = _paired(actual, predicted)
    if any(a == 0 for a in actual_values):
        raise ValueError("actual values must be non-zero for MAPE")
    return (
        sum(abs((a - p) / a) for a, p in zip(actual_values, predicted_values))
        / len(actual_values)
        * 100
    )


def directional_accuracy(
    current: Iterable[float], actual: Iterable[float], predicted: Iterable[float]
) -> float:
    """Fraction of predictions whose direction matches actual direction."""
    current_values, actual_values = _paired(current, actual)
    _, predicted_values = _paired(current_values, predicted)
    matches = 0
    for current_value, actual_value, predicted_value in zip(
        current_values, actual_values, predicted_values
    ):
        actual_direction = actual_value >= current_value
        predicted_direction = predicted_value >= current_value
        matches += int(actual_direction == predicted_direction)
    return matches / len(current_values)
