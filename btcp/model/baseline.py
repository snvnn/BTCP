"""Baseline predictors for comparing model forecasts."""

from __future__ import annotations

from typing import Iterable, Sequence


def _prices_list(prices: Iterable[float]) -> list[float]:
    values = [float(price) for price in prices]
    if not values:
        raise ValueError("prices must not be empty")
    return values


def _horizon_dict(value: float, pred_offsets: Sequence[int]) -> dict[str, float]:
    return {f"{offset}m": float(value) for offset in pred_offsets}


def last_price_baseline(prices: Iterable[float], pred_offsets: Sequence[int]) -> dict[str, float]:
    """Predict every horizon as the latest observed price."""
    values = _prices_list(prices)
    return _horizon_dict(values[-1], pred_offsets)


def moving_average_baseline(
    prices: Iterable[float], pred_offsets: Sequence[int], window: int = 10
) -> dict[str, float]:
    """Predict every horizon as the recent moving average."""
    if window <= 0:
        raise ValueError("window must be positive")

    values = _prices_list(prices)
    window_values = values[-window:]
    average = sum(window_values) / len(window_values)
    return _horizon_dict(average, pred_offsets)
