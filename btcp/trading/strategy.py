"""Trading signal helpers."""

from __future__ import annotations

Signal = str


def threshold_signal(current_price: float, predicted_price: float, threshold: float = 0.01) -> Signal:
    """Return buy/sell/hold based on predicted return threshold."""
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    expected_return = (predicted_price - current_price) / current_price
    if expected_return > threshold:
        return "buy"
    if expected_return < -threshold:
        return "sell"
    return "hold"
