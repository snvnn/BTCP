import pytest

from btcp.model.baseline import last_price_baseline, moving_average_baseline


def test_last_price_baseline_repeats_latest_price_for_all_horizons():
    result = last_price_baseline(prices=[100.0, 101.0, 102.0], pred_offsets=[5, 15])

    assert result == {"5m": 102.0, "15m": 102.0}


def test_moving_average_baseline_uses_recent_window_average():
    result = moving_average_baseline(
        prices=[100.0, 110.0, 120.0, 130.0], pred_offsets=[5, 15], window=2
    )

    assert result == {"5m": 125.0, "15m": 125.0}


def test_baselines_reject_empty_prices():
    with pytest.raises(ValueError, match="prices must not be empty"):
        last_price_baseline(prices=[], pred_offsets=[5])


def test_moving_average_baseline_rejects_invalid_window():
    with pytest.raises(ValueError, match="window must be positive"):
        moving_average_baseline(prices=[100.0], pred_offsets=[5], window=0)
