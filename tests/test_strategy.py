from btcp.trading.strategy import threshold_signal


def test_threshold_signal_buys_when_expected_return_above_threshold():
    assert threshold_signal(current_price=100.0, predicted_price=102.0, threshold=0.01) == "buy"


def test_threshold_signal_sells_when_expected_return_below_negative_threshold():
    assert threshold_signal(current_price=100.0, predicted_price=98.0, threshold=0.01) == "sell"


def test_threshold_signal_holds_inside_threshold_band():
    assert threshold_signal(current_price=100.0, predicted_price=100.5, threshold=0.01) == "hold"
