import pytest

from btcp.model.metrics import directional_accuracy, mae, mape, rmse


def test_mae_calculates_mean_absolute_error():
    assert mae([100, 110], [90, 115]) == pytest.approx(7.5)


def test_rmse_calculates_root_mean_squared_error():
    assert rmse([100, 110], [90, 115]) == pytest.approx((125 / 2) ** 0.5)


def test_mape_calculates_percentage_error():
    assert mape([100, 200], [90, 220]) == pytest.approx(10.0)


def test_directional_accuracy_compares_actual_and_predicted_direction():
    assert directional_accuracy(current=[100, 100], actual=[110, 90], predicted=[105, 95]) == 1.0
    assert directional_accuracy(current=[100], actual=[110], predicted=[95]) == 0.0


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        mae([1, 2], [1])
