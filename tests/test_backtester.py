from btcp.trading.backtester import backtest_threshold_strategy


def test_backtester_runs_threshold_strategy_and_reports_metrics():
    prices = [100.0, 101.0, 103.0, 99.0]
    predictions = [102.0, 104.0, 98.0, 98.0]

    result = backtest_threshold_strategy(
        prices=prices,
        predictions=predictions,
        initial_cash=1000.0,
        threshold=0.01,
        trade_fraction=0.5,
        fee_rate=0.0,
    )

    assert result["initial_cash"] == 1000.0
    assert result["final_equity"] > 0
    assert result["trade_count"] >= 1
    assert len(result["equity_curve"]) == len(prices)
    assert "total_return" in result
    assert "max_drawdown" in result
