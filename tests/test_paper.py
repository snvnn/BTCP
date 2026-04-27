from btcp.trading.paper import PaperTradingEngine

import pytest


def test_paper_trading_engine_starts_stops_and_processes_ticks():
    engine = PaperTradingEngine(initial_cash=1000.0, threshold=0.01, trade_fraction=0.5)

    assert engine.running is False
    engine.start()
    assert engine.running is True

    event = engine.on_tick(current_price=100.0, predicted_price=102.0)

    assert event["signal"] == "buy"
    assert event["portfolio"]["equity"] > 0
    assert len(engine.trades) == 1

    engine.stop()
    assert engine.running is False


def test_paper_trading_engine_rejects_invalid_trade_fraction():
    with pytest.raises(ValueError, match="trade_fraction must be in"):
        PaperTradingEngine(initial_cash=1000.0, trade_fraction=0.0)


def test_paper_trading_engine_ignores_ticks_when_stopped():
    engine = PaperTradingEngine(initial_cash=1000.0)

    event = engine.on_tick(current_price=100.0, predicted_price=102.0)

    assert event["status"] == "stopped"
    assert engine.trades == []
