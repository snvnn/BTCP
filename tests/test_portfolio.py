import pytest

from btcp.trading.portfolio import VirtualPortfolio


def test_portfolio_buys_btc_with_fee():
    portfolio = VirtualPortfolio(initial_cash=1000.0, fee_rate=0.001)

    fill = portfolio.buy(price=100.0, cash_amount=500.0)

    assert fill.side == "buy"
    assert fill.price == 100.0
    assert fill.quantity == pytest.approx(4.995)
    assert fill.fee == pytest.approx(0.5)
    assert portfolio.cash == pytest.approx(500.0)
    assert portfolio.btc == pytest.approx(4.995)


def test_portfolio_sells_btc_with_fee_and_realized_pnl():
    portfolio = VirtualPortfolio(initial_cash=1000.0, fee_rate=0.001)
    portfolio.buy(price=100.0, cash_amount=500.0)

    fill = portfolio.sell(price=120.0, quantity=2.0)

    assert fill.side == "sell"
    assert fill.quantity == 2.0
    assert fill.fee == pytest.approx(0.24)
    assert portfolio.cash == pytest.approx(739.76)
    assert portfolio.btc == pytest.approx(2.995)
    assert portfolio.realized_pnl == pytest.approx(39.5598, rel=1e-4)


def test_portfolio_round_trip_pnl_matches_equity_loss_with_fees():
    portfolio = VirtualPortfolio(initial_cash=1000.0, fee_rate=0.001)
    portfolio.buy(price=100.0, cash_amount=500.0)
    portfolio.sell(price=100.0, quantity=portfolio.btc)

    status = portfolio.status(current_price=100.0)

    assert portfolio.realized_pnl == pytest.approx(status["equity"] - portfolio.initial_cash)
    assert portfolio.realized_pnl < 0


def test_portfolio_rejects_overspend_and_oversell():
    portfolio = VirtualPortfolio(initial_cash=100.0)

    with pytest.raises(ValueError, match="cash_amount exceeds available cash"):
        portfolio.buy(price=10.0, cash_amount=101.0)

    with pytest.raises(ValueError, match="quantity exceeds available BTC"):
        portfolio.sell(price=10.0, quantity=1.0)


def test_portfolio_status_includes_equity_and_unrealized_pnl():
    portfolio = VirtualPortfolio(initial_cash=1000.0, fee_rate=0.0)
    portfolio.buy(price=100.0, cash_amount=500.0)

    status = portfolio.status(current_price=120.0)

    assert status["cash"] == pytest.approx(500.0)
    assert status["btc"] == pytest.approx(5.0)
    assert status["equity"] == pytest.approx(1100.0)
    assert status["unrealized_pnl"] == pytest.approx(100.0)
