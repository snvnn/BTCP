"""Simple backtesting utilities for simulated trading."""

from __future__ import annotations

from collections.abc import Sequence

from btcp.trading.portfolio import VirtualPortfolio
from btcp.trading.strategy import threshold_signal


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """Return max drawdown as a positive fraction."""
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    worst = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst


def backtest_threshold_strategy(
    prices: Sequence[float],
    predictions: Sequence[float],
    initial_cash: float = 10_000.0,
    threshold: float = 0.01,
    trade_fraction: float = 0.5,
    fee_rate: float = 0.001,
) -> dict:
    """Backtest a long-only threshold strategy over aligned prices/predictions."""
    if len(prices) != len(predictions):
        raise ValueError("prices and predictions must have the same length")
    if not prices:
        raise ValueError("prices must not be empty")
    if not 0 < trade_fraction <= 1:
        raise ValueError("trade_fraction must be in (0, 1]")

    portfolio = VirtualPortfolio(initial_cash=initial_cash, fee_rate=fee_rate)
    equity_curve: list[float] = []
    signals: list[str] = []

    for price, predicted in zip(prices, predictions):
        signal = threshold_signal(price, predicted, threshold)
        signals.append(signal)
        if signal == "buy" and portfolio.cash > 0:
            portfolio.buy(price=price, cash_amount=portfolio.cash * trade_fraction)
        elif signal == "sell" and portfolio.btc > 0:
            portfolio.sell(price=price, quantity=portfolio.btc * trade_fraction)
        equity_curve.append(portfolio.equity(price))

    final_equity = equity_curve[-1]
    return {
        "initial_cash": float(initial_cash),
        "final_equity": float(final_equity),
        "total_return": float((final_equity - initial_cash) / initial_cash) if initial_cash else 0.0,
        "max_drawdown": float(max_drawdown(equity_curve)),
        "trade_count": len(portfolio.trades),
        "equity_curve": [float(value) for value in equity_curve],
        "signals": signals,
        "trades": [trade.__dict__ for trade in portfolio.trades],
    }
