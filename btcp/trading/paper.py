"""In-memory paper trading engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from btcp.trading.portfolio import VirtualPortfolio
from btcp.trading.strategy import threshold_signal


@dataclass
class PaperTradingEngine:
    """Long-only in-memory paper trading engine."""

    initial_cash: float = 10_000.0
    threshold: float = 0.01
    trade_fraction: float = 0.5
    fee_rate: float = 0.001
    running: bool = False
    portfolio: VirtualPortfolio = field(init=False)

    def __post_init__(self) -> None:
        if not 0 < self.trade_fraction <= 1:
            raise ValueError("trade_fraction must be in (0, 1]")
        self.portfolio = VirtualPortfolio(initial_cash=self.initial_cash, fee_rate=self.fee_rate)

    @property
    def trades(self):
        return self.portfolio.trades

    def start(self) -> dict:
        self.running = True
        return {"status": "running"}

    def stop(self) -> dict:
        self.running = False
        return {"status": "stopped"}

    def status(self, current_price: float | None = None) -> dict:
        if current_price is None:
            current_price = self.portfolio.average_entry_price or 1.0
        return {
            "running": self.running,
            "portfolio": self.portfolio.status(current_price),
            "trade_count": len(self.trades),
        }

    def on_tick(self, current_price: float, predicted_price: float) -> dict:
        """Process one price/prediction tick and optionally simulate a trade."""
        if not self.running:
            return {"status": "stopped"}

        signal = threshold_signal(current_price, predicted_price, self.threshold)
        fill = None
        if signal == "buy" and self.portfolio.cash > 0:
            fill = self.portfolio.buy(
                price=current_price,
                cash_amount=self.portfolio.cash * self.trade_fraction,
            )
        elif signal == "sell" and self.portfolio.btc > 0:
            fill = self.portfolio.sell(
                price=current_price,
                quantity=self.portfolio.btc * self.trade_fraction,
            )

        return {
            "status": "running",
            "signal": signal,
            "fill": fill.__dict__ if fill else None,
            "portfolio": self.portfolio.status(current_price),
        }
