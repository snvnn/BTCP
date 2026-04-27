"""Virtual portfolio for paper trading."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Fill:
    """A simulated trade execution."""

    side: str
    price: float
    quantity: float
    fee: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class VirtualPortfolio:
    """Long-only BTC/USDT virtual portfolio."""

    initial_cash: float
    fee_rate: float = 0.001
    cash: float = field(init=False)
    btc: float = 0.0
    cost_basis: float = 0.0
    realized_pnl: float = 0.0
    trades: list[Fill] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")
        if self.fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        self.cash = float(self.initial_cash)

    @property
    def average_entry_price(self) -> float:
        if self.btc == 0:
            return 0.0
        return self.cost_basis / self.btc

    def buy(self, price: float, cash_amount: float) -> Fill:
        self._validate_positive_price(price)
        if cash_amount <= 0:
            raise ValueError("cash_amount must be positive")
        if cash_amount > self.cash:
            raise ValueError("cash_amount exceeds available cash")

        fee = cash_amount * self.fee_rate
        net_cash = cash_amount - fee
        quantity = net_cash / price
        self.cash -= cash_amount
        self.btc += quantity
        # Cost basis includes buy-side fees so realized/unrealized PnL matches equity.
        self.cost_basis += cash_amount
        fill = Fill(side="buy", price=float(price), quantity=quantity, fee=fee)
        self.trades.append(fill)
        return fill

    def sell(self, price: float, quantity: float) -> Fill:
        self._validate_positive_price(price)
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.btc:
            raise ValueError("quantity exceeds available BTC")

        proceeds = price * quantity
        fee = proceeds * self.fee_rate
        average_cost = self.average_entry_price
        self.cash += proceeds - fee
        self.btc -= quantity
        self.cost_basis -= average_cost * quantity
        if abs(self.btc) < 1e-12:
            self.btc = 0.0
            self.cost_basis = 0.0
        self.realized_pnl += (price - average_cost) * quantity - fee
        fill = Fill(side="sell", price=float(price), quantity=quantity, fee=fee)
        self.trades.append(fill)
        return fill

    def equity(self, current_price: float) -> float:
        self._validate_positive_price(current_price)
        return self.cash + self.btc * current_price

    def status(self, current_price: float) -> dict[str, float]:
        current_equity = self.equity(current_price)
        unrealized_pnl = (current_price - self.average_entry_price) * self.btc
        return {
            "initial_cash": float(self.initial_cash),
            "cash": float(self.cash),
            "btc": float(self.btc),
            "average_entry_price": float(self.average_entry_price),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(unrealized_pnl),
            "equity": float(current_equity),
            "total_return": float((current_equity - self.initial_cash) / self.initial_cash)
            if self.initial_cash
            else 0.0,
        }

    @staticmethod
    def _validate_positive_price(price: float) -> None:
        if price <= 0:
            raise ValueError("price must be positive")
