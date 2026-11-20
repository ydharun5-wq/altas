"""Immutable assumptions and result types for historical backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import fsum, isfinite
from numbers import Real

from atlas.strategies.models import StrategySignal


class FeeBasis(str, Enum):
    PER_EXECUTION = "per_execution"
    NOTIONAL_RATE = "notional_rate"
    PRIOR_CHARGES_RATE = "prior_charges_rate"


class FillSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is PositionSide.LONG else -1


class TradeExitReason(str, Enum):
    SIGNAL = "signal"
    STOP_LOSS = "stop_loss"
    TARGET = "target"


class ProfitFactorStatus(str, Enum):
    DEFINED = "defined"
    NO_TRADES = "no_trades"
    NO_LOSSES = "no_losses"


@dataclass(frozen=True)
class FeeComponent:
    """Explicit fee assumption; no tax rate or local tariff is implied.

    PER_EXECUTION amount is a fixed currency charge per filled order.
    NOTIONAL_RATE amount is a decimal fraction applied to order notional.
    PRIOR_CHARGES_RATE is applied to the sum of earlier components in the
    supplied tuple, allowing explicit taxes on an explicitly supplied fee base.
    """

    name: str
    basis: FeeBasis
    amount: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("fee component name must be non-empty")
        if not isinstance(self.basis, FeeBasis):
            raise ValueError("fee component basis must be a FeeBasis")
        _nonnegative_finite("fee component amount", self.amount)


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Caller-specified fills and costs for each simulated execution.

    The full spread is in basis points and half is charged adversely on each
    side. Slippage is an additional adverse basis-point adjustment. Fee
    components are independently explicit and charged on each filled order.
    """

    spread_bps: int | float
    slippage_bps: int | float
    fee_components: tuple[FeeComponent, ...]

    def __post_init__(self) -> None:
        _nonnegative_finite("spread_bps", self.spread_bps)
        _nonnegative_finite("slippage_bps", self.slippage_bps)
        spread = _finite_float(self.spread_bps)
        slippage = _finite_float(self.slippage_bps)
        if spread / 2.0 + slippage >= 10000.0:
            raise ValueError("spread and slippage assumptions make sell fills non-positive")
        if not isinstance(self.fee_components, tuple):
            raise ValueError("fee_components must be a tuple")
        if any(not isinstance(item, FeeComponent) for item in self.fee_components):
            raise ValueError("fee_components must contain FeeComponent values")
        names = [item.name for item in self.fee_components]
        if len(names) != len(set(names)):
            raise ValueError("fee component names must be unique")

    @classmethod
    def zero_cost_research_mode(cls) -> ExecutionAssumptions:
        """Return an explicitly selected zero-spread, zero-slippage, zero-fee mode."""
        return cls(spread_bps=0.0, slippage_bps=0.0, fee_components=())

    def fill_price(self, reference_price: int | float, side: FillSide) -> float:
        reference = _positive_finite("reference price", reference_price)
        if not isinstance(side, FillSide):
            raise ValueError("side must be a FillSide")
        adverse_bps = _finite_float(self.spread_bps) / 2.0 + _finite_float(
            self.slippage_bps
        )
        adjustment = reference * adverse_bps / 10000.0
        fill = reference + adjustment if side is FillSide.BUY else reference - adjustment
        if not isfinite(fill) or fill <= 0.0:
            raise ValueError("execution assumptions produce a non-positive fill")
        return fill

    def fee_charges(self, notional: int | float) -> tuple[FeeCharge, ...]:
        value = _nonnegative_finite("order notional", notional)
        charges = []
        prior_charges = 0.0
        for component in self.fee_components:
            amount = _finite_float(component.amount)
            if component.basis is FeeBasis.PER_EXECUTION:
                cost = amount
            elif component.basis is FeeBasis.NOTIONAL_RATE:
                cost = value * amount
            else:
                cost = prior_charges * amount
            if not isfinite(cost) or cost < 0.0:
                raise ValueError("fee assumptions produce an invalid charge")
            charges.append(FeeCharge(component=component.name, amount=cost))
            prior_charges = _finite_sum("fee charges", prior_charges, cost)
        return tuple(charges)


@dataclass(frozen=True)
class FeeCharge:
    component: str
    amount: float

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("fee charge component must be non-empty")
        _nonnegative_finite("fee charge amount", self.amount)


@dataclass(frozen=True)
class BacktestConfig:
    """Explicit capital, quantity, price-point value, execution and target inputs.

    Positions are linear index-point proxies. Quantity is a count of simulated
    units, and unit_multiplier is currency P&L per one index-price point per
    unit. This model has no contract lot size, financing, or margin rules.
    """

    initial_capital: int | float
    quantity: int
    unit_multiplier: int | float
    execution: ExecutionAssumptions
    target_distance: int | float | None

    def __post_init__(self) -> None:
        _positive_finite("initial_capital", self.initial_capital)
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int):
            raise ValueError("quantity must be a positive integer")
        if self.quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        _positive_finite("unit_multiplier", self.unit_multiplier)
        if not isinstance(self.execution, ExecutionAssumptions):
            raise ValueError("execution must be an ExecutionAssumptions value")
        if self.target_distance is not None:
            _positive_finite("target_distance", self.target_distance)


@dataclass(frozen=True)
class ExecutionRecord:
    sequence: int
    timestamp: datetime
    action: str
    side: PositionSide
    quantity: int
    reference_price: float
    fill_price: float | None
    fee_charges: tuple[FeeCharge, ...]
    filled: bool
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence <= 0:
            raise ValueError("execution sequence must be a positive integer")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("execution timestamp must be a datetime")
        if not isinstance(self.action, str) or not self.action.strip():
            raise ValueError("execution action must be non-empty")
        if not isinstance(self.side, PositionSide):
            raise ValueError("execution side must be a PositionSide")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("execution quantity must be a positive integer")
        _positive_finite("reference_price", self.reference_price)
        if self.fill_price is not None:
            _positive_finite("fill_price", self.fill_price)
        if not isinstance(self.fee_charges, tuple) or any(
            not isinstance(item, FeeCharge) for item in self.fee_charges
        ):
            raise ValueError("fee_charges must contain FeeCharge values")
        if not isinstance(self.filled, bool):
            raise ValueError("filled must be a bool")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("execution reason must be non-empty")
        if self.filled != (self.fill_price is not None):
            raise ValueError("filled must match whether a fill_price is present")

    @property
    def total_fees(self) -> float:
        return sum(item.amount for item in self.fee_charges)


@dataclass(frozen=True)
class TradeRecord:
    trade_id: int
    side: PositionSide
    quantity: int
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_bar_index: int
    exit_bar_index: int
    entry_fill_price: float
    exit_fill_price: float
    gross_pnl: float
    entry_fees: float
    exit_fees: float
    net_pnl: float
    exit_reason: TradeExitReason
    entry_reason: str
    holding_bars: int
    holding_duration: timedelta


@dataclass(frozen=True)
class OpenPositionMark:
    side: PositionSide
    quantity: int
    entry_timestamp: datetime
    entry_bar_index: int
    entry_fill_price: float
    mark_timestamp: datetime
    mark_price: float
    unrealized_gross_pnl: float
    entry_fees: float
    stop_price: float | None
    target_price: float | None
    holding_bars: int
    holding_duration: timedelta


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    bar_index: int
    cash: float
    unrealized_pnl: float
    equity: float
    signed_quantity: int
    mark_price: float
    gross_exposure: float


@dataclass(frozen=True)
class BacktestMetrics:
    net_pnl: float
    total_return: float
    trade_count: int
    win_rate: float | None
    average_win: float | None
    average_loss: float | None
    expectancy: float | None
    profit_factor: float | None
    profit_factor_status: ProfitFactorStatus
    maximum_drawdown: float
    maximum_drawdown_fraction: float
    realized_trade_pnl: float
    unrealized_pnl: float
    exposure_bars: int
    exposure_fraction: float
    average_gross_exposure: float
    maximum_gross_exposure: float
    average_closed_holding_duration: timedelta | None


@dataclass(frozen=True)
class BacktestResult:
    strategy_name: str
    strategy_version: str
    symbol: str
    timeframe: timedelta
    underlying: str
    requested_start: datetime
    requested_end: datetime
    trade_start: datetime
    data_source: str
    is_synthetic: bool
    config: BacktestConfig
    initial_capital: float
    final_cash: float
    final_equity: float
    metrics: BacktestMetrics
    trades: tuple[TradeRecord, ...]
    executions: tuple[ExecutionRecord, ...]
    signals: tuple[StrategySignal, ...]
    equity_curve: tuple[EquityPoint, ...]
    open_position: OpenPositionMark | None
    notes: tuple[str, ...]


def _finite_float(value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("value must be finite")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("value must be finite") from exc
    if not isfinite(converted):
        raise ValueError("value must be finite")
    return converted


def _positive_finite(name: str, value: int | float) -> float:
    converted = _finite_float(value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _nonnegative_finite(name: str, value: int | float) -> float:
    converted = _finite_float(value)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _finite_sum(name: str, *values: float) -> float:
    try:
        result = fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} exceeds numeric range") from exc
    if not isfinite(result):
        raise ValueError(f"{name} exceeds numeric range")
    return result


__all__ = [
    "BacktestConfig",
    "BacktestMetrics",
    "BacktestResult",
    "EquityPoint",
    "ExecutionAssumptions",
    "ExecutionRecord",
    "FeeBasis",
    "FeeCharge",
    "FeeComponent",
    "FillSide",
    "OpenPositionMark",
    "PositionSide",
    "ProfitFactorStatus",
    "TradeExitReason",
    "TradeRecord",
]
