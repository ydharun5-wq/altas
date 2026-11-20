"""Immutable domain models for the local, paper-only execution ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, DecimalException, MAX_EMAX, MIN_EMIN, localcontext
from enum import Enum

from atlas.derivatives.models import DerivativeContract
from atlas.derivatives.validation import validate_positive_integer
from atlas.market_data.models import (
    FuturesQuoteResponse,
    OptionsQuoteResponse,
    QuoteFreshnessPolicy,
)
from atlas.risk.models import ProposedTrade, RiskDecision, RiskPolicy


class PaperOrderStatus(str, Enum):
    PENDING_APPROVAL = "pending_approval"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PaperOrderAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    STOP_LOSS = "stop_loss"
    TARGET = "target"


class PaperFillSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PaperPositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is PaperPositionSide.LONG else -1


class PaperFeeBasis(str, Enum):
    PER_FILL = "per_fill"
    NOTIONAL_RATE = "notional_rate"
    PRIOR_CHARGES_RATE = "prior_charges_rate"


def validate_aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    try:
        offset = value.utcoffset()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have valid timezone information") from exc
    if offset is None:
        raise ValueError(f"{name} must be timezone-aware")


def validate_finite_decimal(name: str, value: Decimal, *, allow_zero: bool = True) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if value < 0 or (not allow_zero and value == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def _identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be a non-empty identifier without whitespace")


@dataclass(frozen=True)
class PaperFeeComponent:
    """Explicit Decimal fee rule; no exchange or broker charge is implied.

    ``amount`` is account currency for PER_FILL and a fraction for either rate
    basis. Rate amounts are applied exactly as supplied; callers choose units.
    """

    name: str
    basis: PaperFeeBasis
    amount: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("fee component name must be non-empty")
        if not isinstance(self.basis, PaperFeeBasis):
            raise ValueError("basis must be a PaperFeeBasis")
        validate_finite_decimal("fee component amount", self.amount)


@dataclass(frozen=True)
class PaperExecutionConfig:
    """Explicit fill-cost assumptions; bid/ask spread comes from each quote.

    ``slippage_bps`` is an explicit basis-point rate. The bid/ask spread comes
    from each supplied quote. Every zero must be chosen by the caller. No
    default freshness threshold, brokerage, fee, spread, or slippage assumption
    is implied.
    """

    slippage_bps: Decimal
    fee_components: tuple[PaperFeeComponent, ...]
    quote_freshness_policy: QuoteFreshnessPolicy | None

    def __post_init__(self) -> None:
        validate_finite_decimal("slippage_bps", self.slippage_bps)
        if self.slippage_bps >= Decimal("10000"):
            raise ValueError("slippage_bps must be below 10000")
        if not isinstance(self.fee_components, tuple) or any(
            not isinstance(item, PaperFeeComponent) for item in self.fee_components
        ):
            raise ValueError("fee_components must be a tuple of PaperFeeComponent values")
        names = [item.name for item in self.fee_components]
        if len(names) != len(set(names)):
            raise ValueError("fee component names must be unique")
        if self.quote_freshness_policy is not None and not isinstance(
            self.quote_freshness_policy, QuoteFreshnessPolicy
        ):
            raise ValueError("quote_freshness_policy must be QuoteFreshnessPolicy or None")


@dataclass(frozen=True)
class PaperAccountConfig:
    """Explicit initial cash, risk policy, execution costs, and session instant."""

    initial_cash: Decimal
    risk_policy: RiskPolicy
    execution: PaperExecutionConfig
    started_at: datetime

    def __post_init__(self) -> None:
        validate_finite_decimal("initial_cash", self.initial_cash, allow_zero=False)
        if not isinstance(self.risk_policy, RiskPolicy):
            raise ValueError("risk_policy must be a RiskPolicy")
        if not isinstance(self.execution, PaperExecutionConfig):
            raise ValueError("execution must be a PaperExecutionConfig")
        validate_aware_datetime("started_at", self.started_at)


@dataclass(frozen=True)
class PaperQuoteObservation:
    """Explicit quote event used for a paper fill; no data is fetched here.

    ``approved`` records caller approval for a non-synthetic quote. Synthetic
    data remains separately identified and does not become real data through
    this flag.
    """

    response: FuturesQuoteResponse | OptionsQuoteResponse
    data_source: str
    as_of: datetime
    approved: bool

    def __post_init__(self) -> None:
        if not isinstance(self.response, (FuturesQuoteResponse, OptionsQuoteResponse)):
            raise ValueError("response must be a futures or options quote response")
        if not isinstance(self.data_source, str) or not self.data_source.strip():
            raise ValueError("data_source must be a non-empty string")
        validate_aware_datetime("as_of", self.as_of)
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be a bool")

    @property
    def is_synthetic(self) -> bool:
        return self.response.is_synthetic

    @property
    def timestamp(self) -> datetime:
        return self.response.timestamp


@dataclass(frozen=True)
class PaperOpenOrderRequest:
    client_order_id: str
    proposal: ProposedTrade
    submitted_at: datetime
    quote_approved: bool

    def __post_init__(self) -> None:
        _identifier("client_order_id", self.client_order_id)
        if not isinstance(self.proposal, ProposedTrade):
            raise ValueError("proposal must be a ProposedTrade")
        if self.proposal.quantity is None:
            raise ValueError("opening order proposal must have a sized quantity")
        validate_aware_datetime("submitted_at", self.submitted_at)
        if not isinstance(self.quote_approved, bool):
            raise ValueError("quote_approved must be a bool")


@dataclass(frozen=True)
class PaperCloseOrderRequest:
    client_order_id: str
    position_id: str
    quantity: int
    quote: PaperQuoteObservation
    submitted_at: datetime

    def __post_init__(self) -> None:
        _identifier("client_order_id", self.client_order_id)
        _identifier("position_id", self.position_id)
        validate_positive_integer("quantity", self.quantity)
        if not isinstance(self.quote, PaperQuoteObservation):
            raise ValueError("quote must be a PaperQuoteObservation")
        validate_aware_datetime("submitted_at", self.submitted_at)


@dataclass(frozen=True)
class PaperOrder:
    client_order_id: str
    action: PaperOrderAction
    contract: DerivativeContract
    direction: PaperPositionSide
    requested_quantity: int
    filled_quantity: int
    status: PaperOrderStatus
    submitted_at: datetime
    updated_at: datetime
    data_source: str | None
    is_synthetic: bool | None
    quote_approved: bool | None
    proposal: ProposedTrade | None
    position_id: str
    risk_decision: RiskDecision | None
    risk_per_contract: Decimal | None
    exposure_per_contract: Decimal | None
    reserved_risk: Decimal
    reserved_exposure: Decimal
    reserved_cash: Decimal
    limit_price: Decimal | None
    rejection_reason: str | None

    def __post_init__(self) -> None:
        _identifier("client_order_id", self.client_order_id)
        _identifier("position_id", self.position_id)
        if not isinstance(self.action, PaperOrderAction):
            raise ValueError("action must be a PaperOrderAction")
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        if not isinstance(self.direction, PaperPositionSide):
            raise ValueError("direction must be a PaperPositionSide")
        validate_positive_integer("requested_quantity", self.requested_quantity)
        if isinstance(self.filled_quantity, bool) or not isinstance(self.filled_quantity, int):
            raise ValueError("filled_quantity must be an integer")
        if not 0 <= self.filled_quantity <= self.requested_quantity:
            raise ValueError("filled_quantity must be between zero and requested_quantity")
        if not isinstance(self.status, PaperOrderStatus):
            raise ValueError("status must be a PaperOrderStatus")
        validate_aware_datetime("submitted_at", self.submitted_at)
        validate_aware_datetime("updated_at", self.updated_at)
        if self.updated_at < self.submitted_at:
            raise ValueError("updated_at must not precede submitted_at")
        if self.data_source is not None and (
            not isinstance(self.data_source, str) or not self.data_source.strip()
        ):
            raise ValueError("data_source must be a non-empty string or None")
        if self.is_synthetic is not None and not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool or None")
        if self.quote_approved is not None and not isinstance(self.quote_approved, bool):
            raise ValueError("quote_approved must be a bool or None")
        if self.proposal is not None and not isinstance(self.proposal, ProposedTrade):
            raise ValueError("proposal must be a ProposedTrade or None")
        if self.risk_decision is not None and not isinstance(
            self.risk_decision, RiskDecision
        ):
            raise ValueError("risk_decision must be a RiskDecision or None")
        for name in ("risk_per_contract", "exposure_per_contract"):
            value = getattr(self, name)
            if value is not None:
                validate_finite_decimal(name, value)
        validate_finite_decimal("reserved_risk", self.reserved_risk)
        validate_finite_decimal("reserved_exposure", self.reserved_exposure)
        validate_finite_decimal("reserved_cash", self.reserved_cash)
        if self.limit_price is not None:
            validate_finite_decimal("limit_price", self.limit_price, allow_zero=False)
        if self.rejection_reason is not None and (
            not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip()
        ):
            raise ValueError("rejection_reason must be non-empty or None")
        if self.action is PaperOrderAction.OPEN and self.proposal is None:
            raise ValueError("opening orders require their exact proposal")
        if self.action is not PaperOrderAction.OPEN and self.proposal is not None:
            raise ValueError("closing orders must not contain a new-trade proposal")
        if self.status in (
            PaperOrderStatus.PENDING_APPROVAL,
            PaperOrderStatus.REJECTED,
            PaperOrderStatus.CANCELLED,
            PaperOrderStatus.FILLED,
        ) and (
            self.reserved_risk != 0
            or self.reserved_exposure != 0
            or self.reserved_cash != 0
        ):
            raise ValueError("terminal or pending orders must not reserve resources")

    @property
    def remaining_quantity(self) -> int:
        return self.requested_quantity - self.filled_quantity


@dataclass(frozen=True)
class PaperFeeCharge:
    component: str
    amount: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("component must be non-empty")
        validate_finite_decimal("fee amount", self.amount)


@dataclass(frozen=True)
class PaperFill:
    """Applied fill; reference is the executable bid/ask and price includes slippage."""

    fill_id: str
    client_order_id: str
    position_id: str
    contract: DerivativeContract
    action: PaperOrderAction
    side: PaperFillSide
    quantity: int
    timestamp: datetime
    reference_price: Decimal
    fill_price: Decimal
    notional: Decimal
    fee_charges: tuple[PaperFeeCharge, ...]
    entry_fees_allocated: Decimal
    gross_realized_pnl: Decimal
    net_realized_pnl: Decimal
    data_source: str
    is_synthetic: bool
    quote_approved: bool

    def __post_init__(self) -> None:
        _identifier("fill_id", self.fill_id)
        _identifier("client_order_id", self.client_order_id)
        _identifier("position_id", self.position_id)
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        if not isinstance(self.action, PaperOrderAction):
            raise ValueError("action must be a PaperOrderAction")
        if not isinstance(self.side, PaperFillSide):
            raise ValueError("side must be a PaperFillSide")
        validate_positive_integer("quantity", self.quantity)
        validate_aware_datetime("timestamp", self.timestamp)
        for name in ("reference_price", "fill_price"):
            validate_finite_decimal(name, getattr(self, name), allow_zero=False)
        for name in ("notional", "entry_fees_allocated"):
            validate_finite_decimal(name, getattr(self, name))
        for name in ("gross_realized_pnl", "net_realized_pnl"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
        if not isinstance(self.fee_charges, tuple) or any(
            not isinstance(item, PaperFeeCharge) for item in self.fee_charges
        ):
            raise ValueError("fee_charges must be a tuple of PaperFeeCharge values")
        if not isinstance(self.data_source, str) or not self.data_source.strip():
            raise ValueError("data_source must be non-empty")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool")
        if not isinstance(self.quote_approved, bool):
            raise ValueError("quote_approved must be a bool")

    @property
    def fees(self) -> Decimal:
        values = tuple(item.amount for item in self.fee_charges)
        if not values:
            return Decimal(0)
        least_exponent = min(value.as_tuple().exponent for value in values)
        greatest_adjusted = max(value.adjusted() for value in values)
        precision = max(80, greatest_adjusted - least_exponent + len(values) + 2)
        if precision > 10000:
            raise ValueError("fee total exceeds safe Decimal precision")
        try:
            with localcontext() as context:
                context.prec = precision
                context.Emax = MAX_EMAX
                context.Emin = MIN_EMIN
                result = sum(values, Decimal(0))
        except DecimalException as exc:
            raise ValueError("fee total exceeds finite Decimal range") from exc
        if not result.is_finite():
            raise ValueError("fee total exceeds finite Decimal range")
        return result


@dataclass(frozen=True)
class PaperPosition:
    position_id: str
    contract: DerivativeContract
    direction: PaperPositionSide
    quantity: int
    average_entry_price: Decimal
    point_value: Decimal | None
    stop_price: Decimal | None
    target_price: Decimal | None
    opened_at: datetime
    updated_at: datetime
    mark_price: Decimal | None
    mark_timestamp: datetime | None
    remaining_entry_fees: Decimal
    realized_gross_pnl: Decimal
    data_sources: tuple[str, ...]
    is_synthetic: bool
    all_real_quotes_approved: bool

    def __post_init__(self) -> None:
        _identifier("position_id", self.position_id)
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        if not isinstance(self.direction, PaperPositionSide):
            raise ValueError("direction must be a PaperPositionSide")
        validate_positive_integer("quantity", self.quantity)
        for name in ("average_entry_price",):
            validate_finite_decimal(name, getattr(self, name), allow_zero=False)
        for name in ("point_value", "stop_price", "target_price", "mark_price"):
            value = getattr(self, name)
            if value is not None:
                validate_finite_decimal(name, value, allow_zero=False)
        validate_finite_decimal("remaining_entry_fees", self.remaining_entry_fees)
        if not isinstance(self.realized_gross_pnl, Decimal) or not self.realized_gross_pnl.is_finite():
            raise ValueError("realized_gross_pnl must be finite")
        validate_aware_datetime("opened_at", self.opened_at)
        validate_aware_datetime("updated_at", self.updated_at)
        if self.mark_timestamp is not None:
            validate_aware_datetime("mark_timestamp", self.mark_timestamp)
        if self.updated_at < self.opened_at:
            raise ValueError("updated_at must not precede opened_at")
        if self.mark_timestamp is not None and self.mark_timestamp > self.updated_at:
            raise ValueError("mark_timestamp must not follow updated_at")
        if not isinstance(self.data_sources, tuple) or not self.data_sources or any(
            not isinstance(item, str) or not item.strip() for item in self.data_sources
        ):
            raise ValueError("data_sources must be a non-empty tuple of source identities")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool")
        if not isinstance(self.all_real_quotes_approved, bool):
            raise ValueError("all_real_quotes_approved must be a bool")


@dataclass(frozen=True)
class PaperTradeHistory:
    """One explicit close event; quote_approved means each real quote used was approved."""

    trade_id: int
    position_id: str
    contract: DerivativeContract
    direction: PaperPositionSide
    quantity_closed: int
    opened_at: datetime
    closed_at: datetime
    average_entry_price: Decimal
    average_exit_price: Decimal
    gross_pnl: Decimal
    entry_fees: Decimal
    exit_fees: Decimal
    net_pnl: Decimal
    exit_reason: PaperOrderAction
    data_sources: tuple[str, ...]
    is_synthetic: bool
    quote_approved: bool
    position_fully_closed: bool

    def __post_init__(self) -> None:
        if isinstance(self.trade_id, bool) or not isinstance(self.trade_id, int) or self.trade_id <= 0:
            raise ValueError("trade_id must be a positive integer")
        _identifier("position_id", self.position_id)
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        if not isinstance(self.direction, PaperPositionSide):
            raise ValueError("direction must be a PaperPositionSide")
        validate_positive_integer("quantity_closed", self.quantity_closed)
        validate_aware_datetime("opened_at", self.opened_at)
        validate_aware_datetime("closed_at", self.closed_at)
        if self.closed_at < self.opened_at:
            raise ValueError("closed_at must not precede opened_at")
        for name in ("average_entry_price", "average_exit_price"):
            validate_finite_decimal(name, getattr(self, name), allow_zero=False)
        for name in ("entry_fees", "exit_fees"):
            validate_finite_decimal(name, getattr(self, name))
        for name in ("gross_pnl", "net_pnl"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
        if not isinstance(self.exit_reason, PaperOrderAction):
            raise ValueError("exit_reason must be a PaperOrderAction")
        if self.exit_reason is PaperOrderAction.OPEN:
            raise ValueError("exit_reason cannot be OPEN")
        if not isinstance(self.data_sources, tuple) or not self.data_sources:
            raise ValueError("data_sources must be a non-empty tuple")
        if any(
            not isinstance(value, bool)
            for value in (self.is_synthetic, self.quote_approved, self.position_fully_closed)
        ):
            raise ValueError("provenance and close status fields must be bools")


@dataclass(frozen=True)
class PaperPortfolioSnapshot:
    """Account view with explicit accounting conventions.

    ``realized_pnl`` is gross realized P&L less all fees paid; ``unrealized_pnl``
    is gross mark-to-market P&L. Equity is cash plus long-option market value
    plus futures unrealized P&L. Cash already includes option premium debits,
    option sale proceeds, futures realized P&L, and fees.
    """

    initial_cash: Decimal
    execution_config: PaperExecutionConfig
    cash: Decimal
    equity: Decimal
    realized_pnl: Decimal
    gross_realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_paid: Decimal
    reserved_risk: Decimal
    reserved_exposure: Decimal
    reserved_cash: Decimal
    peak_equity: Decimal
    as_of: datetime
    positions: tuple[PaperPosition, ...]
    orders: tuple[PaperOrder, ...]
    fills: tuple[PaperFill, ...]
    trade_history: tuple[PaperTradeHistory, ...]

    def __post_init__(self) -> None:
        validate_finite_decimal("initial_cash", self.initial_cash, allow_zero=False)
        if not isinstance(self.execution_config, PaperExecutionConfig):
            raise ValueError("execution_config must be a PaperExecutionConfig")
        for name in (
            "cash",
            "equity",
            "realized_pnl",
            "gross_realized_pnl",
            "unrealized_pnl",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"{name} must be a finite Decimal")
        validate_finite_decimal("fees_paid", self.fees_paid)
        validate_finite_decimal("reserved_risk", self.reserved_risk)
        validate_finite_decimal("reserved_exposure", self.reserved_exposure)
        validate_finite_decimal("reserved_cash", self.reserved_cash)
        if not isinstance(self.peak_equity, Decimal) or not self.peak_equity.is_finite():
            raise ValueError("peak_equity must be a finite Decimal")
        validate_aware_datetime("as_of", self.as_of)
        for name, expected in (
            ("positions", PaperPosition),
            ("orders", PaperOrder),
            ("fills", PaperFill),
            ("trade_history", PaperTradeHistory),
        ):
            value = getattr(self, name)
            if not isinstance(value, tuple) or any(not isinstance(item, expected) for item in value):
                raise ValueError(f"{name} must be a tuple of {expected.__name__} values")


class PaperTradingError(RuntimeError):
    """Base class for paper-ledger state and execution errors."""


class DuplicateSubmissionError(PaperTradingError):
    """An idempotency key was reused with different request terms."""


class OrderStateError(PaperTradingError):
    """An operation is not permitted in the order's current state."""


class InvalidPaperQuoteError(PaperTradingError):
    """An explicitly supplied quote is invalid for the requested fill."""


__all__ = [
    "DuplicateSubmissionError",
    "InvalidPaperQuoteError",
    "OrderStateError",
    "PaperAccountConfig",
    "PaperCloseOrderRequest",
    "PaperExecutionConfig",
    "PaperFeeBasis",
    "PaperFeeCharge",
    "PaperFeeComponent",
    "PaperFill",
    "PaperFillSide",
    "PaperOpenOrderRequest",
    "PaperOrder",
    "PaperOrderAction",
    "PaperOrderStatus",
    "PaperPosition",
    "PaperPositionSide",
    "PaperPortfolioSnapshot",
    "PaperQuoteObservation",
    "PaperTradeHistory",
    "PaperTradingError",
]
