"""Immutable inputs and results for deterministic pre-trade risk checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from atlas.derivatives.models import DerivativeContract, InstrumentType, UnderlyingIndex
from atlas.derivatives.validation import (
    validate_decimal,
    validate_nonnegative_integer,
    validate_positive_integer,
)
from atlas.market_data.models import FuturesQuoteResponse, OptionsQuoteResponse
from atlas.strategies.models import StrategySignal


class TradingHorizon(str, Enum):
    INTRADAY = "intraday"
    SWING = "swing"


class TradeStructure(str, Enum):
    SINGLE_LEG = "single_leg"
    MULTI_LEG = "multi_leg"


class RiskDecisionStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    NO_TRADE = "no_trade"


class RiskReasonCode(str, Enum):
    RISK_CHECKS_PASSED = "risk_checks_passed"
    ACCOUNT_EQUITY_MISSING = "account_equity_missing"
    ACCOUNT_EQUITY_ZERO = "account_equity_zero"
    ACCOUNT_DAILY_BASELINE_MISSING = "account_daily_baseline_missing"
    ACCOUNT_DAILY_BASELINE_ZERO = "account_daily_baseline_zero"
    ACCOUNT_PEAK_EQUITY_MISSING = "account_peak_equity_missing"
    ACCOUNT_PEAK_EQUITY_ZERO = "account_peak_equity_zero"
    DAILY_PNL_MISSING = "daily_pnl_missing"
    OPEN_POSITION_COUNT_MISSING = "open_position_count_missing"
    EXPOSURE_DATA_MISSING = "exposure_data_missing"
    SNAPSHOT_TIMESTAMP_MISSING = "snapshot_timestamp_missing"
    INVALID_TIMESTAMP = "invalid_timestamp"
    FUTURE_ACCOUNT_SNAPSHOT = "future_account_snapshot"
    SIGNAL_TIMESTAMP_MISSING = "signal_timestamp_missing"
    MARKET_DATA_MISSING = "market_data_missing"
    MARKET_DATA_SOURCE_MISSING = "market_data_source_missing"
    MARKET_PRICE_MISSING = "market_price_missing"
    QUOTE_CONTRACT_MISMATCH = "quote_contract_mismatch"
    PROVENANCE_MISMATCH = "provenance_mismatch"
    SIGNAL_AFTER_EVALUATION = "signal_after_evaluation"
    SIGNAL_NOT_READY = "signal_not_ready"
    INCOMPLETE_CANDLE = "incomplete_candle"
    NO_TRADE_SIGNAL = "no_trade_signal"
    UNDERLYING_MISMATCH = "underlying_mismatch"
    UNDERLYING_NOT_ALLOWED = "underlying_not_allowed"
    INSTRUMENT_NOT_ALLOWED = "instrument_not_allowed"
    HORIZON_NOT_ALLOWED = "horizon_not_allowed"
    SYNTHETIC_DATA_NOT_ALLOWED = "synthetic_data_not_allowed"
    SYNTHETIC_STATUS_MISSING = "synthetic_status_missing"
    MULTI_LEG_UNSUPPORTED = "multi_leg_unsupported"
    SHORT_OPTION_UNSUPPORTED = "short_option_unsupported"
    ENTRY_REFERENCE_MISSING = "entry_reference_missing"
    STOP_REFERENCE_MISSING = "stop_reference_missing"
    QUANTITY_MISSING = "quantity_missing"
    POINT_VALUE_MISSING = "point_value_missing"
    INVALID_PRICE_REFERENCE = "invalid_price_reference"
    INVALID_STOP_PLACEMENT = "invalid_stop_placement"
    INVALID_TARGET_PLACEMENT = "invalid_target_placement"
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"
    DAILY_LOSS_LIMIT_REACHED = "daily_loss_limit_reached"
    DAILY_RISK_ALLOWANCE_EXCEEDED = "daily_risk_allowance_exceeded"
    MAXIMUM_DRAWDOWN_REACHED = "maximum_drawdown_reached"
    OPEN_POSITION_LIMIT_REACHED = "open_position_limit_reached"
    UNDERLYING_EXPOSURE_LIMIT_EXCEEDED = "underlying_exposure_limit_exceeded"
    USER_APPROVAL_REQUIRED = "user_approval_required"
    USER_REJECTED = "user_rejected"
    INVALID_APPROVAL_STATE = "invalid_approval_state"
    NUMERIC_CALCULATION_FAILED = "numeric_calculation_failed"


@dataclass(frozen=True)
class RiskPolicy:
    """Caller-configured limits; no account size or risk percentages are implied.

    Exposure caps and snapshot exposure values use the caller's account-currency
    units. Their methodology must match the estimates documented by the risk
    calculation (futures notional versus long-option premium deployed).
    Paper-trading proposals always require explicit user approval; the
    ``user_approval_required`` field is retained as an auditable policy value
    and must be ``True``.
    """

    policy_id: str
    version: str
    max_risk_per_trade_fraction: Decimal
    max_daily_loss_fraction: Decimal
    max_account_drawdown_fraction: Decimal
    max_open_positions: int
    max_exposure_per_underlying: Decimal
    allowed_underlyings: frozenset[UnderlyingIndex]
    allowed_instrument_types: frozenset[InstrumentType]
    allowed_horizons: frozenset[TradingHorizon]
    allow_synthetic_data: bool
    user_approval_required: bool

    def __post_init__(self) -> None:
        for name in ("policy_id", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "max_risk_per_trade_fraction",
            "max_daily_loss_fraction",
            "max_account_drawdown_fraction",
        ):
            value = _nonnegative_decimal(name, getattr(self, name))
            if value > 1:
                raise ValueError(f"{name} must be between 0 and 1 inclusive")
        validate_nonnegative_integer("max_open_positions", self.max_open_positions)
        _nonnegative_decimal(
            "max_exposure_per_underlying", self.max_exposure_per_underlying
        )
        _validate_frozenset(
            "allowed_underlyings", self.allowed_underlyings, UnderlyingIndex
        )
        _validate_frozenset(
            "allowed_instrument_types", self.allowed_instrument_types, InstrumentType
        )
        _validate_frozenset("allowed_horizons", self.allowed_horizons, TradingHorizon)
        if not isinstance(self.allow_synthetic_data, bool):
            raise ValueError("allow_synthetic_data must be a bool")
        if self.user_approval_required is not True:
            raise ValueError(
                "user_approval_required must be True for paper-trading proposals"
            )


@dataclass(frozen=True)
class AccountRiskSnapshot:
    """Point-in-time account inputs for risk evaluation.

    Equity is already marked to market. Daily realised and unrealised P&L are
    signed attribution values used only for the daily-loss check; they are not
    added to equity. Exposure values are non-negative gross currency amounts,
    with an explicit entry for every supported underlying. The caller must use
    the same account-currency exposure basis as its RiskPolicy.
    """

    current_equity: Decimal | None
    starting_daily_equity: Decimal | None
    peak_equity: Decimal | None
    daily_realised_pnl: Decimal | None
    daily_unrealised_pnl: Decimal | None
    open_position_count: int | None
    exposure_by_underlying: Mapping[UnderlyingIndex, Decimal] | None
    snapshot_timestamp: datetime | None

    def __post_init__(self) -> None:
        for name in ("current_equity", "starting_daily_equity", "peak_equity"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_decimal(name, value)
        for name in ("daily_realised_pnl", "daily_unrealised_pnl"):
            value = getattr(self, name)
            if value is not None:
                _finite_decimal(name, value)
        count = self.open_position_count
        if count is not None:
            validate_nonnegative_integer("open_position_count", count)
        if self.snapshot_timestamp is not None and not isinstance(
            self.snapshot_timestamp, datetime
        ):
            raise ValueError("snapshot_timestamp must be a datetime or None")
        if self.exposure_by_underlying is not None:
            if not isinstance(self.exposure_by_underlying, Mapping):
                raise ValueError("exposure_by_underlying must be a mapping or None")
            exposures = dict(self.exposure_by_underlying)
            if set(exposures) != set(UnderlyingIndex):
                raise ValueError(
                    "exposure_by_underlying must explicitly include every supported underlying"
                )
            for underlying, amount in exposures.items():
                if not isinstance(underlying, UnderlyingIndex):
                    raise ValueError("exposure keys must be UnderlyingIndex values")
                _nonnegative_decimal("underlying exposure", amount)
            object.__setattr__(
                self, "exposure_by_underlying", MappingProxyType(exposures)
            )
            if count == 0 and any(exposures.values()):
                raise ValueError("zero open positions must have zero exposure")
            if count is not None and count > 0 and not any(exposures.values()):
                raise ValueError("open positions must have positive gross exposure")
        if (
            self.current_equity is not None
            and self.peak_equity is not None
            and self.peak_equity < self.current_equity
        ):
            raise ValueError("peak_equity must be at least current_equity")


@dataclass(frozen=True)
class ProposedTrade:
    """Single-contract proposal. Prices and point value use explicit Decimals."""

    contract: DerivativeContract
    signal: StrategySignal
    horizon: TradingHorizon
    structure: TradeStructure
    market_data: FuturesQuoteResponse | OptionsQuoteResponse | None
    market_data_source: str | None
    entry_reference_price: Decimal | None
    stop_loss_reference_price: Decimal | None
    target_reference_price: Decimal | None
    quantity: int | None
    point_value: Decimal | None
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        if not isinstance(self.signal, StrategySignal):
            raise ValueError("signal must be a StrategySignal")
        if not isinstance(self.horizon, TradingHorizon):
            raise ValueError("horizon must be a TradingHorizon")
        if not isinstance(self.structure, TradeStructure):
            raise ValueError("structure must be a TradeStructure")
        if self.market_data is not None and not isinstance(
            self.market_data, (FuturesQuoteResponse, OptionsQuoteResponse)
        ):
            raise ValueError(
                "market_data must be a futures/options quote response or None"
            )
        if self.market_data_source is not None and (
            not isinstance(self.market_data_source, str)
            or not self.market_data_source.strip()
        ):
            raise ValueError("market_data_source must be a non-empty string or None")
        for name in (
            "entry_reference_price",
            "stop_loss_reference_price",
            "target_reference_price",
            "point_value",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(name, value, allow_zero=False)
        if self.quantity is not None:
            validate_positive_integer("quantity", self.quantity)
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must be a tuple of non-empty strings")


@dataclass(frozen=True)
class RiskReason:
    code: RiskReasonCode
    explanation: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, RiskReasonCode):
            raise ValueError("reason code must be a RiskReasonCode")
        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("reason explanation must be non-empty")


@dataclass(frozen=True)
class RiskDecision:
    """Risk clearance only; APPROVED never submits or authorises an order."""

    status: RiskDecisionStatus
    reasons: tuple[RiskReason, ...]
    estimated_risk_amount: Decimal | None
    estimated_exposure_amount: Decimal | None
    risk_budget: Decimal | None
    remaining_daily_risk_allowance: Decimal | None
    evaluated_at: datetime | None
    policy_id: str
    policy_version: str
    contract_id: str
    data_source: str | None
    signal_data_source: str
    is_synthetic: bool | None
    user_approval_required: bool
    user_approved: bool | None
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.status, RiskDecisionStatus):
            raise ValueError("status must be a RiskDecisionStatus")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(item, RiskReason) for item in self.reasons
        ):
            raise ValueError("reasons must contain RiskReason values")
        for name in (
            "estimated_risk_amount",
            "estimated_exposure_amount",
            "risk_budget",
        ):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_decimal(name, value)
        if self.remaining_daily_risk_allowance is not None:
            _finite_decimal(
                "remaining_daily_risk_allowance", self.remaining_daily_risk_allowance
            )
        if self.evaluated_at is not None and not isinstance(self.evaluated_at, datetime):
            raise ValueError("evaluated_at must be a datetime or None")
        for name in ("policy_id", "policy_version", "contract_id", "signal_data_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.data_source is not None and (
            not isinstance(self.data_source, str) or not self.data_source.strip()
        ):
            raise ValueError("data_source must be a non-empty string or None")
        if self.is_synthetic is not None and not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool or None")
        if not isinstance(self.user_approval_required, bool):
            raise ValueError("user_approval_required must be a bool")
        if self.user_approved is not None and not isinstance(self.user_approved, bool):
            raise ValueError("user_approved must be a bool or None")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must be a tuple of non-empty strings")


class RiskCalculationError(ValueError):
    """A fail-closed calculation failure with an explicit decision reason."""

    def __init__(self, code: RiskReasonCode, explanation: str) -> None:
        super().__init__(explanation)
        self.code = code
        self.explanation = explanation


def _finite_decimal(name: str, value: Decimal) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    return value


def _nonnegative_decimal(name: str, value: Decimal) -> Decimal:
    validate_decimal(name, value, allow_zero=True)
    return value


def _validate_frozenset(name: str, value: frozenset, expected_type: type) -> None:
    if not isinstance(value, frozenset):
        raise ValueError(f"{name} must be a frozenset")
    if any(not isinstance(item, expected_type) for item in value):
        raise ValueError(f"{name} contains an invalid value")


__all__ = [
    "AccountRiskSnapshot",
    "ProposedTrade",
    "RiskCalculationError",
    "RiskDecision",
    "RiskDecisionStatus",
    "RiskPolicy",
    "RiskReason",
    "RiskReasonCode",
    "TradeStructure",
    "TradingHorizon",
]
