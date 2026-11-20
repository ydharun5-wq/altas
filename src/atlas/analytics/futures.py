"""Deterministic index-futures basis, carry, and rollover calculations.

All prices and calculated values are per underlying unit; no lot multiplier is
applied. Annualization uses actual elapsed seconds divided by a 365-day year
(ACT/365 Fixed). Futures expiry is a date in the existing contract model, so
calculations requiring remaining time also require an explicit, timezone-aware
expiry instant whose local date matches that contract date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from math import exp, fsum, isfinite
from numbers import Real
from typing import ClassVar, Iterable

from atlas.derivatives.models import DerivativeContract, InstrumentType, UnderlyingIndex
from atlas.market_data.models import FuturesQuoteResponse


_SECONDS_PER_ACT_365_YEAR = 365.0 * 24.0 * 60.0 * 60.0


@dataclass(frozen=True)
class FuturesObservation:
    """A futures response paired with its required source identifier."""

    response: FuturesQuoteResponse
    data_source: str

    def __post_init__(self) -> None:
        if not isinstance(self.response, FuturesQuoteResponse):
            raise ValueError("response must be a FuturesQuoteResponse")
        _validate_data_source(self.data_source)
        if self.response.quote.last_traded_price is None:
            raise ValueError("futures last_traded_price is missing")
        _positive_finite("futures last_traded_price", self.response.quote.last_traded_price)

    @property
    def contract(self) -> DerivativeContract:
        return self.response.quote.contract

    @property
    def price(self) -> float:
        if self.response.quote.last_traded_price is None:
            raise ValueError("futures last_traded_price is missing")
        return _positive_finite(
            "futures last_traded_price", self.response.quote.last_traded_price
        )

    @property
    def observed_at(self) -> datetime:
        return self.response.quote.timestamp

    @property
    def is_synthetic(self) -> bool:
        return self.response.is_synthetic


@dataclass(frozen=True)
class UnderlyingSpotObservation:
    """Explicit spot input, since the current domain has no spot-quote model."""

    underlying: UnderlyingIndex
    price: int | float | Decimal
    observed_at: datetime
    data_source: str
    is_synthetic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.underlying, UnderlyingIndex):
            raise ValueError("underlying must be an UnderlyingIndex")
        _positive_finite("spot price", self.price)
        _validate_aware_datetime("spot observed_at", self.observed_at)
        _validate_data_source(self.data_source)
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("spot is_synthetic must be a bool")

    @property
    def numeric_price(self) -> float:
        return _positive_finite("spot price", self.price)


@dataclass(frozen=True)
class CarryRatePeriod:
    """Caller-supplied continuous annual rates for one explicit time interval.

    Rates are annualized decimals, such as 0.05 for 5%. A cost-of-carry
    schedule may vary these values between contiguous periods; no global
    interest-rate or dividend-yield value is selected by this module.
    """

    start_at: datetime
    end_at: datetime
    risk_free_rate: int | float | Decimal
    dividend_yield: int | float | Decimal

    def __post_init__(self) -> None:
        _validate_aware_datetime("carry period start_at", self.start_at)
        _validate_aware_datetime("carry period end_at", self.end_at)
        if _seconds_between(self.start_at, self.end_at) <= 0.0:
            raise ValueError("carry period end_at must be later than start_at")
        _finite_number("risk_free_rate", self.risk_free_rate)
        _finite_number("dividend_yield", self.dividend_yield)


@dataclass(frozen=True)
class FuturesBasisAnalysis:
    """Observed futures-versus-spot basis, separate from theoretical fair value."""

    contract: DerivativeContract
    futures_price: float
    spot_price: float
    absolute_basis: float
    percentage_basis: float
    annualized_basis_percentage: float
    time_to_expiry_years: float
    expiry_at: datetime
    futures_observed_at: datetime
    spot_observed_at: datetime
    futures_data_source: str
    spot_data_source: str
    futures_is_synthetic: bool
    spot_is_synthetic: bool

    day_count_convention: ClassVar[str] = "ACT/365 Fixed"
    annualization_method: ClassVar[str] = "simple linear annualization"


@dataclass(frozen=True)
class CostOfCarryAnalysis:
    """Model-implied fair value from spot and an explicit carry schedule.

    This result contains no observed futures price or observed market basis.
    The model assumes continuous compounding and caller-supplied rates that
    are constant only within each explicit schedule period. It excludes
    transaction costs, taxes, and other carry components.
    """

    contract: DerivativeContract
    spot_price: float
    model_implied_fair_value: float
    time_to_expiry_years: float
    expiry_at: datetime
    spot_observed_at: datetime
    data_source: str
    is_synthetic: bool
    rate_periods: tuple[CarryRatePeriod, ...]

    formula: ClassVar[str] = "F = S * exp(sum((r_i - q_i) * T_i))"
    day_count_convention: ClassVar[str] = "ACT/365 Fixed"
    compounding_assumption: ClassVar[str] = "continuous"


@dataclass(frozen=True)
class RolloverAnalysis:
    """Calendar spread and long-roll cash sign for two supplied futures quotes."""

    near_contract: DerivativeContract
    next_contract: DerivativeContract
    near_price: float
    next_price: float
    absolute_calendar_spread: float
    percentage_calendar_spread: float
    long_rollover_cost: float
    near_observed_at: datetime
    next_observed_at: datetime
    near_data_source: str
    next_data_source: str
    near_is_synthetic: bool
    next_is_synthetic: bool

    percentage_reference: ClassVar[str] = "near-month futures price"
    sign_convention: ClassVar[str] = (
        "positive is a debit to a long roll; negative is a credit"
    )


def analyze_futures_basis(
    futures: FuturesObservation,
    spot: UnderlyingSpotObservation,
    *,
    expiry_at: datetime,
) -> FuturesBasisAnalysis:
    """Calculate observed basis and simple ACT/365F annualized percentage.

    Absolute basis is ``futures - spot``. Percentage basis is that difference
    divided by spot, times 100. Annualized basis percentage is the percentage
    basis divided by actual time-to-expiry in ACT/365F years. The annualization
    interval starts at the spot observation timestamp; both input timestamps
    are retained in the result, and no freshness threshold is imposed.
    """
    _validate_futures_observation(futures, "futures")
    if not isinstance(spot, UnderlyingSpotObservation):
        raise ValueError("spot must be an UnderlyingSpotObservation")
    if futures.contract.underlying is not spot.underlying:
        raise ValueError("futures and spot underlyings must match")
    _validate_expiry_instant(futures.contract, expiry_at)
    time_to_expiry = _positive_time_to_expiry(spot.observed_at, expiry_at)
    _positive_time_to_expiry(futures.observed_at, expiry_at)

    futures_price = futures.price
    spot_price = spot.numeric_price
    absolute_basis = futures_price - spot_price
    percentage_basis = _finite_result(
        "percentage basis", absolute_basis / spot_price * 100.0
    )
    annualized_basis = _finite_result(
        "annualized basis", percentage_basis / time_to_expiry
    )
    return FuturesBasisAnalysis(
        contract=futures.contract,
        futures_price=futures_price,
        spot_price=spot_price,
        absolute_basis=_finite_result("absolute basis", absolute_basis),
        percentage_basis=percentage_basis,
        annualized_basis_percentage=annualized_basis,
        time_to_expiry_years=time_to_expiry,
        expiry_at=expiry_at,
        futures_observed_at=futures.observed_at,
        spot_observed_at=spot.observed_at,
        futures_data_source=futures.data_source,
        spot_data_source=spot.data_source,
        futures_is_synthetic=futures.is_synthetic,
        spot_is_synthetic=spot.is_synthetic,
    )


def analyze_cost_of_carry(
    contract: DerivativeContract,
    spot: UnderlyingSpotObservation,
    *,
    expiry_at: datetime,
    rate_periods: Iterable[CarryRatePeriod],
) -> CostOfCarryAnalysis:
    """Compute model fair value from explicit piecewise rate/yield inputs.

    The supplied periods must cover the spot observation instant through
    expiry exactly, with no gaps or overlaps. Period rates are continuously
    compounded annual decimals. This supports a varying schedule and never
    supplies an interest rate or dividend yield implicitly.
    """
    _validate_futures_contract(contract, "contract")
    if not isinstance(spot, UnderlyingSpotObservation):
        raise ValueError("spot must be an UnderlyingSpotObservation")
    if contract.underlying is not spot.underlying:
        raise ValueError("futures contract and spot underlyings must match")
    _validate_expiry_instant(contract, expiry_at)
    time_to_expiry = _positive_time_to_expiry(spot.observed_at, expiry_at)
    periods = _normalize_rate_periods(rate_periods, spot.observed_at, expiry_at)

    exponent_parts: list[float] = []
    for period in periods:
        period_years = _seconds_between(period.start_at, period.end_at) / _SECONDS_PER_ACT_365_YEAR
        exponent_parts.append(
            (
                _finite_number("risk_free_rate", period.risk_free_rate)
                - _finite_number("dividend_yield", period.dividend_yield)
            )
            * period_years
        )
    try:
        exponent = _finite_result("carry exponent", fsum(exponent_parts))
    except (OverflowError, ValueError) as exc:
        raise ValueError("carry exponent exceeds numeric range") from exc
    try:
        carry_factor = exp(exponent)
    except OverflowError as exc:
        raise ValueError("cost-of-carry fair value exceeds numeric range") from exc
    fair_value = _finite_result("model-implied fair value", spot.numeric_price * carry_factor)
    if fair_value <= 0.0:
        raise ValueError("model-implied fair value must remain positive")

    return CostOfCarryAnalysis(
        contract=contract,
        spot_price=spot.numeric_price,
        model_implied_fair_value=fair_value,
        time_to_expiry_years=time_to_expiry,
        expiry_at=expiry_at,
        spot_observed_at=spot.observed_at,
        data_source=spot.data_source,
        is_synthetic=spot.is_synthetic,
        rate_periods=periods,
    )


def analyze_rollover(
    near: FuturesObservation,
    next_month: FuturesObservation,
) -> RolloverAnalysis:
    """Compare two explicit contracts and estimate long-roll cash cost.

    Calendar spread is ``next-month price - near-month price``. Its percentage
    uses near-month price as the denominator. For a long roll, positive spread
    means the next contract costs more (a debit); negative spread is a credit.
    Values are per underlying unit, with no lot-size multiplier.
    """
    _validate_futures_observation(near, "near")
    _validate_futures_observation(next_month, "next_month")
    near_contract = near.contract
    next_contract = next_month.contract
    if near_contract.exchange is not next_contract.exchange:
        raise ValueError("near and next-month contracts must use the same exchange")
    if near_contract.underlying is not next_contract.underlying:
        raise ValueError("near and next-month contracts must have the same underlying")
    if near_contract.expiry >= next_contract.expiry:
        raise ValueError("near contract expiry must be earlier than next-month expiry")

    near_price = near.price
    next_price = next_month.price
    spread = _finite_result("calendar spread", next_price - near_price)
    percentage = _finite_result(
        "percentage calendar spread", spread / near_price * 100.0
    )
    return RolloverAnalysis(
        near_contract=near_contract,
        next_contract=next_contract,
        near_price=near_price,
        next_price=next_price,
        absolute_calendar_spread=spread,
        percentage_calendar_spread=percentage,
        long_rollover_cost=spread,
        near_observed_at=near.observed_at,
        next_observed_at=next_month.observed_at,
        near_data_source=near.data_source,
        next_data_source=next_month.data_source,
        near_is_synthetic=near.is_synthetic,
        next_is_synthetic=next_month.is_synthetic,
    )


def _normalize_rate_periods(
    rate_periods: Iterable[CarryRatePeriod],
    start_at: datetime,
    end_at: datetime,
) -> tuple[CarryRatePeriod, ...]:
    if isinstance(rate_periods, (str, bytes, bytearray)):
        raise ValueError("rate_periods must be an iterable of CarryRatePeriod values")
    try:
        periods = tuple(rate_periods)
    except TypeError as exc:
        raise ValueError(
            "rate_periods must be an iterable of CarryRatePeriod values"
        ) from exc
    if not periods:
        raise ValueError("an explicit interest-rate and dividend-yield schedule is required")
    if any(not isinstance(period, CarryRatePeriod) for period in periods):
        raise ValueError("rate_periods must contain CarryRatePeriod values")

    if not _same_instant(periods[0].start_at, start_at):
        raise ValueError("rate schedule must begin at the spot observation time")
    if not _same_instant(periods[-1].end_at, end_at):
        raise ValueError("rate schedule must end at expiry_at")
    for previous, current in zip(periods, periods[1:]):
        if not _same_instant(previous.end_at, current.start_at):
            raise ValueError("rate schedule periods must be contiguous and non-overlapping")
    return periods


def _validate_futures_observation(value: FuturesObservation, name: str) -> None:
    if not isinstance(value, FuturesObservation):
        raise ValueError(f"{name} must be a FuturesObservation")
    _validate_futures_contract(value.contract, f"{name} contract")
    _validate_aware_datetime(f"{name} observed_at", value.observed_at)
    _positive_finite(f"{name} futures price", value.response.quote.last_traded_price)


def _validate_futures_contract(contract: DerivativeContract, name: str) -> None:
    if not isinstance(contract, DerivativeContract):
        raise ValueError(f"{name} must be a DerivativeContract")
    if contract.instrument_type is not InstrumentType.FUT:
        raise ValueError(f"{name} must be a futures contract")


def _validate_expiry_instant(contract: DerivativeContract, expiry_at: datetime) -> None:
    _validate_aware_datetime("expiry_at", expiry_at)
    if expiry_at.date() != contract.expiry:
        raise ValueError("expiry_at local date must match the futures contract expiry")


def _positive_time_to_expiry(start_at: datetime, expiry_at: datetime) -> float:
    seconds = _seconds_between(start_at, expiry_at)
    if seconds <= 0.0:
        raise ValueError("time to expiry must be positive")
    return _finite_result("time to expiry", seconds / _SECONDS_PER_ACT_365_YEAR)


def _seconds_between(start_at: datetime, end_at: datetime) -> float:
    _validate_aware_datetime("start_at", start_at)
    _validate_aware_datetime("end_at", end_at)
    try:
        seconds = (
            end_at.astimezone(timezone.utc) - start_at.astimezone(timezone.utc)
        ).total_seconds()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("timestamps must be comparable instants") from exc
    return _finite_result("elapsed seconds", seconds)


def _same_instant(first: datetime, second: datetime) -> bool:
    return first.astimezone(timezone.utc) == second.astimezone(timezone.utc)


def _validate_aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    try:
        offset = value.utcoffset()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have valid timezone information") from exc
    if offset is None:
        raise ValueError(f"{name} must be timezone-aware")


def _validate_data_source(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("data_source must be a non-empty string")


def _finite_number(name: str, value: int | float | Decimal) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not isfinite(converted):
        raise ValueError(f"{name} must be a finite real number")
    return converted


def _positive_finite(name: str, value: int | float | Decimal | None) -> float:
    converted = _finite_number(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _finite_result(name: str, value: float) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} exceeds numeric range")
    return value


__all__ = [
    "CarryRatePeriod",
    "CostOfCarryAnalysis",
    "FuturesBasisAnalysis",
    "FuturesObservation",
    "RolloverAnalysis",
    "UnderlyingSpotObservation",
    "analyze_cost_of_carry",
    "analyze_futures_basis",
    "analyze_rollover",
]
