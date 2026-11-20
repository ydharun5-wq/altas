"""Actual option-position exposure and deterministic scenario P&L analytics.

Positions are supplied explicitly as signed long/short contract quantities.
Public open interest is not a position direction and is never used by these
calculations. Exposures scale by the lot size on the supplied contract.
Scenario P&L is the modeled option-value change from the supplied base state,
not P&L relative to an entry cost or a guaranteed forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from math import fsum, isfinite
from numbers import Real
from typing import ClassVar, Iterable

from atlas.analytics.options.black_scholes import (
    BlackScholesGreeks,
    OptionRight,
    black_scholes_greeks,
    black_scholes_price,
)
from atlas.derivatives.models import (
    DerivativeContract,
    InstrumentType,
    OptionType,
    UnderlyingIndex,
)


_SECONDS_PER_YEAR_ACT_365F = 365.0 * 24.0 * 60.0 * 60.0


class PositionDirection(str, Enum):
    """Explicit side of an actual supplied option position."""

    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is PositionDirection.LONG else -1


@dataclass(frozen=True)
class OptionPosition:
    """A supplied portfolio position; ``quantity`` counts contracts."""

    contract: DerivativeContract
    direction: PositionDirection
    quantity: int
    observed_at: datetime
    data_source: str
    is_synthetic: bool

    def __post_init__(self) -> None:
        _validate_option_contract(self.contract, "position contract")
        if not isinstance(self.direction, PositionDirection):
            raise ValueError("direction must be an explicit PositionDirection")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("quantity must be a positive integer contract count")
        _validate_aware_datetime("position observed_at", self.observed_at)
        _validate_data_source(self.data_source, "position data_source")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("position is_synthetic must be a bool")


@dataclass(frozen=True)
class OptionMarketInputs:
    """Explicit model inputs and provenance for one option contract.

    ``expiry_at`` must be timezone-aware and its local calendar date must
    equal the contract's supplied expiry date. Remaining time is measured
    between UTC instants using ACT/365F; no exchange timezone is inferred.
    Spot is expressed in the contract's price units. Implied volatility, the
    continuously compounded risk-free rate, and dividend yield are annualized
    decimals. The base model state is timestamped at ``observed_at``.
    """

    contract: DerivativeContract
    underlying_price: int | float | Decimal
    implied_volatility: int | float | Decimal
    risk_free_rate: int | float | Decimal
    dividend_yield: int | float | Decimal
    observed_at: datetime
    expiry_at: datetime
    data_source: str
    is_synthetic: bool

    def __post_init__(self) -> None:
        _validate_option_contract(self.contract, "market contract")
        _positive_finite("underlying_price", self.underlying_price)
        _positive_finite("implied_volatility", self.implied_volatility)
        _finite_number("risk_free_rate", self.risk_free_rate)
        _finite_number("dividend_yield", self.dividend_yield)
        _validate_aware_datetime("market observed_at", self.observed_at)
        _validate_aware_datetime("expiry_at", self.expiry_at)
        if self.expiry_at.date() != self.contract.expiry:
            raise ValueError("expiry_at local date must match the option contract expiry")
        _positive_time_to_expiry(self.observed_at, self.expiry_at)
        _validate_data_source(self.data_source, "market data_source")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("market is_synthetic must be a bool")


@dataclass(frozen=True)
class OptionPositionValuation:
    """Pair a supplied position with matching, explicit market model inputs."""

    position: OptionPosition
    market: OptionMarketInputs

    def __post_init__(self) -> None:
        if not isinstance(self.position, OptionPosition):
            raise ValueError("position must be an OptionPosition")
        if not isinstance(self.market, OptionMarketInputs):
            raise ValueError("market must be OptionMarketInputs")
        if self.position.contract != self.market.contract:
            raise ValueError("position and market contract identities must match")


@dataclass(frozen=True)
class OptionScenario:
    """Absolute underlying/IV shocks and an explicit future scenario instant.

    ``underlying_price_change`` is an absolute price-unit change.
    ``implied_volatility_change`` is an additive volatility-decimal change
    (0.05 means a five-volatility-point increase). Risk-free rate and dividend
    yield remain at their supplied base values.
    """

    underlying_price_change: int | float | Decimal
    implied_volatility_change: int | float | Decimal
    scenario_at: datetime

    def __post_init__(self) -> None:
        _finite_number("underlying_price_change", self.underlying_price_change)
        _finite_number("implied_volatility_change", self.implied_volatility_change)
        _validate_aware_datetime("scenario_at", self.scenario_at)


@dataclass(frozen=True)
class PositionExposure:
    """Signed Greeks for one actual supplied position, scaled by contract lot.

    Exposure units are: delta, currency P&L per one underlying-price unit;
    gamma, currency P&L per squared underlying-price unit; vega, currency P&L
    per 1.00 volatility change; theta, currency P&L per calendar day on a
    365-day convention. These values are model sensitivities, not forecasts.
    """

    contract: DerivativeContract
    direction: PositionDirection
    quantity_contracts: int
    lot_size_multiplier: int
    signed_underlying_units: int
    per_unit_greeks: BlackScholesGreeks
    delta_exposure: float
    gamma_exposure: float
    vega_exposure: float
    theta_exposure: float
    time_to_expiry_years: float
    calculated_at: datetime
    position_observed_at: datetime
    market_observed_at: datetime
    position_data_source: str
    market_data_source: str
    position_is_synthetic: bool
    market_is_synthetic: bool

    exposure_basis: ClassVar[str] = "actual supplied signed position quantity"
    model_assumption: ClassVar[str] = "European Black-Scholes model sensitivities"
    theta_day_count: ClassVar[str] = "365 calendar days per year"
    time_to_expiry_convention: ClassVar[str] = "ACT/365F between UTC instants"
    delta_unit: ClassVar[str] = "currency P&L per one underlying-price unit"
    gamma_unit: ClassVar[str] = "currency P&L per squared underlying-price unit"
    vega_unit: ClassVar[str] = "currency P&L per 1.00 volatility change"
    theta_unit: ClassVar[str] = "currency P&L per calendar day"


@dataclass(frozen=True)
class PortfolioExposure:
    """Aggregate signed Greek exposures for positions on one underlying."""

    underlying: UnderlyingIndex
    positions: tuple[PositionExposure, ...]
    total_delta_exposure: float
    total_gamma_exposure: float
    total_vega_exposure: float
    total_theta_exposure: float
    calculated_at: datetime
    contains_synthetic_inputs: bool

    exposure_basis: ClassVar[str] = "actual supplied signed position quantities"
    model_assumption: ClassVar[str] = "European Black-Scholes model sensitivities"


@dataclass(frozen=True)
class PositionScenarioResult:
    """Model-value scenario P&L for one supplied actual position."""

    contract: DerivativeContract
    direction: PositionDirection
    quantity_contracts: int
    lot_size_multiplier: int
    signed_underlying_units: int
    base_option_price: float
    scenario_option_price: float
    scenario_pnl: float
    base_underlying_price: float
    scenario_underlying_price: float
    base_implied_volatility: float
    scenario_implied_volatility: float
    base_time_to_expiry_years: float
    scenario_time_to_expiry_years: float
    scenario_at: datetime
    calculated_at: datetime
    position_observed_at: datetime
    market_observed_at: datetime
    position_data_source: str
    market_data_source: str
    position_is_synthetic: bool
    market_is_synthetic: bool

    pnl_basis: ClassVar[str] = "scenario model value minus base model value"
    pnl_unit: ClassVar[str] = "currency"
    model_assumption: ClassVar[str] = (
        "European Black-Scholes; risk-free rate and dividend yield held at base inputs"
    )


@dataclass(frozen=True)
class PortfolioScenarioResult:
    """Aggregate scenario P&L for positions on one underlying."""

    underlying: UnderlyingIndex
    positions: tuple[PositionScenarioResult, ...]
    total_scenario_pnl: float
    scenario: OptionScenario
    calculated_at: datetime
    contains_synthetic_inputs: bool

    pnl_basis: ClassVar[str] = "scenario model value minus base model value"
    pnl_unit: ClassVar[str] = "currency"


def calculate_position_exposure(
    valuation: OptionPositionValuation,
    *,
    calculated_at: datetime,
) -> PositionExposure:
    """Calculate signed position-scaled Black-Scholes Greeks.

    Time remaining is measured from the market input's observation instant,
    so the market values and model time describe the same base state.
    """
    _validate_valuation(valuation, calculated_at)
    market = valuation.market
    position = valuation.position
    time_to_expiry = _positive_time_to_expiry(market.observed_at, market.expiry_at)
    greeks = black_scholes_greeks(
        _option_right(position.contract.option_type),
        spot=market.underlying_price,
        strike=position.contract.strike,
        time_to_expiry_years=time_to_expiry,
        risk_free_rate=market.risk_free_rate,
        dividend_yield=market.dividend_yield,
        volatility=market.implied_volatility,
    )
    signed_units = _signed_underlying_units(position)
    return PositionExposure(
        contract=position.contract,
        direction=position.direction,
        quantity_contracts=position.quantity,
        lot_size_multiplier=position.contract.lot_size,
        signed_underlying_units=signed_units,
        per_unit_greeks=greeks,
        delta_exposure=_scaled_exposure("delta exposure", greeks.delta, signed_units),
        gamma_exposure=_scaled_exposure("gamma exposure", greeks.gamma, signed_units),
        vega_exposure=_scaled_exposure("vega exposure", greeks.vega, signed_units),
        theta_exposure=_scaled_exposure("theta exposure", greeks.theta, signed_units),
        time_to_expiry_years=time_to_expiry,
        calculated_at=calculated_at,
        position_observed_at=position.observed_at,
        market_observed_at=market.observed_at,
        position_data_source=position.data_source,
        market_data_source=market.data_source,
        position_is_synthetic=position.is_synthetic,
        market_is_synthetic=market.is_synthetic,
    )


def calculate_portfolio_exposure(
    valuations: Iterable[OptionPositionValuation],
    *,
    calculated_at: datetime,
) -> PortfolioExposure:
    """Sum actual supplied-position exposures for a single underlying."""
    materialized = _materialize_valuations(valuations)
    underlying = materialized[0].position.contract.underlying
    if any(item.position.contract.underlying is not underlying for item in materialized):
        raise ValueError("portfolio exposure aggregation requires one underlying")
    position_exposures = tuple(
        calculate_position_exposure(item, calculated_at=calculated_at)
        for item in materialized
    )
    return PortfolioExposure(
        underlying=underlying,
        positions=position_exposures,
        total_delta_exposure=_finite_sum(
            "portfolio delta exposure",
            (item.delta_exposure for item in position_exposures),
        ),
        total_gamma_exposure=_finite_sum(
            "portfolio gamma exposure",
            (item.gamma_exposure for item in position_exposures),
        ),
        total_vega_exposure=_finite_sum(
            "portfolio vega exposure",
            (item.vega_exposure for item in position_exposures),
        ),
        total_theta_exposure=_finite_sum(
            "portfolio theta exposure",
            (item.theta_exposure for item in position_exposures),
        ),
        calculated_at=calculated_at,
        contains_synthetic_inputs=any(
            item.position_is_synthetic or item.market_is_synthetic
            for item in position_exposures
        ),
    )


def calculate_position_scenario_pnl(
    valuation: OptionPositionValuation,
    scenario: OptionScenario,
    *,
    calculated_at: datetime,
) -> PositionScenarioResult:
    """Reprice base and scenario states with the existing Black-Scholes model.

    Base time is measured from the market observation instant. The scenario
    uses its explicit timestamp and the specified spot/volatility shocks.
    """
    _validate_valuation(valuation, calculated_at)
    if not isinstance(scenario, OptionScenario):
        raise ValueError("scenario must be an OptionScenario")
    if _instant(scenario.scenario_at) < _instant(calculated_at):
        raise ValueError("scenario_at must not be earlier than calculated_at")

    market = valuation.market
    position = valuation.position
    base_spot = _positive_finite("underlying_price", market.underlying_price)
    scenario_spot = _positive_finite(
        "scenario underlying price",
        base_spot + _finite_number(
            "underlying_price_change", scenario.underlying_price_change
        ),
    )
    base_volatility = _positive_finite(
        "implied_volatility", market.implied_volatility
    )
    scenario_volatility = _positive_finite(
        "scenario implied volatility",
        base_volatility
        + _finite_number(
            "implied_volatility_change", scenario.implied_volatility_change
        ),
    )
    base_time = _positive_time_to_expiry(market.observed_at, market.expiry_at)
    scenario_time = _positive_time_to_expiry(scenario.scenario_at, market.expiry_at)
    right = _option_right(position.contract.option_type)
    common_model_inputs = {
        "strike": position.contract.strike,
        "risk_free_rate": market.risk_free_rate,
        "dividend_yield": market.dividend_yield,
    }
    base_price = black_scholes_price(
        right,
        spot=base_spot,
        time_to_expiry_years=base_time,
        volatility=base_volatility,
        **common_model_inputs,
    )
    scenario_price = black_scholes_price(
        right,
        spot=scenario_spot,
        time_to_expiry_years=scenario_time,
        volatility=scenario_volatility,
        **common_model_inputs,
    )
    signed_units = _signed_underlying_units(position)
    scenario_pnl = _finite_result(
        "scenario P&L", (scenario_price - base_price) * signed_units
    )
    return PositionScenarioResult(
        contract=position.contract,
        direction=position.direction,
        quantity_contracts=position.quantity,
        lot_size_multiplier=position.contract.lot_size,
        signed_underlying_units=signed_units,
        base_option_price=base_price,
        scenario_option_price=scenario_price,
        scenario_pnl=scenario_pnl,
        base_underlying_price=base_spot,
        scenario_underlying_price=scenario_spot,
        base_implied_volatility=base_volatility,
        scenario_implied_volatility=scenario_volatility,
        base_time_to_expiry_years=base_time,
        scenario_time_to_expiry_years=scenario_time,
        scenario_at=scenario.scenario_at,
        calculated_at=calculated_at,
        position_observed_at=position.observed_at,
        market_observed_at=market.observed_at,
        position_data_source=position.data_source,
        market_data_source=market.data_source,
        position_is_synthetic=position.is_synthetic,
        market_is_synthetic=market.is_synthetic,
    )


def calculate_portfolio_scenario_pnl(
    valuations: Iterable[OptionPositionValuation],
    scenario: OptionScenario,
    *,
    calculated_at: datetime,
) -> PortfolioScenarioResult:
    """Aggregate full-repricing scenario P&L for positions on one underlying.

    The same absolute spot change and additive volatility change apply to all
    positions. Each position retains its own contract expiry and base inputs.
    """
    materialized = _materialize_valuations(valuations)
    underlying = materialized[0].position.contract.underlying
    if any(item.position.contract.underlying is not underlying for item in materialized):
        raise ValueError("portfolio scenario aggregation requires one underlying")
    position_results = tuple(
        calculate_position_scenario_pnl(
            item,
            scenario,
            calculated_at=calculated_at,
        )
        for item in materialized
    )
    return PortfolioScenarioResult(
        underlying=underlying,
        positions=position_results,
        total_scenario_pnl=_finite_sum(
            "portfolio scenario P&L",
            (item.scenario_pnl for item in position_results),
        ),
        scenario=scenario,
        calculated_at=calculated_at,
        contains_synthetic_inputs=any(
            item.position_is_synthetic or item.market_is_synthetic
            for item in position_results
        ),
    )


def _materialize_valuations(
    valuations: Iterable[OptionPositionValuation],
) -> tuple[OptionPositionValuation, ...]:
    if isinstance(valuations, (str, bytes, bytearray)):
        raise ValueError("valuations must be an iterable of OptionPositionValuation values")
    try:
        items = tuple(valuations)
    except TypeError as exc:
        raise ValueError(
            "valuations must be an iterable of OptionPositionValuation values"
        ) from exc
    if not items:
        raise ValueError("at least one position valuation is required")
    if any(not isinstance(item, OptionPositionValuation) for item in items):
        raise ValueError("valuations must contain OptionPositionValuation values")
    return items


def _validate_valuation(
    valuation: OptionPositionValuation,
    calculated_at: datetime,
) -> None:
    if not isinstance(valuation, OptionPositionValuation):
        raise ValueError("valuation must be an OptionPositionValuation")
    _validate_aware_datetime("calculated_at", calculated_at)
    if _instant(calculated_at) < _instant(valuation.position.observed_at):
        raise ValueError("calculated_at must not precede position observed_at")
    if _instant(calculated_at) < _instant(valuation.market.observed_at):
        raise ValueError("calculated_at must not precede market observed_at")
    if _instant(calculated_at) >= _instant(valuation.market.expiry_at):
        raise ValueError("calculated_at must be earlier than expiry_at")


def _validate_option_contract(contract: DerivativeContract, name: str) -> None:
    if not isinstance(contract, DerivativeContract):
        raise ValueError(f"{name} must be a DerivativeContract")
    if contract.instrument_type is not InstrumentType.OPT:
        raise ValueError(f"{name} must be an options contract")
    if contract.option_type not in (OptionType.CE, OptionType.PE):
        raise ValueError(f"{name} must have option_type CE or PE")


def _option_right(option_type: OptionType | None) -> OptionRight:
    if option_type is OptionType.CE:
        return OptionRight.CALL
    if option_type is OptionType.PE:
        return OptionRight.PUT
    raise ValueError("option contract must specify CE or PE")


def _signed_underlying_units(position: OptionPosition) -> int:
    return position.direction.sign * position.quantity * position.contract.lot_size


def _scaled_exposure(name: str, greek: float, signed_units: int) -> float:
    try:
        result = greek * signed_units
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds numeric range") from exc
    return _finite_result(name, result)


def _positive_time_to_expiry(start_at: datetime, expiry_at: datetime) -> float:
    seconds = _seconds_between(start_at, expiry_at)
    if seconds <= 0.0:
        raise ValueError("time to expiry must be positive")
    return _finite_result("time to expiry", seconds / _SECONDS_PER_YEAR_ACT_365F)


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


def _instant(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _validate_aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    try:
        offset = value.utcoffset()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have valid timezone information") from exc
    if offset is None:
        raise ValueError(f"{name} must be timezone-aware")


def _validate_data_source(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


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


def _positive_finite(name: str, value: int | float | Decimal) -> float:
    converted = _finite_number(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _finite_result(name: str, value: float) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} exceeds numeric range")
    return value


def _finite_sum(name: str, values: Iterable[float]) -> float:
    try:
        result = fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} exceeds numeric range") from exc
    return _finite_result(name, result)


__all__ = [
    "OptionMarketInputs",
    "OptionPosition",
    "OptionPositionValuation",
    "OptionScenario",
    "PositionDirection",
    "PositionExposure",
    "PortfolioExposure",
    "PositionScenarioResult",
    "PortfolioScenarioResult",
    "calculate_position_exposure",
    "calculate_portfolio_exposure",
    "calculate_position_scenario_pnl",
    "calculate_portfolio_scenario_pnl",
]
