"""European Black-Scholes pricing and Greeks with continuous dividends.

Prices and sensitivities are per one unit of the underlying, without a
contract multiplier. Rates, dividend yield, and volatility are annualized
decimals (for example, 0.05 means 5%). Time to expiry is measured in years.

This is a European exercise model with continuously compounded rates and a
continuous dividend yield. Its outputs are model values, not guaranteed
executable market prices, and it does not assert that any exchange-traded
contract has European exercise terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from math import erfc, exp, fsum, isfinite, log, log1p, pi, sqrt
from numbers import Real


class OptionRight(str, Enum):
    """Mathematical right of a European option, independent of exchange codes."""

    CALL = "call"
    PUT = "put"


@dataclass(frozen=True)
class BlackScholesGreeks:
    """Model sensitivities, with theta per day and vega/rho per unit change.

    * ``delta``: price change per one-unit underlying price change.
    * ``gamma``: delta change per one-unit underlying price change.
    * ``theta``: price change per calendar day, using 365 days per year.
    * ``vega``: price change per 1.00 change in annualized volatility.
    * ``rho``: price change per 1.00 change in annual risk-free rate.
    """

    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


@dataclass(frozen=True)
class _PreparedInputs:
    spot: float
    strike: float
    time_to_expiry_years: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    sqrt_time: float
    volatility_time: float
    discounted_spot: float
    discounted_strike: float
    d1: float
    d2: float


_SQRT_2 = sqrt(2.0)
_SQRT_2_PI = sqrt(2.0 * pi)
_THETA_DAYS_PER_YEAR = 365.0


def black_scholes_price(
    option_right: OptionRight | str,
    *,
    spot: int | float | Decimal,
    strike: int | float | Decimal,
    time_to_expiry_years: int | float | Decimal,
    risk_free_rate: int | float | Decimal,
    dividend_yield: int | float | Decimal,
    volatility: int | float | Decimal,
) -> float:
    """Return a European call or put model price per underlying unit.

    ``option_right`` is ``OptionRight.CALL``/``"call"`` or
    ``OptionRight.PUT``/``"put"``. All numeric inputs must be finite; spot,
    strike, time, and volatility must be positive. Rates and dividend yield
    may be negative, zero, or positive. No market inputs are defaulted.
    """
    right = _option_right(option_right)
    inputs = _prepare_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )
    return _price_from_prepared(right, inputs)


def black_scholes_greeks(
    option_right: OptionRight | str,
    *,
    spot: int | float | Decimal,
    strike: int | float | Decimal,
    time_to_expiry_years: int | float | Decimal,
    risk_free_rate: int | float | Decimal,
    dividend_yield: int | float | Decimal,
    volatility: int | float | Decimal,
) -> BlackScholesGreeks:
    """Return European Black-Scholes Greeks in the documented units.

    Vega is per 1.00 volatility change, not per percentage point. Rho is per
    1.00 annual-rate change, not per percentage point. Theta is per calendar
    day on a 365-day convention.
    """
    right = _option_right(option_right)
    inputs = _prepare_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry_years=time_to_expiry_years,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=volatility,
    )

    density = _normal_pdf(inputs.d1)
    delta = (
        inputs.discounted_spot / inputs.spot * _normal_cdf(inputs.d1)
        if right is OptionRight.CALL
        else -inputs.discounted_spot / inputs.spot * _normal_cdf(-inputs.d1)
    )
    gamma = (
        (density / inputs.spot)
        * (inputs.discounted_spot / inputs.spot)
        / inputs.volatility_time
    )
    vega_scale = _finite_intermediate(
        "vega scale", density * inputs.sqrt_time
    )
    vega = _finite_intermediate(
        "vega", inputs.discounted_spot * vega_scale
    )

    diffusion_theta = _finite_intermediate(
        "theta", -0.5 * vega * inputs.volatility / inputs.time_to_expiry_years
    )
    if right is OptionRight.CALL:
        rate_theta = _finite_intermediate(
            "theta",
            -inputs.risk_free_rate
            * (inputs.discounted_strike * _normal_cdf(inputs.d2)),
        )
        dividend_theta = _finite_intermediate(
            "theta",
            inputs.dividend_yield
            * (inputs.discounted_spot * _normal_cdf(inputs.d1)),
        )
        rho = _finite_intermediate(
            "rho",
            inputs.discounted_strike
            * (inputs.time_to_expiry_years * _normal_cdf(inputs.d2)),
        )
    else:
        rate_theta = _finite_intermediate(
            "theta",
            inputs.risk_free_rate
            * (inputs.discounted_strike * _normal_cdf(-inputs.d2)),
        )
        dividend_theta = _finite_intermediate(
            "theta",
            -inputs.dividend_yield
            * (inputs.discounted_spot * _normal_cdf(-inputs.d1)),
        )
        rho = _finite_intermediate(
            "rho",
            -inputs.discounted_strike
            * (inputs.time_to_expiry_years * _normal_cdf(-inputs.d2)),
        )

    theta_year = _finite_sum(
        "theta", diffusion_theta, rate_theta, dividend_theta
    )
    result = BlackScholesGreeks(
        delta=_finite_intermediate("delta", delta),
        gamma=_finite_intermediate("gamma", gamma),
        theta=_finite_intermediate(
            "theta", theta_year / _THETA_DAYS_PER_YEAR
        ),
        vega=vega,
        rho=rho,
    )
    return result


def _price_from_prepared(
    option_right: OptionRight,
    inputs: _PreparedInputs,
) -> float:
    # Price the out-of-the-money side directly to reduce cancellation, then
    # derive the in-the-money side from European put-call parity.
    if inputs.discounted_spot <= inputs.discounted_strike:
        call_price = max(
            0.0,
            _finite_sum(
                "option price",
                inputs.discounted_spot * _normal_cdf(inputs.d1),
                -inputs.discounted_strike * _normal_cdf(inputs.d2),
            ),
        )
        put_price = _finite_sum(
            "option price",
            call_price,
            inputs.discounted_strike,
            -inputs.discounted_spot,
        )
    else:
        put_price = max(
            0.0,
            _finite_sum(
                "option price",
                inputs.discounted_strike * _normal_cdf(-inputs.d2),
                -inputs.discounted_spot * _normal_cdf(-inputs.d1),
            ),
        )
        call_price = _finite_sum(
            "option price",
            put_price,
            inputs.discounted_spot,
            -inputs.discounted_strike,
        )

    price = call_price if option_right is OptionRight.CALL else put_price
    return _finite_intermediate("option price", price)


def _prepare_inputs(
    *,
    spot: int | float | Decimal,
    strike: int | float | Decimal,
    time_to_expiry_years: int | float | Decimal,
    risk_free_rate: int | float | Decimal,
    dividend_yield: int | float | Decimal,
    volatility: int | float | Decimal,
) -> _PreparedInputs:
    spot_value = _positive_finite("spot", spot)
    strike_value = _positive_finite("strike", strike)
    time_value = _positive_finite("time_to_expiry_years", time_to_expiry_years)
    risk_free_value = _finite_number("risk_free_rate", risk_free_rate)
    dividend_value = _finite_number("dividend_yield", dividend_yield)
    volatility_value = _positive_finite("volatility", volatility)

    sqrt_time = sqrt(time_value)
    volatility_time = _finite_intermediate(
        "volatility-time scale", volatility_value * sqrt_time
    )
    if volatility_time <= 0.0:
        raise ValueError("volatility-time scale is outside the supported range")

    rate_spread = _finite_intermediate(
        "rate spread", risk_free_value - dividend_value
    )
    variance = _finite_intermediate(
        "volatility variance", volatility_value * volatility_value
    )
    drift_time = _finite_intermediate(
        "drift-time product",
        (rate_spread + 0.5 * variance) * time_value,
    )
    spot_strike_difference = spot_value - strike_value
    if abs(spot_strike_difference) <= strike_value:
        # log1p preserves precision for near-the-money inputs. Sterbenz's
        # lemma makes the subtraction exact when the positive operands are
        # within a factor of two.
        log_moneyness = log1p(spot_strike_difference / strike_value)
    else:
        # Avoid forming spot/strike when that ratio could overflow or underflow.
        log_moneyness = log(spot_value) - log(strike_value)
    d1 = _finite_intermediate(
        "d1", (log_moneyness + drift_time) / volatility_time
    )
    d2 = _finite_intermediate("d2", d1 - volatility_time)

    discounted_spot = _finite_intermediate(
        "discounted spot",
        spot_value * _discount_factor(dividend_value, time_value, "dividend_yield"),
    )
    discounted_strike = _finite_intermediate(
        "discounted strike",
        strike_value * _discount_factor(risk_free_value, time_value, "risk_free_rate"),
    )
    if discounted_spot <= 0.0 or discounted_strike <= 0.0:
        raise ValueError("discounted spot and strike must remain positive")

    return _PreparedInputs(
        spot=spot_value,
        strike=strike_value,
        time_to_expiry_years=time_value,
        risk_free_rate=risk_free_value,
        dividend_yield=dividend_value,
        volatility=volatility_value,
        sqrt_time=sqrt_time,
        volatility_time=volatility_time,
        discounted_spot=discounted_spot,
        discounted_strike=discounted_strike,
        d1=d1,
        d2=d2,
    )


def _discount_factor(rate: float, time: float, name: str) -> float:
    rate_time = _finite_intermediate(f"{name} time product", rate * time)
    try:
        factor = exp(-rate_time)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the supported numeric range") from exc
    if not isfinite(factor) or factor <= 0.0:
        raise ValueError(f"{name} is outside the supported numeric range")
    return factor


def _option_right(value: OptionRight | str) -> OptionRight:
    try:
        return OptionRight(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("option_right must be 'call' or 'put'") from exc


def _finite_number(name: str, value: int | float | Decimal) -> float:
    if isinstance(value, bool) or not isinstance(value, (Real, Decimal)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _positive_finite(name: str, value: int | float | Decimal) -> float:
    result = _finite_number(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_intermediate(name: str, value: float) -> float:
    if not isfinite(value):
        raise ValueError(f"{name} is outside the supported numeric range")
    return value


def _finite_sum(name: str, *values: float) -> float:
    try:
        result = fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is outside the supported numeric range") from exc
    return _finite_intermediate(name, result)


def _normal_cdf(value: float) -> float:
    """Normal CDF evaluated with erfc to retain accuracy in both tails."""
    return 0.5 * erfc(-value / _SQRT_2)


def _normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / _SQRT_2_PI
