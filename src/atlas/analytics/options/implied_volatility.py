"""Deterministic implied-volatility inversion for European Black-Scholes.

The default search interval is annualized volatility [1e-8, 5.0] (up to
500%). This is a numerical solver interval, not an assumed market volatility.
The solver uses bisection, an absolute volatility-width tolerance of 1e-10
and at most 256 iterations. Inputs outside the interval or without a finite
solution are rejected rather than assigned a fallback volatility.
"""

from __future__ import annotations

from decimal import Decimal
from atlas.analytics.options.black_scholes import (
    OptionRight,
    _discount_factor,
    _finite_number,
    _finite_sum,
    _option_right,
    _positive_finite,
    black_scholes_price,
)


def implied_volatility(
    option_right: OptionRight | str,
    *,
    option_price: int | float | Decimal,
    spot: int | float | Decimal,
    strike: int | float | Decimal,
    time_to_expiry_years: int | float | Decimal,
    risk_free_rate: int | float | Decimal,
    dividend_yield: int | float | Decimal,
    min_volatility: int | float | Decimal = 1e-8,
    max_volatility: int | float | Decimal = 5.0,
    volatility_tolerance: int | float | Decimal = 1e-10,
    max_iterations: int = 256,
) -> float:
    """Infer annualized volatility from a European option model price.

    Prices are per one underlying unit. Rates and dividend yield are
    continuously compounded annual decimals, and expiry is in years. The
    returned volatility is an annualized decimal (for example, ``0.2`` is
    20%). Market inputs have no defaults.

    The valid European no-arbitrage interval is checked before solving. A
    price at either theoretical bound has no finite strictly positive
    volatility solution and is rejected. ``ValueError`` indicates invalid,
    impossible, or out-of-search-interval input; ``RuntimeError`` indicates
    that bisection did not converge within ``max_iterations``.
    """
    right = _option_right(option_right)
    market_price = _positive_finite("option_price", option_price)
    spot_value = _positive_finite("spot", spot)
    strike_value = _positive_finite("strike", strike)
    time_value = _positive_finite("time_to_expiry_years", time_to_expiry_years)
    rate_value = _finite_number("risk_free_rate", risk_free_rate)
    dividend_value = _finite_number("dividend_yield", dividend_yield)
    low = _positive_finite("min_volatility", min_volatility)
    high = _positive_finite("max_volatility", max_volatility)
    volatility_tol = _positive_finite(
        "volatility_tolerance", volatility_tolerance
    )
    if low >= high:
        raise ValueError("min_volatility must be less than max_volatility")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    discounted_spot = spot_value * _discount_factor(
        dividend_value, time_value, "dividend_yield"
    )
    discounted_strike = strike_value * _discount_factor(
        rate_value, time_value, "risk_free_rate"
    )
    if right is OptionRight.CALL:
        lower_bound = max(
            0.0,
            _finite_sum("option price bound", discounted_spot, -discounted_strike),
        )
        upper_bound = discounted_spot
    else:
        lower_bound = max(
            0.0,
            _finite_sum("option price bound", discounted_strike, -discounted_spot),
        )
        upper_bound = discounted_strike
    if market_price <= lower_bound or market_price >= upper_bound:
        raise ValueError("option_price is outside the finite-volatility bounds")

    model_args = {
        "spot": spot_value,
        "strike": strike_value,
        "time_to_expiry_years": time_value,
        "risk_free_rate": rate_value,
        "dividend_yield": dividend_value,
    }
    low_price = black_scholes_price(
        right, **model_args, volatility=low
    )
    high_price = black_scholes_price(
        right, **model_args, volatility=high
    )
    if market_price < low_price or market_price > high_price:
        raise ValueError("option_price has no solution in the volatility interval")
    if market_price == low_price:
        return low
    if market_price == high_price:
        return high

    left = low
    right_edge = high
    for _ in range(max_iterations):
        midpoint = left + (right_edge - left) / 2.0
        model_price = black_scholes_price(
            right, **model_args, volatility=midpoint
        )
        if model_price < market_price:
            left = midpoint
        else:
            right_edge = midpoint
        if right_edge - left <= volatility_tol:
            return left + (right_edge - left) / 2.0

    raise RuntimeError("implied volatility solver did not converge")


__all__ = ["implied_volatility"]
