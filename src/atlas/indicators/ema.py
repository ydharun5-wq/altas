"""Exponential moving average calculation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def ema(closing_prices: Sequence[float], period: int) -> list[float]:
    """Return the EMA values, starting with the SMA seed at ``period - 1``."""
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")

    try:
        prices = tuple(closing_prices)
    except TypeError as exc:
        raise ValueError("closing_prices must be a sequence of positive numbers") from exc

    if len(prices) < period:
        raise ValueError(
            f"at least {period} closing prices are required; received {len(prices)}"
        )

    numeric_prices: list[float] = []
    for index, price in enumerate(prices):
        if isinstance(price, (bool, str, bytes, bytearray)):
            raise ValueError(f"closing price at index {index} must be finite and positive")
        try:
            numeric_price = float(price)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"closing price at index {index} must be finite and positive"
            ) from exc
        if not math.isfinite(numeric_price) or numeric_price <= 0:
            raise ValueError(f"closing price at index {index} must be finite and positive")
        numeric_prices.append(numeric_price)

    seed_prices = numeric_prices[:period]
    scale = max(seed_prices)
    seed = scale * (sum(price / scale for price in seed_prices) / period)
    multiplier = 2.0 / (period + 1)

    values = [seed]
    for price in numeric_prices[period:]:
        previous = values[-1]
        values.append(previous + multiplier * (price - previous))
    return values
