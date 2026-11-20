"""Relative strength index using Wilder smoothing."""

from __future__ import annotations

import math
from collections.abc import Sequence


def rsi(closing_prices: Sequence[float], period: int) -> list[float]:
    """Return Wilder's RSI values, beginning after the first ``period`` changes."""
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")

    try:
        prices = tuple(closing_prices)
    except TypeError as exc:
        raise ValueError("closing_prices must be a sequence of positive numbers") from exc

    if len(prices) < period + 1:
        raise ValueError(
            f"at least {period + 1} closing prices are required; received {len(prices)}"
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

    changes = [
        numeric_prices[index] - numeric_prices[index - 1]
        for index in range(1, len(numeric_prices))
    ]
    gains = [max(change, 0.0) for change in changes]
    losses = [max(-change, 0.0) for change in changes]

    average_gain = sum(gain / period for gain in gains[:period])
    average_loss = sum(loss / period for loss in losses[:period])
    values = [_calculate_rsi(average_gain, average_loss)]

    for gain, loss in zip(gains[period:], losses[period:]):
        average_gain += (gain - average_gain) / period
        average_loss += (loss - average_loss) / period
        values.append(_calculate_rsi(average_gain, average_loss))

    return values


def _calculate_rsi(average_gain: float, average_loss: float) -> float:
    if average_gain == 0 and average_loss == 0:
        return 50.0
    if average_loss == 0:
        return 100.0
    if average_gain == 0:
        return 0.0
    relative_strength = average_gain / average_loss
    return 100.0 - (100.0 / (1.0 + relative_strength))
