"""Average True Range calculation using Wilder smoothing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from atlas.data.ohlcv import OHLCVBar


def atr(bars: Sequence[OHLCVBar], period: int) -> list[float]:
    """Return ATR values, aligned with bars starting at ``bars[period]``."""
    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise ValueError("period must be a positive integer")

    try:
        records = tuple(bars)
    except TypeError as exc:
        raise ValueError("bars must be a sequence of OHLCVBar objects") from exc

    if len(records) < period + 1:
        raise ValueError(
            f"at least {period + 1} bars are required; received {len(records)}"
        )

    validated: list[tuple[datetime, float, float, float, float]] = []
    previous_timestamp: datetime | None = None
    symbol: str | None = None
    for index, bar in enumerate(records):
        if not isinstance(bar, OHLCVBar):
            raise ValueError(f"bar at index {index} must be an OHLCVBar")
        if index == 0:
            symbol = bar.symbol
        elif bar.symbol != symbol:
            raise ValueError("all bars must have the same symbol")
        if not isinstance(bar.timestamp, datetime):
            raise ValueError(f"timestamp at index {index} must be a datetime")
        if previous_timestamp is not None:
            try:
                is_not_increasing = bar.timestamp <= previous_timestamp
            except TypeError as exc:
                raise ValueError("bar timestamps must be comparable and strictly increasing") from exc
            if is_not_increasing:
                raise ValueError("bar timestamps must be strictly increasing")
        previous_timestamp = bar.timestamp

        prices = []
        for field in ("open", "high", "low", "close"):
            raw_price = getattr(bar, field)
            if isinstance(raw_price, (bool, str, bytes, bytearray)):
                raise ValueError(f"{field} at index {index} must be finite and positive")
            try:
                price = float(raw_price)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"{field} at index {index} must be finite and positive"
                ) from exc
            if not math.isfinite(price) or price <= 0:
                raise ValueError(f"{field} at index {index} must be finite and positive")
            prices.append(price)

        open_price, high, low, close = prices
        if high < max(open_price, low, close) or low > min(open_price, high, close):
            raise ValueError(f"bar at index {index} has inconsistent OHLC prices")
        validated.append((bar.timestamp, open_price, high, low, close))

    true_ranges = []
    for index in range(1, len(validated)):
        previous_close = validated[index - 1][4]
        _, _, high, low, _ = validated[index]
        true_ranges.append(
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        )

    average_true_range = sum(value / period for value in true_ranges[:period])
    values = [average_true_range]
    for current_true_range in true_ranges[period:]:
        average_true_range += (current_true_range - average_true_range) / period
        values.append(average_true_range)

    return values
