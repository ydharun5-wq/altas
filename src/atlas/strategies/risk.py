"""ATR-based stop-distance helpers; these do not size positions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Sequence

from atlas.data.ohlcv import OHLCVBar
from atlas.indicators.atr import atr
from atlas.strategies.models import SignalDirection


@dataclass(frozen=True)
class ATRStopConfig:
    """Explicit Wilder ATR period and positive distance multiplier."""

    period: int
    multiplier: int | float

    def __post_init__(self) -> None:
        if isinstance(self.period, bool) or not isinstance(self.period, int) or self.period <= 0:
            raise ValueError("ATR period must be a positive integer")
        if isinstance(self.multiplier, bool) or not isinstance(self.multiplier, Real):
            raise ValueError("ATR multiplier must be finite and positive")
        try:
            multiplier = float(self.multiplier)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("ATR multiplier must be finite and positive") from exc
        if not isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("ATR multiplier must be finite and positive")


def calculate_atr_stop_distance(
    bars: Sequence[OHLCVBar],
    *,
    period: int,
    multiplier: int | float,
) -> float:
    """Return latest Wilder ATR multiplied by an explicit multiplier.

    The result is in the same price units as OHLCV inputs. No position size is
    inferred. The caller must pass only bars that are valid for the analysis
    timestamp and whose candle-completion status is known.
    """
    config = ATRStopConfig(period=period, multiplier=multiplier)
    values = atr(bars, config.period)
    distance = values[-1] * float(config.multiplier)
    if not isfinite(distance) or distance <= 0.0:
        raise ValueError("ATR stop distance must be finite and positive")
    return distance


def calculate_stop_reference_price(
    direction: SignalDirection,
    *,
    entry_reference_price: int | float,
    stop_distance: int | float,
) -> float:
    """Return a long stop below entry or a short stop above entry."""
    if direction not in (SignalDirection.LONG, SignalDirection.SHORT):
        raise ValueError("direction must be LONG or SHORT")
    entry = _positive_finite("entry_reference_price", entry_reference_price)
    distance = _positive_finite("stop_distance", stop_distance)
    stop = entry - distance if direction is SignalDirection.LONG else entry + distance
    if not isfinite(stop) or stop <= 0.0:
        raise ValueError("stop reference price must be finite and positive")
    return stop


def _positive_finite(name: str, value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


__all__ = [
    "ATRStopConfig",
    "calculate_atr_stop_distance",
    "calculate_stop_reference_price",
]
