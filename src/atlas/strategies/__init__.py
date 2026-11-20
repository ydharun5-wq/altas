"""Deterministic research strategies; no backtesting or execution support."""

from atlas.strategies.engine import StrategyResearchEngine
from atlas.strategies.models import (
    IndicatorValue,
    SignalDirection,
    SignalEvent,
    SignalStatus,
    StrategySignal,
    WarmupStatus,
)
from atlas.strategies.risk import (
    ATRStopConfig,
    calculate_atr_stop_distance,
    calculate_stop_reference_price,
)
from atlas.strategies.strategies import (
    EMATrendStrategy,
    RSIMeanReversionStrategy,
    Strategy,
)

__all__ = [
    "ATRStopConfig",
    "EMATrendStrategy",
    "IndicatorValue",
    "RSIMeanReversionStrategy",
    "SignalDirection",
    "SignalEvent",
    "SignalStatus",
    "Strategy",
    "StrategyResearchEngine",
    "StrategySignal",
    "WarmupStatus",
    "calculate_atr_stop_distance",
    "calculate_stop_reference_price",
]
