"""Immutable inputs and outputs for deterministic strategy research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite
from numbers import Real

from atlas.data.ohlcv import OHLCVBar
from atlas.derivatives.models import UnderlyingIndex
from atlas.market_data.models import HistoricalOHLCVResponse


class SignalDirection(str, Enum):
    """Directional research result; NO_TRADE never means an order instruction."""

    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


class SignalStatus(str, Enum):
    READY = "ready"
    NO_TRADE = "no_trade"
    MISSING_DATA = "missing_data"
    INSUFFICIENT_HISTORY = "insufficient_history"
    INCOMPLETE_CANDLE = "incomplete_candle"
    INVALID_DATA = "invalid_data"


class WarmupStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    NOT_ASSESSED = "not_assessed"


class SignalEvent(str, Enum):
    """A strategy condition that fired; exits only describe an existing thesis."""

    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"


_VALIDATED_PROVIDER_TOKEN = object()


@dataclass(frozen=True, init=False)
class StrategyMarketData:
    """Validated historical bars plus caller-supplied context absent from OHLCV.

    The research engine obtains the response from ValidatedMarketDataProvider.
    Timeframe, source identity, evaluation time, and last-candle completeness
    are explicit because the historical response model lacks those facts.
    """

    response: HistoricalOHLCVResponse
    underlying: UnderlyingIndex
    timeframe: timedelta
    data_source: str
    evaluation_at: datetime
    last_bar_complete: bool

    def __init__(
        self,
        response: HistoricalOHLCVResponse,
        underlying: UnderlyingIndex,
        timeframe: timedelta,
        data_source: str,
        evaluation_at: datetime,
        last_bar_complete: bool,
        *,
        _validated_provider_token: object | None = None,
    ) -> None:
        if _validated_provider_token is not _VALIDATED_PROVIDER_TOKEN:
            raise TypeError(
                "StrategyMarketData is created by StrategyResearchEngine from validated data"
            )
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "underlying", underlying)
        object.__setattr__(self, "timeframe", timeframe)
        object.__setattr__(self, "data_source", data_source)
        object.__setattr__(self, "evaluation_at", evaluation_at)
        object.__setattr__(self, "last_bar_complete", last_bar_complete)
        self.__post_init__()

    @classmethod
    def _from_validated_provider(
        cls,
        *,
        response: HistoricalOHLCVResponse,
        underlying: UnderlyingIndex,
        timeframe: timedelta,
        data_source: str,
        evaluation_at: datetime,
        last_bar_complete: bool,
    ) -> StrategyMarketData:
        return cls(
            response=response,
            underlying=underlying,
            timeframe=timeframe,
            data_source=data_source,
            evaluation_at=evaluation_at,
            last_bar_complete=last_bar_complete,
            _validated_provider_token=_VALIDATED_PROVIDER_TOKEN,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.response, HistoricalOHLCVResponse):
            raise ValueError("response must be a HistoricalOHLCVResponse")
        if not isinstance(self.underlying, UnderlyingIndex):
            raise ValueError("underlying must be an UnderlyingIndex")
        if not isinstance(self.timeframe, timedelta) or self.timeframe <= timedelta(0):
            raise ValueError("timeframe must be a positive timedelta")
        if not isinstance(self.data_source, str) or not self.data_source.strip():
            raise ValueError("data_source must be a non-empty string")
        if not isinstance(self.evaluation_at, datetime):
            raise ValueError("evaluation_at must be a datetime")
        if not isinstance(self.last_bar_complete, bool):
            raise ValueError("last_bar_complete must be a bool")
        if not isinstance(self.response.symbol, str) or not self.response.symbol.strip():
            raise ValueError("response symbol must be a non-empty string")
        if not isinstance(self.response.start, datetime) or not isinstance(
            self.response.end, datetime
        ):
            raise ValueError("response bounds must be datetimes")
        try:
            if self.response.start > self.response.end:
                raise ValueError("response start must not follow response end")
            if self.response.end > self.evaluation_at:
                raise ValueError("response end must not be later than evaluation_at")
        except TypeError as exc:
            raise ValueError("response bounds and evaluation_at must be comparable") from exc

        if not self.response.bars:
            raise ValueError("response must contain at least one bar")

        previous_timestamp: datetime | None = None
        for index, bar in enumerate(self.response.bars):
            if not isinstance(bar, OHLCVBar):
                raise ValueError(f"bar at index {index} must be an OHLCVBar")
            if bar.symbol != self.response.symbol:
                raise ValueError(f"bar at index {index} has a different symbol")
            if not isinstance(bar.timestamp, datetime):
                raise ValueError(f"bar timestamp at index {index} must be a datetime")
            try:
                if not self.response.start <= bar.timestamp <= self.response.end:
                    raise ValueError(f"bar at index {index} is outside response bounds")
                if bar.timestamp > self.evaluation_at:
                    raise ValueError(f"bar at index {index} is later than evaluation_at")
                if previous_timestamp is not None and bar.timestamp <= previous_timestamp:
                    raise ValueError("bar timestamps must be unique and strictly increasing")
            except TypeError as exc:
                raise ValueError("bar timestamps and bounds must be comparable") from exc
            previous_timestamp = bar.timestamp

            prices: dict[str, float] = {}
            for field in ("open", "high", "low", "close"):
                value = getattr(bar, field)
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(f"{field} at index {index} must be finite and positive")
                try:
                    numeric = float(value)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{field} at index {index} must be finite and positive"
                    ) from exc
                if not isfinite(numeric) or numeric <= 0.0:
                    raise ValueError(f"{field} at index {index} must be finite and positive")
                prices[field] = numeric
            if prices["high"] < max(prices["open"], prices["low"], prices["close"]):
                raise ValueError(f"bar at index {index} has inconsistent OHLC prices")
            if prices["low"] > min(prices["open"], prices["high"], prices["close"]):
                raise ValueError(f"bar at index {index} has inconsistent OHLC prices")
            if (
                isinstance(bar.volume, bool)
                or not isinstance(bar.volume, int)
                or bar.volume < 0
            ):
                raise ValueError(f"volume at index {index} must be a non-negative integer")

    @property
    def bars(self) -> tuple[OHLCVBar, ...]:
        return self.response.bars

    @property
    def symbol(self) -> str:
        return self.response.symbol

    @property
    def is_synthetic(self) -> bool:
        return self.response.is_synthetic

    @property
    def observation_timestamp(self) -> datetime:
        return self.bars[-1].timestamp


@dataclass(frozen=True)
class IndicatorValue:
    name: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("indicator name must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise ValueError("indicator value must be finite")
        try:
            numeric = float(self.value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("indicator value must be finite") from exc
        if not isfinite(numeric):
            raise ValueError("indicator value must be finite")


@dataclass(frozen=True)
class StrategySignal:
    """Auditable deterministic signal; this is not an order or profit forecast."""

    strategy_name: str
    strategy_version: str
    underlying: UnderlyingIndex
    input_symbol: str
    timeframe: timedelta
    observation_timestamp: datetime | None
    direction: SignalDirection
    events: tuple[SignalEvent, ...]
    status: SignalStatus
    warmup_status: WarmupStatus
    indicator_values: tuple[IndicatorValue, ...]
    entry_reference_price: float | None
    stop_reference_price: float | None
    entry_conditions: tuple[str, ...]
    exit_conditions: tuple[str, ...]
    stop_loss_method: str
    input_requirements: tuple[str, ...]
    reasons: tuple[str, ...]
    missing_data: tuple[str, ...]
    invalid_conditions: tuple[str, ...]
    data_source: str
    is_synthetic: bool | None
    used_incomplete_candle: bool
    evaluation_at: datetime

    def __post_init__(self) -> None:
        for name in ("strategy_name", "strategy_version", "input_symbol", "data_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.underlying, UnderlyingIndex):
            raise ValueError("underlying must be an UnderlyingIndex")
        if not isinstance(self.timeframe, timedelta) or self.timeframe <= timedelta(0):
            raise ValueError("timeframe must be a positive timedelta")
        if self.observation_timestamp is not None and not isinstance(
            self.observation_timestamp, datetime
        ):
            raise ValueError("observation_timestamp must be a datetime or None")
        if not isinstance(self.evaluation_at, datetime):
            raise ValueError("evaluation_at must be a datetime")
        if not isinstance(self.direction, SignalDirection):
            raise ValueError("direction must be a SignalDirection")
        if not isinstance(self.status, SignalStatus):
            raise ValueError("status must be a SignalStatus")
        if not isinstance(self.warmup_status, WarmupStatus):
            raise ValueError("warmup_status must be a WarmupStatus")
        if self.is_synthetic is not None and not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool or None")
        if not isinstance(self.used_incomplete_candle, bool):
            raise ValueError("used_incomplete_candle must be a bool")
        for name in (
            "events",
            "indicator_values",
            "entry_conditions",
            "exit_conditions",
            "input_requirements",
            "reasons",
            "missing_data",
            "invalid_conditions",
        ):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be a tuple")
        if any(not isinstance(event, SignalEvent) for event in self.events):
            raise ValueError("events must contain SignalEvent values")
        if self.direction is SignalDirection.LONG and SignalEvent.ENTER_LONG not in self.events:
            raise ValueError("LONG signals must include an ENTER_LONG event")
        if self.direction is SignalDirection.SHORT and SignalEvent.ENTER_SHORT not in self.events:
            raise ValueError("SHORT signals must include an ENTER_SHORT event")
        if self.direction is SignalDirection.NO_TRADE and any(
            event in (SignalEvent.ENTER_LONG, SignalEvent.ENTER_SHORT)
            for event in self.events
        ):
            raise ValueError("NO_TRADE signals cannot include an entry event")
        if any(not isinstance(item, IndicatorValue) for item in self.indicator_values):
            raise ValueError("indicator_values must contain IndicatorValue values")
        if any(
            not isinstance(item, str) or not item.strip()
            for name in (
                "entry_conditions",
                "exit_conditions",
                "input_requirements",
                "reasons",
                "missing_data",
                "invalid_conditions",
            )
            for item in getattr(self, name)
        ):
            raise ValueError("signal descriptions must be non-empty strings")
        if not isinstance(self.stop_loss_method, str) or not self.stop_loss_method.strip():
            raise ValueError("stop_loss_method must be a non-empty string")
        for name in ("entry_reference_price", "stop_reference_price"):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(f"{name} must be finite and positive or None")
                try:
                    numeric = float(value)
                except (OverflowError, TypeError, ValueError) as exc:
                    raise ValueError(f"{name} must be finite and positive or None") from exc
                if not isfinite(numeric) or numeric <= 0.0:
                    raise ValueError(f"{name} must be finite and positive or None")
        if self.direction is SignalDirection.NO_TRADE and (
            self.entry_reference_price is not None or self.stop_reference_price is not None
        ):
            raise ValueError("NO_TRADE signals cannot contain entry or stop reference prices")
        if self.direction is not SignalDirection.NO_TRADE and (
            self.entry_reference_price is None
        ):
            raise ValueError("directional signals require an entry reference price")


__all__ = [
    "IndicatorValue",
    "SignalDirection",
    "SignalEvent",
    "SignalStatus",
    "StrategyMarketData",
    "StrategySignal",
    "WarmupStatus",
]
