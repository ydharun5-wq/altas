"""Deterministic EMA trend and RSI mean-reversion strategies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Protocol, runtime_checkable

from atlas.derivatives.models import UnderlyingIndex
from atlas.indicators.ema import ema
from atlas.indicators.rsi import rsi
from atlas.strategies.models import (
    IndicatorValue,
    SignalDirection,
    SignalEvent,
    SignalStatus,
    StrategyMarketData,
    StrategySignal,
    WarmupStatus,
)
from atlas.strategies.risk import (
    ATRStopConfig,
    calculate_atr_stop_distance,
    calculate_stop_reference_price,
)


@runtime_checkable
class Strategy(Protocol):
    """Provider-neutral deterministic strategy interface."""

    name: str
    version: str
    underlying: UnderlyingIndex
    symbol: str
    timeframe: timedelta
    allow_incomplete_candles: bool
    supports_short: bool

    @property
    def input_requirements(self) -> tuple[str, ...]: ...

    @property
    def entry_conditions(self) -> tuple[str, ...]: ...

    @property
    def exit_conditions(self) -> tuple[str, ...]: ...

    @property
    def stop_loss_method(self) -> str: ...

    def is_compatible_with(
        self, underlying: UnderlyingIndex, timeframe: timedelta, symbol: str
    ) -> bool: ...

    def evaluate(self, data: StrategyMarketData) -> StrategySignal: ...


@dataclass(frozen=True)
class EMATrendStrategy:
    """Enter on a fast/slow EMA crossover; use no default periods or timeframe."""

    underlying: UnderlyingIndex
    symbol: str
    timeframe: timedelta
    fast_period: int
    slow_period: int
    stop: ATRStopConfig | None = None
    allow_incomplete_candles: bool = False

    name = "ema_trend"
    version = "1.0"
    supports_short = True

    def __post_init__(self) -> None:
        _validate_binding(self.underlying, self.symbol, self.timeframe)
        _validate_period("fast_period", self.fast_period)
        _validate_period("slow_period", self.slow_period)
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be lower than slow_period")
        _validate_stop(self.stop)
        _validate_incomplete_setting(self.allow_incomplete_candles)

    @property
    def required_history(self) -> int:
        periods = [self.slow_period + 1]
        if self.stop is not None:
            periods.append(self.stop.period + 1)
        return max(periods)

    @property
    def input_requirements(self) -> tuple[str, ...]:
        return (
            f"at least {self.required_history} chronological OHLCV bars",
            "explicit timeframe, source identity, evaluation timestamp, and candle completeness",
            "fast and slow EMA values on the latest two bars",
        )

    @property
    def entry_conditions(self) -> tuple[str, ...]:
        return (
            "Enter long when fast EMA crosses from at-or-below to above slow EMA.",
            "Enter short when fast EMA crosses from at-or-above to below slow EMA.",
        )

    @property
    def exit_conditions(self) -> tuple[str, ...]:
        return (
            "A downward crossover invalidates an existing long thesis.",
            "An upward crossover invalidates an existing short thesis.",
            "Exit conditions describe research conditions; no position state is managed.",
        )

    @property
    def stop_loss_method(self) -> str:
        if self.stop is None:
            return "not configured; no stop reference is produced"
        return f"Wilder ATR({self.stop.period}) x {self.stop.multiplier} price units"

    def is_compatible_with(
        self, underlying: UnderlyingIndex, timeframe: timedelta, symbol: str
    ) -> bool:
        return (
            self.underlying is underlying
            and self.timeframe == timeframe
            and self.symbol == symbol
        )

    def evaluate(self, data: StrategyMarketData) -> StrategySignal:
        _validate_compatibility(self, data)
        blocked = _precondition_signal(self, data)
        if blocked is not None:
            return blocked

        try:
            closes = [bar.close for bar in data.bars]
            fast_values = ema(closes, self.fast_period)
            slow_values = ema(closes, self.slow_period)
        except (OverflowError, ValueError):
            return _invalid_calculation_signal(self, data)
        previous_fast, current_fast = fast_values[-2], fast_values[-1]
        previous_slow, current_slow = slow_values[-2], slow_values[-1]
        indicators = (
            IndicatorValue("fast_ema_previous", previous_fast),
            IndicatorValue("fast_ema", current_fast),
            IndicatorValue("slow_ema_previous", previous_slow),
            IndicatorValue("slow_ema", current_slow),
        )

        direction = SignalDirection.NO_TRADE
        events: tuple[SignalEvent, ...] = ()
        reason = "No EMA crossover occurred on the latest two evaluated bars."
        if previous_fast <= previous_slow and current_fast > current_slow:
            direction = SignalDirection.LONG
            events = (SignalEvent.EXIT_SHORT, SignalEvent.ENTER_LONG)
            reason = "Fast EMA crossed above slow EMA on the latest evaluated bar."
        elif previous_fast >= previous_slow and current_fast < current_slow:
            direction = SignalDirection.SHORT
            events = (SignalEvent.EXIT_LONG, SignalEvent.ENTER_SHORT)
            reason = "Fast EMA crossed below slow EMA on the latest evaluated bar."

        return _signal_for_result(
            self,
            data,
            direction=direction,
            events=events,
            indicators=indicators,
            reasons=(reason,),
            warmup_status=WarmupStatus.SUFFICIENT,
        )


@dataclass(frozen=True)
class RSIMeanReversionStrategy:
    """Threshold-crossing research rules; extremes do not guarantee reversals."""

    underlying: UnderlyingIndex
    symbol: str
    timeframe: timedelta
    period: int
    oversold_threshold: float
    overbought_threshold: float
    long_exit_threshold: float
    short_exit_threshold: float
    stop: ATRStopConfig | None = None
    allow_incomplete_candles: bool = False

    name = "rsi_mean_reversion"
    version = "1.0"
    supports_short = True

    def __post_init__(self) -> None:
        _validate_binding(self.underlying, self.symbol, self.timeframe)
        _validate_period("period", self.period)
        thresholds = (
            ("oversold_threshold", self.oversold_threshold),
            ("overbought_threshold", self.overbought_threshold),
            ("long_exit_threshold", self.long_exit_threshold),
            ("short_exit_threshold", self.short_exit_threshold),
        )
        for name, value in thresholds:
            _validate_rsi_threshold(name, value)
        if not (
            self.oversold_threshold
            < self.long_exit_threshold
            <= self.short_exit_threshold
            < self.overbought_threshold
        ):
            raise ValueError(
                "thresholds must satisfy oversold < long_exit <= short_exit < overbought"
            )
        _validate_stop(self.stop)
        _validate_incomplete_setting(self.allow_incomplete_candles)

    @property
    def required_history(self) -> int:
        periods = [self.period + 2]
        if self.stop is not None:
            periods.append(self.stop.period + 1)
        return max(periods)

    @property
    def input_requirements(self) -> tuple[str, ...]:
        return (
            f"at least {self.required_history} chronological OHLCV bars",
            "explicit timeframe, source identity, evaluation timestamp, and candle completeness",
            "RSI values on the latest two bars",
        )

    @property
    def entry_conditions(self) -> tuple[str, ...]:
        return (
            f"Enter long when RSI crosses down through {self.oversold_threshold}.",
            f"Enter short when RSI crosses up through {self.overbought_threshold}.",
            "An RSI extreme is descriptive and does not guarantee a reversal.",
        )

    @property
    def exit_conditions(self) -> tuple[str, ...]:
        return (
            f"Exit-long condition: RSI crosses up through {self.long_exit_threshold}.",
            f"Exit-short condition: RSI crosses down through {self.short_exit_threshold}.",
            "Exit conditions describe research conditions; no position state is managed.",
        )

    @property
    def stop_loss_method(self) -> str:
        if self.stop is None:
            return "not configured; no stop reference is produced"
        return f"Wilder ATR({self.stop.period}) x {self.stop.multiplier} price units"

    def is_compatible_with(
        self, underlying: UnderlyingIndex, timeframe: timedelta, symbol: str
    ) -> bool:
        return (
            self.underlying is underlying
            and self.timeframe == timeframe
            and self.symbol == symbol
        )

    def evaluate(self, data: StrategyMarketData) -> StrategySignal:
        _validate_compatibility(self, data)
        blocked = _precondition_signal(self, data)
        if blocked is not None:
            return blocked

        try:
            values = rsi([bar.close for bar in data.bars], self.period)
        except (OverflowError, ValueError):
            return _invalid_calculation_signal(self, data)
        previous_rsi, current_rsi = values[-2], values[-1]
        indicators = (
            IndicatorValue("rsi_previous", previous_rsi),
            IndicatorValue("rsi", current_rsi),
        )
        events: list[SignalEvent] = []
        direction = SignalDirection.NO_TRADE

        if previous_rsi > self.oversold_threshold and current_rsi <= self.oversold_threshold:
            events.append(SignalEvent.ENTER_LONG)
            direction = SignalDirection.LONG
        if previous_rsi < self.overbought_threshold and current_rsi >= self.overbought_threshold:
            events.append(SignalEvent.ENTER_SHORT)
            direction = SignalDirection.SHORT
        if previous_rsi < self.long_exit_threshold and current_rsi >= self.long_exit_threshold:
            events.append(SignalEvent.EXIT_LONG)
        if previous_rsi > self.short_exit_threshold and current_rsi <= self.short_exit_threshold:
            events.append(SignalEvent.EXIT_SHORT)

        if events:
            reason = "RSI crossed one or more configured entry/exit thresholds."
        else:
            reason = "No configured RSI entry or exit threshold crossed on the latest bar."
        reasons = (
            reason,
            "RSI threshold crossings are not proof that a market reversal will occur.",
        )
        return _signal_for_result(
            self,
            data,
            direction=direction,
            events=tuple(events),
            indicators=indicators,
            reasons=reasons,
            warmup_status=WarmupStatus.SUFFICIENT,
        )


def unavailable_signal(
    strategy: Strategy,
    *,
    input_symbol: str,
    data_source: str,
    evaluation_at: datetime,
    status: SignalStatus,
    reason: str,
    missing_data: tuple[str, ...] = (),
    invalid_conditions: tuple[str, ...] = (),
) -> StrategySignal:
    """Build a no-trade result when provider data is absent or rejected."""
    return StrategySignal(
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        underlying=strategy.underlying,
        input_symbol=input_symbol,
        timeframe=strategy.timeframe,
        observation_timestamp=None,
        direction=SignalDirection.NO_TRADE,
        events=(),
        status=status,
        warmup_status=WarmupStatus.NOT_ASSESSED,
        indicator_values=(),
        entry_reference_price=None,
        stop_reference_price=None,
        entry_conditions=strategy.entry_conditions,
        exit_conditions=strategy.exit_conditions,
        stop_loss_method=strategy.stop_loss_method,
        input_requirements=strategy.input_requirements,
        reasons=(reason,),
        missing_data=missing_data,
        invalid_conditions=invalid_conditions,
        data_source=data_source,
        is_synthetic=None,
        used_incomplete_candle=False,
        evaluation_at=evaluation_at,
    )


def _precondition_signal(
    strategy: Strategy, data: StrategyMarketData
) -> StrategySignal | None:
    if not data.last_bar_complete and not strategy.allow_incomplete_candles:
        return _signal_for_result(
            strategy,
            data,
            direction=SignalDirection.NO_TRADE,
            events=(),
            indicators=(),
            reasons=("Latest candle is incomplete; no signal was calculated from it.",),
            status=SignalStatus.INCOMPLETE_CANDLE,
            warmup_status=WarmupStatus.NOT_ASSESSED,
            missing_data=("latest candle is incomplete",),
            used_incomplete_candle=False,
        )
    if len(data.bars) < strategy.required_history:
        required = strategy.required_history
        return _signal_for_result(
            strategy,
            data,
            direction=SignalDirection.NO_TRADE,
            events=(),
            indicators=(),
            reasons=(f"Insufficient history: {required} bars are required.",),
            status=SignalStatus.INSUFFICIENT_HISTORY,
            warmup_status=WarmupStatus.INSUFFICIENT,
            missing_data=(f"need {required} bars; received {len(data.bars)}",),
            used_incomplete_candle=False,
        )
    return None


def _signal_for_result(
    strategy: Strategy,
    data: StrategyMarketData,
    *,
    direction: SignalDirection,
    events: tuple[SignalEvent, ...],
    indicators: tuple[IndicatorValue, ...],
    reasons: tuple[str, ...],
    status: SignalStatus | None = None,
    warmup_status: WarmupStatus,
    missing_data: tuple[str, ...] = (),
    invalid_conditions: tuple[str, ...] = (),
    used_incomplete_candle: bool | None = None,
) -> StrategySignal:
    if status is None:
        status = (
            SignalStatus.READY
            if direction is not SignalDirection.NO_TRADE or events
            else SignalStatus.NO_TRADE
        )
    entry_price: float | None = None
    stop_price: float | None = None
    final_indicators = list(indicators)
    if direction is not SignalDirection.NO_TRADE:
        entry_price = float(data.bars[-1].close)
        if strategy.stop is not None:
            try:
                distance = calculate_atr_stop_distance(
                    data.bars,
                    period=strategy.stop.period,
                    multiplier=strategy.stop.multiplier,
                )
                stop_price = calculate_stop_reference_price(
                    direction, entry_reference_price=entry_price, stop_distance=distance
                )
            except (OverflowError, ValueError):
                return _invalid_calculation_signal(
                    strategy,
                    data,
                    indicators=indicators,
                    reason="Configured ATR stop could not produce a valid price.",
                )
            final_indicators.append(IndicatorValue("atr_stop_distance", distance))

    if used_incomplete_candle is None:
        used_incomplete_candle = (
            not data.last_bar_complete and strategy.allow_incomplete_candles
        )
    return StrategySignal(
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        underlying=data.underlying,
        input_symbol=data.symbol,
        timeframe=data.timeframe,
        observation_timestamp=data.observation_timestamp,
        direction=direction,
        events=events,
        status=status,
        warmup_status=warmup_status,
        indicator_values=tuple(final_indicators),
        entry_reference_price=entry_price,
        stop_reference_price=stop_price,
        entry_conditions=strategy.entry_conditions,
        exit_conditions=strategy.exit_conditions,
        stop_loss_method=strategy.stop_loss_method,
        input_requirements=strategy.input_requirements,
        reasons=reasons,
        missing_data=missing_data,
        invalid_conditions=invalid_conditions,
        data_source=data.data_source,
        is_synthetic=data.is_synthetic,
        used_incomplete_candle=used_incomplete_candle,
        evaluation_at=data.evaluation_at,
    )


def _validate_binding(
    underlying: UnderlyingIndex, symbol: str, timeframe: timedelta
) -> None:
    if not isinstance(underlying, UnderlyingIndex):
        raise ValueError("underlying must be an UnderlyingIndex")
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbol must be a non-empty string")
    if not isinstance(timeframe, timedelta) or timeframe <= timedelta(0):
        raise ValueError("timeframe must be a positive timedelta")


def _validate_period(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_rsi_threshold(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and between 0 and 100")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and between 0 and 100") from exc
    if not isfinite(numeric) or not 0.0 <= numeric <= 100.0:
        raise ValueError(f"{name} must be finite and between 0 and 100")


def _validate_stop(stop: ATRStopConfig | None) -> None:
    if stop is not None and not isinstance(stop, ATRStopConfig):
        raise ValueError("stop must be an ATRStopConfig or None")


def _validate_incomplete_setting(value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError("allow_incomplete_candles must be a bool")


def _validate_compatibility(strategy: Strategy, data: StrategyMarketData) -> None:
    if not isinstance(data, StrategyMarketData):
        raise ValueError("data must be StrategyMarketData from validated history")
    if not strategy.is_compatible_with(data.underlying, data.timeframe, data.symbol):
        raise ValueError("strategy underlying/symbol/timeframe does not match input data")


def _invalid_calculation_signal(
    strategy: Strategy,
    data: StrategyMarketData,
    *,
    indicators: tuple[IndicatorValue, ...] = (),
    reason: str = "Indicator calculation rejected invalid or unrepresentable input.",
) -> StrategySignal:
    return _signal_for_result(
        strategy,
        data,
        direction=SignalDirection.NO_TRADE,
        events=(),
        indicators=indicators,
        reasons=(reason,),
        status=SignalStatus.INVALID_DATA,
        warmup_status=WarmupStatus.SUFFICIENT,
        invalid_conditions=(reason,),
        used_incomplete_candle=(
            not data.last_bar_complete and strategy.allow_incomplete_candles
        ),
    )


__all__ = [
    "EMATrendStrategy",
    "RSIMeanReversionStrategy",
    "Strategy",
    "unavailable_signal",
]
