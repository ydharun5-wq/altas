"""Validated market-data entry point for strategy research."""

from __future__ import annotations

from datetime import datetime, timedelta

from atlas.derivatives.models import UnderlyingIndex
from atlas.market_data.models import HistoricalOHLCVRequest
from atlas.market_data.provider import (
    InvalidProviderResponseError,
    MissingMarketDataError,
    ValidatedMarketDataProvider,
)
from atlas.strategies.models import (
    SignalStatus,
    StrategyMarketData,
    StrategySignal,
)
from atlas.strategies.strategies import Strategy, unavailable_signal


class StrategyResearchEngine:
    """Generate research signals only from the validated provider boundary.

    This engine fetches requested historical bars through a
    ValidatedMarketDataProvider. It does not backtest, manage positions,
    produce sizing, or submit orders.
    """

    def __init__(self, market_data: ValidatedMarketDataProvider) -> None:
        if not isinstance(market_data, ValidatedMarketDataProvider):
            raise TypeError("market_data must be a ValidatedMarketDataProvider")
        self._market_data = market_data

    def generate_signal(
        self,
        strategy: Strategy,
        request: HistoricalOHLCVRequest,
        *,
        underlying: UnderlyingIndex,
        timeframe: timedelta,
        data_source: str,
        evaluation_at: datetime,
        last_bar_complete: bool,
    ) -> StrategySignal:
        """Fetch validated bars and evaluate one strategy at an explicit as-of.

        The request end must not be later than evaluation_at. Timeframe,
        source identifier, and last-candle completeness are explicit because
        those properties are not included in historical OHLCV responses.
        """
        _validate_strategy(strategy)
        if not isinstance(request, HistoricalOHLCVRequest):
            raise ValueError("request must be a HistoricalOHLCVRequest")
        if not isinstance(underlying, UnderlyingIndex):
            raise ValueError("underlying must be an UnderlyingIndex")
        if not isinstance(timeframe, timedelta) or timeframe <= timedelta(0):
            raise ValueError("timeframe must be a positive timedelta")
        if not isinstance(data_source, str) or not data_source.strip():
            raise ValueError("data_source must be a non-empty string")
        if not isinstance(evaluation_at, datetime):
            raise ValueError("evaluation_at must be a datetime")
        if not isinstance(last_bar_complete, bool):
            raise ValueError("last_bar_complete must be a bool")
        if not strategy.is_compatible_with(underlying, timeframe, request.symbol):
            raise ValueError(
                "strategy underlying/symbol/timeframe does not match request context"
            )
        try:
            if request.end > evaluation_at:
                raise ValueError("request end must not be later than evaluation_at")
        except TypeError as exc:
            raise ValueError("request end and evaluation_at must be comparable") from exc

        try:
            response = self._market_data.get_historical_ohlcv(request)
        except MissingMarketDataError:
            return unavailable_signal(
                strategy,
                input_symbol=request.symbol,
                data_source=data_source,
                evaluation_at=evaluation_at,
                status=SignalStatus.MISSING_DATA,
                reason="No historical bars are available for the requested range.",
                missing_data=("validated provider returned no historical bars",),
            )
        except InvalidProviderResponseError:
            return unavailable_signal(
                strategy,
                input_symbol=request.symbol,
                data_source=data_source,
                evaluation_at=evaluation_at,
                status=SignalStatus.INVALID_DATA,
                reason="Validated market-data checks rejected the historical response.",
                invalid_conditions=("provider response failed market-data validation",),
            )

        try:
            data = StrategyMarketData._from_validated_provider(
                response=response,
                underlying=underlying,
                timeframe=timeframe,
                data_source=data_source,
                evaluation_at=evaluation_at,
                last_bar_complete=last_bar_complete,
            )
        except (OverflowError, TypeError, ValueError):
            return unavailable_signal(
                strategy,
                input_symbol=request.symbol,
                data_source=data_source,
                evaluation_at=evaluation_at,
                status=SignalStatus.INVALID_DATA,
                reason="Historical data failed strategy-input validation.",
                invalid_conditions=("malformed, inconsistent, or future market data",),
            )
        try:
            return strategy.evaluate(data)
        except (OverflowError, TypeError, ValueError):
            return unavailable_signal(
                strategy,
                input_symbol=request.symbol,
                data_source=data_source,
                evaluation_at=evaluation_at,
                status=SignalStatus.INVALID_DATA,
                reason="Strategy evaluation rejected invalid or unrepresentable inputs.",
                invalid_conditions=("indicator or stop calculation failed validation",),
            )


def _validate_strategy(strategy: Strategy) -> None:
    required_attributes = (
        "name",
        "version",
        "underlying",
        "symbol",
        "timeframe",
        "input_requirements",
        "entry_conditions",
        "exit_conditions",
        "stop_loss_method",
        "allow_incomplete_candles",
        "is_compatible_with",
        "evaluate",
    )
    if any(not hasattr(strategy, name) for name in required_attributes):
        raise TypeError("strategy must implement the Strategy interface")
    if not callable(strategy.evaluate) or not callable(strategy.is_compatible_with):
        raise TypeError("strategy evaluate and compatibility members must be callable")


__all__ = ["StrategyResearchEngine"]
