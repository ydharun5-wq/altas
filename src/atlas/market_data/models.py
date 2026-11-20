"""Immutable request and response models for market-data providers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from atlas.data.ohlcv import OHLCVBar
from atlas.derivatives.models import (
    DerivativeContract,
    FuturesQuote,
    FuturesQuoteRequest,
    InstrumentType,
    OptionsQuote,
    OptionsQuoteRequest,
)


def _require_aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    try:
        offset = value.utcoffset()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have valid timezone information") from exc
    if offset is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True)
class HistoricalOHLCVRequest:
    """Request inclusive historical bounds for one exact symbol."""

    symbol: str
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.symbol, str)
            or not self.symbol
            or any(character.isspace() for character in self.symbol)
        ):
            raise ValueError("symbol must be a non-empty identifier without whitespace")
        if not isinstance(self.start, datetime) or not isinstance(self.end, datetime):
            raise ValueError("start and end must be datetimes")
        try:
            invalid_range = self.start > self.end
        except TypeError as exc:
            raise ValueError("start and end must use comparable timezones") from exc
        if invalid_range:
            raise ValueError("start must not be later than end")


@dataclass(frozen=True)
class HistoricalOHLCVResponse:
    """Historical bars and the exact symbol and inclusive bounds requested."""

    symbol: str
    start: datetime
    end: datetime
    bars: tuple[OHLCVBar, ...]
    is_synthetic: bool

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "bars", tuple(self.bars))
        except TypeError as exc:
            raise ValueError("bars must be an iterable of OHLCVBar objects") from exc
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool")


@dataclass(frozen=True)
class FuturesQuoteResponse:
    """Futures quote response; its timestamp is ``quote.timestamp``."""

    quote: FuturesQuote
    is_synthetic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.quote, FuturesQuote):
            raise ValueError("quote must be a FuturesQuote")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool")

    @property
    def timestamp(self) -> datetime:
        return self.quote.timestamp


@dataclass(frozen=True)
class OptionsQuoteResponse:
    """Options quote response; its timestamp is ``quote.timestamp``."""

    quote: OptionsQuote
    is_synthetic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.quote, OptionsQuote):
            raise ValueError("quote must be an OptionsQuote")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool")

    @property
    def timestamp(self) -> datetime:
        return self.quote.timestamp


@dataclass(frozen=True)
class QuoteFreshnessPolicy:
    """Caller-selected maximum quote age; no universal default is imposed."""

    max_age: timedelta

    def __post_init__(self) -> None:
        if not isinstance(self.max_age, timedelta) or self.max_age <= timedelta(0):
            raise ValueError("max_age must be a positive timedelta")
