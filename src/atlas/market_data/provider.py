"""Provider protocol, market-data errors, and deterministic response checks."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Protocol, runtime_checkable

from atlas.data.ohlcv import OHLCVBar
from atlas.derivatives.models import (
    DerivativeContract,
    FuturesQuote,
    InstrumentType,
    OptionsQuote,
)
from atlas.market_data.models import (
    FuturesQuoteRequest,
    FuturesQuoteResponse,
    HistoricalOHLCVRequest,
    HistoricalOHLCVResponse,
    OptionsQuoteRequest,
    OptionsQuoteResponse,
    QuoteFreshnessPolicy,
)


class MarketDataError(Exception):
    """Base class for expected market-data failures."""


class UnsupportedRequestError(MarketDataError):
    """The provider does not support the requested instrument or operation."""


class MissingMarketDataError(MarketDataError):
    """No data is available for the requested identity and time range."""


class InvalidProviderResponseError(MarketDataError):
    """A provider response is malformed or conflicts with its request."""


class StaleMarketDataError(InvalidProviderResponseError):
    """A quote exceeds the caller-specified freshness limit."""


class ProviderUnavailableError(MarketDataError):
    """The provider cannot currently serve requests."""


@runtime_checkable
class MarketDataProvider(Protocol):
    """Provider-neutral retrieval interface; implementations perform no repair."""

    def get_historical_ohlcv(
        self, request: HistoricalOHLCVRequest
    ) -> HistoricalOHLCVResponse:
        """Return historical bars for the requested symbol and inclusive range."""
        ...

    def get_futures_quote(
        self, request: FuturesQuoteRequest
    ) -> FuturesQuoteResponse:
        """Return a futures quote for the requested contract."""
        ...

    def get_options_quote(
        self, request: OptionsQuoteRequest
    ) -> OptionsQuoteResponse:
        """Return an options quote for the requested contract."""
        ...


class ValidatedMarketDataProvider:
    """Validate every delegated provider response before exposing it to callers."""

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        freshness_policy: QuoteFreshnessPolicy | None = None,
    ) -> None:
        if not isinstance(provider, MarketDataProvider):
            raise TypeError("provider must implement MarketDataProvider")
        provider_methods = (
            "get_historical_ohlcv",
            "get_futures_quote",
            "get_options_quote",
        )
        if any(
            not callable(getattr(provider, name, None))
            for name in provider_methods
        ):
            raise TypeError("provider methods must be callable")
        if freshness_policy is not None and not isinstance(
            freshness_policy, QuoteFreshnessPolicy
        ):
            raise TypeError("freshness_policy must be a QuoteFreshnessPolicy or None")
        self._provider = provider
        self._freshness_policy = freshness_policy

    def get_historical_ohlcv(
        self, request: HistoricalOHLCVRequest
    ) -> HistoricalOHLCVResponse:
        if not isinstance(request, HistoricalOHLCVRequest):
            raise ValueError("request must be a HistoricalOHLCVRequest")
        response = self._provider.get_historical_ohlcv(request)
        return validate_historical_ohlcv_response(request, response)

    def get_futures_quote(
        self, request: FuturesQuoteRequest
    ) -> FuturesQuoteResponse:
        if not isinstance(request, FuturesQuoteRequest):
            raise ValueError("request must be a FuturesQuoteRequest")
        response = self._provider.get_futures_quote(request)
        return validate_futures_quote_response(
            request, response, freshness_policy=self._freshness_policy
        )

    def get_options_quote(
        self, request: OptionsQuoteRequest
    ) -> OptionsQuoteResponse:
        if not isinstance(request, OptionsQuoteRequest):
            raise ValueError("request must be an OptionsQuoteRequest")
        response = self._provider.get_options_quote(request)
        return validate_options_quote_response(
            request, response, freshness_policy=self._freshness_policy
        )


def create_validated_market_data_provider(
    provider: MarketDataProvider,
    *,
    freshness_policy: QuoteFreshnessPolicy | None = None,
) -> ValidatedMarketDataProvider:
    """Build the application-facing boundary around a raw provider adapter."""
    return ValidatedMarketDataProvider(
        provider, freshness_policy=freshness_policy
    )


def validate_historical_ohlcv_response(
    request: HistoricalOHLCVRequest,
    response: object,
) -> HistoricalOHLCVResponse:
    """Validate identity, range, ordering, uniqueness, and every returned bar."""
    if not isinstance(request, HistoricalOHLCVRequest):
        raise ValueError("request must be a HistoricalOHLCVRequest")
    if not isinstance(response, HistoricalOHLCVResponse):
        raise InvalidProviderResponseError(
            "provider must return a HistoricalOHLCVResponse"
        )
    if response.symbol != request.symbol:
        raise InvalidProviderResponseError("response symbol does not match request")
    if response.start != request.start or response.end != request.end:
        raise InvalidProviderResponseError("response time range does not match request")
    if not response.bars:
        raise MissingMarketDataError(
            f"provider returned no bars for {request.symbol} in the requested range"
        )

    seen_timestamps: set[datetime] = set()
    previous_timestamp: datetime | None = None
    for index, bar in enumerate(response.bars):
        if not isinstance(bar, OHLCVBar):
            raise InvalidProviderResponseError(
                f"bar at index {index} is not an OHLCVBar"
            )
        if bar.symbol != request.symbol:
            raise InvalidProviderResponseError(
                f"bar at index {index} has symbol {bar.symbol!r}, "
                f"expected {request.symbol!r}"
            )
        if not isinstance(bar.timestamp, datetime):
            raise InvalidProviderResponseError(
                f"bar timestamp at index {index} is not a datetime"
            )
        try:
            outside_range = bar.timestamp < request.start or bar.timestamp > request.end
        except TypeError as exc:
            raise InvalidProviderResponseError(
                f"bar timestamp at index {index} is not comparable to request bounds"
            ) from exc
        if outside_range:
            raise InvalidProviderResponseError(
                f"bar at index {index} falls outside the requested time range"
            )
        try:
            duplicate = bar.timestamp in seen_timestamps
        except TypeError as exc:
            raise InvalidProviderResponseError(
                f"bar timestamp at index {index} cannot be compared"
            ) from exc
        if duplicate:
            raise InvalidProviderResponseError(
                f"duplicate timestamp for {request.symbol}"
            )
        seen_timestamps.add(bar.timestamp)
        if previous_timestamp is not None:
            try:
                out_of_order = bar.timestamp < previous_timestamp
            except TypeError as exc:
                raise InvalidProviderResponseError(
                    "bar timestamps are not mutually comparable"
                ) from exc
            if out_of_order:
                raise InvalidProviderResponseError(
                    "historical bars are not in chronological order"
                )
        previous_timestamp = bar.timestamp
        _validate_bar(bar, index)
    return response


def validate_futures_quote_response(
    request: FuturesQuoteRequest,
    response: object,
    *,
    freshness_policy: QuoteFreshnessPolicy | None = None,
) -> FuturesQuoteResponse:
    """Validate futures quote type, identity, request time, and optional freshness."""
    if not isinstance(request, FuturesQuoteRequest):
        raise ValueError("request must be a FuturesQuoteRequest")
    if not isinstance(response, FuturesQuoteResponse):
        raise InvalidProviderResponseError(
            "provider must return a FuturesQuoteResponse"
        )
    _validate_quote(
        request.contract,
        request.as_of,
        response.quote,
        FuturesQuote,
        freshness_policy,
    )
    return response


def validate_options_quote_response(
    request: OptionsQuoteRequest,
    response: object,
    *,
    freshness_policy: QuoteFreshnessPolicy | None = None,
) -> OptionsQuoteResponse:
    """Validate options quote type, identity, request time, and optional freshness."""
    if not isinstance(request, OptionsQuoteRequest):
        raise ValueError("request must be an OptionsQuoteRequest")
    if not isinstance(response, OptionsQuoteResponse):
        raise InvalidProviderResponseError(
            "provider must return an OptionsQuoteResponse"
        )
    _validate_quote(
        request.contract,
        request.as_of,
        response.quote,
        OptionsQuote,
        freshness_policy,
    )
    return response


def _validate_quote(
    requested_contract: DerivativeContract,
    as_of: datetime,
    quote: FuturesQuote | OptionsQuote,
    expected_type: type[FuturesQuote] | type[OptionsQuote],
    freshness_policy: QuoteFreshnessPolicy | None,
) -> None:
    if not isinstance(quote, expected_type):
        raise InvalidProviderResponseError("provider returned the wrong quote type")
    if quote.contract != requested_contract:
        raise InvalidProviderResponseError(
            "quote contract does not match requested contract"
        )
    try:
        is_future = quote.timestamp > as_of
    except TypeError as exc:
        raise InvalidProviderResponseError(
            "quote timestamp is not comparable to request as_of"
        ) from exc
    if is_future:
        raise InvalidProviderResponseError(
            "quote timestamp is later than the requested as_of time"
        )
    if freshness_policy is not None:
        if not isinstance(freshness_policy, QuoteFreshnessPolicy):
            raise ValueError("freshness_policy must be a QuoteFreshnessPolicy")
        age = as_of - quote.timestamp
        if age > freshness_policy.max_age:
            raise StaleMarketDataError(
                f"quote age {age} exceeds configured maximum "
                f"{freshness_policy.max_age}"
            )


def _validate_bar(bar: OHLCVBar, index: int) -> None:
    prices: list[float] = []
    for field in ("open", "high", "low", "close"):
        raw_price = getattr(bar, field)
        if isinstance(raw_price, (bool, str, bytes, bytearray)):
            raise InvalidProviderResponseError(
                f"{field} at index {index} must be finite and positive"
            )
        try:
            price = float(raw_price)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidProviderResponseError(
                f"{field} at index {index} must be finite and positive"
            ) from exc
        if not math.isfinite(price) or price <= 0:
            raise InvalidProviderResponseError(
                f"{field} at index {index} must be finite and positive"
            )
        prices.append(price)

    volume = bar.volume
    if isinstance(volume, bool) or not isinstance(volume, int) or volume < 0:
        raise InvalidProviderResponseError(
            f"volume at index {index} must be a non-negative integer"
        )
    open_price, high, low, close = prices
    if high < max(open_price, low, close) or low > min(open_price, high, close):
        raise InvalidProviderResponseError(
            f"bar at index {index} has inconsistent OHLC prices"
        )
