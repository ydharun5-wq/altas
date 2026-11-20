"""Deterministic in-memory provider for explicitly synthetic test fixtures."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

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
)
from atlas.market_data.provider import (
    InvalidProviderResponseError,
    MissingMarketDataError,
    UnsupportedRequestError,
)


class MockMarketDataProvider:
    """Serve caller-supplied synthetic fixtures without network or credentials."""

    def __init__(
        self,
        *,
        historical_bars: Iterable[OHLCVBar],
        futures_quotes: Iterable[FuturesQuote],
        options_quotes: Iterable[OptionsQuote],
    ) -> None:
        grouped_bars: dict[str, list[OHLCVBar]] = defaultdict(list)
        for bar in historical_bars:
            if not isinstance(bar, OHLCVBar):
                raise ValueError("historical_bars must contain OHLCVBar objects")
            grouped_bars[bar.symbol].append(bar)
        self._historical_bars = {
            symbol: tuple(bars) for symbol, bars in grouped_bars.items()
        }
        self._futures_quotes = self._index_quotes(futures_quotes, FuturesQuote)
        self._options_quotes = self._index_quotes(options_quotes, OptionsQuote)

    def get_historical_ohlcv(
        self, request: HistoricalOHLCVRequest
    ) -> HistoricalOHLCVResponse:
        if not isinstance(request, HistoricalOHLCVRequest):
            raise UnsupportedRequestError(
                "get_historical_ohlcv requires a HistoricalOHLCVRequest"
            )
        available = self._historical_bars.get(request.symbol)
        if available is None:
            raise MissingMarketDataError(f"no synthetic bars for {request.symbol}")
        selected: list[OHLCVBar] = []
        for bar in available:
            try:
                in_range = request.start <= bar.timestamp <= request.end
            except TypeError as exc:
                raise InvalidProviderResponseError(
                    "synthetic bar timestamps are not comparable to request bounds"
                ) from exc
            if in_range:
                selected.append(bar)
        if not selected:
            raise MissingMarketDataError(
                f"no synthetic bars for {request.symbol} in the requested range"
            )
        return HistoricalOHLCVResponse(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            bars=tuple(selected),
            is_synthetic=True,
        )

    def get_futures_quote(
        self, request: FuturesQuoteRequest
    ) -> FuturesQuoteResponse:
        if not isinstance(request, FuturesQuoteRequest):
            raise UnsupportedRequestError(
                "get_futures_quote requires a FuturesQuoteRequest"
            )
        quote = self._futures_quotes.get(request.contract)
        if quote is None:
            raise MissingMarketDataError(
                f"no synthetic futures quote for {request.contract.contract_id}"
            )
        return FuturesQuoteResponse(quote=quote, is_synthetic=True)

    def get_options_quote(
        self, request: OptionsQuoteRequest
    ) -> OptionsQuoteResponse:
        if not isinstance(request, OptionsQuoteRequest):
            raise UnsupportedRequestError(
                "get_options_quote requires an OptionsQuoteRequest"
            )
        quote = self._options_quotes.get(request.contract)
        if quote is None:
            raise MissingMarketDataError(
                f"no synthetic options quote for {request.contract.contract_id}"
            )
        return OptionsQuoteResponse(quote=quote, is_synthetic=True)

    @staticmethod
    def _index_quotes(
        quotes: Iterable[FuturesQuote] | Iterable[OptionsQuote],
        expected_type: type[FuturesQuote] | type[OptionsQuote],
    ) -> dict[DerivativeContract, FuturesQuote | OptionsQuote]:
        indexed: dict[DerivativeContract, FuturesQuote | OptionsQuote] = {}
        for quote in quotes:
            if not isinstance(quote, expected_type):
                raise ValueError(
                    f"quote fixtures must contain {expected_type.__name__} objects"
                )
            if quote.contract in indexed:
                raise ValueError(
                    f"duplicate synthetic quote for {quote.contract.contract_id}"
                )
            indexed[quote.contract] = quote
        return indexed
