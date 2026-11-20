"""Local CSV history adapter for validated strategy research and backtests."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from atlas.data.ohlcv import DataValidationError, OHLCVBar, read_ohlcv_csv
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


class CSVHistoricalMarketDataProvider:
    """Serve caller-selected local OHLCV files through the market-data protocol.

    ``data_source`` and ``is_synthetic`` are mandatory so local fixtures cannot
    silently be presented as verified real-market history. Quote operations
    are intentionally unsupported; no broker or network access is performed.
    Each mapping key names the one symbol expected in its corresponding file.
    """

    def __init__(
        self,
        symbol_files: Mapping[str, str | Path],
        *,
        data_source: str,
        is_synthetic: bool,
    ) -> None:
        if not isinstance(symbol_files, Mapping) or not symbol_files:
            raise ValueError("symbol_files must be a non-empty symbol-to-path mapping")
        if not isinstance(data_source, str) or not data_source.strip():
            raise ValueError("data_source must be a non-empty string")
        if not isinstance(is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool")

        loaded: dict[str, tuple[OHLCVBar, ...]] = {}
        for symbol, path in symbol_files.items():
            if (
                not isinstance(symbol, str)
                or not symbol
                or symbol != symbol.strip().upper()
                or any(character.isspace() for character in symbol)
            ):
                raise ValueError("symbol_files keys must be uppercase identifiers")
            if symbol in loaded:
                raise ValueError(f"duplicate local history for {symbol}")
            bars = read_ohlcv_csv(path)
            if any(bar.symbol != symbol for bar in bars):
                raise DataValidationError(
                    f"CSV configured for {symbol} contains a different symbol"
                )
            loaded[symbol] = tuple(bars)

        self._bars_by_symbol = MappingProxyType(loaded)
        self._data_source = data_source.strip()
        self._is_synthetic = is_synthetic

    @property
    def data_source(self) -> str:
        return self._data_source

    @property
    def is_synthetic(self) -> bool:
        return self._is_synthetic

    def get_historical_ohlcv(
        self, request: HistoricalOHLCVRequest
    ) -> HistoricalOHLCVResponse:
        if not isinstance(request, HistoricalOHLCVRequest):
            raise UnsupportedRequestError(
                "get_historical_ohlcv requires a HistoricalOHLCVRequest"
            )
        available = self._bars_by_symbol.get(request.symbol)
        if available is None:
            raise MissingMarketDataError(
                f"no local CSV history is configured for {request.symbol}"
            )
        try:
            selected = tuple(
                bar
                for bar in available
                if request.start <= bar.timestamp <= request.end
            )
        except TypeError as exc:
            raise InvalidProviderResponseError(
                "local CSV timestamps are not comparable to request bounds"
            ) from exc
        if not selected:
            raise MissingMarketDataError(
                f"no local CSV bars for {request.symbol} in the requested range"
            )
        return HistoricalOHLCVResponse(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            bars=selected,
            is_synthetic=self.is_synthetic,
        )

    def get_futures_quote(
        self, request: FuturesQuoteRequest
    ) -> FuturesQuoteResponse:
        if not isinstance(request, FuturesQuoteRequest):
            raise UnsupportedRequestError(
                "get_futures_quote requires a FuturesQuoteRequest"
            )
        raise UnsupportedRequestError("local CSV history does not provide futures quotes")

    def get_options_quote(
        self, request: OptionsQuoteRequest
    ) -> OptionsQuoteResponse:
        if not isinstance(request, OptionsQuoteRequest):
            raise UnsupportedRequestError(
                "get_options_quote requires an OptionsQuoteRequest"
            )
        raise UnsupportedRequestError("local CSV history does not provide options quotes")


__all__ = ["CSVHistoricalMarketDataProvider"]
