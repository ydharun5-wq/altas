"""Validated market-data interfaces and local/mock history providers."""

from atlas.market_data.mock import MockMarketDataProvider
from atlas.market_data.csv_history import CSVHistoricalMarketDataProvider
from atlas.market_data.models import (
    FuturesQuoteRequest,
    FuturesQuoteResponse,
    HistoricalOHLCVRequest,
    HistoricalOHLCVResponse,
    OptionsQuoteRequest,
    OptionsQuoteResponse,
    QuoteFreshnessPolicy,
)
from atlas.market_data.provider import (
    InvalidProviderResponseError,
    MarketDataError,
    MarketDataProvider,
    MissingMarketDataError,
    ProviderUnavailableError,
    StaleMarketDataError,
    UnsupportedRequestError,
    ValidatedMarketDataProvider,
    create_validated_market_data_provider,
    validate_futures_quote_response,
    validate_historical_ohlcv_response,
    validate_options_quote_response,
)

__all__ = [
    "FuturesQuoteRequest",
    "FuturesQuoteResponse",
    "CSVHistoricalMarketDataProvider",
    "HistoricalOHLCVRequest",
    "HistoricalOHLCVResponse",
    "InvalidProviderResponseError",
    "MarketDataError",
    "MarketDataProvider",
    "MissingMarketDataError",
    "MockMarketDataProvider",
    "OptionsQuoteRequest",
    "OptionsQuoteResponse",
    "ProviderUnavailableError",
    "QuoteFreshnessPolicy",
    "StaleMarketDataError",
    "UnsupportedRequestError",
    "ValidatedMarketDataProvider",
    "create_validated_market_data_provider",
    "validate_futures_quote_response",
    "validate_historical_ohlcv_response",
    "validate_options_quote_response",
]
