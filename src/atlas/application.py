"""Application composition for provider-neutral ATLAS consumers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from atlas.data.ohlcv import OHLCVBar
from atlas.derivatives.models import FuturesQuote, OptionsQuote
from atlas.market_data import (
    CSVHistoricalMarketDataProvider,
    MockMarketDataProvider,
    QuoteFreshnessPolicy,
    ValidatedMarketDataProvider,
    create_validated_market_data_provider,
)


@dataclass(frozen=True)
class AtlasApplication:
    """Application-facing services; market data is always validated."""

    market_data: ValidatedMarketDataProvider

    def __post_init__(self) -> None:
        if not isinstance(self.market_data, ValidatedMarketDataProvider):
            raise TypeError(
                "market_data must be a ValidatedMarketDataProvider"
            )


def create_application(
    *,
    provider_name: str,
    csv_symbol_files: Mapping[str, str | Path] | None = None,
    csv_data_source: str | None = None,
    csv_is_synthetic: bool | None = None,
    mock_historical_bars: Iterable[OHLCVBar] | None = None,
    mock_futures_quotes: Iterable[FuturesQuote] | None = None,
    mock_options_quotes: Iterable[OptionsQuote] | None = None,
    freshness_policy: QuoteFreshnessPolicy | None = None,
) -> AtlasApplication:
    """Select and validate a provider for application consumers.

    ``mock`` serves explicitly supplied synthetic fixtures. ``csv`` serves
    caller-selected local OHLCV history and requires explicit provenance. No
    source or freshness policy is selected by default.
    """
    if not isinstance(provider_name, str) or not provider_name:
        raise ValueError("provider_name must be a non-empty string")
    if freshness_policy is not None and not isinstance(
        freshness_policy, QuoteFreshnessPolicy
    ):
        raise ValueError("freshness_policy must be a QuoteFreshnessPolicy or None")

    if provider_name == "mock":
        mock_fixtures = (
            ("mock_historical_bars", mock_historical_bars),
            ("mock_futures_quotes", mock_futures_quotes),
            ("mock_options_quotes", mock_options_quotes),
        )
        for name, fixtures in mock_fixtures:
            if fixtures is None:
                raise ValueError(f"{name} must be supplied for the mock provider")
            if not isinstance(fixtures, Iterable) or isinstance(
                fixtures, (str, bytes, bytearray)
            ):
                raise ValueError(f"{name} must be an iterable fixture collection")
        raw_provider = MockMarketDataProvider(
            historical_bars=mock_historical_bars,
            futures_quotes=mock_futures_quotes,
            options_quotes=mock_options_quotes,
        )
    elif provider_name == "csv":
        if csv_symbol_files is None:
            raise ValueError("csv_symbol_files must be supplied for the CSV provider")
        if csv_data_source is None:
            raise ValueError("csv_data_source must be supplied for the CSV provider")
        if csv_is_synthetic is None:
            raise ValueError("csv_is_synthetic must be supplied for the CSV provider")
        raw_provider = CSVHistoricalMarketDataProvider(
            csv_symbol_files,
            data_source=csv_data_source,
            is_synthetic=csv_is_synthetic,
        )
    else:
        raise ValueError(f"unsupported market-data provider: {provider_name!r}")

    validated_provider = create_validated_market_data_provider(
        raw_provider,
        freshness_policy=freshness_policy,
    )
    return AtlasApplication(market_data=validated_provider)
