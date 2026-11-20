from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.application import AtlasApplication, create_application
from atlas.data.ohlcv import OHLCVBar
from atlas.derivatives.models import (
    DerivativeContract,
    Exchange,
    FuturesQuote,
    InstrumentType,
    OptionType,
    OptionsQuote,
    UnderlyingIndex,
)
from atlas.market_data import (
    FuturesQuoteRequest,
    HistoricalOHLCVRequest,
    InvalidProviderResponseError,
    MissingMarketDataError,
    MockMarketDataProvider,
    OptionsQuoteRequest,
    QuoteFreshnessPolicy,
    StaleMarketDataError,
    ValidatedMarketDataProvider,
)


class ApplicationCompositionTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 2, 5, 9, 15)
        self.end = self.start + timedelta(minutes=5)
        self.as_of = datetime(2026, 2, 5, 4, 0, tzinfo=timezone.utc)
        self.quote_time = self.as_of - timedelta(minutes=1)
        self.bars = (
            OHLCVBar(self.start, "NIFTY", 100.0, 102.0, 99.0, 101.0, 100),
        )
        self.futures_contract = self._contract(InstrumentType.FUT)
        self.options_contract = self._contract(
            InstrumentType.OPT,
            strike=Decimal("1000"),
            option_type=OptionType.CE,
        )
        self.futures_quotes = (self._quote(FuturesQuote, self.futures_contract),)
        self.options_quotes = (self._quote(OptionsQuote, self.options_contract),)

    @staticmethod
    def _contract(instrument_type, *, strike=None, option_type=None):
        contract_id = (
            "NIFTY-FUT-SYNTH"
            if instrument_type is InstrumentType.FUT
            else "NIFTY-OPT-SYNTH"
        )
        return DerivativeContract(
            exchange=Exchange.NSE,
            underlying=UnderlyingIndex.NIFTY,
            instrument_type=instrument_type,
            expiry=date(2030, 1, 31),
            contract_id=contract_id,
            lot_size=1,
            tick_size=Decimal("0.05"),
            strike=strike,
            option_type=option_type,
        )

    def _quote(self, quote_type, contract):
        return quote_type(
            contract=contract,
            timestamp=self.quote_time,
            bid=Decimal("99.95"),
            ask=Decimal("100.00"),
            last_traded_price=Decimal("99.95"),
            volume=10,
            open_interest=20,
            change_in_open_interest=0,
        )

    def _mock_fixtures(self):
        return {
            "mock_historical_bars": self.bars,
            "mock_futures_quotes": self.futures_quotes,
            "mock_options_quotes": self.options_quotes,
        }

    def _create_mock_application(self, **kwargs):
        return create_application(
            provider_name="mock",
            **self._mock_fixtures(),
            **kwargs,
        )

    def test_mock_selection_exposes_only_validated_synthetic_data(self):
        app = self._create_mock_application()

        self.assertIsInstance(app, AtlasApplication)
        self.assertIsInstance(app.market_data, ValidatedMarketDataProvider)
        self.assertFalse(hasattr(app, "raw_provider"))

        history = app.market_data.get_historical_ohlcv(
            HistoricalOHLCVRequest("NIFTY", self.start, self.end)
        )
        futures = app.market_data.get_futures_quote(
            FuturesQuoteRequest(self.futures_contract, self.as_of)
        )
        options = app.market_data.get_options_quote(
            OptionsQuoteRequest(self.options_contract, self.as_of)
        )

        self.assertTrue(history.is_synthetic)
        self.assertTrue(futures.is_synthetic)
        self.assertTrue(options.is_synthetic)

    def test_csv_selection_requires_and_preserves_explicit_provenance(self):
        sample = Path(__file__).resolve().parents[1] / "data" / "sample_ohlcv.csv"
        app = create_application(
            provider_name="csv",
            csv_symbol_files={"RELIANCE": sample},
            csv_data_source="sample-csv-fixture",
            csv_is_synthetic=True,
        )

        history = app.market_data.get_historical_ohlcv(
            HistoricalOHLCVRequest(
                "RELIANCE", datetime(2026, 1, 5), datetime(2026, 1, 5, 9, 20)
            )
        )

        self.assertIsInstance(app.market_data, ValidatedMarketDataProvider)
        self.assertTrue(history.is_synthetic)
        self.assertEqual(len(history.bars), 2)

    def test_csv_selection_requires_source_and_synthetic_status(self):
        sample = Path(__file__).resolve().parents[1] / "data" / "sample_ohlcv.csv"
        with self.assertRaisesRegex(ValueError, "csv_data_source"):
            create_application(
                provider_name="csv",
                csv_symbol_files={"RELIANCE": sample},
                csv_is_synthetic=True,
            )
        with self.assertRaisesRegex(ValueError, "csv_is_synthetic"):
            create_application(
                provider_name="csv",
                csv_symbol_files={"RELIANCE": sample},
                csv_data_source="sample-csv-fixture",
            )

    def test_application_container_rejects_raw_provider(self):
        raw_provider = MockMarketDataProvider(
            historical_bars=self.bars,
            futures_quotes=self.futures_quotes,
            options_quotes=self.options_quotes,
        )

        with self.assertRaisesRegex(TypeError, "ValidatedMarketDataProvider"):
            AtlasApplication(market_data=raw_provider)

    def test_unknown_provider_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            create_application(provider_name="groww", **self._mock_fixtures())

    def test_mock_provider_requires_all_fixture_collections(self):
        with self.assertRaisesRegex(ValueError, "must be supplied"):
            create_application(provider_name="mock")

    def test_invalid_mock_fixture_collections_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "iterable"):
            create_application(
                provider_name="mock",
                mock_historical_bars=object(),
                mock_futures_quotes=self.futures_quotes,
                mock_options_quotes=self.options_quotes,
            )

        with self.assertRaises(ValueError):
            create_application(
                provider_name="mock",
                mock_historical_bars=(object(),),
                mock_futures_quotes=self.futures_quotes,
                mock_options_quotes=self.options_quotes,
            )

    def test_invalid_freshness_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "freshness_policy"):
            self._create_mock_application(freshness_policy=object())

    def test_configured_freshness_policy_is_applied(self):
        app = self._create_mock_application(
            freshness_policy=QuoteFreshnessPolicy(timedelta(seconds=30))
        )

        with self.assertRaises(StaleMarketDataError):
            app.market_data.get_futures_quote(
                FuturesQuoteRequest(self.futures_contract, self.as_of)
            )

    def test_invalid_mock_fixture_response_is_rejected_at_application_boundary(self):
        bad_bar = OHLCVBar(
            self.start, "NIFTY", 100.0, 99.0, 98.0, 101.0, 100
        )
        app = create_application(
            provider_name="mock",
            mock_historical_bars=(bad_bar,),
            mock_futures_quotes=self.futures_quotes,
            mock_options_quotes=self.options_quotes,
        )

        with self.assertRaises(InvalidProviderResponseError):
            app.market_data.get_historical_ohlcv(
                HistoricalOHLCVRequest("NIFTY", self.start, self.end)
            )

    def test_provider_missing_data_error_is_preserved(self):
        app = self._create_mock_application()

        with self.assertRaises(MissingMarketDataError):
            app.market_data.get_historical_ohlcv(
                HistoricalOHLCVRequest("UNKNOWN", self.start, self.end)
            )

if __name__ == "__main__":
    unittest.main()
