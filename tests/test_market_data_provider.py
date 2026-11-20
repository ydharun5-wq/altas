from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    FuturesQuoteResponse,
    HistoricalOHLCVRequest,
    HistoricalOHLCVResponse,
    InvalidProviderResponseError,
    MarketDataProvider,
    MissingMarketDataError,
    MockMarketDataProvider,
    OptionsQuoteRequest,
    OptionsQuoteResponse,
    ProviderUnavailableError,
    QuoteFreshnessPolicy,
    StaleMarketDataError,
    UnsupportedRequestError,
    ValidatedMarketDataProvider,
    create_validated_market_data_provider,
    validate_futures_quote_response,
    validate_historical_ohlcv_response,
    validate_options_quote_response,
)


class _CustomProvider:
    def __init__(
        self,
        historical_response,
        futures_response,
        options_response,
        error=None,
    ):
        self.historical_response = historical_response
        self.futures_response = futures_response
        self.options_response = options_response
        self.error = error
        self.calls = []

    def get_historical_ohlcv(self, request):
        self.calls.append("historical")
        if self.error is not None:
            raise self.error
        return self.historical_response

    def get_futures_quote(self, request):
        self.calls.append("futures")
        if self.error is not None:
            raise self.error
        return self.futures_response

    def get_options_quote(self, request):
        self.calls.append("options")
        if self.error is not None:
            raise self.error
        return self.options_response


class _NonCallableProvider:
    get_historical_ohlcv = "not callable"
    get_futures_quote = "not callable"
    get_options_quote = "not callable"


class MarketDataProviderTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 1, 5, 9, 15)
        self.end = datetime(2026, 1, 5, 9, 25)
        self.as_of = datetime(2026, 1, 5, 4, 0, tzinfo=timezone.utc)
        self.bars = (
            OHLCVBar(self.start, "NIFTY", 100.0, 102.0, 99.0, 101.0, 100),
            OHLCVBar(
                self.start + timedelta(minutes=5),
                "NIFTY",
                101.0,
                103.0,
                100.0,
                102.0,
                110,
            ),
            OHLCVBar(self.start, "SENSEX", 200.0, 202.0, 199.0, 201.0, 200),
        )
        self.nifty_future = self._contract(
            "NIFTY-FUT-SYNTH",
            UnderlyingIndex.NIFTY,
            InstrumentType.FUT,
        )
        self.banknifty_future = self._contract(
            "BANKNIFTY-FUT-SYNTH",
            UnderlyingIndex.BANKNIFTY,
            InstrumentType.FUT,
        )
        self.nifty_option = self._contract(
            "NIFTY-OPT-SYNTH",
            UnderlyingIndex.NIFTY,
            InstrumentType.OPT,
            strike=Decimal("1000"),
            option_type=OptionType.CE,
        )
        quote_time = datetime(2026, 1, 5, 3, 59, tzinfo=timezone.utc)
        self.futures_quotes = (
            self._quote(FuturesQuote, self.nifty_future, quote_time),
            self._quote(FuturesQuote, self.banknifty_future, quote_time),
        )
        self.options_quotes = (
            self._quote(OptionsQuote, self.nifty_option, quote_time),
        )
        self.provider = MockMarketDataProvider(
            historical_bars=self.bars,
            futures_quotes=self.futures_quotes,
            options_quotes=self.options_quotes,
        )

    @staticmethod
    def _contract(
        contract_id,
        underlying,
        instrument_type,
        *,
        strike=None,
        option_type=None,
    ):
        exchange = Exchange.BSE if underlying is UnderlyingIndex.SENSEX else Exchange.NSE
        return DerivativeContract(
            exchange=exchange,
            underlying=underlying,
            instrument_type=instrument_type,
            expiry=date(2030, 1, 31),
            contract_id=contract_id,
            lot_size=1,
            tick_size=Decimal("0.05"),
            strike=strike,
            option_type=option_type,
        )

    @staticmethod
    def _quote(quote_type, contract, timestamp):
        return quote_type(
            contract=contract,
            timestamp=timestamp,
            bid=Decimal("99.95"),
            ask=Decimal("100.00"),
            last_traded_price=Decimal("99.95"),
            volume=10,
            open_interest=20,
            change_in_open_interest=0,
        )

    def test_mock_historical_retrieval_is_labeled_synthetic(self):
        request = HistoricalOHLCVRequest("NIFTY", self.start, self.end)

        response = self.provider.get_historical_ohlcv(request)

        self.assertEqual(response.bars, self.bars[:2])
        self.assertTrue(response.is_synthetic)
        self.assertEqual((response.symbol, response.start, response.end), ("NIFTY", self.start, self.end))

    def test_mock_futures_quote_retrieval_supports_multiple_contracts(self):
        for contract, expected in zip(
            (self.nifty_future, self.banknifty_future), self.futures_quotes
        ):
            with self.subTest(contract=contract.contract_id):
                response = self.provider.get_futures_quote(
                    FuturesQuoteRequest(contract, self.as_of)
                )
                self.assertEqual(response.quote, expected)
                self.assertTrue(response.is_synthetic)
                self.assertEqual(response.timestamp, expected.timestamp)

    def test_mock_options_quote_retrieval(self):
        response = self.provider.get_options_quote(
            OptionsQuoteRequest(self.nifty_option, self.as_of)
        )

        self.assertEqual(response.quote, self.options_quotes[0])
        self.assertTrue(response.is_synthetic)

    def test_multiple_historical_symbols_remain_separate(self):
        response = self.provider.get_historical_ohlcv(
            HistoricalOHLCVRequest("SENSEX", self.start, self.end)
        )

        self.assertEqual(len(response.bars), 1)
        self.assertEqual(response.bars[0].symbol, "SENSEX")

    def test_identical_requests_return_deterministic_results(self):
        request = HistoricalOHLCVRequest("NIFTY", self.start, self.end)

        self.assertEqual(
            self.provider.get_historical_ohlcv(request),
            self.provider.get_historical_ohlcv(request),
        )
        quote_request = FuturesQuoteRequest(self.nifty_future, self.as_of)
        self.assertEqual(
            self.provider.get_futures_quote(quote_request),
            self.provider.get_futures_quote(quote_request),
        )

    def test_provider_satisfies_runtime_protocol(self):
        self.assertIsInstance(self.provider, MarketDataProvider)

    def test_valid_mock_responses_pass_through_validated_boundary(self):
        validated = create_validated_market_data_provider(self.provider)
        self.assertIsInstance(validated, ValidatedMarketDataProvider)

        history = validated.get_historical_ohlcv(
            HistoricalOHLCVRequest("NIFTY", self.start, self.end)
        )
        futures = validated.get_futures_quote(
            FuturesQuoteRequest(self.nifty_future, self.as_of)
        )
        options = validated.get_options_quote(
            OptionsQuoteRequest(self.nifty_option, self.as_of)
        )

        self.assertTrue(history.is_synthetic)
        self.assertEqual(history.bars, self.bars[:2])
        self.assertTrue(futures.is_synthetic)
        self.assertEqual(futures.quote.contract, self.nifty_future)
        self.assertTrue(options.is_synthetic)
        self.assertEqual(options.quote.contract, self.nifty_option)

    def test_factory_rejects_provider_with_non_callable_operations(self):
        with self.assertRaisesRegex(TypeError, "callable"):
            create_validated_market_data_provider(_NonCallableProvider())

    def test_validated_boundary_rejects_invalid_requests_before_delegation(self):
        provider = self._custom_provider()
        validated = create_validated_market_data_provider(provider)

        for operation in (
            validated.get_historical_ohlcv,
            validated.get_futures_quote,
            validated.get_options_quote,
        ):
            with self.subTest(operation=operation.__name__):
                with self.assertRaises(ValueError):
                    operation(object())

        self.assertEqual(provider.calls, [])

    def test_validated_boundary_rejects_malformed_custom_historical_responses(self):
        request = HistoricalOHLCVRequest("NIFTY", self.start, self.end)
        malformed_responses = (
            self._history_response((self.bars[0], self.bars[0])),
            self._history_response(tuple(reversed(self.bars[:2]))),
            self._history_response((self.bars[2],)),
            self._history_response(
                (
                    OHLCVBar(
                        self.bars[0].timestamp,
                        "NIFTY",
                        100.0,
                        99.0,
                        98.0,
                        101.0,
                        100,
                    ),
                )
            ),
            self._history_response(
                (
                    OHLCVBar(
                        self.end + timedelta(minutes=1),
                        "NIFTY",
                        101.0,
                        103.0,
                        100.0,
                        102.0,
                        10,
                    ),
                )
            ),
        )
        for response in malformed_responses:
            with self.subTest(response=response):
                provider = self._custom_provider(historical_response=response)
                with self.assertRaises(InvalidProviderResponseError):
                    create_validated_market_data_provider(
                        provider
                    ).get_historical_ohlcv(request)

    def test_validated_boundary_rejects_custom_quote_contract_mismatch(self):
        other_contract = self._contract(
            "OTHER-FUT-SYNTH", UnderlyingIndex.NIFTY, InstrumentType.FUT
        )
        wrong_futures_response = FuturesQuoteResponse(
            self._quote(FuturesQuote, other_contract, self.as_of),
            is_synthetic=True,
        )
        provider = self._custom_provider(futures_response=wrong_futures_response)

        with self.assertRaisesRegex(InvalidProviderResponseError, "contract"):
            create_validated_market_data_provider(provider).get_futures_quote(
                FuturesQuoteRequest(self.nifty_future, self.as_of)
            )

        other_option = self._contract(
            "OTHER-OPT-SYNTH",
            UnderlyingIndex.NIFTY,
            InstrumentType.OPT,
            strike=Decimal("1000"),
            option_type=OptionType.PE,
        )
        wrong_options_response = OptionsQuoteResponse(
            self._quote(OptionsQuote, other_option, self.as_of),
            is_synthetic=True,
        )
        provider = self._custom_provider(options_response=wrong_options_response)

        with self.assertRaisesRegex(InvalidProviderResponseError, "contract"):
            create_validated_market_data_provider(provider).get_options_quote(
                OptionsQuoteRequest(self.nifty_option, self.as_of)
            )

    def test_validated_boundary_applies_configured_freshness_policy(self):
        provider = self._custom_provider()
        validated = create_validated_market_data_provider(
            provider,
            freshness_policy=QuoteFreshnessPolicy(timedelta(seconds=30)),
        )

        with self.assertRaises(StaleMarketDataError):
            validated.get_futures_quote(
                FuturesQuoteRequest(self.nifty_future, self.as_of)
            )

    def test_validated_boundary_preserves_provider_errors(self):
        errors = (
            MissingMarketDataError("missing"),
            UnsupportedRequestError("unsupported"),
            ProviderUnavailableError("unavailable"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                provider = self._custom_provider(error=error)
                with self.assertRaises(type(error)) as raised:
                    create_validated_market_data_provider(
                        provider
                    ).get_historical_ohlcv(
                        HistoricalOHLCVRequest("NIFTY", self.start, self.end)
                    )
                self.assertIs(raised.exception, error)

    def test_mock_reports_missing_data_and_unsupported_requests(self):
        with self.assertRaises(MissingMarketDataError):
            self.provider.get_historical_ohlcv(
                HistoricalOHLCVRequest("UNKNOWN", self.start, self.end)
            )
        with self.assertRaises(UnsupportedRequestError):
            self.provider.get_futures_quote(object())

    def test_rejects_wrong_response_type(self):
        request = HistoricalOHLCVRequest("NIFTY", self.start, self.end)

        with self.assertRaisesRegex(InvalidProviderResponseError, "HistoricalOHLCVResponse"):
            validate_historical_ohlcv_response(request, object())

        with self.assertRaisesRegex(InvalidProviderResponseError, "FuturesQuoteResponse"):
            validate_futures_quote_response(
                FuturesQuoteRequest(self.nifty_future, self.as_of), object()
            )

    def test_rejects_duplicate_timestamps(self):
        duplicate = self.bars[0]
        response = self._history_response((duplicate, duplicate))

        with self.assertRaisesRegex(InvalidProviderResponseError, "duplicate timestamp"):
            validate_historical_ohlcv_response(
                HistoricalOHLCVRequest("NIFTY", self.start, self.end), response
            )

    def test_rejects_out_of_order_bars(self):
        response = self._history_response(tuple(reversed(self.bars[:2])))

        with self.assertRaisesRegex(InvalidProviderResponseError, "chronological order"):
            validate_historical_ohlcv_response(
                HistoricalOHLCVRequest("NIFTY", self.start, self.end), response
            )

    def test_rejects_wrong_symbol_or_contract_identity(self):
        wrong_symbol_response = self._history_response((self.bars[2],))
        with self.assertRaisesRegex(InvalidProviderResponseError, "symbol"):
            validate_historical_ohlcv_response(
                HistoricalOHLCVRequest("NIFTY", self.start, self.end),
                wrong_symbol_response,
            )

        other_contract = self._contract(
            "OTHER-FUT-SYNTH", UnderlyingIndex.NIFTY, InstrumentType.FUT
        )
        quote_response = FuturesQuoteResponse(
            quote=self.futures_quotes[0], is_synthetic=True
        )
        with self.assertRaisesRegex(InvalidProviderResponseError, "contract"):
            validate_futures_quote_response(
                FuturesQuoteRequest(other_contract, self.as_of), quote_response
            )

    def test_rejects_bars_outside_requested_range(self):
        later = OHLCVBar(
            self.end + timedelta(minutes=1), "NIFTY", 101, 103, 100, 102, 10
        )
        response = self._history_response((later,))

        with self.assertRaisesRegex(InvalidProviderResponseError, "outside"):
            validate_historical_ohlcv_response(
                HistoricalOHLCVRequest("NIFTY", self.start, self.end), response
            )

    def test_empty_response_is_missing_data(self):
        response = self._history_response(())

        with self.assertRaises(MissingMarketDataError):
            validate_historical_ohlcv_response(
                HistoricalOHLCVRequest("NIFTY", self.start, self.end), response
            )

    def test_rejects_quote_after_requested_as_of(self):
        future_quote = self._quote(
            FuturesQuote,
            self.nifty_future,
            self.as_of + timedelta(seconds=1),
        )
        response = FuturesQuoteResponse(future_quote, is_synthetic=True)

        with self.assertRaisesRegex(InvalidProviderResponseError, "later than"):
            validate_futures_quote_response(
                FuturesQuoteRequest(self.nifty_future, self.as_of), response
            )

    def test_freshness_is_explicit_and_configurable(self):
        request = FuturesQuoteRequest(self.nifty_future, self.as_of)
        response = FuturesQuoteResponse(self.futures_quotes[0], is_synthetic=True)

        validate_futures_quote_response(request, response)
        validate_futures_quote_response(
            request, response, freshness_policy=QuoteFreshnessPolicy(timedelta(minutes=2))
        )
        with self.assertRaises(StaleMarketDataError):
            validate_futures_quote_response(
                request, response, freshness_policy=QuoteFreshnessPolicy(timedelta(seconds=30))
            )

    def test_options_response_validation_checks_contract(self):
        request = OptionsQuoteRequest(self.nifty_option, self.as_of)
        response = OptionsQuoteResponse(self.options_quotes[0], is_synthetic=True)

        self.assertIs(validate_options_quote_response(request, response), response)

    def test_requests_and_responses_are_immutable(self):
        request = HistoricalOHLCVRequest("NIFTY", self.start, self.end)
        response = self.provider.get_historical_ohlcv(request)

        self.assertIsInstance(response.bars, tuple)
        with self.assertRaises(FrozenInstanceError):
            request.symbol = "SENSEX"
        with self.assertRaises(FrozenInstanceError):
            response.symbol = "SENSEX"
        with self.assertRaises(FrozenInstanceError):
            response.bars[0].close = 0

    def test_response_rejects_mutable_bar_list_after_copying(self):
        bars = [self.bars[0]]
        response = self._history_response(bars)
        bars.clear()

        self.assertEqual(response.bars, (self.bars[0],))

    def _history_response(self, bars):
        return HistoricalOHLCVResponse(
            symbol="NIFTY",
            start=self.start,
            end=self.end,
            bars=bars,
            is_synthetic=True,
        )

    def _custom_provider(
        self,
        *,
        historical_response=None,
        futures_response=None,
        options_response=None,
        error=None,
    ):
        return _CustomProvider(
            historical_response
            if historical_response is not None
            else self._history_response(self.bars[:2]),
            futures_response
            if futures_response is not None
            else FuturesQuoteResponse(self.futures_quotes[0], is_synthetic=True),
            options_response
            if options_response is not None
            else OptionsQuoteResponse(self.options_quotes[0], is_synthetic=True),
            error=error,
        )


if __name__ == "__main__":
    unittest.main()
