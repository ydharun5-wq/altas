from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.derivatives.models import (
    DerivativeContract,
    Exchange,
    FuturesQuote,
    InstrumentType,
    OptionType,
    OptionsQuote,
    UnderlyingIndex,
)


class DerivativeModelTests(unittest.TestCase):
    def _contract(
        self,
        *,
        exchange=Exchange.NSE,
        underlying=UnderlyingIndex.NIFTY,
        instrument_type=InstrumentType.FUT,
        expiry=date(2030, 1, 31),
        contract_id="NIFTY-FUT-20300131",
        lot_size=25,
        tick_size=Decimal("0.05"),
        strike=None,
        option_type=None,
    ):
        return DerivativeContract(
            exchange=exchange,
            underlying=underlying,
            instrument_type=instrument_type,
            expiry=expiry,
            contract_id=contract_id,
            lot_size=lot_size,
            tick_size=tick_size,
            strike=strike,
            option_type=option_type,
        )

    def _quote_fields(self, contract):
        return {
            "contract": contract,
            "timestamp": datetime(2030, 1, 1, 10, 0, tzinfo=timezone.utc),
            "bid": Decimal("99.95"),
            "ask": Decimal("100.00"),
            "last_traded_price": Decimal("99.95"),
            "volume": 120,
            "open_interest": 300,
            "change_in_open_interest": -10,
        }

    def test_valid_futures_contract(self):
        contract = self._contract()

        self.assertEqual(contract.instrument_type, InstrumentType.FUT)
        self.assertIsNone(contract.strike)
        self.assertIsNone(contract.option_type)

    def test_valid_banknifty_contract(self):
        contract = self._contract(underlying=UnderlyingIndex.BANKNIFTY)

        self.assertEqual(contract.exchange, Exchange.NSE)
        self.assertEqual(contract.underlying, UnderlyingIndex.BANKNIFTY)

    def test_valid_ce_contract(self):
        contract = self._contract(
            instrument_type=InstrumentType.OPT,
            contract_id="NIFTY-CE-20300131",
            strike=Decimal("25123.45"),
            option_type=OptionType.CE,
        )

        self.assertEqual(contract.option_type, OptionType.CE)
        self.assertEqual(contract.strike, Decimal("25123.45"))

    def test_valid_pe_contract(self):
        contract = self._contract(
            exchange=Exchange.BSE,
            underlying=UnderlyingIndex.SENSEX,
            instrument_type=InstrumentType.OPT,
            contract_id="SENSEX-PE-20300131",
            strike=Decimal("80000"),
            option_type=OptionType.PE,
        )

        self.assertEqual(contract.exchange, Exchange.BSE)
        self.assertEqual(contract.option_type, OptionType.PE)

    def test_futures_reject_option_fields(self):
        for fields in (
            {"strike": Decimal("25000")},
            {"option_type": OptionType.CE},
        ):
            with self.subTest(fields=fields):
                with self.assertRaisesRegex(ValueError, "futures must not have"):
                    self._contract(**fields)

    def test_options_require_strike_and_option_type(self):
        base = {"instrument_type": InstrumentType.OPT}
        cases = (
            {"strike": None, "option_type": OptionType.CE},
            {"strike": Decimal("25000"), "option_type": None},
        )
        for option_fields in cases:
            with self.subTest(option_fields=option_fields):
                with self.assertRaises(ValueError):
                    self._contract(**base, **option_fields)

    def test_rejects_exchange_underlying_mismatch(self):
        with self.assertRaisesRegex(ValueError, "SENSEX contracts must use the BSE exchange"):
            self._contract(exchange=Exchange.NSE, underlying=UnderlyingIndex.SENSEX)

    def test_rejects_invalid_lot_sizes(self):
        for lot_size in (0, -1, True, 1.5):
            with self.subTest(lot_size=lot_size):
                with self.assertRaisesRegex(ValueError, "lot_size must be a positive integer"):
                    self._contract(lot_size=lot_size)

    def test_rejects_invalid_tick_sizes(self):
        for tick_size in (
            Decimal("0"),
            Decimal("-0.05"),
            Decimal("NaN"),
            Decimal("Infinity"),
            0.05,
        ):
            with self.subTest(tick_size=tick_size):
                with self.assertRaises(ValueError):
                    self._contract(tick_size=tick_size)

    def test_rejects_invalid_strikes(self):
        for strike in (None, Decimal("0"), Decimal("-1"), Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(strike=strike):
                with self.assertRaises(ValueError):
                    self._contract(
                        instrument_type=InstrumentType.OPT,
                        strike=strike,
                        option_type=OptionType.CE,
                    )

    def test_rejects_invalid_contract_identity(self):
        for contract_id in ("", " ", " NIFTY", "NIFTY ", "NIFTY 50"):
            with self.subTest(contract_id=contract_id):
                with self.assertRaisesRegex(ValueError, "contract_id"):
                    self._contract(contract_id=contract_id)

    def test_rejects_invalid_expiry(self):
        for expiry in ("2030-01-31", datetime(2030, 1, 31)):
            with self.subTest(expiry=expiry):
                with self.assertRaisesRegex(ValueError, "expiry must be a date"):
                    self._contract(expiry=expiry)

    def test_valid_futures_and_options_quotes(self):
        futures = FuturesQuote(**self._quote_fields(self._contract()))
        options_contract = self._contract(
            instrument_type=InstrumentType.OPT,
            contract_id="NIFTY-CE-20300131",
            strike=Decimal("25000"),
            option_type=OptionType.CE,
        )
        options = OptionsQuote(**self._quote_fields(options_contract))

        self.assertEqual(futures.contract.instrument_type, InstrumentType.FUT)
        self.assertEqual(options.contract.instrument_type, InstrumentType.OPT)

    def test_quote_models_require_matching_contract_type(self):
        futures_contract = self._contract()
        options_contract = self._contract(
            instrument_type=InstrumentType.OPT,
            contract_id="NIFTY-CE-20300131",
            strike=Decimal("25000"),
            option_type=OptionType.CE,
        )

        with self.assertRaisesRegex(ValueError, "requires an OPT contract"):
            OptionsQuote(**self._quote_fields(futures_contract))
        with self.assertRaisesRegex(ValueError, "requires a FUT contract"):
            FuturesQuote(**self._quote_fields(options_contract))

    def test_rejects_invalid_quote_prices_and_bid_ask(self):
        contract = self._contract()
        for field in ("bid", "ask", "last_traded_price"):
            for price in (Decimal("-0.01"), Decimal("NaN"), Decimal("Infinity")):
                with self.subTest(field=field, price=price):
                    quote_fields = self._quote_fields(contract)
                    quote_fields[field] = price
                    with self.assertRaises(ValueError):
                        FuturesQuote(**quote_fields)

        quote_fields = self._quote_fields(contract)
        quote_fields["bid"] = Decimal("101")
        with self.assertRaisesRegex(ValueError, "bid must not exceed ask"):
            FuturesQuote(**quote_fields)

    def test_rejects_invalid_quote_timestamp(self):
        quote_fields = self._quote_fields(self._contract())
        quote_fields["timestamp"] = "2030-01-01T10:00:00"

        with self.assertRaisesRegex(ValueError, "timestamp must be a datetime"):
            FuturesQuote(**quote_fields)

    def test_rejects_naive_quote_timestamp(self):
        quote_fields = self._quote_fields(self._contract())
        quote_fields["timestamp"] = datetime(2030, 1, 1, 10, 0)

        with self.assertRaisesRegex(ValueError, "timestamp must be timezone-aware"):
            FuturesQuote(**quote_fields)

    def test_accepts_utc_quote_timestamp(self):
        timestamp = datetime(2030, 1, 1, 10, 0, tzinfo=timezone.utc)
        quote_fields = self._quote_fields(self._contract())
        quote_fields["timestamp"] = timestamp

        self.assertEqual(FuturesQuote(**quote_fields).timestamp, timestamp)

    def test_accepts_ist_quote_timestamp(self):
        ist = timezone(timedelta(hours=5, minutes=30))
        timestamp = datetime(2030, 1, 1, 10, 0, tzinfo=ist)
        contract = self._contract(
            instrument_type=InstrumentType.OPT,
            contract_id="NIFTY-CE-20300131",
            strike=Decimal("25000"),
            option_type=OptionType.CE,
        )
        quote_fields = self._quote_fields(contract)
        quote_fields["timestamp"] = timestamp

        self.assertEqual(OptionsQuote(**quote_fields).timestamp, timestamp)

    def test_allows_each_optional_quote_price_to_be_missing(self):
        for field in ("bid", "ask", "last_traded_price"):
            with self.subTest(field=field):
                quote_fields = self._quote_fields(self._contract())
                quote_fields[field] = None
                quote = FuturesQuote(**quote_fields)
                self.assertIsNone(getattr(quote, field))

        quote_fields = self._quote_fields(self._contract())
        quote_fields.update(bid=None, ask=None, last_traded_price=None)
        quote = FuturesQuote(**quote_fields)
        self.assertIsNone(quote.bid)
        self.assertIsNone(quote.ask)
        self.assertIsNone(quote.last_traded_price)

    def test_accepts_all_quote_prices_when_tick_aligned(self):
        quote = FuturesQuote(**self._quote_fields(self._contract()))

        self.assertEqual(quote.bid, Decimal("99.95"))
        self.assertEqual(quote.ask, Decimal("100.00"))
        self.assertEqual(quote.last_traded_price, Decimal("99.95"))

    def test_rejects_quote_prices_not_aligned_to_tick_size(self):
        for field in ("bid", "ask", "last_traded_price"):
            with self.subTest(field=field):
                quote_fields = self._quote_fields(self._contract())
                quote_fields[field] = Decimal("99.96")
                with self.assertRaisesRegex(ValueError, "exact multiple of contract tick_size"):
                    FuturesQuote(**quote_fields)

    def test_rejects_invalid_quote_price_types(self):
        for field in ("bid", "ask", "last_traded_price"):
            for value in (0.05, "99.95"):
                with self.subTest(field=field, value=value):
                    quote_fields = self._quote_fields(self._contract())
                    quote_fields[field] = value
                    with self.assertRaisesRegex(ValueError, f"{field} must be a Decimal"):
                        FuturesQuote(**quote_fields)

    def test_rejects_negative_volume_and_open_interest(self):
        for field in ("volume", "open_interest"):
            with self.subTest(field=field):
                quote_fields = self._quote_fields(self._contract())
                quote_fields[field] = -1
                with self.assertRaisesRegex(ValueError, f"{field} must be a non-negative integer"):
                    FuturesQuote(**quote_fields)

    def test_preserves_decimal_precision_and_allows_negative_oi_change(self):
        contract = self._contract(tick_size=Decimal("0.00000000000000000001"))
        quote_fields = self._quote_fields(contract)
        quote_fields["bid"] = Decimal("0.10000000000000000001")
        quote_fields["ask"] = Decimal("0.10000000000000000002")
        quote_fields["last_traded_price"] = Decimal("0.10000000000000000001")
        quote = FuturesQuote(**quote_fields)

        self.assertIsInstance(quote.bid, Decimal)
        self.assertEqual(quote.bid, Decimal("0.10000000000000000001"))
        self.assertEqual(quote.change_in_open_interest, -10)

    def test_models_are_immutable(self):
        contract = self._contract()
        quote = FuturesQuote(**self._quote_fields(contract))

        with self.assertRaises(FrozenInstanceError):
            contract.lot_size = 1
        with self.assertRaises(FrozenInstanceError):
            quote.bid = Decimal("1")


if __name__ == "__main__":
    unittest.main()
