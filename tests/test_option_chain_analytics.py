"""Deterministic option-chain analytics tests using synthetic quote fixtures."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.analytics.options import (
    OptionChain,
    OptionStrike,
    RatioStatus,
    analyze_option_chain,
)
from atlas.derivatives.models import (
    DerivativeContract,
    Exchange,
    InstrumentType,
    OptionType,
    OptionsQuote,
    UnderlyingIndex,
)


class OptionChainAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.expiry = date(2027, 6, 25)
        self.observed_at = datetime(2026, 11, 19, 10, 0, tzinfo=timezone.utc)

    def _quote(
        self,
        right,
        strike,
        *,
        open_interest=0,
        change_in_open_interest=0,
        volume=0,
        timestamp=None,
        bid=Decimal("100.00"),
        ask=Decimal("100.05"),
        last_traded_price=Decimal("100.00"),
        contract_id=None,
        underlying=UnderlyingIndex.NIFTY,
        expiry=None,
    ):
        strike = strike if isinstance(strike, Decimal) else Decimal(str(strike))
        expiry = self.expiry if expiry is None else expiry
        exchange = (
            Exchange.BSE
            if underlying is UnderlyingIndex.SENSEX
            else Exchange.NSE
        )
        right_name = right.value
        contract = DerivativeContract(
            exchange=exchange,
            underlying=underlying,
            instrument_type=InstrumentType.OPT,
            expiry=expiry,
            contract_id=contract_id
            or f"{underlying.value}-{expiry:%Y%m%d}-{strike}-{right_name}",
            lot_size=1,
            tick_size=Decimal("0.05"),
            strike=strike,
            option_type=right,
        )
        quote_time = (
            self.observed_at - timedelta(seconds=1)
            if timestamp is None
            else timestamp
        )
        return OptionsQuote(
            contract=contract,
            timestamp=quote_time,
            bid=bid,
            ask=ask,
            last_traded_price=last_traded_price,
            volume=volume,
            open_interest=open_interest,
            change_in_open_interest=change_in_open_interest,
        )

    def _row(
        self,
        strike,
        *,
        call_oi=0,
        call_change=0,
        call_volume=0,
        put_oi=0,
        put_change=0,
        put_volume=0,
        call_quote_options=None,
        put_quote_options=None,
    ):
        call_options = {
            "open_interest": call_oi,
            "change_in_open_interest": call_change,
            "volume": call_volume,
        }
        put_options = {
            "open_interest": put_oi,
            "change_in_open_interest": put_change,
            "volume": put_volume,
        }
        if call_quote_options:
            call_options.update(call_quote_options)
        if put_quote_options:
            put_options.update(put_quote_options)
        return OptionStrike(
            strike=Decimal(str(strike)),
            call_quote=self._quote(OptionType.CE, strike, **call_options),
            put_quote=self._quote(OptionType.PE, strike, **put_options),
        )

    def _chain(self, strikes, *, is_synthetic=True, **kwargs):
        values = {
            "underlying": UnderlyingIndex.NIFTY,
            "expiry": self.expiry,
            "observed_at": self.observed_at,
            "data_source": "mock-fixture",
            "is_synthetic": is_synthetic,
            "strikes": strikes,
        }
        values.update(kwargs)
        return OptionChain(**values)

    def test_known_oi_pcr_and_volume_totals(self):
        chain = self._chain(
            [
                self._row(
                    100,
                    call_oi=40,
                    call_change=3,
                    call_volume=100,
                    put_oi=80,
                    put_change=-1,
                    put_volume=200,
                ),
                self._row(
                    105,
                    call_oi=60,
                    call_change=-2,
                    call_volume=300,
                    put_oi=70,
                    put_change=5,
                    put_volume=100,
                ),
            ]
        )

        result = analyze_option_chain(chain)

        self.assertEqual(result.total_call_open_interest, 100)
        self.assertEqual(result.total_put_open_interest, 150)
        self.assertEqual(result.total_call_change_in_open_interest, 1)
        self.assertEqual(result.total_put_change_in_open_interest, 4)
        self.assertEqual(result.total_call_volume, 400)
        self.assertEqual(result.total_put_volume, 300)
        self.assertEqual(result.oi_pcr.status, RatioStatus.AVAILABLE)
        self.assertEqual(result.oi_pcr.value, 1.5)
        self.assertEqual(result.volume_pcr.status, RatioStatus.AVAILABLE)
        self.assertEqual(result.volume_pcr.value, 0.75)

    def test_strike_summaries_compare_rights_and_are_sorted(self):
        chain = self._chain(
            [
                self._row(
                    105,
                    call_oi=60,
                    call_change=-2,
                    call_volume=300,
                    put_oi=70,
                    put_change=5,
                    put_volume=100,
                ),
                self._row(
                    100,
                    call_oi=40,
                    call_change=3,
                    call_volume=100,
                    put_oi=80,
                    put_change=-1,
                    put_volume=200,
                ),
            ]
        )

        result = analyze_option_chain(chain)

        self.assertEqual([row.strike for row in result.strikes], [Decimal("100"), Decimal("105")])
        first = result.strikes[0]
        self.assertEqual(first.call.open_interest, 40)
        self.assertEqual(first.put.open_interest, 80)
        self.assertEqual(first.call_minus_put_open_interest, -40)
        self.assertEqual(first.call_minus_put_change_in_open_interest, 4)
        self.assertEqual(first.call_minus_put_volume, -100)
        self.assertIn("Descriptive comparison only", first.descriptive_observation)
        self.assertIn("not a trading signal", first.descriptive_observation)
        self.assertIn("PE open interest exceeds CE", first.descriptive_observation)

    def test_zero_call_denominators_are_explicit_for_both_pcrs(self):
        result = analyze_option_chain(
            self._chain(
                [self._row(100, call_oi=0, call_volume=0, put_oi=20, put_volume=10)]
            )
        )

        self.assertEqual(result.oi_pcr.status, RatioStatus.ZERO_DENOMINATOR)
        self.assertIsNone(result.oi_pcr.value)
        self.assertEqual(result.volume_pcr.status, RatioStatus.ZERO_DENOMINATOR)
        self.assertIsNone(result.volume_pcr.value)

    def test_zero_put_totals_return_zero_pcr_when_denominators_are_nonzero(self):
        result = analyze_option_chain(
            self._chain(
                [self._row(100, call_oi=20, call_volume=10, put_oi=0, put_volume=0)]
            )
        )

        self.assertEqual(result.oi_pcr.status, RatioStatus.AVAILABLE)
        self.assertEqual(result.oi_pcr.value, 0.0)
        self.assertEqual(result.volume_pcr.value, 0.0)

    def test_missing_market_prices_are_preserved_and_not_fabricated(self):
        chain = self._chain(
            [
                self._row(
                    100,
                    call_oi=12,
                    call_volume=34,
                    call_quote_options={
                        "bid": None,
                        "ask": None,
                        "last_traded_price": None,
                    },
                )
            ]
        )

        result = analyze_option_chain(chain)

        self.assertIsNone(result.strikes[0].call.bid)
        self.assertIsNone(result.strikes[0].call.ask)
        self.assertIsNone(result.strikes[0].call.last_traded_price)
        self.assertEqual(result.strikes[0].call.open_interest, 12)
        self.assertEqual(result.strikes[0].call.volume, 34)

    def test_missing_option_side_or_required_oi_is_rejected(self):
        call = self._quote(OptionType.CE, 100)
        with self.assertRaisesRegex(ValueError, "put_quote"):
            OptionStrike(
                strike=Decimal("100"),
                call_quote=call,
                put_quote=None,
            )

        with self.assertRaisesRegex(ValueError, "open_interest"):
            self._quote(OptionType.CE, 100, open_interest=None)
        with self.assertRaisesRegex(ValueError, "volume"):
            self._quote(OptionType.CE, 100, volume=None)

    def test_extreme_pcr_underflow_is_not_reported_as_zero(self):
        chain = self._chain(
            [
                self._row(
                    100,
                    call_oi=10**400,
                    put_oi=1,
                    call_volume=1,
                    put_volume=1,
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "below the supported numeric range"):
            analyze_option_chain(chain)

    def test_duplicate_strikes_and_contract_identifiers_are_rejected(self):
        pair = self._row(100)
        with self.assertRaisesRegex(ValueError, "duplicate option-chain strike"):
            self._chain([pair, pair])

        first = self._row(100)
        second = self._row(
            105,
            call_quote_options={"contract_id": "SHARED-CALL-ID"},
        )
        third = self._row(
            110,
            call_quote_options={"contract_id": "SHARED-CALL-ID"},
        )
        with self.assertRaisesRegex(ValueError, "duplicate option contract identifier"):
            self._chain([first, second, third])

    def test_rejects_inconsistent_contracts_and_invalid_chain_metadata(self):
        call = self._quote(OptionType.CE, 100)
        wrong_strike_put = self._quote(OptionType.PE, 105)
        with self.assertRaisesRegex(ValueError, "strike does not match"):
            OptionStrike(Decimal("100"), call, wrong_strike_put)

        banknifty_put = self._quote(
            OptionType.PE,
            100,
            underlying=UnderlyingIndex.BANKNIFTY,
        )
        with self.assertRaisesRegex(ValueError, "share exchange, underlying, and expiry"):
            OptionStrike(Decimal("100"), call, banknifty_put)

        banknifty_row = self._row(
            100,
            call_quote_options={"underlying": UnderlyingIndex.BANKNIFTY},
            put_quote_options={"underlying": UnderlyingIndex.BANKNIFTY},
        )
        with self.assertRaisesRegex(ValueError, "underlying does not match chain"):
            self._chain([banknifty_row])

        alternate_expiry = date(2027, 7, 30)
        alternate_expiry_row = self._row(
            100,
            call_quote_options={"expiry": alternate_expiry},
            put_quote_options={"expiry": alternate_expiry},
        )
        with self.assertRaisesRegex(ValueError, "expiry does not match chain"):
            self._chain([alternate_expiry_row])

        with self.assertRaisesRegex(ValueError, "data_source"):
            self._chain([self._row(100)], data_source="  ")
        with self.assertRaisesRegex(ValueError, "is_synthetic"):
            self._chain([self._row(100)], is_synthetic=1)
        with self.assertRaisesRegex(ValueError, "expiry must be a date"):
            self._chain([self._row(100)], expiry=datetime(2027, 6, 25))

    def test_observation_timestamps_must_be_aware_and_quotes_not_future(self):
        with self.assertRaisesRegex(ValueError, "observed_at must be timezone-aware"):
            self._chain(
                [self._row(100)],
                observed_at=datetime(2026, 11, 19, 10, 0),
            )

        future_time = self.observed_at + timedelta(seconds=1)
        future_row = OptionStrike(
            strike=Decimal("100"),
            call_quote=self._quote(OptionType.CE, 100, timestamp=future_time),
            put_quote=self._quote(OptionType.PE, 100),
        )
        with self.assertRaisesRegex(ValueError, "later than observed_at"):
            self._chain([future_row])

    def test_chain_copies_and_sorts_rows_without_mutating_input(self):
        original_rows = [self._row(105), self._row(100)]
        before = original_rows.copy()

        chain = self._chain(original_rows)
        result = analyze_option_chain(chain)

        self.assertEqual(original_rows, before)
        self.assertIsInstance(chain.strikes, tuple)
        self.assertEqual([row.strike for row in chain.strikes], [Decimal("100"), Decimal("105")])
        self.assertEqual(len(result.strikes), 2)

    def test_synthetic_and_source_identification_are_preserved(self):
        synthetic = analyze_option_chain(self._chain([self._row(100)]))
        real_source = analyze_option_chain(
            self._chain(
                [self._row(100)],
                is_synthetic=False,
                data_source="authorized-csv",
            )
        )

        self.assertTrue(synthetic.is_synthetic)
        self.assertEqual(synthetic.data_source, "mock-fixture")
        self.assertFalse(real_source.is_synthetic)
        self.assertEqual(real_source.data_source, "authorized-csv")


if __name__ == "__main__":
    unittest.main()
