"""Futures basis, carry, and rollover tests using explicit synthetic fixtures."""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import math
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.analytics import (
    CarryRatePeriod,
    FuturesObservation,
    UnderlyingSpotObservation,
    analyze_cost_of_carry,
    analyze_futures_basis,
    analyze_rollover,
)
from atlas.derivatives.models import (
    DerivativeContract,
    Exchange,
    FuturesQuote,
    InstrumentType,
    UnderlyingIndex,
)
from atlas.market_data.models import FuturesQuoteResponse


class FuturesAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.expiry = date(2027, 1, 1)
        self.expiry_at = datetime(2027, 1, 1, tzinfo=timezone.utc)

    def _contract(
        self,
        expiry=None,
        *,
        underlying=UnderlyingIndex.NIFTY,
        exchange=None,
        contract_id=None,
    ):
        expiry = self.expiry if expiry is None else expiry
        if exchange is None:
            exchange = (
                Exchange.BSE
                if underlying is UnderlyingIndex.SENSEX
                else Exchange.NSE
            )
        return DerivativeContract(
            exchange=exchange,
            underlying=underlying,
            instrument_type=InstrumentType.FUT,
            expiry=expiry,
            contract_id=contract_id or f"{underlying.value}-FUT-{expiry:%Y%m%d}",
            lot_size=1,
            tick_size=Decimal("0.05"),
        )

    def _futures(
        self,
        price,
        *,
        expiry=None,
        underlying=UnderlyingIndex.NIFTY,
        timestamp=None,
        data_source="mock-futures",
        is_synthetic=True,
        contract_id=None,
    ):
        contract = self._contract(
            expiry,
            underlying=underlying,
            contract_id=contract_id,
        )
        quote_time = self.start if timestamp is None else timestamp
        last_price = None if price is None else Decimal(str(price))
        quote = FuturesQuote(
            contract=contract,
            timestamp=quote_time,
            bid=Decimal("100.00"),
            ask=Decimal("100.05"),
            last_traded_price=last_price,
            volume=0,
            open_interest=0,
            change_in_open_interest=0,
        )
        response = FuturesQuoteResponse(quote=quote, is_synthetic=is_synthetic)
        return FuturesObservation(response=response, data_source=data_source)

    def _spot(
        self,
        price=100,
        *,
        underlying=UnderlyingIndex.NIFTY,
        observed_at=None,
        data_source="mock-spot",
        is_synthetic=True,
    ):
        return UnderlyingSpotObservation(
            underlying=underlying,
            price=price,
            observed_at=self.start if observed_at is None else observed_at,
            data_source=data_source,
            is_synthetic=is_synthetic,
        )

    def test_basis_and_annualization_known_values_and_provenance(self):
        future = self._futures(100.5, timestamp=self.start)
        spot = self._spot(100, data_source="authorized-spot", is_synthetic=False)

        result = analyze_futures_basis(future, spot, expiry_at=self.expiry_at)

        self.assertEqual(result.absolute_basis, 0.5)
        self.assertEqual(result.percentage_basis, 0.5)
        self.assertEqual(result.annualized_basis_percentage, 0.5)
        self.assertEqual(result.time_to_expiry_years, 1.0)
        self.assertEqual(result.day_count_convention, "ACT/365 Fixed")
        self.assertEqual(result.contract, future.contract)
        self.assertEqual(result.futures_observed_at, future.observed_at)
        self.assertEqual(result.spot_observed_at, spot.observed_at)
        self.assertEqual(result.futures_data_source, "mock-futures")
        self.assertEqual(result.spot_data_source, "authorized-spot")
        self.assertTrue(result.futures_is_synthetic)
        self.assertFalse(result.spot_is_synthetic)

    def test_actual_elapsed_time_uses_act_365_fixed(self):
        halfway_expiry = datetime(2026, 7, 2, 12, tzinfo=timezone.utc)
        contract = self._contract(halfway_expiry.date())
        future = self._futures(100.5, expiry=contract.expiry, timestamp=self.start)

        result = analyze_futures_basis(
            future,
            self._spot(),
            expiry_at=halfway_expiry,
        )

        self.assertEqual(result.time_to_expiry_years, 0.5)
        self.assertEqual(result.annualized_basis_percentage, 1.0)

    def test_basis_rejects_zero_or_negative_time_and_mismatched_expiry_instant(self):
        same_day_expiry = datetime(2026, 1, 1, tzinfo=timezone.utc)
        same_day_future = self._futures(
            100,
            expiry=same_day_expiry.date(),
            timestamp=self.start,
        )
        with self.assertRaisesRegex(ValueError, "time to expiry must be positive"):
            analyze_futures_basis(same_day_future, self._spot(), expiry_at=same_day_expiry)

        past_expiry = datetime(2025, 12, 31, tzinfo=timezone.utc)
        expired_future = self._futures(
            100,
            expiry=past_expiry.date(),
            timestamp=self.start,
        )
        with self.assertRaisesRegex(ValueError, "time to expiry must be positive"):
            analyze_futures_basis(expired_future, self._spot(), expiry_at=past_expiry)

        wrong_date = datetime(2027, 1, 2, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "local date must match"):
            analyze_futures_basis(self._futures(100), self._spot(), expiry_at=wrong_date)

        with self.assertRaisesRegex(ValueError, "expiry_at must be timezone-aware"):
            analyze_futures_basis(
                self._futures(100),
                self._spot(),
                expiry_at=datetime(2027, 1, 1),
            )

    def test_basis_rejects_missing_or_invalid_prices_and_underlyings(self):
        future = self._futures(100)
        for price in (0, -1, math.nan, math.inf, "100", None):
            with self.subTest(spot_price=price):
                with self.assertRaises(ValueError):
                    self._spot(price)

        with self.assertRaisesRegex(ValueError, "spot observed_at must be timezone-aware"):
            self._spot(observed_at=datetime(2026, 1, 1))

        with self.assertRaisesRegex(ValueError, "last_traded_price must be positive"):
            self._futures(0)
        with self.assertRaisesRegex(ValueError, "last_traded_price must be finite"):
            self._futures(math.inf)
        with self.assertRaisesRegex(ValueError, "last_traded_price is missing"):
            self._futures(None)

        with self.assertRaisesRegex(ValueError, "underlyings must match"):
            analyze_futures_basis(
                future,
                self._spot(underlying=UnderlyingIndex.BANKNIFTY),
                expiry_at=self.expiry_at,
            )

    def test_cost_of_carry_uses_explicit_varying_rate_periods(self):
        middle = datetime(2026, 7, 2, 12, tzinfo=timezone.utc)
        contract = self._contract()
        spot = self._spot(100)
        periods = [
            CarryRatePeriod(self.start, middle, 0.05, 0.01),
            CarryRatePeriod(middle, self.expiry_at, 0.04, 0.02),
        ]

        result = analyze_cost_of_carry(
            contract,
            spot,
            expiry_at=self.expiry_at,
            rate_periods=periods,
        )

        self.assertAlmostEqual(
            result.model_implied_fair_value,
            103.0454533953517,
            places=11,
        )
        self.assertEqual(result.time_to_expiry_years, 1.0)
        self.assertEqual(result.formula, "F = S * exp(sum((r_i - q_i) * T_i))")
        self.assertEqual(result.compounding_assumption, "continuous")
        self.assertEqual(result.data_source, "mock-spot")
        self.assertTrue(result.is_synthetic)
        self.assertFalse(hasattr(result, "absolute_basis"))
        self.assertEqual(result.rate_periods, tuple(periods))

    def test_cost_of_carry_requires_explicit_rates_and_dividend_yields(self):
        middle = datetime(2026, 7, 2, 12, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "risk_free_rate"):
            CarryRatePeriod(self.start, middle, None, 0.01)
        with self.assertRaisesRegex(ValueError, "dividend_yield"):
            CarryRatePeriod(self.start, middle, 0.05, None)
        with self.assertRaisesRegex(ValueError, "explicit interest-rate"):
            analyze_cost_of_carry(
                self._contract(),
                self._spot(),
                expiry_at=self.expiry_at,
                rate_periods=(),
            )

    def test_cost_of_carry_requires_complete_contiguous_period_schedule(self):
        middle = datetime(2026, 7, 2, 12, tzinfo=timezone.utc)
        gap = middle + timedelta(seconds=1)
        periods_with_gap = [
            CarryRatePeriod(self.start, middle, 0.05, 0.02),
            CarryRatePeriod(gap, self.expiry_at, 0.04, 0.01),
        ]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            analyze_cost_of_carry(
                self._contract(),
                self._spot(),
                expiry_at=self.expiry_at,
                rate_periods=periods_with_gap,
            )

        with self.assertRaisesRegex(ValueError, "end_at must be later"):
            CarryRatePeriod(middle, middle, 0.05, 0.02)

    def test_cost_of_carry_rejects_expired_or_mismatched_underlying_inputs(self):
        expired_at = self.start
        expired_contract = self._contract(expired_at.date())
        with self.assertRaisesRegex(ValueError, "time to expiry must be positive"):
            analyze_cost_of_carry(
                expired_contract,
                self._spot(),
                expiry_at=expired_at,
                rate_periods=(),
            )

        with self.assertRaisesRegex(ValueError, "underlyings must match"):
            analyze_cost_of_carry(
                self._contract(),
                self._spot(underlying=UnderlyingIndex.BANKNIFTY),
                expiry_at=self.expiry_at,
                rate_periods=(),
            )

    def test_rollover_known_calendar_spread_and_long_cost_sign(self):
        near_expiry = date(2026, 6, 25)
        next_expiry = date(2026, 7, 30)
        near = self._futures(
            100,
            expiry=near_expiry,
            contract_id="NIFTY-NEAR",
            data_source="near-feed",
        )
        next_month = self._futures(
            102,
            expiry=next_expiry,
            contract_id="NIFTY-NEXT",
            data_source="next-feed",
            is_synthetic=False,
        )

        result = analyze_rollover(near, next_month)

        self.assertEqual(result.absolute_calendar_spread, 2.0)
        self.assertEqual(result.percentage_calendar_spread, 2.0)
        self.assertEqual(result.long_rollover_cost, 2.0)
        self.assertIn("positive is a debit", result.sign_convention)
        self.assertEqual(result.near_contract, near.contract)
        self.assertEqual(result.next_contract, next_month.contract)
        self.assertEqual(result.near_observed_at, near.observed_at)
        self.assertEqual(result.next_observed_at, next_month.observed_at)
        self.assertEqual(result.near_data_source, "near-feed")
        self.assertEqual(result.next_data_source, "next-feed")
        self.assertTrue(result.near_is_synthetic)
        self.assertFalse(result.next_is_synthetic)

    def test_backwardation_is_a_credit_for_a_long_roll(self):
        near = self._futures(100, expiry=date(2026, 6, 25))
        next_month = self._futures(
            98,
            expiry=date(2026, 7, 30),
            contract_id="NIFTY-NEXT",
        )

        result = analyze_rollover(near, next_month)

        self.assertEqual(result.absolute_calendar_spread, -2.0)
        self.assertEqual(result.percentage_calendar_spread, -2.0)
        self.assertEqual(result.long_rollover_cost, -2.0)

    def test_rollover_rejects_equal_or_reversed_expiries(self):
        near = self._futures(100, expiry=date(2026, 6, 25))
        equal_expiry = self._futures(
            101,
            expiry=date(2026, 6, 25),
            contract_id="NIFTY-EQUAL",
        )
        earlier_expiry = self._futures(
            101,
            expiry=date(2026, 6, 20),
            contract_id="NIFTY-EARLIER",
        )
        for other in (equal_expiry, earlier_expiry):
            with self.subTest(expiry=other.contract.expiry):
                with self.assertRaisesRegex(ValueError, "expiry must be earlier"):
                    analyze_rollover(near, other)

    def test_rollover_rejects_different_underlyings_and_exchanges(self):
        near = self._futures(100, expiry=date(2026, 6, 25))
        different_underlying_same_exchange = self._futures(
            101,
            expiry=date(2026, 7, 30),
            underlying=UnderlyingIndex.BANKNIFTY,
        )
        with self.assertRaisesRegex(ValueError, "same underlying"):
            analyze_rollover(near, different_underlying_same_exchange)

        different_exchange = self._futures(
            101,
            expiry=date(2026, 7, 30),
            underlying=UnderlyingIndex.SENSEX,
        )
        with self.assertRaisesRegex(ValueError, "same exchange"):
            analyze_rollover(near, different_exchange)

    def test_rollover_rejects_missing_futures_prices(self):
        with self.assertRaisesRegex(ValueError, "last_traded_price is missing"):
            self._futures(None)

    def test_inputs_are_immutable_and_schedule_is_copied(self):
        middle = datetime(2026, 7, 2, 12, tzinfo=timezone.utc)
        periods = [
            CarryRatePeriod(self.start, middle, 0.05, 0.02),
            CarryRatePeriod(middle, self.expiry_at, 0.04, 0.01),
        ]
        before = periods.copy()
        contract = self._contract()
        spot = self._spot()

        result = analyze_cost_of_carry(
            contract,
            spot,
            expiry_at=self.expiry_at,
            rate_periods=periods,
        )

        self.assertEqual(periods, before)
        self.assertIsInstance(result.rate_periods, tuple)
        self.assertEqual(result.rate_periods, tuple(before))
        self.assertEqual(spot.price, 100)
        self.assertEqual(contract.expiry, self.expiry)


if __name__ == "__main__":
    unittest.main()
