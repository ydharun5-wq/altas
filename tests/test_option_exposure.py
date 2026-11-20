"""Tests for explicit option-position exposure and scenario analytics."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.analytics.options import (
    OptionMarketInputs,
    OptionPosition,
    OptionPositionValuation,
    OptionScenario,
    PositionDirection,
    calculate_position_exposure,
    calculate_portfolio_exposure,
    calculate_position_scenario_pnl,
    calculate_portfolio_scenario_pnl,
)
from atlas.derivatives.models import (
    DerivativeContract,
    Exchange,
    InstrumentType,
    OptionType,
    OptionsQuote,
    UnderlyingIndex,
)


UTC = timezone.utc
BASE_AT = datetime(2026, 11, 19, 0, 0, tzinfo=UTC)
EXPIRY_AT = datetime(2027, 11, 19, 0, 0, tzinfo=UTC)
CALCULATED_AT = BASE_AT


def contract(
    *,
    right=OptionType.CE,
    strike=Decimal("100"),
    expiry=date(2027, 11, 19),
    lot_size=50,
    underlying=UnderlyingIndex.NIFTY,
    instrument_type=InstrumentType.OPT,
    exchange=Exchange.NSE,
    suffix="",
):
    return DerivativeContract(
        exchange=exchange,
        underlying=underlying,
        instrument_type=instrument_type,
        expiry=expiry,
        contract_id=(
            f"{underlying.value}-{expiry:%Y%m%d}-{strike}-{right}-{suffix}"
        ),
        lot_size=lot_size,
        tick_size=Decimal("0.05"),
        strike=strike if instrument_type is InstrumentType.OPT else None,
        option_type=right if instrument_type is InstrumentType.OPT else None,
    )


def make_valuation(
    *,
    right=OptionType.CE,
    quantity=2,
    direction=PositionDirection.LONG,
    lot_size=50,
    spot=100.0,
    volatility=0.2,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    observed_at=BASE_AT,
    expiry_at=EXPIRY_AT,
    contract_expiry=date(2027, 11, 19),
    underlying=UnderlyingIndex.NIFTY,
    synthetic=True,
    data_source="synthetic-test",
    contract_override=None,
):
    item_contract = contract(
        right=right,
        expiry=contract_expiry,
        lot_size=lot_size,
        underlying=underlying,
        suffix="a",
    ) if contract_override is None else contract_override
    position = OptionPosition(
        contract=item_contract,
        direction=direction,
        quantity=quantity,
        observed_at=observed_at,
        data_source=data_source,
        is_synthetic=synthetic,
    )
    market = OptionMarketInputs(
        contract=item_contract,
        underlying_price=spot,
        implied_volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        observed_at=observed_at,
        expiry_at=expiry_at,
        data_source=data_source,
        is_synthetic=synthetic,
    )
    return OptionPositionValuation(position=position, market=market)


def reference_call_price(spot, strike, years, rate, dividend, volatility):
    """Independent textbook Black-Scholes calculation for test expectations."""
    root_time = math.sqrt(years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend + volatility * volatility / 2.0) * years
    ) / (volatility * root_time)
    d2 = d1 - volatility * root_time
    cdf = lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))
    return (
        spot * math.exp(-dividend * years) * cdf(d1)
        - strike * math.exp(-rate * years) * cdf(d2)
    )


class OptionExposureTests(unittest.TestCase):
    def test_known_greek_exposures_apply_quantity_and_contract_lot(self):
        result = calculate_position_exposure(
            make_valuation(), calculated_at=CALCULATED_AT
        )
        self.assertEqual(result.signed_underlying_units, 100)
        self.assertEqual(result.quantity_contracts, 2)
        self.assertEqual(result.lot_size_multiplier, 50)
        self.assertAlmostEqual(result.delta_exposure, 63.68306511756191, places=10)
        self.assertAlmostEqual(result.gamma_exposure, 1.87620173458469, places=11)
        self.assertAlmostEqual(result.vega_exposure, 3752.40346916938, places=8)
        self.assertAlmostEqual(result.theta_exposure, -1.75726782094, places=9)
        self.assertIn("actual supplied", result.exposure_basis)

    def test_short_exposures_are_exact_sign_inverse_of_long(self):
        long_result = calculate_position_exposure(
            make_valuation(direction=PositionDirection.LONG),
            calculated_at=CALCULATED_AT,
        )
        short_result = calculate_position_exposure(
            make_valuation(direction=PositionDirection.SHORT),
            calculated_at=CALCULATED_AT,
        )
        for field in (
            "delta_exposure",
            "gamma_exposure",
            "vega_exposure",
            "theta_exposure",
        ):
            self.assertAlmostEqual(
                getattr(short_result, field), -getattr(long_result, field)
            )

    def test_quantity_and_lot_size_scale_exposure(self):
        one_contract = calculate_position_exposure(
            make_valuation(quantity=1), calculated_at=CALCULATED_AT
        )
        two_contracts = calculate_position_exposure(
            make_valuation(quantity=2), calculated_at=CALCULATED_AT
        )
        larger_lot = calculate_position_exposure(
            make_valuation(quantity=2, lot_size=100),
            calculated_at=CALCULATED_AT,
        )
        self.assertAlmostEqual(two_contracts.delta_exposure, 2 * one_contract.delta_exposure)
        self.assertAlmostEqual(larger_lot.delta_exposure, 2 * two_contracts.delta_exposure)

    def test_actual_position_quantity_is_separate_from_public_oi(self):
        valuation = make_valuation(quantity=2)
        public_quote = OptionsQuote(
            contract=valuation.position.contract,
            timestamp=BASE_AT,
            bid=Decimal("9.95"),
            ask=Decimal("10.05"),
            last_traded_price=Decimal("10.00"),
            volume=1200,
            open_interest=900000,
            change_in_open_interest=1000,
        )
        # The exposure API accepts explicit position quantity only. Public OI
        # is neither the position quantity nor a direction assumption.
        result = calculate_position_exposure(
            valuation, calculated_at=CALCULATED_AT
        )
        self.assertEqual(public_quote.open_interest, 900000)
        self.assertEqual(result.quantity_contracts, 2)
        self.assertEqual(result.exposure_basis, "actual supplied signed position quantity")

    def test_synthetic_status_sources_and_timestamps_are_preserved(self):
        result = calculate_position_exposure(
            make_valuation(data_source="synthetic-fixture", synthetic=True),
            calculated_at=CALCULATED_AT,
        )
        self.assertEqual(result.position_data_source, "synthetic-fixture")
        self.assertEqual(result.market_data_source, "synthetic-fixture")
        self.assertTrue(result.position_is_synthetic)
        self.assertTrue(result.market_is_synthetic)
        self.assertEqual(result.calculated_at, CALCULATED_AT)
        self.assertEqual(result.market_observed_at, BASE_AT)
        self.assertEqual(result.time_to_expiry_convention, "ACT/365F between UTC instants")

    def test_non_synthetic_inputs_are_identified_as_such(self):
        result = calculate_position_exposure(
            make_valuation(synthetic=False, data_source="explicit-market-input"),
            calculated_at=CALCULATED_AT,
        )
        self.assertFalse(result.position_is_synthetic)
        self.assertFalse(result.market_is_synthetic)

    def test_portfolio_exposure_sums_explicit_positions(self):
        long_position = make_valuation(quantity=1)
        short_position = make_valuation(
            quantity=1, direction=PositionDirection.SHORT, right=OptionType.PE
        )
        result = calculate_portfolio_exposure(
            [long_position, short_position], calculated_at=CALCULATED_AT
        )
        expected_delta = sum(
            calculate_position_exposure(item, calculated_at=CALCULATED_AT).delta_exposure
            for item in (long_position, short_position)
        )
        self.assertEqual(len(result.positions), 2)
        self.assertAlmostEqual(result.total_delta_exposure, expected_delta)
        self.assertTrue(result.contains_synthetic_inputs)

    def test_rejects_empty_and_mixed_underlying_portfolios(self):
        with self.assertRaises(ValueError):
            calculate_portfolio_exposure([], calculated_at=CALCULATED_AT)
        other_underlying = make_valuation(underlying=UnderlyingIndex.BANKNIFTY)
        with self.assertRaises(ValueError):
            calculate_portfolio_exposure(
                [make_valuation(), other_underlying],
                calculated_at=CALCULATED_AT,
            )

    def test_rejects_missing_or_invalid_quantity_and_direction(self):
        values = make_valuation()
        for quantity in (0, -1, True, 1.5, None):
            with self.subTest(quantity=quantity):
                with self.assertRaises(ValueError):
                    OptionPosition(
                        contract=values.position.contract,
                        direction=PositionDirection.LONG,
                        quantity=quantity,
                        observed_at=BASE_AT,
                        data_source="test",
                        is_synthetic=True,
                    )
        with self.assertRaises(ValueError):
            OptionPosition(
                contract=values.position.contract,
                direction="long",
                quantity=1,
                observed_at=BASE_AT,
                data_source="test",
                is_synthetic=True,
            )

    def test_rejects_non_option_contract_and_contract_mismatch(self):
        future = contract(instrument_type=InstrumentType.FUT)
        with self.assertRaises(ValueError):
            OptionPosition(
                contract=future,
                direction=PositionDirection.LONG,
                quantity=1,
                observed_at=BASE_AT,
                data_source="test",
                is_synthetic=True,
            )
        other_contract = contract(strike=Decimal("105"), suffix="different")
        market = OptionMarketInputs(
            contract=other_contract,
            underlying_price=100.0,
            implied_volatility=0.2,
            risk_free_rate=0.05,
            dividend_yield=0.0,
            observed_at=BASE_AT,
            expiry_at=EXPIRY_AT,
            data_source="test",
            is_synthetic=True,
        )
        with self.assertRaises(ValueError):
            OptionPositionValuation(make_valuation().position, market)

    def test_rejects_missing_nonfinite_and_nonpositive_market_inputs(self):
        valid = make_valuation()
        base = valid.market
        invalid_cases = (
            {"underlying_price": None},
            {"underlying_price": 0.0},
            {"underlying_price": math.inf},
            {"implied_volatility": None},
            {"implied_volatility": 0.0},
            {"implied_volatility": math.nan},
            {"risk_free_rate": None},
            {"risk_free_rate": math.inf},
            {"dividend_yield": None},
            {"dividend_yield": math.nan},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                values = {
                    "contract": base.contract,
                    "underlying_price": base.underlying_price,
                    "implied_volatility": base.implied_volatility,
                    "risk_free_rate": base.risk_free_rate,
                    "dividend_yield": base.dividend_yield,
                    "observed_at": base.observed_at,
                    "expiry_at": base.expiry_at,
                    "data_source": base.data_source,
                    "is_synthetic": base.is_synthetic,
                }
                values.update(overrides)
                with self.assertRaises(ValueError):
                    OptionMarketInputs(**values)

    def test_rejects_naive_timestamps_and_expiry_date_mismatch(self):
        values = make_valuation()
        with self.assertRaises(ValueError):
            OptionMarketInputs(
                contract=values.market.contract,
                underlying_price=100,
                implied_volatility=0.2,
                risk_free_rate=0.05,
                dividend_yield=0,
                observed_at=BASE_AT.replace(tzinfo=None),
                expiry_at=EXPIRY_AT,
                data_source="test",
                is_synthetic=True,
            )
        with self.assertRaises(ValueError):
            OptionMarketInputs(
                contract=values.market.contract,
                underlying_price=100,
                implied_volatility=0.2,
                risk_free_rate=0.05,
                dividend_yield=0,
                observed_at=BASE_AT,
                expiry_at=datetime(2027, 11, 20, tzinfo=UTC),
                data_source="test",
                is_synthetic=True,
            )

    def test_rejects_zero_time_to_expiry_and_calculation_at_expiry(self):
        values = make_valuation()
        with self.assertRaises(ValueError):
            OptionMarketInputs(
                contract=values.market.contract,
                underlying_price=100,
                implied_volatility=0.2,
                risk_free_rate=0.05,
                dividend_yield=0,
                observed_at=EXPIRY_AT,
                expiry_at=EXPIRY_AT,
                data_source="test",
                is_synthetic=True,
            )
        with self.assertRaises(ValueError):
            calculate_position_exposure(values, calculated_at=EXPIRY_AT)

    def test_input_objects_remain_immutable(self):
        values = make_valuation()
        original_quantity = values.position.quantity
        original_spot = values.market.underlying_price
        calculate_position_exposure(values, calculated_at=CALCULATED_AT)
        self.assertEqual(values.position.quantity, original_quantity)
        self.assertEqual(values.market.underlying_price, original_spot)
        with self.assertRaises(FrozenInstanceError):
            values.position.quantity = 10


class OptionScenarioTests(unittest.TestCase):
    def test_zero_shock_has_zero_model_value_change(self):
        result = calculate_position_scenario_pnl(
            make_valuation(),
            OptionScenario(0.0, 0.0, BASE_AT),
            calculated_at=CALCULATED_AT,
        )
        self.assertAlmostEqual(result.base_option_price, 10.450583572185565, places=11)
        self.assertAlmostEqual(result.scenario_option_price, result.base_option_price)
        self.assertAlmostEqual(result.scenario_pnl, 0.0, places=12)

    def test_underlying_shock_scenario_matches_independent_reference(self):
        result = calculate_position_scenario_pnl(
            make_valuation(),
            OptionScenario(10.0, 0.0, BASE_AT),
            calculated_at=CALCULATED_AT,
        )
        expected_base = reference_call_price(100, 100, 1, 0.05, 0, 0.2)
        expected_shocked = reference_call_price(110, 100, 1, 0.05, 0, 0.2)
        self.assertAlmostEqual(result.base_option_price, expected_base, places=11)
        self.assertAlmostEqual(result.scenario_option_price, expected_shocked, places=11)
        self.assertAlmostEqual(
            result.scenario_pnl, 100 * (expected_shocked - expected_base), places=9
        )
        self.assertEqual(result.pnl_basis, "scenario model value minus base model value")

    def test_volatility_shock_scenario_reprices_using_additive_decimal_change(self):
        result = calculate_position_scenario_pnl(
            make_valuation(),
            OptionScenario(0.0, 0.05, BASE_AT),
            calculated_at=CALCULATED_AT,
        )
        expected = reference_call_price(100, 100, 1, 0.05, 0, 0.25)
        self.assertAlmostEqual(result.scenario_implied_volatility, 0.25)
        self.assertAlmostEqual(result.scenario_option_price, expected, places=11)
        self.assertGreater(result.scenario_pnl, 0.0)

    def test_time_decay_scenario_reprices_with_actual_elapsed_time(self):
        scenario_at = BASE_AT + timedelta(days=30)
        result = calculate_position_scenario_pnl(
            make_valuation(),
            OptionScenario(0.0, 0.0, scenario_at),
            calculated_at=CALCULATED_AT,
        )
        expected = reference_call_price(100, 100, 335 / 365, 0.05, 0, 0.2)
        self.assertAlmostEqual(result.scenario_time_to_expiry_years, 335 / 365, places=12)
        self.assertAlmostEqual(result.scenario_option_price, expected, places=11)
        self.assertLess(result.scenario_pnl, 0.0)

    def test_scenario_pnl_is_directionally_symmetric(self):
        scenario = OptionScenario(10.0, 0.05, BASE_AT)
        long_result = calculate_position_scenario_pnl(
            make_valuation(direction=PositionDirection.LONG),
            scenario,
            calculated_at=CALCULATED_AT,
        )
        short_result = calculate_position_scenario_pnl(
            make_valuation(direction=PositionDirection.SHORT),
            scenario,
            calculated_at=CALCULATED_AT,
        )
        self.assertAlmostEqual(short_result.scenario_pnl, -long_result.scenario_pnl)

    def test_portfolio_scenario_sums_and_identifies_synthetic_inputs(self):
        scenario = OptionScenario(5.0, 0.02, BASE_AT + timedelta(days=1))
        items = (make_valuation(), make_valuation(right=OptionType.PE, quantity=1))
        result = calculate_portfolio_scenario_pnl(
            iter(items), scenario, calculated_at=CALCULATED_AT
        )
        expected = sum(
            calculate_position_scenario_pnl(
                item, scenario, calculated_at=CALCULATED_AT
            ).scenario_pnl
            for item in items
        )
        self.assertAlmostEqual(result.total_scenario_pnl, expected)
        self.assertTrue(result.contains_synthetic_inputs)

    def test_rejects_invalid_shocked_spot_volatility_and_scenario_time(self):
        values = make_valuation()
        invalid_scenarios = (
            OptionScenario(-100.0, 0.0, BASE_AT),
            OptionScenario(0.0, -0.2, BASE_AT),
            OptionScenario(0.0, 0.0, BASE_AT - timedelta(seconds=1)),
            OptionScenario(0.0, 0.0, EXPIRY_AT),
        )
        for scenario in invalid_scenarios:
            with self.subTest(scenario=scenario):
                with self.assertRaises(ValueError):
                    calculate_position_scenario_pnl(
                        values, scenario, calculated_at=CALCULATED_AT
                    )

    def test_scenario_requires_future_or_current_calculation_instant(self):
        with self.assertRaises(ValueError):
            calculate_position_scenario_pnl(
                make_valuation(),
                OptionScenario(0.0, 0.0, BASE_AT),
                calculated_at=BASE_AT + timedelta(seconds=1),
            )


if __name__ == "__main__":
    unittest.main()
