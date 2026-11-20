"""Reference and boundary tests for the European options analytics module."""

from decimal import Decimal
import math
import unittest

from atlas.analytics.options import (
    OptionRight,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)


class BlackScholesPricingTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry_years": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.0,
            "volatility": 0.2,
        }

    def test_known_call_price(self):
        # Published textbook Black-Scholes reference for S=K=100, r=5%,
        # sigma=20%, T=1, q=0.
        price = black_scholes_price(OptionRight.CALL, **self.inputs)
        self.assertAlmostEqual(price, 10.450583572185565, places=11)

    def test_known_put_price(self):
        price = black_scholes_price(OptionRight.PUT, **self.inputs)
        self.assertAlmostEqual(price, 5.573526022256971, places=11)

    def test_put_call_parity_with_zero_and_positive_dividend_yield(self):
        for dividend_yield in (0.0, 0.03):
            inputs = {**self.inputs, "dividend_yield": dividend_yield}
            call = black_scholes_price(OptionRight.CALL, **inputs)
            put = black_scholes_price(OptionRight.PUT, **inputs)
            expected = (
                inputs["spot"] * math.exp(-dividend_yield)
                - inputs["strike"] * math.exp(-inputs["risk_free_rate"])
            )
            self.assertAlmostEqual(call - put, expected, places=12)

    def test_positive_dividend_yield_changes_call_and_put_in_expected_directions(self):
        zero_dividend_call = black_scholes_price(
            OptionRight.CALL, **self.inputs
        )
        zero_dividend_put = black_scholes_price(OptionRight.PUT, **self.inputs)
        dividend_inputs = {**self.inputs, "dividend_yield": 0.04}
        dividend_call = black_scholes_price(
            OptionRight.CALL, **dividend_inputs
        )
        dividend_put = black_scholes_price(OptionRight.PUT, **dividend_inputs)
        self.assertLess(dividend_call, zero_dividend_call)
        self.assertGreater(dividend_put, zero_dividend_put)

    def test_decimal_inputs_and_string_option_right(self):
        price = black_scholes_price(
            "call",
            spot=Decimal("100"),
            strike=Decimal("100"),
            time_to_expiry_years=Decimal("1"),
            risk_free_rate=Decimal("0.05"),
            dividend_yield=Decimal("0"),
            volatility=Decimal("0.2"),
        )
        self.assertAlmostEqual(price, 10.450583572185565, places=11)

    def test_far_out_of_money_prices_remain_nonnegative_and_finite(self):
        inputs = {**self.inputs, "spot": 1.0, "strike": 1000.0}
        for right in (OptionRight.CALL, OptionRight.PUT):
            price = black_scholes_price(right, **inputs)
            self.assertGreaterEqual(price, 0.0)
            self.assertTrue(math.isfinite(price))

    def test_rejects_invalid_option_right_and_numeric_inputs(self):
        with self.assertRaises(ValueError):
            black_scholes_price("CE", **self.inputs)
        for field in ("spot", "strike", "time_to_expiry_years", "volatility"):
            for bad_value in (0.0, -1.0):
                with self.subTest(field=field, value=bad_value):
                    with self.assertRaises(ValueError):
                        black_scholes_price(
                            OptionRight.CALL,
                            **{**self.inputs, field: bad_value},
                        )
        for field in self.inputs:
            for bad_value in (math.nan, math.inf, -math.inf, True, "1"):
                with self.subTest(field=field, value=bad_value):
                    with self.assertRaises(ValueError):
                        black_scholes_price(
                            OptionRight.CALL,
                            **{**self.inputs, field: bad_value},
                        )

    def test_rejects_unrepresentable_extreme_intermediates(self):
        with self.assertRaises(ValueError):
            black_scholes_price(
                OptionRight.CALL,
                **{**self.inputs, "risk_free_rate": 1000.0},
            )

    def test_negative_rates_and_dividend_yields_are_supported(self):
        inputs = {
            **self.inputs,
            "risk_free_rate": -0.01,
            "dividend_yield": -0.02,
        }
        for right in (OptionRight.CALL, OptionRight.PUT):
            self.assertGreaterEqual(black_scholes_price(right, **inputs), 0.0)


class BlackScholesGreeksTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry_years": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.0,
            "volatility": 0.2,
        }

    def test_known_call_greeks_and_units(self):
        greeks = black_scholes_greeks(OptionRight.CALL, **self.inputs)
        self.assertAlmostEqual(greeks.delta, 0.6368306511756191, places=12)
        self.assertAlmostEqual(greeks.gamma, 0.0187620173458469, places=13)
        self.assertAlmostEqual(greeks.theta, -0.0175726782094, places=12)
        self.assertAlmostEqual(greeks.vega, 37.5240346916938, places=10)
        self.assertAlmostEqual(greeks.rho, 53.2324815453763, places=10)

    def test_known_put_greeks_and_units(self):
        greeks = black_scholes_greeks(OptionRight.PUT, **self.inputs)
        self.assertAlmostEqual(greeks.delta, -0.3631693488243809, places=12)
        self.assertAlmostEqual(greeks.gamma, 0.0187620173458469, places=13)
        self.assertAlmostEqual(greeks.theta, -0.0045421381478, places=12)
        self.assertAlmostEqual(greeks.vega, 37.5240346916938, places=10)
        self.assertAlmostEqual(greeks.rho, -41.8904609046951, places=10)

    def test_call_put_symmetries_and_expected_signs(self):
        call = black_scholes_greeks(OptionRight.CALL, **self.inputs)
        put = black_scholes_greeks(OptionRight.PUT, **self.inputs)
        self.assertAlmostEqual(call.gamma, put.gamma, places=14)
        self.assertAlmostEqual(call.vega, put.vega, places=12)
        self.assertAlmostEqual(call.delta - put.delta, 1.0, places=14)
        self.assertGreater(call.delta, 0.0)
        self.assertLess(put.delta, 0.0)
        self.assertGreater(call.gamma, 0.0)
        self.assertGreater(call.vega, 0.0)
        self.assertGreater(call.rho, 0.0)
        self.assertLess(put.rho, 0.0)
        self.assertLess(call.theta, 0.0)
        self.assertLess(put.theta, 0.0)

    def test_dividend_yield_adjusts_delta_and_theta(self):
        inputs = {**self.inputs, "dividend_yield": 0.03}
        call = black_scholes_greeks(OptionRight.CALL, **inputs)
        put = black_scholes_greeks(OptionRight.PUT, **inputs)
        self.assertAlmostEqual(
            call.delta - put.delta,
            math.exp(-inputs["dividend_yield"]),
            places=14,
        )
        self.assertTrue(math.isfinite(call.theta))
        self.assertTrue(math.isfinite(put.theta))


class ImpliedVolatilityTests(unittest.TestCase):
    def setUp(self):
        self.inputs = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry_years": 1.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.0,
        }

    def test_recovers_volatility_from_independent_call_and_put_prices(self):
        # Hardcoded values from the Black-Scholes reference example tested
        # above; these are not generated by the inverse solver under test.
        call_vol = implied_volatility(
            OptionRight.CALL,
            option_price=10.450583572185565,
            **self.inputs,
        )
        put_vol = implied_volatility(
            OptionRight.PUT,
            option_price=5.573526022256971,
            **self.inputs,
        )
        self.assertAlmostEqual(call_vol, 0.2, delta=1e-9)
        self.assertAlmostEqual(put_vol, 0.2, delta=1e-9)

    def test_recovers_near_intrinsic_value(self):
        inputs = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry_years": 1.0,
            "risk_free_rate": 0.0,
            "dividend_yield": 0.0,
        }
        # Independent small-volatility expansion at sigma=0.0001:
        # 100 * (2*N(0.00005)-1) ~= 0.003989422804.
        volatility = implied_volatility(
            OptionRight.CALL,
            option_price=0.003989422804,
            **inputs,
        )
        self.assertAlmostEqual(volatility, 0.0001, delta=1e-8)

    def test_rejects_prices_at_or_outside_no_arbitrage_bounds(self):
        with self.assertRaises(ValueError):
            implied_volatility(
                OptionRight.CALL, option_price=0.0, **self.inputs
            )
        with self.assertRaises(ValueError):
            implied_volatility(
                OptionRight.CALL, option_price=100.0, **self.inputs
            )
        with self.assertRaises(ValueError):
            implied_volatility(
                OptionRight.CALL, option_price=101.0, **self.inputs
            )
        with self.assertRaises(ValueError):
            implied_volatility(
                OptionRight.PUT, option_price=0.0, **self.inputs
            )
        with self.assertRaises(ValueError):
            implied_volatility(
                OptionRight.PUT, option_price=100.0, **self.inputs
            )

    def test_rejects_price_without_solution_in_search_interval(self):
        # At sigma=6, S=K=100 and r=q=0 the call is about 99.73, inside the
        # mathematical upper bound 100 but above the default sigma=5 ceiling.
        inputs = {
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry_years": 1.0,
            "risk_free_rate": 0.0,
            "dividend_yield": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "volatility interval"):
            implied_volatility(
                OptionRight.CALL, option_price=99.73, **inputs
            )

    def test_reports_nonconvergence(self):
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            implied_volatility(
                OptionRight.CALL,
                option_price=10.450583572185565,
                max_iterations=1,
                **self.inputs,
            )

    def test_rejects_invalid_solver_configuration_and_values(self):
        arguments = {
            "option_price": 10.45,
            **self.inputs,
        }
        invalid_options = (
            {"min_volatility": 0.0},
            {"min_volatility": 1.0, "max_volatility": 1.0},
            {"max_volatility": math.inf},
            {"volatility_tolerance": math.nan},
            {"max_iterations": 0},
            {"max_iterations": True},
        )
        for invalid in invalid_options:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    implied_volatility(
                        OptionRight.CALL, **arguments, **invalid
                    )
        for invalid_price in (math.nan, math.inf, -math.inf, -1.0, True):
            with self.subTest(option_price=invalid_price):
                with self.assertRaises(ValueError):
                    implied_volatility(
                        OptionRight.CALL,
                        **{**arguments, "option_price": invalid_price},
                    )

    def test_rejects_invalid_and_nonfinite_model_inputs(self):
        arguments = {
            "option_price": 10.45,
            **self.inputs,
        }
        for field in ("spot", "strike", "time_to_expiry_years"):
            for value in (0.0, -1.0):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        implied_volatility(
                            OptionRight.CALL,
                            **{**arguments, field: value},
                        )
        for field in arguments:
            for value in (math.nan, math.inf, -math.inf):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        implied_volatility(
                            OptionRight.CALL,
                            **{**arguments, field: value},
                        )


if __name__ == "__main__":
    unittest.main()
