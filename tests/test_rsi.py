from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.indicators.rsi import rsi


class RSITests(unittest.TestCase):
    def test_known_values_use_wilder_smoothing(self):
        values = rsi([10.0, 11.0, 10.0, 12.0], 2)

        self.assertAlmostEqual(values[0], 50.0)
        self.assertAlmostEqual(values[1], 100.0 - (100.0 / 6.0))

    def test_rising_prices_return_one_hundred(self):
        self.assertEqual(rsi([1.0, 2.0, 3.0, 4.0], 2), [100.0, 100.0])

    def test_falling_prices_return_zero(self):
        self.assertEqual(rsi([4.0, 3.0, 2.0, 1.0], 2), [0.0, 0.0])

    def test_flat_prices_return_fifty(self):
        self.assertEqual(rsi([5.0, 5.0, 5.0, 5.0], 2), [50.0, 50.0])

    def test_output_begins_at_first_completed_period(self):
        values = rsi([10.0, 11.0, 10.0, 12.0, 11.0], 2)

        self.assertEqual(len(values), 3)
        self.assertAlmostEqual(values[0], 50.0)

    def test_rejects_invalid_periods(self):
        prices = [1.0, 2.0, 3.0]
        for period in (0, -1, 1.5, True, "2"):
            with self.subTest(period=period):
                with self.assertRaisesRegex(ValueError, "period must be a positive integer"):
                    rsi(prices, period)

    def test_rejects_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, "at least 3 closing prices"):
            rsi([10.0, 11.0], 2)

    def test_rejects_non_finite_prices(self):
        for price in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    rsi([1.0, price, 2.0], 2)

    def test_rejects_zero_and_negative_prices(self):
        for price in (0.0, -1.0):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    rsi([1.0, price, 2.0], 2)

    def test_rejects_non_numeric_prices(self):
        for price in ("not-a-price", None, True):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    rsi([1.0, price, 2.0], 2)

    def test_does_not_mutate_input(self):
        prices = [10.0, 11.0, 10.0, 12.0]
        original = prices.copy()

        rsi(prices, 2)

        self.assertEqual(prices, original)


if __name__ == "__main__":
    unittest.main()
