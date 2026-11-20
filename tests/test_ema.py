from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.indicators import ema


class EMATests(unittest.TestCase):
    def test_known_values_use_sma_seed_and_standard_multiplier(self):
        result = ema([3.0, 6.0, 9.0, 12.0], 2)

        self.assertEqual(result, [4.5, 7.5, 10.5])

    def test_period_one_returns_each_price(self):
        self.assertEqual(ema([2.0, 3.0, 4.0], 1), [2.0, 3.0, 4.0])

    def test_output_starts_at_seed_point(self):
        self.assertEqual(len(ema([1.0, 2.0, 3.0, 4.0, 5.0], 3)), 3)

    def test_rejects_insufficient_data(self):
        with self.assertRaisesRegex(ValueError, "at least 3 closing prices"):
            ema([10.0, 11.0], 3)

    def test_rejects_invalid_periods(self):
        for period in (0, -1, 1.5, True, "2"):
            with self.subTest(period=period):
                with self.assertRaisesRegex(ValueError, "period must be a positive integer"):
                    ema([1.0, 2.0, 3.0], period)

    def test_rejects_non_finite_prices(self):
        for price in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    ema([1.0, price], 2)

    def test_rejects_zero_and_negative_prices(self):
        for price in (0.0, -1.0):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    ema([1.0, price], 2)

    def test_rejects_non_numeric_prices(self):
        for price in ("not-a-price", None, True):
            with self.subTest(price=price):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    ema([1.0, price], 2)

    def test_does_not_mutate_input(self):
        prices = [3.0, 6.0, 9.0, 12.0]
        original = prices.copy()

        ema(prices, 2)

        self.assertEqual(prices, original)


if __name__ == "__main__":
    unittest.main()
