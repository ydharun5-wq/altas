from datetime import datetime, timedelta
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.data.ohlcv import OHLCVBar
from atlas.indicators.atr import atr


class ATRTests(unittest.TestCase):
    def _bar(self, minute, open_price, high, low, close):
        return OHLCVBar(
            timestamp=datetime(2026, 1, 5, 9, 15) + timedelta(minutes=minute),
            symbol="TEST",
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=100,
        )

    def test_known_atr_values_and_chronological_alignment(self):
        bars = [
            self._bar(0, 10, 12, 9, 11),
            self._bar(1, 11, 14, 10, 13),
            self._bar(2, 13, 15, 12, 14),
            self._bar(3, 14, 17, 13, 16),
        ]

        values = atr(bars, 2)

        self.assertAlmostEqual(values[0], 3.5)
        self.assertAlmostEqual(values[1], 3.75)
        self.assertEqual(len(values), len(bars) - 2)

    def test_single_symbol_input_calculates_correctly(self):
        bars = [
            self._bar(0, 10, 10, 10, 10),
            self._bar(1, 11, 13, 10, 12),
        ]

        self.assertEqual(atr(bars, 1), [3.0])

    def test_rejects_mixed_symbol_input(self):
        first = self._bar(0, 10, 11, 9, 10)
        other_symbol = OHLCVBar(
            timestamp=self._bar(1, 20, 21, 19, 20).timestamp,
            symbol="OTHER",
            open=20,
            high=21,
            low=19,
            close=20,
            volume=100,
        )

        with self.assertRaisesRegex(ValueError, "all bars must have the same symbol"):
            atr([first, other_symbol], 1)

    def test_gap_up_uses_previous_close_in_true_range(self):
        bars = [
            self._bar(0, 10, 10, 10, 10),
            self._bar(1, 12, 14, 12, 13),
        ]

        self.assertEqual(atr(bars, 1), [4.0])

    def test_gap_down_uses_previous_close_in_true_range(self):
        bars = [
            self._bar(0, 14, 14, 14, 14),
            self._bar(1, 12, 12, 10, 11),
        ]

        self.assertEqual(atr(bars, 1), [4.0])

    def test_subsequent_atr_uses_wilder_smoothing(self):
        bars = [
            self._bar(0, 10, 10, 10, 10),
            self._bar(1, 9, 12, 8, 11),
            self._bar(2, 11, 12, 10, 11),
            self._bar(3, 12, 14, 11, 13),
            self._bar(4, 12.5, 13, 12, 12.5),
        ]

        values = atr(bars, 3)

        self.assertAlmostEqual(values[0], 3.0)
        self.assertAlmostEqual(values[1], 7.0 / 3.0)

    def test_period_one_returns_each_true_range(self):
        bars = [
            self._bar(0, 10, 10, 10, 10),
            self._bar(1, 11, 13, 10, 12),
            self._bar(2, 12, 14, 11, 13),
        ]

        self.assertEqual(atr(bars, 1), [3.0, 3.0])

    def test_rejects_invalid_periods(self):
        bars = [self._bar(0, 10, 11, 9, 10), self._bar(1, 10, 11, 9, 10)]
        for period in (0, -1, 1.5, True, "1"):
            with self.subTest(period=period):
                with self.assertRaisesRegex(ValueError, "period must be a positive integer"):
                    atr(bars, period)

    def test_rejects_insufficient_bars(self):
        with self.assertRaisesRegex(ValueError, "at least 3 bars"):
            atr([self._bar(0, 10, 11, 9, 10), self._bar(1, 10, 11, 9, 10)], 2)

    def test_rejects_non_finite_and_non_positive_ohlc_prices(self):
        invalid_prices = (float("nan"), float("inf"), float("-inf"), 0.0, -1.0)
        for invalid_price in invalid_prices:
            with self.subTest(price=invalid_price):
                bars = [
                    self._bar(0, 10, 11, 9, 10),
                    self._bar(1, 10, invalid_price, 9, 10),
                ]
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    atr(bars, 1)

    def test_rejects_inconsistent_ohlc_prices(self):
        cases = (
            (11, 10, 9, 9.5),
            (9.5, 10, 9, 11),
            (8.5, 12, 9, 10.5),
            (11, 12, 10.5, 10),
            (10, 9, 11, 10),
        )
        for prices in cases:
            with self.subTest(prices=prices):
                bars = [
                    self._bar(0, 10, 11, 9, 10),
                    self._bar(1, *prices),
                ]
                with self.assertRaisesRegex(ValueError, "inconsistent OHLC"):
                    atr(bars, 1)

    def test_rejects_non_increasing_timestamps(self):
        first = self._bar(0, 10, 11, 9, 10)
        second = self._bar(1, 10, 11, 9, 10)
        with self.assertRaisesRegex(ValueError, "timestamps must be strictly increasing"):
            atr([first, second, second], 1)

    def test_does_not_mutate_input(self):
        bars = [
            self._bar(0, 10, 12, 9, 11),
            self._bar(1, 11, 14, 10, 13),
            self._bar(2, 13, 15, 12, 14),
        ]
        original = bars.copy()

        atr(bars, 2)

        self.assertEqual(bars, original)


if __name__ == "__main__":
    unittest.main()
