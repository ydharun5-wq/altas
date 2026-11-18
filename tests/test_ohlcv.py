from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.data.ohlcv import DataValidationError, read_ohlcv_csv


CSV_HEADER = "timestamp,symbol,open,high,low,close,volume"


def _write_csv(directory, *rows):
    path = Path(directory) / "input.csv"
    contents = CSV_HEADER + "\n"
    if rows:
        contents += "\n".join(rows) + "\n"
    path.write_text(contents, encoding="utf-8")
    return path


class OHLCVReaderTests(unittest.TestCase):
    def test_reads_sample_csv_in_chronological_order(self):
        sample_path = Path(__file__).resolve().parents[1] / "data" / "sample_ohlcv.csv"
        bars = read_ohlcv_csv(sample_path)

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].symbol, "RELIANCE")
        self.assertEqual(bars[0].close, 1505)
        self.assertGreater(bars[1].timestamp, bars[0].timestamp)
        self.assertEqual(bars[1].volume, 14000)

    def test_rejects_missing_required_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_column.csv"
            path.write_text(
                "timestamp,symbol,open,high,low,close\n"
                "2026-01-05 09:15:00,RELIANCE,1500,1510,1495,1505\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataValidationError, "missing required columns"):
                read_ohlcv_csv(path)

    def test_rejects_inconsistent_ohlc_prices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid_prices.csv"
            path.write_text(
                "timestamp,symbol,open,high,low,close,volume\n"
                "2026-01-05 09:15:00,RELIANCE,1500,1490,1495,1505,12000\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataValidationError, "inconsistent OHLC"):
                read_ohlcv_csv(path)

    def test_rejects_negative_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "negative_volume.csv"
            path.write_text(
                "timestamp,symbol,open,high,low,close,volume\n"
                "2026-01-05 09:15:00,RELIANCE,1500,1510,1495,1505,-1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataValidationError, "non-negative integer"):
                read_ohlcv_csv(path)

    def test_rejects_non_increasing_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out_of_order.csv"
            path.write_text(
                "timestamp,symbol,open,high,low,close,volume\n"
                "2026-01-05 09:20:00,RELIANCE,1500,1510,1495,1505,12000\n"
                "2026-01-05 09:15:00,RELIANCE,1505,1515,1500,1512,14000\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataValidationError, "timestamps .* must increase"):
                read_ohlcv_csv(path)

    def test_rejects_invalid_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory, "not-a-timestamp,RELIANCE,10,12,8,11,100")
            with self.assertRaisesRegex(DataValidationError, "invalid timestamp"):
                read_ohlcv_csv(path)

    def test_rejects_missing_numeric_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory, "2026-01-05 09:15:00,RELIANCE,10,12,8,,100")
            with self.assertRaisesRegex(DataValidationError, "missing or extra field"):
                read_ohlcv_csv(path)

    def test_rejects_nan_price(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory, "2026-01-05 09:15:00,RELIANCE,nan,12,8,11,100")
            with self.assertRaisesRegex(DataValidationError, "finite and positive"):
                read_ohlcv_csv(path)

    def test_rejects_infinite_price(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory, "2026-01-05 09:15:00,RELIANCE,inf,12,8,11,100")
            with self.assertRaisesRegex(DataValidationError, "finite and positive"):
                read_ohlcv_csv(path)

    def test_rejects_zero_price(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory, "2026-01-05 09:15:00,RELIANCE,0,12,8,11,100")
            with self.assertRaisesRegex(DataValidationError, "finite and positive"):
                read_ohlcv_csv(path)

    def test_rejects_negative_price(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory, "2026-01-05 09:15:00,RELIANCE,-1,12,8,11,100")
            with self.assertRaisesRegex(DataValidationError, "finite and positive"):
                read_ohlcv_csv(path)

    def test_rejects_duplicate_timestamp_for_same_symbol(self):
        row = "2026-01-05 09:15:00,RELIANCE,10,12,8,11,100"
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory, row, row)
            with self.assertRaisesRegex(DataValidationError, "duplicate timestamp"):
                read_ohlcv_csv(path)

    def test_allows_same_timestamp_for_different_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(
                directory,
                "2026-01-05 09:15:00,RELIANCE,10,12,8,11,100",
                "2026-01-05 09:15:00,TCS,20,22,18,21,200",
            )
            bars = read_ohlcv_csv(path)

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].timestamp, bars[1].timestamp)
        self.assertNotEqual(bars[0].symbol, bars[1].symbol)

    def test_allows_multiple_symbols_with_independent_chronology(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(
                directory,
                "2026-01-05 09:15:00,RELIANCE,10,12,8,11,100",
                "2026-01-05 09:10:00,TCS,20,22,18,21,200",
                "2026-01-05 09:20:00,RELIANCE,11,13,9,12,110",
                "2026-01-05 09:15:00,TCS,21,23,19,22,210",
            )
            bars = read_ohlcv_csv(path)

        self.assertEqual(len(bars), 4)
        self.assertEqual([bar.symbol for bar in bars], ["RELIANCE", "TCS", "RELIANCE", "TCS"])

    def test_rejects_each_invalid_ohlc_relationship(self):
        cases = (
            ("high below open", "11,10,9,9.5"),
            ("high below close", "9.5,10,9,11"),
            ("low above open", "8.5,12,9,10.5"),
            ("low above close", "11,12,10.5,10"),
            ("high below low", "10,9,11,10"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for label, ohlc in cases:
                with self.subTest(relationship=label):
                    row = f"2026-01-05 09:15:00,RELIANCE,{ohlc},100"
                    path = _write_csv(directory, row)
                    with self.assertRaisesRegex(DataValidationError, "inconsistent OHLC"):
                        read_ohlcv_csv(path)

    def test_rejects_malformed_row_with_extra_field(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(
                directory,
                "2026-01-05 09:15:00,RELIANCE,10,12,8,11,100,unexpected",
            )
            with self.assertRaisesRegex(DataValidationError, "missing or extra field"):
                read_ohlcv_csv(path)

    def test_rejects_csv_with_no_data_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _write_csv(directory)
            with self.assertRaisesRegex(DataValidationError, "no OHLCV data rows"):
                read_ohlcv_csv(path)


if __name__ == "__main__":
    unittest.main()
