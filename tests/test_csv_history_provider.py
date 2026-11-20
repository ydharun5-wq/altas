from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.backtesting import BacktestConfig, BacktestEngine, ExecutionAssumptions
from atlas.derivatives.models import (
    DerivativeContract,
    Exchange,
    FuturesQuoteRequest,
    InstrumentType,
    UnderlyingIndex,
)
from atlas.market_data import (
    CSVHistoricalMarketDataProvider,
    HistoricalOHLCVRequest,
    MissingMarketDataError,
    UnsupportedRequestError,
    create_validated_market_data_provider,
)
from atlas.strategies import EMATrendStrategy


BASE = datetime(2026, 1, 5, 9, 15)
TIMEFRAME = timedelta(minutes=5)


class CSVHistoricalProviderTests(unittest.TestCase):
    def _write_history(self, directory, closes):
        path = Path(directory) / "nifty.csv"
        lines = ["timestamp,symbol,open,high,low,close,volume"]
        for index, close in enumerate(closes):
            timestamp = BASE + index * TIMEFRAME
            lines.append(
                f"{timestamp.isoformat()},NIFTY,{close},{close + 1},"
                f"{close - 1},{close},100"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _provider(self, path, *, is_synthetic=True):
        return CSVHistoricalMarketDataProvider(
            {"NIFTY": path},
            data_source="local-csv-fixture",
            is_synthetic=is_synthetic,
        )

    def test_validated_history_preserves_source_and_explicit_synthetic_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_history(directory, [100, 101])
            raw = self._provider(path)
            provider = create_validated_market_data_provider(raw)
            response = provider.get_historical_ohlcv(
                HistoricalOHLCVRequest("NIFTY", BASE, BASE + TIMEFRAME)
            )
            with self.assertRaises(AttributeError):
                raw.is_synthetic = False

        self.assertEqual(len(response.bars), 2)
        self.assertEqual(raw.data_source, "local-csv-fixture")
        self.assertTrue(response.is_synthetic)

    def test_missing_symbol_and_unsupported_quotes_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_history(directory, [100, 101])
            raw_provider = self._provider(path)
            provider = create_validated_market_data_provider(raw_provider)
            contract = DerivativeContract(
                exchange=Exchange.NSE,
                underlying=UnderlyingIndex.NIFTY,
                instrument_type=InstrumentType.FUT,
                expiry=date(2026, 2, 26),
                contract_id="NIFTY-FUT-20260226",
                lot_size=1,
                tick_size=Decimal("0.05"),
            )

            with self.assertRaises(MissingMarketDataError):
                provider.get_historical_ohlcv(
                    HistoricalOHLCVRequest("BANKNIFTY", BASE, BASE + TIMEFRAME)
                )
            with self.assertRaises(UnsupportedRequestError):
                provider.get_futures_quote(
                    FuturesQuoteRequest(
                        contract,
                        datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc),
                    )
                )

    def test_local_csv_history_flows_through_strategy_and_backtest(self):
        closes = [6, 5, 4, 3, 2, 5, 6]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_history(directory, closes)
            raw_provider = self._provider(path)
            provider = create_validated_market_data_provider(raw_provider)
            result = BacktestEngine(provider).run(
                EMATrendStrategy(
                    underlying=UnderlyingIndex.NIFTY,
                    symbol="NIFTY",
                    timeframe=TIMEFRAME,
                    fast_period=2,
                    slow_period=3,
                ),
                HistoricalOHLCVRequest(
                    "NIFTY", BASE, BASE + (len(closes) - 1) * TIMEFRAME
                ),
                timeframe=TIMEFRAME,
                data_source=raw_provider.data_source,
                config=BacktestConfig(
                    initial_capital=10000,
                    quantity=1,
                    unit_multiplier=1,
                    execution=ExecutionAssumptions.zero_cost_research_mode(),
                    target_distance=None,
                ),
                last_bar_complete=True,
            )

        self.assertTrue(result.is_synthetic)
        self.assertEqual(result.data_source, "local-csv-fixture")
        self.assertEqual(len(result.signals), len(closes))
        self.assertIsNotNone(result.open_position)
        self.assertTrue(
            any(item.action == "enter_long" and item.filled for item in result.executions)
        )


if __name__ == "__main__":
    unittest.main()
