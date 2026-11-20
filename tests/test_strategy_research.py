"""Deterministic strategy research tests using explicit synthetic fixtures."""

from datetime import datetime, timedelta
import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.application import create_application
from atlas.data.ohlcv import OHLCVBar
from atlas.derivatives.models import UnderlyingIndex
from atlas.market_data import (
    HistoricalOHLCVRequest,
    HistoricalOHLCVResponse,
    MissingMarketDataError,
    UnsupportedRequestError,
    create_validated_market_data_provider,
)
from atlas.strategies import (
    ATRStopConfig,
    EMATrendStrategy,
    RSIMeanReversionStrategy,
    SignalDirection,
    SignalEvent,
    SignalStatus,
    StrategyResearchEngine,
    WarmupStatus,
    calculate_atr_stop_distance,
    calculate_stop_reference_price,
)
from atlas.strategies.models import StrategyMarketData


BASE = datetime(2026, 1, 5, 9, 15)
TIMEFRAME = timedelta(minutes=5)
SYMBOL = "NIFTY"


def _bar(index, close, *, open_price=None, high=None, low=None, timestamp=None):
    return OHLCVBar(
        timestamp=BASE + index * TIMEFRAME if timestamp is None else timestamp,
        symbol=SYMBOL,
        open=close if open_price is None else open_price,
        high=close + 1 if high is None else high,
        low=close - 1 if low is None else low,
        close=close,
        volume=100 + index,
    )


def _bars(closes):
    return [_bar(index, close) for index, close in enumerate(closes)]


def _ema_strategy(**kwargs):
    values = {
        "underlying": UnderlyingIndex.NIFTY,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "fast_period": 2,
        "slow_period": 3,
    }
    values.update(kwargs)
    return EMATrendStrategy(**values)


def _rsi_strategy(**kwargs):
    values = {
        "underlying": UnderlyingIndex.NIFTY,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "period": 2,
        "oversold_threshold": 30.0,
        "overbought_threshold": 70.0,
        "long_exit_threshold": 45.0,
        "short_exit_threshold": 55.0,
    }
    values.update(kwargs)
    return RSIMeanReversionStrategy(**values)


class StaticHistoryProvider:
    def __init__(self, bars, *, is_synthetic=True):
        self.bars = tuple(bars)
        self.is_synthetic = is_synthetic

    def get_historical_ohlcv(self, request):
        return HistoricalOHLCVResponse(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            bars=self.bars,
            is_synthetic=self.is_synthetic,
        )

    def get_futures_quote(self, request):
        raise UnsupportedRequestError("not used by strategy tests")

    def get_options_quote(self, request):
        raise UnsupportedRequestError("not used by strategy tests")


def _engine_for(bars, *, is_synthetic=True):
    provider = create_validated_market_data_provider(
        StaticHistoryProvider(bars, is_synthetic=is_synthetic)
    )
    return StrategyResearchEngine(provider)


def _run(
    strategy,
    bars,
    *,
    last_bar_complete=True,
    is_synthetic=True,
    data_source="synthetic-fixture",
    evaluation_at=None,
    request_end=None,
):
    if bars:
        start = bars[0].timestamp
        end = bars[-1].timestamp if request_end is None else request_end
    else:
        start = BASE
        end = BASE
    as_of = end if evaluation_at is None else evaluation_at
    request = HistoricalOHLCVRequest(SYMBOL, start, end)
    return _engine_for(bars, is_synthetic=is_synthetic).generate_signal(
        strategy,
        request,
        underlying=UnderlyingIndex.NIFTY,
        timeframe=TIMEFRAME,
        data_source=data_source,
        evaluation_at=as_of,
        last_bar_complete=last_bar_complete,
    )


class EMATrendStrategyTests(unittest.TestCase):
    def test_known_bullish_crossover_emits_long_signal(self):
        bars = _bars([6.0, 5.0, 4.0, 3.0, 2.0, 5.0])
        result = _run(_ema_strategy(), bars)

        self.assertEqual(result.direction, SignalDirection.LONG)
        self.assertEqual(
            result.events, (SignalEvent.EXIT_SHORT, SignalEvent.ENTER_LONG)
        )
        self.assertEqual(result.entry_reference_price, 5.0)
        self.assertEqual(result.status, SignalStatus.READY)
        values = {item.name: item.value for item in result.indicator_values}
        self.assertLess(values["fast_ema_previous"], values["slow_ema_previous"])
        self.assertGreater(values["fast_ema"], values["slow_ema"])

    def test_known_bearish_crossover_emits_short_signal(self):
        bars = _bars([6.0, 7.0, 8.0, 9.0, 10.0, 7.0])
        result = _run(_ema_strategy(), bars)

        self.assertEqual(result.direction, SignalDirection.SHORT)
        self.assertEqual(
            result.events, (SignalEvent.EXIT_LONG, SignalEvent.ENTER_SHORT)
        )
        self.assertEqual(result.entry_reference_price, 7.0)

    def test_no_crossover_is_no_trade(self):
        result = _run(_ema_strategy(), _bars([10.0] * 6))

        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.NO_TRADE)
        self.assertIsNone(result.entry_reference_price)
        self.assertEqual(result.events, ())

    def test_crossover_requires_two_aligned_slow_ema_values(self):
        result = _run(_ema_strategy(), _bars([2.0, 3.0, 2.0]))

        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.INSUFFICIENT_HISTORY)
        self.assertEqual(result.warmup_status, WarmupStatus.INSUFFICIENT)
        self.assertIn("need 4 bars", result.missing_data[0])
        self.assertEqual(result.indicator_values, ())

    def test_atr_stop_is_applied_only_when_explicitly_configured(self):
        bars = _bars([6.0, 5.0, 4.0, 3.0, 2.0, 5.0])
        with_stop = _run(
            _ema_strategy(stop=ATRStopConfig(period=2, multiplier=0.1)), bars
        )
        no_stop = _run(_ema_strategy(), bars)

        self.assertIsNotNone(with_stop.stop_reference_price)
        self.assertAlmostEqual(
            with_stop.stop_reference_price,
            with_stop.entry_reference_price
            - next(
                item.value
                for item in with_stop.indicator_values
                if item.name == "atr_stop_distance"
            ),
        )
        self.assertIsNone(no_stop.stop_reference_price)
        self.assertIn("not configured", no_stop.stop_loss_method)

    def test_invalid_atr_stop_reference_fails_closed_to_no_trade(self):
        bars = _bars([6.0, 5.0, 4.0, 3.0, 2.0, 5.0])
        result = _run(
            _ema_strategy(stop=ATRStopConfig(period=2, multiplier=100.0)), bars
        )
        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.INVALID_DATA)
        self.assertIsNone(result.stop_reference_price)

    def test_invalid_periods_and_compatibility_settings_are_rejected(self):
        for fast, slow in ((0, 3), (2, 2), (4, 3), (True, 3)):
            with self.subTest(fast=fast, slow=slow):
                with self.assertRaises(ValueError):
                    _ema_strategy(fast_period=fast, slow_period=slow)
        with self.assertRaises(ValueError):
            _ema_strategy(symbol="")
        with self.assertRaises(ValueError):
            _ema_strategy(allow_incomplete_candles=1)


class RSIMeanReversionStrategyTests(unittest.TestCase):
    def test_oversold_cross_emits_long_signal_with_configured_threshold(self):
        bars = _bars([100.0, 102.0, 101.0, 100.0])
        strategy = _rsi_strategy(oversold_threshold=40.0)
        result = _run(strategy, bars)

        self.assertEqual(result.direction, SignalDirection.LONG)
        self.assertIn(SignalEvent.ENTER_LONG, result.events)
        values = {item.name: item.value for item in result.indicator_values}
        self.assertGreater(values["rsi_previous"], 40.0)
        self.assertLessEqual(values["rsi"], 40.0)
        self.assertIn("not proof", " ".join(result.reasons))

    def test_overbought_cross_emits_short_signal(self):
        bars = _bars([100.0, 98.0, 99.0, 101.0])
        result = _run(_rsi_strategy(), bars)

        self.assertEqual(result.direction, SignalDirection.SHORT)
        self.assertIn(SignalEvent.ENTER_SHORT, result.events)

    def test_configured_long_exit_condition_is_reported_without_position_state(self):
        bars = _bars([100.0, 102.0, 101.0, 99.0, 104.0])
        strategy = _rsi_strategy(
            overbought_threshold=85.0,
            long_exit_threshold=50.0,
            short_exit_threshold=55.0,
        )
        result = _run(strategy, bars)

        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertIn(SignalEvent.EXIT_LONG, result.events)
        self.assertEqual(result.status, SignalStatus.READY)
        self.assertIsNone(result.entry_reference_price)

    def test_no_threshold_cross_is_no_trade(self):
        result = _run(_rsi_strategy(), _bars([10.0, 10.0, 10.0, 10.0, 10.0]))

        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.NO_TRADE)

    def test_rsi_warmup_requires_two_aligned_values(self):
        result = _run(_rsi_strategy(), _bars([10.0, 11.0, 10.0]))

        self.assertEqual(result.status, SignalStatus.INSUFFICIENT_HISTORY)
        self.assertEqual(result.warmup_status, WarmupStatus.INSUFFICIENT)

    def test_invalid_rsi_thresholds_are_rejected(self):
        for values in (
            {"oversold_threshold": math.nan},
            {"overbought_threshold": math.inf},
            {"oversold_threshold": 50.0, "long_exit_threshold": 40.0},
            {"long_exit_threshold": 60.0, "short_exit_threshold": 50.0},
            {"period": True},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _rsi_strategy(**values)


class ATRRiskDistanceTests(unittest.TestCase):
    def test_known_atr_distance_and_long_short_stop_references(self):
        bars = [
            _bar(0, 11, open_price=10, high=12, low=9),
            _bar(1, 13, open_price=11, high=14, low=10),
            _bar(2, 14, open_price=13, high=15, low=12),
            _bar(3, 16, open_price=14, high=17, low=13),
        ]
        distance = calculate_atr_stop_distance(bars, period=2, multiplier=2.0)

        self.assertAlmostEqual(distance, 7.5)
        self.assertAlmostEqual(
            calculate_stop_reference_price(
                SignalDirection.LONG,
                entry_reference_price=100.0,
                stop_distance=distance,
            ),
            92.5,
        )
        self.assertAlmostEqual(
            calculate_stop_reference_price(
                SignalDirection.SHORT,
                entry_reference_price=100.0,
                stop_distance=distance,
            ),
            107.5,
        )

    def test_atr_helper_rejects_missing_or_invalid_values(self):
        with self.assertRaises(ValueError):
            calculate_atr_stop_distance(_bars([10.0, 11.0]), period=2, multiplier=1.0)
        for multiplier in (0.0, -1.0, math.inf, True):
            with self.subTest(multiplier=multiplier):
                with self.assertRaises(ValueError):
                    calculate_atr_stop_distance(
                        _bars([10.0, 11.0, 12.0]),
                        period=2,
                        multiplier=multiplier,
                    )
        with self.assertRaises(ValueError):
            calculate_stop_reference_price(
                SignalDirection.LONG,
                entry_reference_price=10.0,
                stop_distance=11.0,
            )


class StrategyResearchEngineTests(unittest.TestCase):
    def test_engine_only_accepts_validated_provider(self):
        app = create_application(
            provider_name="mock",
            mock_historical_bars=[],
            mock_futures_quotes=[],
            mock_options_quotes=[],
        )
        with self.assertRaises(TypeError):
            StrategyResearchEngine(object())
        with self.assertRaises(TypeError):
            StrategyResearchEngine(StaticHistoryProvider([]))
        self.assertIsInstance(StrategyResearchEngine(app.market_data), StrategyResearchEngine)

    def test_missing_provider_data_returns_explicit_no_trade(self):
        app = create_application(
            provider_name="mock",
            mock_historical_bars=[],
            mock_futures_quotes=[],
            mock_options_quotes=[],
        )
        result = StrategyResearchEngine(app.market_data).generate_signal(
            _ema_strategy(),
            HistoricalOHLCVRequest(SYMBOL, BASE, BASE),
            underlying=UnderlyingIndex.NIFTY,
            timeframe=TIMEFRAME,
            data_source="local-mock",
            evaluation_at=BASE,
            last_bar_complete=True,
        )

        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.MISSING_DATA)
        self.assertIsNone(result.observation_timestamp)
        self.assertIsNone(result.is_synthetic)
        self.assertTrue(result.missing_data)

    def test_incomplete_candle_is_rejected_unless_strategy_opts_in(self):
        bars = _bars([6.0, 5.0, 4.0, 3.0, 2.0, 5.0])
        rejected = _run(_ema_strategy(), bars, last_bar_complete=False)
        accepted = _run(
            _ema_strategy(allow_incomplete_candles=True),
            bars,
            last_bar_complete=False,
        )

        self.assertEqual(rejected.status, SignalStatus.INCOMPLETE_CANDLE)
        self.assertEqual(rejected.direction, SignalDirection.NO_TRADE)
        self.assertFalse(rejected.used_incomplete_candle)
        self.assertEqual(accepted.direction, SignalDirection.LONG)
        self.assertTrue(accepted.used_incomplete_candle)

    def test_future_request_end_is_rejected_before_provider_access(self):
        bars = _bars([1.0, 2.0, 3.0, 4.0])
        future_end = bars[-1].timestamp + TIMEFRAME
        with self.assertRaisesRegex(ValueError, "request end"):
            _run(
                _ema_strategy(),
                bars,
                request_end=future_end,
                evaluation_at=bars[-1].timestamp,
            )

    def test_strategy_must_match_underlying_symbol_and_timeframe(self):
        bars = _bars([3.0, 3.0, 3.0, 2.0])
        engine = _engine_for(bars)
        request = HistoricalOHLCVRequest(SYMBOL, bars[0].timestamp, bars[-1].timestamp)
        with self.assertRaisesRegex(ValueError, "does not match"):
            engine.generate_signal(
                _ema_strategy(),
                request,
                underlying=UnderlyingIndex.BANKNIFTY,
                timeframe=TIMEFRAME,
                data_source="fixture",
                evaluation_at=bars[-1].timestamp,
                last_bar_complete=True,
            )
        wrong_symbol_strategy = _ema_strategy(symbol="OTHER")
        with self.assertRaisesRegex(ValueError, "does not match"):
            engine.generate_signal(
                wrong_symbol_strategy,
                request,
                underlying=UnderlyingIndex.NIFTY,
                timeframe=TIMEFRAME,
                data_source="fixture",
                evaluation_at=bars[-1].timestamp,
                last_bar_complete=True,
            )

    def test_duplicate_and_out_of_order_timestamps_become_no_trade(self):
        duplicate = _bars([2.0, 3.0, 4.0])
        duplicate[2] = _bar(
            2, 3.0, timestamp=duplicate[1].timestamp
        )
        result = _run(_ema_strategy(), duplicate)
        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.INVALID_DATA)

        out_of_order = _bars([2.0, 3.0, 4.0])
        out_of_order[2] = _bar(
            2, 4.0, timestamp=out_of_order[0].timestamp + timedelta(minutes=1)
        )
        result = _run(
            _ema_strategy(),
            out_of_order,
            request_end=out_of_order[1].timestamp,
            evaluation_at=out_of_order[1].timestamp,
        )
        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.INVALID_DATA)

    def test_invalid_ohlc_becomes_no_trade(self):
        bars = _bars([3.0, 4.0, 5.0])
        bars[-1] = _bar(2, 5.0, high=4.5)
        result = _run(_ema_strategy(), bars)

        self.assertEqual(result.direction, SignalDirection.NO_TRADE)
        self.assertEqual(result.status, SignalStatus.INVALID_DATA)
        self.assertTrue(result.invalid_conditions)

    def test_synthetic_and_real_data_flags_and_source_identity_are_preserved(self):
        bars = _bars([6.0, 5.0, 4.0, 3.0, 2.0, 5.0])
        synthetic = _run(
            _ema_strategy(), bars, data_source="mock-history", is_synthetic=True
        )
        real = _run(
            _ema_strategy(), bars, data_source="authorized-history", is_synthetic=False
        )

        self.assertTrue(synthetic.is_synthetic)
        self.assertEqual(synthetic.data_source, "mock-history")
        self.assertFalse(real.is_synthetic)
        self.assertEqual(real.data_source, "authorized-history")
        self.assertEqual(synthetic.underlying, UnderlyingIndex.NIFTY)
        self.assertEqual(synthetic.timeframe, TIMEFRAME)
        self.assertEqual(synthetic.observation_timestamp, bars[-1].timestamp)

    def test_repeated_execution_is_deterministic_and_does_not_mutate_input(self):
        bars = _bars([3.0, 3.0, 3.0, 2.0, 3.0, 4.0])
        original = bars.copy()
        engine = _engine_for(bars)
        request = HistoricalOHLCVRequest(SYMBOL, bars[0].timestamp, bars[-1].timestamp)
        kwargs = {
            "underlying": UnderlyingIndex.NIFTY,
            "timeframe": TIMEFRAME,
            "data_source": "synthetic-fixture",
            "evaluation_at": bars[-1].timestamp,
            "last_bar_complete": True,
        }
        first = engine.generate_signal(_ema_strategy(), request, **kwargs)
        second = engine.generate_signal(_ema_strategy(), request, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(bars, original)

    def test_strategy_market_data_cannot_be_constructed_around_unvalidated_response(self):
        response = HistoricalOHLCVResponse(
            symbol=SYMBOL,
            start=BASE,
            end=BASE,
            bars=tuple(_bars([10.0])),
            is_synthetic=True,
        )
        with self.assertRaisesRegex(TypeError, "created by StrategyResearchEngine"):
            StrategyMarketData(
                response=response,
                underlying=UnderlyingIndex.NIFTY,
                timeframe=TIMEFRAME,
                data_source="fixture",
                evaluation_at=BASE,
                last_bar_complete=True,
            )


if __name__ == "__main__":
    unittest.main()
