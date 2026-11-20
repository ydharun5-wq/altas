"""Event ordering, execution assumptions, accounting and walk-forward tests."""

from datetime import datetime, timedelta
import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.backtesting import (
    BacktestConfig,
    BacktestEngine,
    ExecutionAssumptions,
    FeeBasis,
    FeeComponent,
    PositionSide,
    ProfitFactorStatus,
    TradeExitReason,
    WalkForwardPlan,
    WalkForwardSplit,
)
from atlas.data.ohlcv import OHLCVBar
from atlas.derivatives.models import UnderlyingIndex
from atlas.market_data import (
    HistoricalOHLCVRequest,
    HistoricalOHLCVResponse,
    UnsupportedRequestError,
    create_validated_market_data_provider,
)
from atlas.strategies import SignalDirection, SignalEvent, SignalStatus, WarmupStatus
from atlas.strategies.models import IndicatorValue, StrategySignal


BASE = datetime(2026, 1, 5, 9, 15)
TIMEFRAME = timedelta(minutes=5)
SYMBOL = "NIFTY"


def _bar(index, *, open_price, high, low, close):
    return OHLCVBar(
        timestamp=BASE + index * TIMEFRAME,
        symbol=SYMBOL,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=100,
    )


def _bars(prices):
    return [
        _bar(
            index,
            open_price=values[0],
            high=values[1],
            low=values[2],
            close=values[3],
        )
        for index, values in enumerate(prices)
    ]


class StaticHistoryProvider:
    def __init__(self, bars, *, is_synthetic=True):
        self.bars = tuple(bars)
        self.is_synthetic = is_synthetic

    def get_historical_ohlcv(self, request):
        selected = tuple(
            bar for bar in self.bars if request.start <= bar.timestamp <= request.end
        )
        return HistoricalOHLCVResponse(
            symbol=request.symbol,
            start=request.start,
            end=request.end,
            bars=selected,
            is_synthetic=self.is_synthetic,
        )

    def get_futures_quote(self, request):
        raise UnsupportedRequestError("not used by backtest")

    def get_options_quote(self, request):
        raise UnsupportedRequestError("not used by backtest")


class ScriptedStrategy:
    name = "scripted"
    version = "test"
    underlying = UnderlyingIndex.NIFTY
    symbol = SYMBOL
    supports_short = True

    def __init__(self, events_by_prefix=None, *, supports_short=True):
        self.events_by_prefix = events_by_prefix or {}
        self.supports_short = supports_short
        self.seen_prefixes = []

    def is_compatible_with(self, underlying, timeframe, symbol):
        return (
            underlying is self.underlying
            and timeframe == TIMEFRAME
            and symbol == self.symbol
        )

    def evaluate(self, data):
        self.seen_prefixes.append(tuple(bar.timestamp for bar in data.bars))
        direction, events, stop_price = self.events_by_prefix.get(
            len(data.bars),
            (SignalDirection.NO_TRADE, (), None),
        )
        is_entry = direction is not SignalDirection.NO_TRADE
        return StrategySignal(
            strategy_name=self.name,
            strategy_version=self.version,
            underlying=data.underlying,
            input_symbol=data.symbol,
            timeframe=data.timeframe,
            observation_timestamp=data.observation_timestamp,
            direction=direction,
            events=events,
            status=(
                SignalStatus.READY
                if is_entry or events
                else SignalStatus.NO_TRADE
            ),
            warmup_status=WarmupStatus.SUFFICIENT,
            indicator_values=(IndicatorValue("close", data.bars[-1].close),),
            entry_reference_price=(
                float(data.bars[-1].close) if is_entry else None
            ),
            stop_reference_price=stop_price if is_entry else None,
            entry_conditions=("scripted entry event",),
            exit_conditions=("scripted exit event",),
            stop_loss_method="explicit test stop" if stop_price else "not configured",
            input_requirements=("validated OHLCV prefix",),
            reasons=("scripted deterministic event",),
            missing_data=(),
            invalid_conditions=(),
            data_source=data.data_source,
            is_synthetic=data.is_synthetic,
            used_incomplete_candle=False,
            evaluation_at=data.evaluation_at,
        )


def _config(
    *,
    initial_capital=10000.0,
    quantity=1,
    unit_multiplier=1.0,
    execution=None,
    target_distance=None,
):
    return BacktestConfig(
        initial_capital=initial_capital,
        quantity=quantity,
        unit_multiplier=unit_multiplier,
        execution=(
            ExecutionAssumptions.zero_cost_research_mode()
            if execution is None
            else execution
        ),
        target_distance=target_distance,
    )


def _engine(bars, *, is_synthetic=True):
    provider = create_validated_market_data_provider(
        StaticHistoryProvider(bars, is_synthetic=is_synthetic)
    )
    return BacktestEngine(provider)


def _run(
    strategy,
    bars,
    *,
    config=None,
    last_bar_complete=True,
    trade_start=None,
    is_synthetic=True,
):
    request = HistoricalOHLCVRequest(
        symbol=SYMBOL,
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
    )
    return _engine(bars, is_synthetic=is_synthetic).run(
        strategy,
        request,
        timeframe=TIMEFRAME,
        data_source="synthetic-test-data" if is_synthetic else "explicit-test-data",
        config=_config() if config is None else config,
        last_bar_complete=last_bar_complete,
        trade_start=trade_start,
    )


class BacktestEventOrderingTests(unittest.TestCase):
    def test_repeated_runs_are_deterministic(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (101, 105, 100, 104),
                (107, 109, 106, 108),
            ]
        )
        events = {
            1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None),
            2: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_LONG,), None),
        }

        first = _run(ScriptedStrategy(events), bars)
        second = _run(ScriptedStrategy(events), bars)

        self.assertEqual(first, second)

    def test_close_signal_executes_at_next_bar_open_and_not_same_close(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (101, 105, 100, 104),
                (107, 109, 106, 108),
            ]
        )
        strategy = ScriptedStrategy(
            {
                1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None),
                2: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_LONG,), None),
            }
        )
        result = _run(strategy, bars, config=_config(quantity=2))

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.entry_bar_index, 1)
        self.assertEqual(trade.exit_bar_index, 2)
        self.assertEqual(trade.entry_fill_price, 101.0)
        self.assertEqual(trade.exit_fill_price, 107.0)
        self.assertEqual(trade.gross_pnl, 12.0)
        self.assertEqual(result.executions[0].timestamp, bars[1].timestamp)

    def test_strategy_receives_only_current_and_prior_bars(self):
        bars = _bars([(10, 11, 9, 10)] * 5)
        strategy = ScriptedStrategy()
        _run(strategy, bars)

        self.assertEqual(len(strategy.seen_prefixes), len(bars))
        for index, prefix in enumerate(strategy.seen_prefixes):
            self.assertEqual(prefix, tuple(bar.timestamp for bar in bars[: index + 1]))
            self.assertLessEqual(prefix[-1], bars[index].timestamp)

    def test_short_positions_are_accounted_with_reversed_pnl_sign(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (99, 100, 94, 95),
                (94, 96, 93, 95),
            ]
        )
        strategy = ScriptedStrategy(
            {
                1: (SignalDirection.SHORT, (SignalEvent.ENTER_SHORT,), None),
                2: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_SHORT,), None),
            }
        )
        trade = _run(strategy, bars).trades[0]

        self.assertEqual(trade.side, PositionSide.SHORT)
        self.assertEqual(trade.entry_fill_price, 99.0)
        self.assertEqual(trade.exit_fill_price, 94.0)
        self.assertEqual(trade.gross_pnl, 5.0)

    def test_strategy_short_permission_is_enforced(self):
        bars = _bars([(100, 101, 99, 100), (99, 101, 98, 100)])
        strategy = ScriptedStrategy(
            {1: (SignalDirection.SHORT, (SignalEvent.ENTER_SHORT,), None)},
            supports_short=False,
        )
        result = _run(strategy, bars)

        self.assertFalse(any(item.filled for item in result.executions))
        self.assertEqual(result.executions[0].reason, "strategy does not permit short positions")
        self.assertIsNone(result.open_position)

    def test_no_trade_period_has_explicit_zero_trade_metrics(self):
        bars = _bars([(10, 11, 9, 10)] * 4)
        result = _run(ScriptedStrategy(), bars)

        self.assertEqual(result.metrics.trade_count, 0)
        self.assertIsNone(result.metrics.win_rate)
        self.assertIsNone(result.metrics.expectancy)
        self.assertIsNone(result.metrics.profit_factor)
        self.assertEqual(
            result.metrics.profit_factor_status, ProfitFactorStatus.NO_TRADES
        )
        self.assertEqual(result.final_equity, result.initial_capital)

    def test_open_position_is_marked_at_end_without_forced_liquidation(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (102, 106, 101, 105),
            ]
        )
        strategy = ScriptedStrategy(
            {1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None)}
        )
        result = _run(strategy, bars, config=_config(quantity=2))

        self.assertEqual(result.trades, ())
        self.assertIsNotNone(result.open_position)
        self.assertEqual(result.open_position.entry_fill_price, 102.0)
        self.assertEqual(result.open_position.unrealized_gross_pnl, 6.0)
        self.assertEqual(result.final_cash, result.initial_capital)
        self.assertEqual(result.final_equity, result.initial_capital + 6.0)
        self.assertTrue(any("not liquidated" in note for note in result.notes))

    def test_incomplete_final_candle_is_excluded(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (101, 104, 100, 103),
            ]
        )
        strategy = ScriptedStrategy(
            {1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None)}
        )
        result = _run(strategy, bars, last_bar_complete=False)

        self.assertEqual(len(result.signals), 1)
        self.assertEqual(result.signals[0].observation_timestamp, bars[0].timestamp)
        self.assertEqual(result.trades, ())
        self.assertIsNone(result.open_position)


class BacktestExecutionModelTests(unittest.TestCase):
    def test_explicit_spread_slippage_and_fee_components_affect_net_pnl(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 106, 99, 105),
                (110, 112, 108, 110),
            ]
        )
        strategy = ScriptedStrategy(
            {
                1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None),
                2: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_LONG,), None),
            }
        )
        execution = ExecutionAssumptions(
            spread_bps=20,
            slippage_bps=10,
            fee_components=(
                FeeComponent("brokerage", FeeBasis.PER_EXECUTION, 1.0),
                FeeComponent("explicit_rate_fee", FeeBasis.NOTIONAL_RATE, 0.001),
            ),
        )
        result = _run(strategy, bars, config=_config(execution=execution))

        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_fill_price, 100.2)
        self.assertAlmostEqual(trade.exit_fill_price, 109.78)
        self.assertAlmostEqual(trade.gross_pnl, 9.58)
        self.assertAlmostEqual(trade.entry_fees, 1.1002)
        self.assertAlmostEqual(trade.exit_fees, 1.10978)
        self.assertAlmostEqual(trade.net_pnl, 7.37002)
        self.assertAlmostEqual(result.final_equity, 10007.37002)

    def test_explicit_fee_rate_can_apply_to_preceding_fee_components(self):
        assumptions = ExecutionAssumptions(
            spread_bps=0,
            slippage_bps=0,
            fee_components=(
                FeeComponent("explicit_brokerage", FeeBasis.PER_EXECUTION, 10.0),
                FeeComponent("explicit_tax_rate", FeeBasis.PRIOR_CHARGES_RATE, 0.1),
            ),
        )

        charges = assumptions.fee_charges(1000.0)

        self.assertEqual([item.amount for item in charges], [10.0, 1.0])

    def test_zero_cost_mode_must_be_selected_explicitly(self):
        zero = ExecutionAssumptions.zero_cost_research_mode()
        self.assertEqual(zero.spread_bps, 0.0)
        self.assertEqual(zero.slippage_bps, 0.0)
        self.assertEqual(zero.fee_components, ())
        with self.assertRaises(ValueError):
            ExecutionAssumptions(spread_bps=-1, slippage_bps=0, fee_components=())
        with self.assertRaises(ValueError):
            FeeComponent("bad", FeeBasis.NOTIONAL_RATE, math.nan)

    def test_gap_through_stop_fills_at_worse_open(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 102, 98, 101),
                (90, 95, 88, 91),
            ]
        )
        strategy = ScriptedStrategy(
            {1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), 95.0)}
        )
        trade = _run(strategy, bars).trades[0]

        self.assertEqual(trade.exit_reason, TradeExitReason.STOP_LOSS)
        self.assertEqual(trade.exit_fill_price, 90.0)
        self.assertEqual(trade.gross_pnl, -10.0)

    def test_target_gap_receives_no_price_improvement(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 101, 99, 100),
                (104, 106, 103, 105),
            ]
        )
        strategy = ScriptedStrategy(
            {1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None)}
        )
        trade = _run(
            strategy, bars, config=_config(target_distance=2.0)
        ).trades[0]

        self.assertEqual(trade.exit_reason, TradeExitReason.TARGET)
        self.assertEqual(trade.exit_fill_price, 102.0)
        self.assertEqual(trade.gross_pnl, 2.0)

    def test_ambiguous_stop_and_target_touch_chooses_stop(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 103, 94, 100),
            ]
        )
        strategy = ScriptedStrategy(
            {1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), 95.0)}
        )
        result = _run(
            strategy, bars, config=_config(target_distance=2.0)
        )

        self.assertEqual(result.trades[0].exit_reason, TradeExitReason.STOP_LOSS)
        self.assertEqual(result.trades[0].exit_fill_price, 95.0)
        self.assertIn("both touched", result.executions[-1].reason)

    def test_signal_stop_breached_before_entry_rejects_entry(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (90, 92, 88, 90),
            ]
        )
        strategy = ScriptedStrategy(
            {1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), 95.0)}
        )
        result = _run(strategy, bars)

        self.assertEqual(result.trades, ())
        self.assertEqual(len(result.executions), 1)
        self.assertFalse(result.executions[0].filled)


class BacktestMetricsTests(unittest.TestCase):
    def test_cash_equity_drawdown_expectancy_and_exposure_metrics(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 111, 99, 110),
                (110, 111, 109, 110),
                (110, 121, 109, 120),
                (120, 121, 119, 120),
            ]
        )
        strategy = ScriptedStrategy(
            {
                1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None),
                2: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_LONG,), None),
                3: (SignalDirection.SHORT, (SignalEvent.ENTER_SHORT,), None),
                4: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_SHORT,), None),
            }
        )
        result = _run(strategy, bars)

        self.assertEqual(result.metrics.trade_count, 2)
        self.assertEqual([trade.net_pnl for trade in result.trades], [10.0, -10.0])
        self.assertAlmostEqual(result.metrics.net_pnl, 0.0)
        self.assertAlmostEqual(result.metrics.total_return, 0.0)
        self.assertEqual(result.metrics.win_rate, 0.5)
        self.assertEqual(result.metrics.average_win, 10.0)
        self.assertEqual(result.metrics.average_loss, -10.0)
        self.assertEqual(result.metrics.expectancy, 0.0)
        self.assertEqual(result.metrics.profit_factor, 1.0)
        self.assertEqual(result.metrics.profit_factor_status, ProfitFactorStatus.DEFINED)
        self.assertEqual(result.metrics.maximum_drawdown, 10.0)
        self.assertEqual(result.metrics.exposure_bars, 2)
        self.assertEqual(result.metrics.exposure_fraction, 0.4)
        self.assertEqual(result.metrics.average_closed_holding_duration, TIMEFRAME)
        self.assertEqual(result.final_cash, result.initial_capital)

    def test_profit_factor_without_losses_is_explicitly_undefined(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 106, 99, 105),
                (105, 106, 104, 105),
            ]
        )
        strategy = ScriptedStrategy(
            {
                1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None),
                2: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_LONG,), None),
            }
        )
        metrics = _run(strategy, bars).metrics
        self.assertIsNone(metrics.profit_factor)
        self.assertEqual(metrics.profit_factor_status, ProfitFactorStatus.NO_LOSSES)


class WalkForwardTests(unittest.TestCase):
    def test_split_requires_separate_train_and_test_periods(self):
        with self.assertRaises(ValueError):
            WalkForwardSplit(BASE, BASE + TIMEFRAME, BASE + TIMEFRAME, BASE + 2 * TIMEFRAME)

    def test_plan_rejects_overlapping_test_and_reused_test_training_periods(self):
        first = WalkForwardSplit(
            BASE,
            BASE + 2 * TIMEFRAME,
            BASE + 3 * TIMEFRAME,
            BASE + 4 * TIMEFRAME,
        )
        second = WalkForwardSplit(
            BASE + 5 * TIMEFRAME,
            BASE + 7 * TIMEFRAME,
            BASE + 8 * TIMEFRAME,
            BASE + 9 * TIMEFRAME,
        )
        self.assertEqual(len(WalkForwardPlan((first, second)).splits), 2)

        reused_test = WalkForwardSplit(
            BASE + 3 * TIMEFRAME,
            BASE + 6 * TIMEFRAME,
            BASE + 7 * TIMEFRAME,
            BASE + 8 * TIMEFRAME,
        )
        with self.assertRaisesRegex(ValueError, "test interval may overlap"):
            WalkForwardPlan((first, reused_test))

        overlapping_tests = WalkForwardSplit(
            BASE,
            BASE + TIMEFRAME,
            BASE + 4 * TIMEFRAME,
            BASE + 6 * TIMEFRAME,
        )
        with self.assertRaises(ValueError):
            WalkForwardPlan((first, overlapping_tests))

    def test_walk_forward_uses_training_bars_for_warmup_but_only_trades_test(self):
        bars = _bars([(100, 101, 99, 100)] * 6)
        split = WalkForwardSplit(
            bars[0].timestamp,
            bars[2].timestamp,
            bars[3].timestamp,
            bars[5].timestamp,
        )
        plan = WalkForwardPlan((split,))
        strategy = ScriptedStrategy(
            {1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None)}
        )
        result = _engine(bars).run_walk_forward(
            strategy,
            plan,
            timeframe=TIMEFRAME,
            data_source="walk-forward-fixture",
            config=_config(),
            last_bar_complete=True,
        )[0]

        self.assertEqual(result.requested_start, split.train_start)
        self.assertEqual(result.trade_start, split.test_start)
        self.assertTrue(all(signal.observation_timestamp >= split.test_start for signal in result.signals))
        self.assertEqual(result.equity_curve[0].timestamp, split.test_start)


class BacktestValidationTests(unittest.TestCase):
    def test_configuration_requires_explicit_positive_capital_quantity_and_point_value(self):
        execution = ExecutionAssumptions.zero_cost_research_mode()
        for values in (
            {"initial_capital": 0, "quantity": 1, "unit_multiplier": 1, "execution": execution, "target_distance": None},
            {"initial_capital": 100, "quantity": 0, "unit_multiplier": 1, "execution": execution, "target_distance": None},
            {"initial_capital": 100, "quantity": 1, "unit_multiplier": 0, "execution": execution, "target_distance": None},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    BacktestConfig(**values)

    def test_data_source_and_final_bar_status_are_explicit(self):
        bars = _bars([(100, 101, 99, 100), (101, 102, 100, 101)])
        engine = _engine(bars)
        request = HistoricalOHLCVRequest(SYMBOL, bars[0].timestamp, bars[-1].timestamp)
        strategy = ScriptedStrategy()
        kwargs = dict(
            timeframe=TIMEFRAME,
            data_source="fixture",
            config=_config(),
            last_bar_complete=True,
        )
        with self.assertRaises(ValueError):
            engine.run(strategy, request, **{**kwargs, "data_source": ""})
        with self.assertRaises(ValueError):
            engine.run(strategy, request, **{**kwargs, "last_bar_complete": 1})

    def test_backtest_requires_validated_provider_and_compatible_timeframe(self):
        bars = _bars([(100, 101, 99, 100), (101, 102, 100, 101)])
        with self.assertRaises(TypeError):
            BacktestEngine(StaticHistoryProvider(bars))
        engine = _engine(bars)
        request = HistoricalOHLCVRequest(SYMBOL, bars[0].timestamp, bars[-1].timestamp)
        with self.assertRaises(ValueError):
            engine.run(
                ScriptedStrategy(),
                request,
                timeframe=timedelta(minutes=1),
                data_source="fixture",
                config=_config(),
                last_bar_complete=True,
            )

    def test_input_bars_are_not_mutated_and_synthetic_provenance_is_kept(self):
        bars = _bars(
            [
                (100, 101, 99, 100),
                (100, 106, 99, 105),
                (105, 106, 104, 105),
            ]
        )
        original = bars.copy()
        strategy = ScriptedStrategy(
            {
                1: (SignalDirection.LONG, (SignalEvent.ENTER_LONG,), None),
                2: (SignalDirection.NO_TRADE, (SignalEvent.EXIT_LONG,), None),
            }
        )
        result = _run(strategy, bars)
        self.assertEqual(bars, original)
        self.assertTrue(result.is_synthetic)
        self.assertEqual(result.data_source, "synthetic-test-data")

    def test_non_synthetic_provenance_is_preserved(self):
        bars = _bars([(100, 101, 99, 100), (100, 101, 99, 100)])
        result = _run(ScriptedStrategy(), bars, is_synthetic=False)
        self.assertFalse(result.is_synthetic)
        self.assertEqual(result.data_source, "explicit-test-data")


if __name__ == "__main__":
    unittest.main()
