"""Deterministic candle-event backtesting over validated historical data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import fsum, isfinite

from atlas.market_data.models import HistoricalOHLCVRequest, HistoricalOHLCVResponse
from atlas.market_data.provider import ValidatedMarketDataProvider
from atlas.strategies.models import (
    SignalDirection,
    SignalEvent,
    SignalStatus,
    StrategyMarketData,
    StrategySignal,
)
from atlas.strategies.strategies import Strategy
from atlas.backtesting.models import (
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    EquityPoint,
    ExecutionRecord,
    FeeCharge,
    FillSide,
    OpenPositionMark,
    PositionSide,
    ProfitFactorStatus,
    TradeExitReason,
    TradeRecord,
    _finite_float,
    _finite_sum,
)
from atlas.backtesting.walk_forward import WalkForwardPlan


@dataclass
class _OpenPosition:
    side: PositionSide
    quantity: int
    entry_timestamp: datetime
    entry_bar_index: int
    entry_fill_price: float
    stop_price: float | None
    target_price: float | None
    entry_fees: float
    entry_reason: str


class BacktestEngine:
    """Event-driven linear index-point simulator; never submits orders.

    Event order for each eligible candle is: gap protection for an existing
    position, previously queued signal at the open, intrabar stop/target checks,
    close mark-to-market, then strategy evaluation at that close. A close-based
    signal can therefore fill no earlier than the next candle's open.
    """

    def __init__(self, market_data: ValidatedMarketDataProvider) -> None:
        if not isinstance(market_data, ValidatedMarketDataProvider):
            raise TypeError("market_data must be a ValidatedMarketDataProvider")
        self._market_data = market_data

    def run(
        self,
        strategy: Strategy,
        request: HistoricalOHLCVRequest,
        *,
        timeframe: timedelta,
        data_source: str,
        config: BacktestConfig,
        last_bar_complete: bool,
        trade_start: datetime | None = None,
    ) -> BacktestResult:
        """Evaluate a strategy using only the validated historical prefix.

        Historical bar timestamps are retained as labels; the existing OHLCV
        model does not specify whether those labels mean bar-open or bar-close
        instants. Execution order is based on bar sequence, not an inferred
        timestamp convention. An incomplete final bar, when declared, is
        excluded from evaluation and execution.
        """
        _validate_run_inputs(
            strategy,
            request,
            timeframe=timeframe,
            data_source=data_source,
            config=config,
            last_bar_complete=last_bar_complete,
        )
        execution_start = request.start if trade_start is None else trade_start
        try:
            if execution_start < request.start or execution_start > request.end:
                raise ValueError("trade_start must lie within the requested range")
        except TypeError as exc:
            raise ValueError("trade_start and request bounds must be comparable") from exc

        response = self._market_data.get_historical_ohlcv(request)
        bars = response.bars
        notes: list[str] = []
        if not last_bar_complete:
            bars = bars[:-1]
            notes.append("The declared incomplete final candle was excluded.")
        if not bars:
            raise ValueError("no complete bars are available for backtesting")

        cash = _finite_float(config.initial_capital)
        position: _OpenPosition | None = None
        pending_signal: StrategySignal | None = None
        trades: list[TradeRecord] = []
        executions: list[ExecutionRecord] = []
        signals: list[StrategySignal] = []
        equity_curve: list[EquityPoint] = []
        sequence = 0

        def record_rejection(
            *,
            action: str,
            side: PositionSide,
            bar,
            reference_price: float,
            reason: str,
        ) -> None:
            nonlocal sequence
            sequence += 1
            executions.append(
                ExecutionRecord(
                    sequence=sequence,
                    timestamp=bar.timestamp,
                    action=action,
                    side=side,
                    quantity=config.quantity,
                    reference_price=reference_price,
                    fill_price=None,
                    fee_charges=(),
                    filled=False,
                    reason=reason,
                )
            )

        def execute_close(
            bar,
            bar_index: int,
            *,
            reference_price: float,
            exit_reason: TradeExitReason,
            action_reason: str,
        ) -> None:
            nonlocal cash, position, sequence
            if position is None:
                return
            side = position.side
            fill_side = FillSide.SELL if side is PositionSide.LONG else FillSide.BUY
            fill = config.execution.fill_price(reference_price, fill_side)
            notional = config.quantity * _finite_float(config.unit_multiplier) * fill
            charges = config.execution.fee_charges(notional)
            exit_fees = _sum_fees(charges)
            gross = (
                side.sign
                * config.quantity
                * _finite_float(config.unit_multiplier)
                * (fill - position.entry_fill_price)
            )
            if not isfinite(gross):
                raise ValueError("trade gross P&L exceeds numeric range")
            cash = _finite_sum("cash", cash, gross, -exit_fees)
            net = _finite_sum(
                "trade net P&L", gross, -position.entry_fees, -exit_fees
            )
            duration = _elapsed(bar.timestamp, position.entry_timestamp)
            if bar_index < position.entry_bar_index:
                raise ValueError("exit event precedes entry event")
            trades.append(
                TradeRecord(
                    trade_id=len(trades) + 1,
                    side=side,
                    quantity=position.quantity,
                    entry_timestamp=position.entry_timestamp,
                    exit_timestamp=bar.timestamp,
                    entry_bar_index=position.entry_bar_index,
                    exit_bar_index=bar_index,
                    entry_fill_price=position.entry_fill_price,
                    exit_fill_price=fill,
                    gross_pnl=gross,
                    entry_fees=position.entry_fees,
                    exit_fees=exit_fees,
                    net_pnl=net,
                    exit_reason=exit_reason,
                    entry_reason=position.entry_reason,
                    holding_bars=bar_index - position.entry_bar_index,
                    holding_duration=duration,
                )
            )
            sequence += 1
            executions.append(
                ExecutionRecord(
                    sequence=sequence,
                    timestamp=bar.timestamp,
                    action=f"exit_{side.value}",
                    side=side,
                    quantity=position.quantity,
                    reference_price=reference_price,
                    fill_price=fill,
                    fee_charges=charges,
                    filled=True,
                    reason=action_reason,
                )
            )
            position = None

        def execute_entry(
            signal: StrategySignal,
            bar,
            bar_index: int,
            side: PositionSide,
        ) -> None:
            nonlocal cash, position, sequence
            if side is PositionSide.SHORT and not strategy.supports_short:
                record_rejection(
                    action="enter_short",
                    side=side,
                    bar=bar,
                    reference_price=bar.open,
                    reason="strategy does not permit short positions",
                )
                return
            if position is not None:
                if position.side is side:
                    return
                execute_close(
                    bar,
                    bar_index,
                    reference_price=bar.open,
                    exit_reason=TradeExitReason.SIGNAL,
                    action_reason="opposite directional signal",
                )

            fill_side = FillSide.BUY if side is PositionSide.LONG else FillSide.SELL
            fill = config.execution.fill_price(bar.open, fill_side)
            stop_price = signal.stop_reference_price
            if stop_price is not None and (
                (side is PositionSide.LONG and stop_price >= fill)
                or (side is PositionSide.SHORT and stop_price <= fill)
            ):
                record_rejection(
                    action=f"enter_{side.value}",
                    side=side,
                    bar=bar,
                    reference_price=bar.open,
                    reason="signal stop is already breached or on the wrong side of entry",
                )
                return

            target_price = None
            if config.target_distance is not None:
                distance = _finite_float(config.target_distance)
                target_price = (
                    fill + distance if side is PositionSide.LONG else fill - distance
                )
                if not isfinite(target_price) or target_price <= 0.0:
                    record_rejection(
                        action=f"enter_{side.value}",
                        side=side,
                        bar=bar,
                        reference_price=bar.open,
                        reason="configured target is outside the positive price range",
                    )
                    return

            notional = config.quantity * _finite_float(config.unit_multiplier) * fill
            charges = config.execution.fee_charges(notional)
            entry_fees = _sum_fees(charges)
            cash = _finite_sum("cash", cash, -entry_fees)
            position = _OpenPosition(
                side=side,
                quantity=config.quantity,
                entry_timestamp=bar.timestamp,
                entry_bar_index=bar_index,
                entry_fill_price=fill,
                stop_price=stop_price,
                target_price=target_price,
                entry_fees=entry_fees,
                entry_reason="; ".join(signal.reasons),
            )
            sequence += 1
            executions.append(
                ExecutionRecord(
                    sequence=sequence,
                    timestamp=bar.timestamp,
                    action=f"enter_{side.value}",
                    side=side,
                    quantity=config.quantity,
                    reference_price=bar.open,
                    fill_price=fill,
                    fee_charges=charges,
                    filled=True,
                    reason="signal from the prior candle executed at this candle's open",
                )
            )

        def process_pending(signal: StrategySignal, bar, bar_index: int) -> None:
            if signal.status not in (SignalStatus.READY, SignalStatus.NO_TRADE):
                return
            if position is not None:
                if (
                    position.side is PositionSide.LONG
                    and SignalEvent.EXIT_LONG in signal.events
                ):
                    execute_close(
                        bar,
                        bar_index,
                        reference_price=bar.open,
                        exit_reason=TradeExitReason.SIGNAL,
                        action_reason="strategy exit-long condition",
                    )
                elif (
                    position.side is PositionSide.SHORT
                    and SignalEvent.EXIT_SHORT in signal.events
                ):
                    execute_close(
                        bar,
                        bar_index,
                        reference_price=bar.open,
                        exit_reason=TradeExitReason.SIGNAL,
                        action_reason="strategy exit-short condition",
                    )

            if signal.direction is SignalDirection.LONG:
                execute_entry(signal, bar, bar_index, PositionSide.LONG)
            elif signal.direction is SignalDirection.SHORT:
                execute_entry(signal, bar, bar_index, PositionSide.SHORT)

        for bar_index, bar in enumerate(bars):
            try:
                eligible = bar.timestamp >= execution_start
            except TypeError as exc:
                raise ValueError("bar timestamp and trade_start must be comparable") from exc
            if not eligible:
                continue

            # 1. Opening gaps can trigger protection carried from a prior bar.
            if position is not None:
                gap_exit = _gap_protection(position, bar)
                if gap_exit is not None:
                    reason, reference = gap_exit
                    execute_close(
                        bar,
                        bar_index,
                        reference_price=reference,
                        exit_reason=reason,
                        action_reason=(
                            "gap-through stop filled at open"
                            if reason is TradeExitReason.STOP_LOSS
                            else "target gap filled at target without price improvement"
                        ),
                    )

            # 2. A signal from the preceding close is eligible at this open.
            if pending_signal is not None:
                process_pending(pending_signal, bar, bar_index)
                pending_signal = None

            # 3. Stop/target touches after the open use OHLC extrema.
            if position is not None:
                intrabar_exit = _intrabar_protection(position, bar)
                if intrabar_exit is not None:
                    reason, reference = intrabar_exit
                    execute_close(
                        bar,
                        bar_index,
                        reference_price=reference,
                        exit_reason=reason,
                        action_reason=(
                            "stop selected because stop and target were both touched"
                            if reason is TradeExitReason.STOP_LOSS
                            and _both_brackets_touched(position, bar)
                            else f"intrabar {reason.value} trigger"
                        ),
                    )

            # 4. Mark positions at the candle close; no same-close execution.
            equity_curve.append(
                _equity_point(
                    bar=bar,
                    bar_index=bar_index,
                    cash=cash,
                    position=position,
                    config=config,
                )
            )

            # 5. Evaluate only a prefix ending at this completed candle.
            prefix = HistoricalOHLCVResponse(
                symbol=response.symbol,
                start=request.start,
                end=bar.timestamp,
                bars=tuple(bars[: bar_index + 1]),
                is_synthetic=response.is_synthetic,
            )
            market_data = StrategyMarketData._from_validated_provider(
                response=prefix,
                underlying=strategy.underlying,
                timeframe=timeframe,
                data_source=data_source,
                evaluation_at=bar.timestamp,
                last_bar_complete=True,
            )
            signal = strategy.evaluate(market_data)
            if not isinstance(signal, StrategySignal):
                raise TypeError("strategy must return a StrategySignal")
            signals.append(signal)
            if signal.status in (SignalStatus.READY, SignalStatus.NO_TRADE) and (
                signal.direction is not SignalDirection.NO_TRADE or signal.events
            ):
                pending_signal = signal

        if pending_signal is not None:
            notes.append("A final-candle signal remained unfilled because no next candle was available.")

        open_mark = None
        unrealized_net = 0.0
        if position is not None:
            final_bar = bars[-1]
            unrealized_gross = _unrealized(position, final_bar.close, config)
            unrealized_net = _finite_sum(
                "unrealized net P&L", unrealized_gross, -position.entry_fees
            )
            open_mark = OpenPositionMark(
                side=position.side,
                quantity=position.quantity,
                entry_timestamp=position.entry_timestamp,
                entry_bar_index=position.entry_bar_index,
                entry_fill_price=position.entry_fill_price,
                mark_timestamp=final_bar.timestamp,
                mark_price=float(final_bar.close),
                unrealized_gross_pnl=unrealized_gross,
                entry_fees=position.entry_fees,
                stop_price=position.stop_price,
                target_price=position.target_price,
                holding_bars=(len(bars) - 1) - position.entry_bar_index,
                holding_duration=_elapsed(final_bar.timestamp, position.entry_timestamp),
            )
            notes.append("Any open position is marked to the final close and is not liquidated.")

        final_equity = (
            _finite_sum(
                "final equity",
                cash,
                _unrealized(position, bars[-1].close, config),
            )
            if position is not None
            else cash
        )
        metrics = _calculate_metrics(
            initial_capital=_finite_float(config.initial_capital),
            final_equity=final_equity,
            trades=trades,
            equity_curve=equity_curve,
            open_unrealized_pnl=unrealized_net,
        )
        return BacktestResult(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            symbol=request.symbol,
            timeframe=timeframe,
            underlying=strategy.underlying.value,
            requested_start=request.start,
            requested_end=request.end,
            trade_start=execution_start,
            data_source=data_source,
            is_synthetic=response.is_synthetic,
            config=config,
            initial_capital=_finite_float(config.initial_capital),
            final_cash=cash,
            final_equity=final_equity,
            metrics=metrics,
            trades=tuple(trades),
            executions=tuple(executions),
            signals=tuple(signals),
            equity_curve=tuple(equity_curve),
            open_position=open_mark,
            notes=tuple(notes),
        )

    def run_walk_forward(
        self,
        strategy: Strategy,
        plan: WalkForwardPlan,
        *,
        timeframe: timedelta,
        data_source: str,
        config: BacktestConfig,
        last_bar_complete: bool,
    ) -> tuple[BacktestResult, ...]:
        """Evaluate a fixed strategy only within each walk-forward test window.

        Training bars are requested solely as indicator warm-up. This API has
        no parameter-fitting callback and never passes test results to a
        selector.
        """
        if not isinstance(plan, WalkForwardPlan):
            raise ValueError("plan must be a WalkForwardPlan")
        results = []
        for split in plan.splits:
            results.append(
                self.run(
                    strategy,
                    split.request(strategy.symbol),
                    timeframe=timeframe,
                    data_source=data_source,
                    config=config,
                    last_bar_complete=last_bar_complete,
                    trade_start=split.test_start,
                )
            )
        return tuple(results)


def _gap_protection(
    position: _OpenPosition,
    bar,
) -> tuple[TradeExitReason, float] | None:
    if position.stop_price is not None:
        stop_gap = (
            bar.open <= position.stop_price
            if position.side is PositionSide.LONG
            else bar.open >= position.stop_price
        )
        if stop_gap:
            return TradeExitReason.STOP_LOSS, float(bar.open)
    if position.target_price is not None:
        target_gap = (
            bar.open >= position.target_price
            if position.side is PositionSide.LONG
            else bar.open <= position.target_price
        )
        if target_gap:
            # A favorable open is not credited beyond the target limit.
            return TradeExitReason.TARGET, position.target_price
    return None


def _intrabar_protection(
    position: _OpenPosition,
    bar,
) -> tuple[TradeExitReason, float] | None:
    stop_touched = position.stop_price is not None and (
        bar.low <= position.stop_price
        if position.side is PositionSide.LONG
        else bar.high >= position.stop_price
    )
    target_touched = position.target_price is not None and (
        bar.high >= position.target_price
        if position.side is PositionSide.LONG
        else bar.low <= position.target_price
    )
    if stop_touched:
        return TradeExitReason.STOP_LOSS, position.stop_price
    if target_touched:
        return TradeExitReason.TARGET, position.target_price
    return None


def _both_brackets_touched(position: _OpenPosition, bar) -> bool:
    if position.stop_price is None or position.target_price is None:
        return False
    stop_touched = (
        bar.low <= position.stop_price
        if position.side is PositionSide.LONG
        else bar.high >= position.stop_price
    )
    target_touched = (
        bar.high >= position.target_price
        if position.side is PositionSide.LONG
        else bar.low <= position.target_price
    )
    return stop_touched and target_touched


def _equity_point(*, bar, bar_index, cash, position, config) -> EquityPoint:
    unrealized = _unrealized(position, bar.close, config) if position else 0.0
    equity = _finite_sum("equity", cash, unrealized)
    signed_quantity = 0 if position is None else position.side.sign * position.quantity
    exposure = (
        0.0
        if position is None
        else abs(
            position.quantity
            * _finite_float(config.unit_multiplier)
            * _finite_float(bar.close)
        )
    )
    if not isfinite(exposure):
        raise ValueError("gross exposure exceeds numeric range")
    return EquityPoint(
        timestamp=bar.timestamp,
        bar_index=bar_index,
        cash=cash,
        unrealized_pnl=unrealized,
        equity=equity,
        signed_quantity=signed_quantity,
        mark_price=float(bar.close),
        gross_exposure=exposure,
    )


def _unrealized(position, mark_price, config) -> float:
    if position is None:
        return 0.0
    result = (
        position.side.sign
        * position.quantity
        * _finite_float(config.unit_multiplier)
        * (_finite_float(mark_price) - position.entry_fill_price)
    )
    if not isfinite(result):
        raise ValueError("unrealized P&L exceeds numeric range")
    return result


def _calculate_metrics(
    *,
    initial_capital: float,
    final_equity: float,
    trades: list[TradeRecord],
    equity_curve: list[EquityPoint],
    open_unrealized_pnl: float,
) -> BacktestMetrics:
    net_pnl = _finite_sum("net P&L", final_equity, -initial_capital)
    total_return = net_pnl / initial_capital
    if not isfinite(total_return):
        raise ValueError("total return exceeds numeric range")
    trade_pnls = [trade.net_pnl for trade in trades]
    wins = [value for value in trade_pnls if value > 0.0]
    losses = [value for value in trade_pnls if value < 0.0]
    count = len(trades)
    win_rate = len(wins) / count if count else None
    average_win = fsum(wins) / len(wins) if wins else None
    average_loss = fsum(losses) / len(losses) if losses else None
    expectancy = fsum(trade_pnls) / count if count else None
    winning_total = fsum(wins)
    losing_total = abs(fsum(losses))
    if count == 0:
        factor = None
        factor_status = ProfitFactorStatus.NO_TRADES
    elif losing_total == 0.0:
        factor = None
        factor_status = ProfitFactorStatus.NO_LOSSES
    else:
        factor = winning_total / losing_total
        factor_status = ProfitFactorStatus.DEFINED

    peak = initial_capital
    maximum_drawdown = 0.0
    maximum_drawdown_fraction = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity)
        drawdown = peak - point.equity
        maximum_drawdown = max(maximum_drawdown, drawdown)
        if peak > 0.0:
            maximum_drawdown_fraction = max(
                maximum_drawdown_fraction, drawdown / peak
            )

    exposure_bars = sum(point.gross_exposure > 0.0 for point in equity_curve)
    point_count = len(equity_curve)
    average_exposure = (
        fsum(point.gross_exposure for point in equity_curve) / point_count
        if point_count
        else 0.0
    )
    maximum_exposure = max(
        (point.gross_exposure for point in equity_curve), default=0.0
    )
    average_duration = (
        sum((trade.holding_duration for trade in trades), timedelta(0)) / count
        if count
        else None
    )
    return BacktestMetrics(
        net_pnl=net_pnl,
        total_return=total_return,
        trade_count=count,
        win_rate=win_rate,
        average_win=average_win,
        average_loss=average_loss,
        expectancy=expectancy,
        profit_factor=factor,
        profit_factor_status=factor_status,
        maximum_drawdown=maximum_drawdown,
        maximum_drawdown_fraction=maximum_drawdown_fraction,
        realized_trade_pnl=fsum(trade_pnls),
        unrealized_pnl=open_unrealized_pnl,
        exposure_bars=exposure_bars,
        exposure_fraction=exposure_bars / point_count if point_count else 0.0,
        average_gross_exposure=average_exposure,
        maximum_gross_exposure=maximum_exposure,
        average_closed_holding_duration=average_duration,
    )


def _validate_run_inputs(
    strategy: Strategy,
    request: HistoricalOHLCVRequest,
    *,
    timeframe: timedelta,
    data_source: str,
    config: BacktestConfig,
    last_bar_complete: bool,
) -> None:
    if not isinstance(request, HistoricalOHLCVRequest):
        raise ValueError("request must be a HistoricalOHLCVRequest")
    if not isinstance(config, BacktestConfig):
        raise ValueError("config must be a BacktestConfig")
    if not isinstance(timeframe, timedelta) or timeframe <= timedelta(0):
        raise ValueError("timeframe must be a positive timedelta")
    if not isinstance(data_source, str) or not data_source.strip():
        raise ValueError("data_source must be a non-empty string")
    if not isinstance(last_bar_complete, bool):
        raise ValueError("last_bar_complete must be a bool")
    required = (
        "name",
        "version",
        "symbol",
        "underlying",
        "supports_short",
        "is_compatible_with",
        "evaluate",
    )
    if any(not hasattr(strategy, name) for name in required):
        raise TypeError("strategy must provide identity, compatibility, and evaluation members")
    if not isinstance(strategy.supports_short, bool):
        raise ValueError("strategy supports_short must be a bool")
    if not callable(strategy.evaluate):
        raise TypeError("strategy evaluate member must be callable")
    if not callable(strategy.is_compatible_with):
        raise TypeError("strategy is_compatible_with member must be callable")
    if not strategy.is_compatible_with(
        strategy.underlying, timeframe, request.symbol
    ):
        raise ValueError("backtest request does not match strategy symbol/timeframe")


def _sum_fees(charges: tuple[FeeCharge, ...]) -> float:
    try:
        result = fsum(item.amount for item in charges)
    except (OverflowError, ValueError) as exc:
        raise ValueError("fee total exceeds numeric range") from exc
    if not isfinite(result):
        raise ValueError("fee total exceeds numeric range")
    return result


def _elapsed(end: datetime, start: datetime) -> timedelta:
    try:
        elapsed = end - start
    except TypeError as exc:
        raise ValueError("bar timestamps must be comparable") from exc
    if elapsed < timedelta(0):
        raise ValueError("bar timestamps must not move backwards")
    return elapsed


__all__ = ["BacktestEngine"]
