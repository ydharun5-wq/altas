"""Deterministic, synthetic-only tests for the M8.3 paper ledger."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atlas.derivatives.models import (
    DerivativeContract,
    Exchange,
    FuturesQuote,
    InstrumentType,
    OptionType,
    OptionsQuote,
    UnderlyingIndex,
)
from atlas.market_data.models import FuturesQuoteResponse, OptionsQuoteResponse, QuoteFreshnessPolicy
from atlas.risk import (
    ProposedTrade,
    RiskDecisionStatus,
    RiskPolicy,
    RiskReasonCode,
    TradeStructure,
    TradingHorizon,
)
from atlas.strategies.models import (
    SignalDirection,
    SignalEvent,
    SignalStatus,
    StrategySignal,
    WarmupStatus,
)
from atlas.paper_trading import (
    DuplicateSubmissionError,
    InvalidPaperQuoteError,
    OrderStateError,
    PaperAccountConfig,
    PaperCloseOrderRequest,
    PaperExecutionConfig,
    PaperFeeBasis,
    PaperFeeComponent,
    PaperFillSide,
    PaperOpenOrderRequest,
    PaperOrderAction,
    PaperOrderStatus,
    PaperPositionSide,
    PaperQuoteObservation,
    PaperTradingError,
    PaperTradingEngine,
)


UTC = timezone.utc
NOW = datetime(2026, 11, 19, 10, 0, tzinfo=UTC)


def _contract(
    instrument=InstrumentType.FUT,
    *,
    contract_id=None,
    lot_size=2,
):
    is_option = instrument is InstrumentType.OPT
    return DerivativeContract(
        exchange=Exchange.NSE,
        underlying=UnderlyingIndex.NIFTY,
        instrument_type=instrument,
        expiry=date(2026, 12, 31),
        contract_id=contract_id or f"NIFTY-{instrument.value}-20261231",
        lot_size=lot_size if not is_option else max(1, lot_size),
        tick_size=Decimal("0.05"),
        strike=Decimal("25000") if is_option else None,
        option_type=OptionType.CE if is_option else None,
    )


def _policy(*, allow_synthetic=True, risk_fraction=Decimal("0.10")):
    return RiskPolicy(
        policy_id="paper-test",
        version="1",
        max_risk_per_trade_fraction=risk_fraction,
        max_daily_loss_fraction=Decimal("0.50"),
        max_account_drawdown_fraction=Decimal("0.50"),
        max_open_positions=5,
        max_exposure_per_underlying=Decimal("10000000"),
        allowed_underlyings=frozenset(UnderlyingIndex),
        allowed_instrument_types=frozenset(InstrumentType),
        allowed_horizons=frozenset(TradingHorizon),
        allow_synthetic_data=allow_synthetic,
        user_approval_required=True,
    )


def _execution(*, slippage_bps=Decimal("0"), fees=(), freshness=None):
    return PaperExecutionConfig(
        slippage_bps=slippage_bps,
        fee_components=tuple(fees),
        quote_freshness_policy=freshness,
    )


def _engine(*, policy=None, execution=None, initial_cash=Decimal("100000")):
    return PaperTradingEngine(
        PaperAccountConfig(
            initial_cash=initial_cash,
            risk_policy=_policy() if policy is None else policy,
            execution=_execution() if execution is None else execution,
            started_at=NOW,
        )
    )


def _quote_response(
    contract,
    timestamp=NOW,
    *,
    bid=Decimal("100"),
    ask=Decimal("100"),
    last=Decimal("100"),
    synthetic=True,
):
    quote_type = FuturesQuote if contract.instrument_type is InstrumentType.FUT else OptionsQuote
    quote = quote_type(
        contract=contract,
        timestamp=timestamp,
        bid=bid,
        ask=ask,
        last_traded_price=last,
        volume=100,
        open_interest=500,
        change_in_open_interest=0,
    )
    response_type = FuturesQuoteResponse if contract.instrument_type is InstrumentType.FUT else OptionsQuoteResponse
    return response_type(quote, is_synthetic=synthetic)


def _observation(
    contract,
    timestamp,
    *,
    bid=Decimal("100"),
    ask=Decimal("100"),
    last=Decimal("100"),
    source="synthetic-fixture",
    synthetic=True,
    approved=False,
    as_of=None,
):
    return PaperQuoteObservation(
        response=_quote_response(
            contract,
            timestamp,
            bid=bid,
            ask=ask,
            last=last,
            synthetic=synthetic,
        ),
        data_source=source,
        as_of=timestamp if as_of is None else as_of,
        approved=approved,
    )


def _proposal(
    *,
    instrument=InstrumentType.FUT,
    direction=SignalDirection.LONG,
    contract=None,
    entry=None,
    stop=None,
    target=None,
    quantity=1,
    structure=TradeStructure.SINGLE_LEG,
    synthetic=True,
    source="synthetic-fixture",
    market_data=None,
):
    selected_contract = contract or _contract(instrument)
    if entry is None:
        entry = Decimal("5") if selected_contract.instrument_type is InstrumentType.OPT else Decimal("100")
    if stop is None and selected_contract.instrument_type is InstrumentType.FUT:
        stop = Decimal("90") if direction is SignalDirection.LONG else Decimal("110")
    if market_data is None:
        market_data = _quote_response(
            selected_contract,
            bid=entry,
            ask=entry,
            last=entry,
            synthetic=synthetic,
        )
    no_trade = direction is SignalDirection.NO_TRADE
    event = (
        ()
        if no_trade
        else (SignalEvent.ENTER_SHORT,)
        if direction is SignalDirection.SHORT
        else (SignalEvent.ENTER_LONG,)
    )
    signal = StrategySignal(
        strategy_name="paper-fixture",
        strategy_version="1",
        underlying=selected_contract.underlying,
        input_symbol=selected_contract.underlying.value,
        timeframe=timedelta(minutes=5),
        observation_timestamp=NOW,
        direction=direction,
        events=event,
        status=SignalStatus.NO_TRADE if no_trade else SignalStatus.READY,
        warmup_status=WarmupStatus.SUFFICIENT,
        indicator_values=(),
        entry_reference_price=None if no_trade else float(entry),
        stop_reference_price=None if no_trade or stop is None else float(stop),
        entry_conditions=("synthetic test entry",),
        exit_conditions=("synthetic test exit",),
        stop_loss_method="explicit fixture stop",
        input_requirements=("explicit quote",),
        reasons=("unit test",),
        missing_data=(),
        invalid_conditions=(),
        data_source=source,
        is_synthetic=synthetic,
        used_incomplete_candle=False,
        evaluation_at=NOW,
    )
    return ProposedTrade(
        contract=selected_contract,
        signal=signal,
        horizon=TradingHorizon.INTRADAY,
        structure=structure,
        market_data=market_data,
        market_data_source=source,
        entry_reference_price=entry,
        stop_loss_reference_price=stop,
        target_reference_price=target,
        quantity=quantity,
        point_value=(Decimal("10") if selected_contract.instrument_type is InstrumentType.FUT else None),
        assumptions=("synthetic unit-test assumptions",),
    )


def _approve(engine, proposal, *, client_order_id="open-1", quote_approved=False):
    order = engine.submit_open_order(
        PaperOpenOrderRequest(
            client_order_id=client_order_id,
            proposal=proposal,
            submitted_at=NOW,
            quote_approved=quote_approved,
        )
    )
    return engine.approve_open_order(
        client_order_id,
        user_approved=True,
        evaluated_at=NOW,
    )


def _open_and_fill(
    *,
    engine=None,
    proposal=None,
    client_order_id="open-1",
    quote_approved=False,
    fill_at=None,
    bid=None,
    ask=None,
):
    selected_engine = _engine() if engine is None else engine
    selected_proposal = _proposal() if proposal is None else proposal
    approved_order = _approve(
        selected_engine,
        selected_proposal,
        client_order_id=client_order_id,
        quote_approved=quote_approved,
    )
    at = fill_at or (NOW + timedelta(minutes=1))
    price = selected_proposal.entry_reference_price
    observation = _observation(
        selected_proposal.contract,
        at,
        bid=price if bid is None else bid,
        ask=price if ask is None else ask,
        last=price,
        source=selected_proposal.market_data_source,
        synthetic=selected_proposal.market_data.is_synthetic,
        approved=quote_approved,
    )
    fill = selected_engine.fill_order(
        client_order_id,
        fill_id=f"fill-{client_order_id}",
        quantity=selected_proposal.quantity,
        quote=observation,
    )
    return selected_engine, approved_order, fill


class PaperTradingEngineTests(unittest.TestCase):
    def test_order_staging_emits_a_paper_audit_log(self):
        engine = _engine()
        with self.assertLogs("atlas.paper_trading.engine", level="INFO") as logs:
            engine.submit_open_order(
                PaperOpenOrderRequest("logged-open", _proposal(), NOW, False)
            )

        self.assertIn("paper order staged for approval", logs.output[0])
        self.assertIn("logged-open", logs.output[0])

    def test_initial_snapshot_has_explicit_cash_and_no_positions(self):
        execution = _execution()
        engine = _engine(initial_cash=Decimal("12345.67"), execution=execution)
        self.assertEqual(engine.snapshot.cash, Decimal("12345.67"))
        self.assertEqual(engine.snapshot.equity, Decimal("12345.67"))
        self.assertEqual(engine.snapshot.positions, ())
        self.assertIs(engine.snapshot.execution_config, execution)

    def test_open_requires_explicit_user_approval_and_m81_pass(self):
        engine = _engine()
        order = engine.submit_open_order(
            PaperOpenOrderRequest("o1", _proposal(), NOW, quote_approved=False)
        )
        self.assertEqual(order.status, PaperOrderStatus.PENDING_APPROVAL)
        self.assertEqual(engine.approve_open_order("o1", user_approved=None, evaluated_at=NOW).status,
                         PaperOrderStatus.PENDING_APPROVAL)
        with self.assertRaises(OrderStateError):
            engine.fill_order(
                "o1",
                fill_id="pending-fill",
                quantity=1,
                quote=_observation(_contract(), NOW + timedelta(minutes=1)),
            )
        approved = engine.approve_open_order("o1", user_approved=True, evaluated_at=NOW)
        self.assertEqual(approved.status, PaperOrderStatus.OPEN)
        self.assertEqual(approved.risk_decision.status, RiskDecisionStatus.APPROVED)
        self.assertIs(approved.risk_decision.user_approved, True)

    def test_explicit_user_denial_rejects_opening(self):
        engine = _engine()
        engine.submit_open_order(PaperOpenOrderRequest("o1", _proposal(), NOW, False))
        rejected = engine.approve_open_order("o1", user_approved=False, evaluated_at=NOW)
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)
        self.assertIn("approval was denied", rejected.rejection_reason)

    def test_invalid_policy_and_synthetic_data_fail_m81_gate(self):
        engine = _engine(policy=_policy(allow_synthetic=False))
        rejected = _approve(engine, _proposal())
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)
        self.assertEqual(rejected.risk_decision.status, RiskDecisionStatus.REJECTED)
        self.assertIn(RiskReasonCode.SYNTHETIC_DATA_NOT_ALLOWED,
                      {item.code for item in rejected.risk_decision.reasons})

    def test_proposal_that_is_not_exactly_sized_is_rejected(self):
        engine = _engine()
        proposal = _proposal(quantity=60)
        rejected = _approve(engine, proposal)
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)
        self.assertIn("not correctly sized", rejected.rejection_reason)
        self.assertEqual(engine.snapshot.positions, ())

    def test_multi_leg_and_short_option_orders_are_rejected(self):
        multi = _approve(_engine(), _proposal(structure=TradeStructure.MULTI_LEG), client_order_id="multi")
        short_option = _approve(
            _engine(),
            _proposal(
                instrument=InstrumentType.OPT,
                direction=SignalDirection.SHORT,
                entry=Decimal("5"),
                quantity=1,
            ),
            client_order_id="short-option",
        )
        self.assertEqual(multi.status, PaperOrderStatus.REJECTED)
        self.assertIn("multi-leg", multi.rejection_reason)
        self.assertEqual(short_option.status, PaperOrderStatus.REJECTED)
        self.assertIn("short-option", short_option.rejection_reason)

    def test_known_long_futures_position_and_decimal_pnl(self):
        engine, order, fill = _open_and_fill()
        position = engine.snapshot.positions[0]
        self.assertEqual(order.reserved_risk, Decimal("200"))
        self.assertEqual(fill.side, PaperFillSide.BUY)
        self.assertEqual(position.direction, PaperPositionSide.LONG)
        self.assertEqual(position.quantity, 1)
        self.assertEqual(position.average_entry_price, Decimal("100"))
        self.assertEqual(engine.snapshot.unrealized_pnl, Decimal("0"))
        self.assertTrue(position.is_synthetic)
        self.assertEqual(position.data_sources, ("synthetic-fixture",))

    def test_known_short_futures_uses_sell_entry_and_symmetric_pnl(self):
        proposal = _proposal(direction=SignalDirection.SHORT)
        engine, _, fill = _open_and_fill(proposal=proposal, client_order_id="short")
        self.assertEqual(fill.side, PaperFillSide.SELL)
        close_at = NOW + timedelta(minutes=2)
        close_quote = _observation(proposal.contract, close_at, bid=Decimal("90"), ask=Decimal("90"))
        request = PaperCloseOrderRequest("close-short", "pos-short", 1, close_quote, close_at)
        order = engine.submit_close_order(request, user_approved=True)
        result = engine.fill_order("close-short", fill_id="close-fill-short", quantity=1, quote=close_quote)
        self.assertEqual(order.direction, PaperPositionSide.SHORT)
        self.assertEqual(result.side, PaperFillSide.BUY)
        self.assertEqual(result.gross_realized_pnl, Decimal("200"))
        self.assertEqual(engine.snapshot.positions, ())

    def test_long_futures_realized_pnl_and_equity(self):
        engine, _, _ = _open_and_fill()
        at = NOW + timedelta(minutes=2)
        quote = _observation(_contract(), at, bid=Decimal("110"), ask=Decimal("110"))
        order = engine.submit_close_order(
            PaperCloseOrderRequest("close-1", "pos-open-1", 1, quote, at),
            user_approved=True,
        )
        fill = engine.fill_order("close-1", fill_id="close-fill-1", quantity=1, quote=quote)
        self.assertEqual(order.action, PaperOrderAction.CLOSE)
        self.assertEqual(fill.gross_realized_pnl, Decimal("200"))
        self.assertEqual(engine.snapshot.realized_pnl, Decimal("200"))
        self.assertEqual(engine.snapshot.gross_realized_pnl, Decimal("200"))
        self.assertEqual(engine.snapshot.cash, Decimal("100200"))
        self.assertEqual(engine.snapshot.equity, Decimal("100200"))
        self.assertEqual(len(engine.snapshot.trade_history), 1)

    def test_long_option_debit_mark_and_close_never_open_short_option(self):
        contract = _contract(InstrumentType.OPT, lot_size=10)
        proposal = _proposal(instrument=InstrumentType.OPT, contract=contract, entry=Decimal("5"))
        engine, order, fill = _open_and_fill(proposal=proposal, client_order_id="option")
        self.assertEqual(order.reserved_cash, Decimal("50"))
        self.assertEqual(fill.notional, Decimal("50"))
        self.assertEqual(engine.snapshot.cash, Decimal("99950"))
        self.assertEqual(engine.snapshot.equity, Decimal("100000"))
        low_mark_at = NOW + timedelta(minutes=2)
        low_mark = _observation(
            contract,
            low_mark_at,
            bid=Decimal("0.05"),
            ask=Decimal("0.05"),
            last=Decimal("0.05"),
        )
        engine.mark_position("pos-option", low_mark)
        self.assertEqual(engine.snapshot.unrealized_pnl, Decimal("-49.50"))
        self.assertGreaterEqual(engine.snapshot.unrealized_pnl, Decimal("-50"))
        at = NOW + timedelta(minutes=3)
        quote = _observation(contract, at, bid=Decimal("6"), ask=Decimal("6"))
        engine.submit_close_order(
            PaperCloseOrderRequest("option-close", "pos-option", 1, quote, at),
            user_approved=True,
        )
        engine.fill_order("option-close", fill_id="option-close-fill", quantity=1, quote=quote)
        self.assertEqual(engine.snapshot.positions, ())
        self.assertEqual(engine.snapshot.cash, Decimal("100010"))
        self.assertEqual(engine.snapshot.trade_history[0].gross_pnl, Decimal("10"))

    def test_partial_fills_update_quantity_average_entry_and_reservations(self):
        engine = _engine()
        proposal = _proposal(quantity=2)
        approved = _approve(engine, proposal)
        self.assertEqual(approved.reserved_risk, Decimal("400"))
        self.assertEqual(approved.reserved_exposure, Decimal("4000"))
        at1 = NOW + timedelta(minutes=1)
        first = _observation(proposal.contract, at1, bid=Decimal("99"), ask=Decimal("99"))
        engine.fill_order("open-1", fill_id="partial-1", quantity=1, quote=first)
        self.assertEqual(engine.snapshot.orders[0].status, PaperOrderStatus.PARTIALLY_FILLED)
        self.assertEqual(engine.snapshot.orders[0].reserved_risk, Decimal("200"))
        self.assertEqual(engine.snapshot.orders[0].reserved_exposure, Decimal("2000"))
        at2 = NOW + timedelta(minutes=2)
        second = _observation(proposal.contract, at2, bid=Decimal("100"), ask=Decimal("100"))
        engine.fill_order("open-1", fill_id="partial-2", quantity=1, quote=second)
        self.assertEqual(engine.snapshot.positions[0].quantity, 2)
        self.assertEqual(engine.snapshot.positions[0].average_entry_price, Decimal("99.5"))
        self.assertEqual(engine.snapshot.orders[0].status, PaperOrderStatus.FILLED)
        self.assertEqual(engine.snapshot.reserved_risk, Decimal("0"))
        self.assertEqual(engine.snapshot.reserved_exposure, Decimal("0"))

    def test_risk_reservations_across_pending_openings_share_policy_budget(self):
        engine = _engine()
        first = _proposal(quantity=40, contract=_contract(contract_id="future-a"))
        second = _proposal(quantity=15, contract=_contract(contract_id="future-b"))
        accepted = _approve(engine, first, client_order_id="a")
        rejected = _approve(engine, second, client_order_id="b")
        self.assertEqual(accepted.status, PaperOrderStatus.OPEN)
        self.assertEqual(accepted.reserved_risk, Decimal("8000"))
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)
        self.assertIn("shared per-trade risk reservation", rejected.rejection_reason)

    def test_cancel_releases_open_order_reservations(self):
        engine = _engine()
        approved = _approve(engine, _proposal(quantity=2))
        self.assertGreater(approved.reserved_risk, 0)
        cancelled = engine.cancel_order("open-1", cancelled_at=NOW + timedelta(minutes=1))
        self.assertEqual(cancelled.status, PaperOrderStatus.CANCELLED)
        self.assertEqual(engine.snapshot.reserved_risk, Decimal("0"))
        self.assertEqual(engine.snapshot.reserved_cash, Decimal("0"))

    def test_explicit_paper_rejection_preserves_partial_position_and_releases_reserves(self):
        engine = _engine()
        approved = _approve(engine, _proposal(quantity=2))
        at = NOW + timedelta(minutes=1)
        engine.fill_order(
            "open-1",
            fill_id="one-contract",
            quantity=1,
            quote=_observation(approved.contract, at),
        )
        rejected = engine.reject_order(
            "open-1",
            reason="paper venue rejected remaining quantity",
            rejected_at=NOW + timedelta(minutes=2),
        )
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)
        self.assertEqual(rejected.filled_quantity, 1)
        self.assertEqual(rejected.reserved_risk, Decimal("0"))
        self.assertEqual(engine.snapshot.positions[0].quantity, 1)
        self.assertEqual(engine.snapshot.reserved_risk, Decimal("0"))
        with self.assertRaises(OrderStateError):
            engine.fill_order(
                "open-1",
                fill_id="after-reject",
                quantity=1,
                quote=_observation(approved.contract, NOW + timedelta(minutes=3)),
            )

    def test_quote_provenance_cannot_change_between_approval_and_fill(self):
        engine = _engine()
        approved = _approve(engine, _proposal())
        wrong_provenance = _observation(
            approved.contract,
            NOW + timedelta(minutes=1),
            source="real-provider",
            synthetic=False,
            approved=True,
        )
        with self.assertRaises(InvalidPaperQuoteError):
            engine.fill_order("open-1", fill_id="wrong-provenance", quantity=1, quote=wrong_provenance)

    def test_non_synthetic_quotes_require_explicit_quote_approval(self):
        contract = _contract()
        real_response = _quote_response(contract, synthetic=False)
        proposal = _proposal(synthetic=False, source="reviewed-source", market_data=real_response)
        engine = _engine()
        rejected = _approve(engine, proposal, quote_approved=False)
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)
        self.assertIn("explicit approval", rejected.rejection_reason)

        engine2 = _engine()
        accepted = _approve(engine2, proposal, client_order_id="real-open", quote_approved=True)
        self.assertEqual(accepted.status, PaperOrderStatus.OPEN)
        unapproved_fill_quote = _observation(
            contract,
            NOW + timedelta(minutes=1),
            source="reviewed-source",
            synthetic=False,
            approved=False,
        )
        with self.assertRaises(InvalidPaperQuoteError):
            engine2.fill_order("real-open", fill_id="not-approved", quantity=1, quote=unapproved_fill_quote)

    def test_order_rejects_proposal_quote_without_executable_two_sided_market(self):
        contract = _contract()
        response = _quote_response(contract, bid=Decimal("100"), ask=None, last=Decimal("100"))
        proposal = _proposal(market_data=response)
        engine = _engine()
        rejected = _approve(engine, proposal)
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)
        self.assertIn("two-sided", rejected.rejection_reason)

    def test_invalid_quote_contract_and_stale_quotes_fail_closed(self):
        contract = _contract()
        other = _contract(contract_id="other-future")
        mismatched = _proposal(market_data=_quote_response(other))
        rejected = _approve(_engine(), mismatched)
        self.assertEqual(rejected.status, PaperOrderStatus.REJECTED)

        freshness = QuoteFreshnessPolicy(timedelta(minutes=1))
        stale_engine = _engine(execution=_execution(freshness=freshness))
        stale = _approve(stale_engine, _proposal())
        self.assertEqual(stale.status, PaperOrderStatus.OPEN)
        stale_fill = _observation(
            contract,
            NOW + timedelta(minutes=1),
            bid=Decimal("100"),
            ask=Decimal("100"),
            as_of=NOW + timedelta(minutes=3),
        )
        with self.assertRaises(InvalidPaperQuoteError):
            stale_engine.fill_order("open-1", fill_id="stale-fill", quantity=1, quote=stale_fill)

    def test_close_order_requires_approval_and_cannot_reverse_position(self):
        engine, _, _ = _open_and_fill()
        at = NOW + timedelta(minutes=2)
        quote = _observation(_contract(), at, bid=Decimal("100"), ask=Decimal("100"))
        pending = engine.submit_close_order(
            PaperCloseOrderRequest("close-pending", "pos-open-1", 1, quote, at),
            user_approved=None,
        )
        self.assertEqual(pending.status, PaperOrderStatus.PENDING_APPROVAL)
        with self.assertRaises(OrderStateError):
            engine.fill_order("close-pending", fill_id="pending-close-fill", quantity=1, quote=quote)
        accepted = engine.approve_close_order(
            "close-pending",
            user_approved=True,
            quote=quote,
            evaluated_at=at,
        )
        self.assertEqual(accepted.status, PaperOrderStatus.OPEN)
        engine.fill_order("close-pending", fill_id="accepted-close-fill", quantity=1, quote=quote)
        self.assertEqual(engine.snapshot.positions, ())

    def test_close_order_rejects_quote_from_after_its_submission_time(self):
        engine, _, _ = _open_and_fill()
        submitted_at = NOW + timedelta(minutes=2)
        future_quote = _observation(
            _contract(),
            NOW + timedelta(minutes=3),
            as_of=NOW + timedelta(minutes=3),
        )
        with self.assertRaises(PaperTradingError):
            engine.submit_close_order(
                PaperCloseOrderRequest(
                    "future-close",
                    "pos-open-1",
                    1,
                    future_quote,
                    submitted_at,
                ),
                user_approved=True,
            )

    def test_close_quantity_cannot_exceed_open_position(self):
        engine, _, _ = _open_and_fill()
        at = NOW + timedelta(minutes=2)
        quote = _observation(_contract(), at)
        with self.assertRaises(PaperTradingError):
            engine.submit_close_order(
                PaperCloseOrderRequest("too-large", "pos-open-1", 2, quote, at),
                user_approved=True,
            )

    def test_manual_partial_close_records_partial_trade_and_remaining_position(self):
        engine, _, _ = _open_and_fill(proposal=_proposal(quantity=2))
        at = NOW + timedelta(minutes=2)
        quote = _observation(_contract(), at, bid=Decimal("105"), ask=Decimal("105"))
        engine.submit_close_order(
            PaperCloseOrderRequest("partial-close", "pos-open-1", 1, quote, at),
            user_approved=True,
        )
        fill = engine.fill_order("partial-close", fill_id="partial-close-fill", quantity=1, quote=quote)
        self.assertEqual(fill.gross_realized_pnl, Decimal("100"))
        self.assertEqual(engine.snapshot.positions[0].quantity, 1)
        self.assertFalse(engine.snapshot.trade_history[0].position_fully_closed)

    def test_entry_fees_are_allocated_across_partial_closes_without_duplication(self):
        fees = (PaperFeeComponent("fixed", PaperFeeBasis.PER_FILL, Decimal("2")),)
        engine = _engine(execution=_execution(fees=fees))
        _, _, opening_fill = _open_and_fill(engine=engine, proposal=_proposal(quantity=2))
        self.assertEqual(opening_fill.fees, Decimal("2"))

        at1 = NOW + timedelta(minutes=2)
        quote1 = _observation(_contract(), at1, bid=Decimal("105"), ask=Decimal("105"))
        engine.submit_close_order(
            PaperCloseOrderRequest("fee-close-1", "pos-open-1", 1, quote1, at1),
            user_approved=True,
        )
        first = engine.fill_order(
            "fee-close-1", fill_id="fee-close-fill-1", quantity=1, quote=quote1
        )
        self.assertEqual(first.entry_fees_allocated, Decimal("1"))
        self.assertEqual(first.net_realized_pnl, Decimal("97"))

        at2 = NOW + timedelta(minutes=3)
        quote2 = _observation(_contract(), at2, bid=Decimal("110"), ask=Decimal("110"))
        engine.submit_close_order(
            PaperCloseOrderRequest("fee-close-2", "pos-open-1", 1, quote2, at2),
            user_approved=True,
        )
        second = engine.fill_order(
            "fee-close-2", fill_id="fee-close-fill-2", quantity=1, quote=quote2
        )
        self.assertEqual(second.entry_fees_allocated, Decimal("1"))
        self.assertEqual(second.net_realized_pnl, Decimal("197"))
        self.assertEqual(engine.snapshot.fees_paid, Decimal("6"))
        self.assertEqual(engine.snapshot.realized_pnl, Decimal("294"))
        self.assertEqual(engine.snapshot.gross_realized_pnl, Decimal("300"))
        self.assertEqual(engine.snapshot.positions, ())

    def test_target_gap_closes_conservatively_at_explicit_target(self):
        contract = _contract()
        engine, _, _ = _open_and_fill(
            proposal=_proposal(target=Decimal("110")),
        )
        at = NOW + timedelta(minutes=2)
        quote = _observation(contract, at, bid=Decimal("115"), ask=Decimal("115"))
        order = engine.check_protective_exit("pos-open-1", quote, client_order_id="target")
        self.assertEqual(order.action, PaperOrderAction.TARGET)
        fill = engine.fill_order("target", fill_id="target-fill", quantity=1, quote=quote)
        self.assertEqual(fill.fill_price, Decimal("110"))
        self.assertEqual(engine.snapshot.positions, ())

    def test_stop_gap_fills_at_market_quote_and_never_favorable_stop_price(self):
        engine, _, _ = _open_and_fill(proposal=_proposal(stop=Decimal("90")))
        at = NOW + timedelta(minutes=2)
        quote = _observation(_contract(), at, bid=Decimal("80"), ask=Decimal("80"))
        order = engine.check_protective_exit("pos-open-1", quote, client_order_id="stop")
        self.assertEqual(order.action, PaperOrderAction.STOP_LOSS)
        fill = engine.fill_order("stop", fill_id="stop-fill", quantity=1, quote=quote)
        self.assertEqual(fill.fill_price, Decimal("80"))
        self.assertEqual(fill.gross_realized_pnl, Decimal("-400"))

    def test_protective_stop_cancels_a_manual_close_and_closes_remaining_quantity(self):
        engine, _, _ = _open_and_fill(proposal=_proposal(quantity=2))
        manual_at = NOW + timedelta(minutes=2)
        manual_quote = _observation(_contract(), manual_at)
        manual = engine.submit_close_order(
            PaperCloseOrderRequest("manual-close", "pos-open-1", 1, manual_quote, manual_at),
            user_approved=True,
        )
        self.assertEqual(manual.status, PaperOrderStatus.OPEN)

        stop_at = NOW + timedelta(minutes=3)
        stop_quote = _observation(
            _contract(), stop_at, bid=Decimal("80"), ask=Decimal("80")
        )
        stop = engine.check_protective_exit(
            "pos-open-1", stop_quote, client_order_id="protective-stop"
        )
        self.assertEqual(engine.get_order("manual-close").status, PaperOrderStatus.CANCELLED)
        self.assertEqual(stop.requested_quantity, 2)
        engine.fill_order(
            "protective-stop",
            fill_id="protective-stop-fill",
            quantity=2,
            quote=stop_quote,
        )
        self.assertEqual(engine.snapshot.positions, ())

    def test_explicit_slippage_and_fee_components_change_cash_and_net_pnl(self):
        fees = (
            PaperFeeComponent("per-fill", PaperFeeBasis.PER_FILL, Decimal("1")),
            PaperFeeComponent("notional-rate", PaperFeeBasis.NOTIONAL_RATE, Decimal("0.001")),
        )
        engine = _engine(execution=_execution(slippage_bps=Decimal("100"), fees=fees))
        # Opening is at the approved ask and cannot worsen past the reference;
        # use zero spread and a limit-executable reference for that fill.
        order = engine.submit_open_order(
            PaperOpenOrderRequest("cost-open", _proposal(), NOW, quote_approved=False)
        )
        engine.approve_open_order("cost-open", user_approved=True, evaluated_at=NOW)
        open_quote = _observation(_contract(), NOW + timedelta(minutes=1), bid=Decimal("99"), ask=Decimal("99"))
        opening_fill = engine.fill_order(
            "cost-open", fill_id="cost-open-fill", quantity=1, quote=open_quote
        )
        self.assertEqual(opening_fill.fill_price, Decimal("100.00"))
        position = engine.snapshot.positions[0]
        # Close execution is a sell: 100 bps adverse slippage on 110 gives 108.90.
        close_at = NOW + timedelta(minutes=2)
        close_quote = _observation(_contract(), close_at, bid=Decimal("110"), ask=Decimal("110"))
        engine.submit_close_order(
            PaperCloseOrderRequest("cost-close", position.position_id, 1, close_quote, close_at),
            user_approved=True,
        )
        fill = engine.fill_order("cost-close", fill_id="cost-close-fill", quantity=1, quote=close_quote)
        self.assertEqual(fill.fill_price, Decimal("108.90"))
        self.assertEqual(tuple(charge.amount for charge in fill.fee_charges),
                         (Decimal("1"), Decimal("2.178")))
        self.assertEqual(fill.net_realized_pnl, Decimal("171.822"))
        self.assertEqual(engine.snapshot.fees_paid, Decimal("6.178"))

    def test_fill_and_order_idempotency_and_duplicate_keys(self):
        engine = _engine()
        proposal = _proposal()
        request = PaperOpenOrderRequest("same", proposal, NOW, False)
        first = engine.submit_open_order(request)
        self.assertIs(engine.submit_open_order(request), first)
        with self.assertRaises(DuplicateSubmissionError):
            engine.submit_open_order(PaperOpenOrderRequest("same", _proposal(quantity=2), NOW, False))
        engine.approve_open_order("same", user_approved=True, evaluated_at=NOW)
        quote = _observation(_contract(), NOW + timedelta(minutes=1))
        fill = engine.fill_order("same", fill_id="idempotent-fill", quantity=1, quote=quote)
        self.assertIs(engine.fill_order("same", fill_id="idempotent-fill", quantity=1, quote=quote), fill)
        self.assertEqual(engine.submit_open_order(request).status, PaperOrderStatus.FILLED)
        with self.assertRaises(DuplicateSubmissionError):
            engine.fill_order("same", fill_id="idempotent-fill", quantity=1,
                              quote=_observation(_contract(), NOW + timedelta(minutes=2)))

    def test_overfill_and_close_order_overreservation_are_rejected(self):
        engine = _engine()
        approved = _approve(engine, _proposal(quantity=2))
        quote = _observation(approved.contract, NOW + timedelta(minutes=1))
        with self.assertRaises(PaperTradingError):
            engine.fill_order("open-1", fill_id="too-much", quantity=3, quote=quote)
        engine.fill_order("open-1", fill_id="one", quantity=1, quote=quote)
        at = NOW + timedelta(minutes=2)
        close_quote = _observation(approved.contract, at)
        engine.submit_close_order(
            PaperCloseOrderRequest("close-a", "pos-open-1", 1, close_quote, at),
            user_approved=True,
        )
        with self.assertRaises(PaperTradingError):
            engine.submit_close_order(
                PaperCloseOrderRequest("close-b", "pos-open-1", 1, close_quote, at),
                user_approved=True,
            )

    def test_future_or_backwards_quote_events_are_rejected(self):
        engine = _engine()
        approved = _approve(engine, _proposal())
        future_asof = NOW + timedelta(minutes=1)
        future_quote = _observation(
            approved.contract,
            future_asof + timedelta(seconds=1),
            as_of=future_asof,
        )
        with self.assertRaises(InvalidPaperQuoteError):
            engine.fill_order("open-1", fill_id="future-quote", quantity=1, quote=future_quote)
        good = _observation(approved.contract, future_asof)
        engine.fill_order("open-1", fill_id="good", quantity=1, quote=good)
        backdated = _observation(approved.contract, NOW)
        with self.assertRaises(InvalidPaperQuoteError):
            engine.mark_position("pos-open-1", backdated)

    def test_manual_close_cannot_change_synthetic_status(self):
        engine, _, _ = _open_and_fill()
        at = NOW + timedelta(minutes=2)
        real = _observation(
            _contract(), at, source="real-source", synthetic=False, approved=True
        )
        with self.assertRaises(InvalidPaperQuoteError):
            engine.submit_close_order(
                PaperCloseOrderRequest("bad-close", "pos-open-1", 1, real, at),
                user_approved=True,
            )

    def test_long_close_requires_a_bid_even_when_ltp_exists(self):
        engine, _, _ = _open_and_fill()
        at = NOW + timedelta(minutes=2)
        no_bid = _observation(
            _contract(),
            at,
            bid=None,
            ask=Decimal("100"),
            last=Decimal("100"),
        )
        with self.assertRaises(InvalidPaperQuoteError):
            engine.submit_close_order(
                PaperCloseOrderRequest("no-bid-close", "pos-open-1", 1, no_bid, at),
                user_approved=True,
            )

    def test_snapshot_and_request_models_are_immutable(self):
        engine = _engine()
        proposal = _proposal()
        request = PaperOpenOrderRequest("immutable", proposal, NOW, False)
        with self.assertRaises(FrozenInstanceError):
            request.client_order_id = "changed"
        approved = _approve(engine, proposal, client_order_id="immutable")
        with self.assertRaises(FrozenInstanceError):
            engine.snapshot.cash = Decimal("1")
        self.assertIsNotNone(approved.proposal)
        self.assertEqual(proposal.quantity, 1)
        self.assertEqual(engine.snapshot.orders[0].proposal, proposal)

    def test_repeated_independent_runs_are_deterministic(self):
        first, _, first_fill = _open_and_fill()
        second, _, second_fill = _open_and_fill()
        self.assertEqual(first.snapshot, second.snapshot)
        self.assertEqual(first_fill, second_fill)

    def test_engine_exposes_no_broker_or_live_submission_path(self):
        engine = _engine()
        self.assertFalse(hasattr(engine, "submit_live_order"))
        self.assertFalse(hasattr(engine, "submit_broker_order"))
        self.assertFalse(hasattr(engine, "broker"))

    def test_no_trade_signal_never_becomes_an_order(self):
        signal_direction = SignalDirection.NO_TRADE
        proposal = _proposal(direction=signal_direction)
        with self.assertRaises(PaperTradingError):
            _engine().submit_open_order(PaperOpenOrderRequest("no-trade", proposal, NOW, False))

    def test_fee_rules_are_explicit_and_decimal(self):
        fees = (
            PaperFeeComponent("base", PaperFeeBasis.PER_FILL, Decimal("0.10")),
            PaperFeeComponent("tax", PaperFeeBasis.PRIOR_CHARGES_RATE, Decimal("0.18")),
        )
        engine, _, fill = _open_and_fill(
            engine=_engine(execution=_execution(fees=fees)),
        )
        self.assertEqual(fill.fees, Decimal("0.1180"))
        self.assertEqual(engine.snapshot.fees_paid, Decimal("0.1180"))


if __name__ == "__main__":
    unittest.main()
