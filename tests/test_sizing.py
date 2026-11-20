"""Deterministic position-sizing tests using synthetic fixtures only."""

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
from atlas.market_data.models import FuturesQuoteResponse, OptionsQuoteResponse
from atlas.risk import (
    AccountRiskSnapshot,
    ProposedTrade,
    RiskDecisionStatus,
    RiskPolicy,
    RiskReasonCode,
    TradeStructure,
    TradingHorizon,
    evaluate_pre_trade_risk,
)
from atlas.sizing import (
    PositionSizingRequest,
    SizingReasonCode,
    SizingStatus,
    size_position,
)
from atlas.strategies.models import (
    SignalDirection,
    SignalEvent,
    SignalStatus,
    StrategySignal,
    WarmupStatus,
)


UTC = timezone.utc
NOW = datetime(2026, 11, 19, 10, 0, tzinfo=UTC)


def _contract(instrument=InstrumentType.FUT, *, lot_size=25, tick_size=Decimal("0.05")):
    is_option = instrument is InstrumentType.OPT
    return DerivativeContract(
        exchange=Exchange.NSE,
        underlying=UnderlyingIndex.NIFTY,
        instrument_type=instrument,
        expiry=date(2026, 12, 31),
        contract_id=f"NIFTY-{instrument.value}-20261231",
        lot_size=lot_size,
        tick_size=tick_size,
        strike=Decimal("25000") if is_option else None,
        option_type=OptionType.CE if is_option else None,
    )


def _policy(*, risk_fraction=Decimal("0.10")):
    return RiskPolicy(
        policy_id="sizing-test",
        version="1",
        max_risk_per_trade_fraction=risk_fraction,
        max_daily_loss_fraction=Decimal("0.10"),
        max_account_drawdown_fraction=Decimal("0.20"),
        max_open_positions=3,
        max_exposure_per_underlying=Decimal("1000000"),
        allowed_underlyings=frozenset(UnderlyingIndex),
        allowed_instrument_types=frozenset(InstrumentType),
        allowed_horizons=frozenset(TradingHorizon),
        allow_synthetic_data=True,
        user_approval_required=True,
    )


def _account(*, equity=Decimal("100000")):
    return AccountRiskSnapshot(
        current_equity=equity,
        starting_daily_equity=equity,
        peak_equity=equity,
        daily_realised_pnl=Decimal("0"),
        daily_unrealised_pnl=Decimal("0"),
        open_position_count=0,
        exposure_by_underlying={
            underlying: Decimal("0") for underlying in UnderlyingIndex
        },
        snapshot_timestamp=NOW,
    )


def _signal(direction=SignalDirection.LONG, *, underlying=UnderlyingIndex.NIFTY):
    event = SignalEvent.ENTER_SHORT if direction is SignalDirection.SHORT else SignalEvent.ENTER_LONG
    no_trade = direction is SignalDirection.NO_TRADE
    return StrategySignal(
        strategy_name="sizing-fixture",
        strategy_version="1",
        underlying=underlying,
        input_symbol=underlying.value,
        timeframe=timedelta(minutes=5),
        observation_timestamp=NOW,
        direction=direction,
        events=() if no_trade else (event,),
        status=SignalStatus.NO_TRADE if no_trade else SignalStatus.READY,
        warmup_status=WarmupStatus.SUFFICIENT,
        indicator_values=(),
        entry_reference_price=None if no_trade else 100.0,
        stop_reference_price=None if no_trade else 98.0,
        entry_conditions=("synthetic test entry",),
        exit_conditions=("synthetic test exit",),
        stop_loss_method="explicit test stop",
        input_requirements=("synthetic quote",),
        reasons=("test only",),
        missing_data=(),
        invalid_conditions=(),
        data_source="synthetic-test-source",
        is_synthetic=True,
        used_incomplete_candle=False,
        evaluation_at=NOW,
    )


def _proposal(
    *,
    instrument=InstrumentType.FUT,
    direction=SignalDirection.LONG,
    contract=None,
    entry=Decimal("100"),
    stop=Decimal("98"),
    quantity=None,
    point_value=Decimal("50"),
    market_data_source="synthetic-test-quote",
    market_data="default",
    structure=TradeStructure.SINGLE_LEG,
):
    selected_contract = _contract(instrument) if contract is None else contract
    signal = _signal(direction, underlying=selected_contract.underlying)
    if market_data == "default":
        if instrument is InstrumentType.FUT:
            quote = FuturesQuote(
                contract=selected_contract,
                timestamp=NOW,
                bid=Decimal("99.95"),
                ask=Decimal("100.00"),
                last_traded_price=Decimal("100.00"),
                volume=100,
                open_interest=500,
                change_in_open_interest=0,
            )
            market_data = FuturesQuoteResponse(quote, is_synthetic=True)
        else:
            quote = OptionsQuote(
                contract=selected_contract,
                timestamp=NOW,
                bid=entry - selected_contract.tick_size,
                ask=entry,
                last_traded_price=entry,
                volume=100,
                open_interest=500,
                change_in_open_interest=0,
            )
            market_data = OptionsQuoteResponse(quote, is_synthetic=True)
    return ProposedTrade(
        contract=selected_contract,
        signal=signal,
        horizon=TradingHorizon.INTRADAY,
        structure=structure,
        market_data=market_data,
        market_data_source=market_data_source,
        entry_reference_price=entry,
        stop_loss_reference_price=stop,
        target_reference_price=None,
        quantity=quantity,
        point_value=point_value,
        assumptions=("synthetic unit-test input",),
    )


def _size(proposal=None, *, account=None, policy=None):
    return size_position(
        PositionSizingRequest(
            policy=_policy() if policy is None else policy,
            account=_account() if account is None else account,
            proposal=_proposal() if proposal is None else proposal,
        )
    )


def _reason_codes(result):
    return {reason.code for reason in result.reasons}


class PositionSizingTests(unittest.TestCase):
    def test_known_futures_sizing_uses_m81_formula(self):
        result = _size()

        self.assertEqual(result.status, SizingStatus.SIZED)
        self.assertEqual(result.risk_budget, Decimal("10000.00"))
        self.assertEqual(result.risk_per_contract, Decimal("2500"))
        self.assertEqual(result.maximum_permitted_contract_quantity, 4)
        self.assertEqual(result.selected_quantity, 4)
        self.assertEqual(result.total_estimated_risk, Decimal("10000"))
        self.assertEqual(result.unused_risk_budget, Decimal("0.00"))
        self.assertEqual(result.sized_proposal.quantity, 4)

    def test_long_and_short_futures_use_absolute_stop_distance(self):
        long_result = _size(_proposal(direction=SignalDirection.LONG, stop=Decimal("98")))
        short_result = _size(_proposal(direction=SignalDirection.SHORT, stop=Decimal("102")))

        self.assertEqual(long_result.risk_per_contract, Decimal("2500"))
        self.assertEqual(short_result.risk_per_contract, Decimal("2500"))

    def test_known_long_option_uses_premium_and_preserves_identity(self):
        contract = _contract(InstrumentType.OPT)
        proposal = _proposal(
            instrument=InstrumentType.OPT,
            contract=contract,
            entry=Decimal("12.50"),
            stop=None,
            point_value=None,
        )
        result = _size(proposal)

        self.assertEqual(result.status, SizingStatus.SIZED)
        self.assertEqual(result.risk_per_contract, Decimal("312.50"))
        self.assertEqual(result.maximum_permitted_contract_quantity, 32)
        self.assertEqual(result.total_estimated_risk, Decimal("10000.00"))
        self.assertEqual(result.contract, contract)
        self.assertEqual(result.sized_proposal.contract, contract)
        self.assertTrue(any("premium paid" in item for item in result.assumptions))

    def test_quantity_rounds_down_and_reports_unused_budget(self):
        proposal = _proposal(point_value=Decimal("60"))
        result = _size(proposal)

        self.assertEqual(result.risk_per_contract, Decimal("3000"))
        self.assertEqual(result.maximum_permitted_contract_quantity, 3)
        self.assertEqual(result.total_estimated_risk, Decimal("9000"))
        self.assertEqual(result.unused_risk_budget, Decimal("1000.00"))
        self.assertLessEqual(result.total_estimated_risk, result.risk_budget)

    def test_requested_quantity_is_capped_without_mutating_input(self):
        proposal = _proposal(quantity=9)
        result = _size(proposal)

        self.assertEqual(proposal.quantity, 9)
        self.assertEqual(result.selected_quantity, 4)
        self.assertIn(SizingReasonCode.REQUESTED_QUANTITY_CAPPED, _reason_codes(result))

    def test_one_contract_over_budget_returns_no_trade(self):
        result = _size(account=_account(equity=Decimal("1000")))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertEqual(result.maximum_permitted_contract_quantity, 0)
        self.assertEqual(result.selected_quantity, 0)
        self.assertIsNone(result.sized_proposal)
        self.assertIn(SizingReasonCode.NO_CONTRACT_WITHIN_RISK_BUDGET, _reason_codes(result))

    def test_missing_equity_fails_closed(self):
        result = _size(account=_account(equity=None))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIsNone(result.risk_budget)
        self.assertIn(RiskReasonCode.ACCOUNT_EQUITY_MISSING, _reason_codes(result))

    def test_zero_equity_fails_closed(self):
        result = _size(account=_account(equity=Decimal("0")))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.ACCOUNT_EQUITY_ZERO, _reason_codes(result))

    def test_missing_futures_point_value_fails_closed(self):
        result = _size(_proposal(point_value=None))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.POINT_VALUE_MISSING, _reason_codes(result))

    def test_missing_market_quote_fails_closed(self):
        result = _size(_proposal(market_data=None))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.MARKET_DATA_MISSING, _reason_codes(result))

    def test_quote_without_any_market_price_fails_closed(self):
        contract = _contract()
        quote = FuturesQuote(
            contract=contract,
            timestamp=NOW,
            bid=None,
            ask=None,
            last_traded_price=None,
            volume=0,
            open_interest=0,
            change_in_open_interest=0,
        )
        result = _size(
            _proposal(market_data=FuturesQuoteResponse(quote, is_synthetic=True))
        )

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.MARKET_PRICE_MISSING, _reason_codes(result))

    def test_zero_only_quote_prices_are_not_usable_market_data(self):
        contract = _contract()
        quote = FuturesQuote(
            contract=contract,
            timestamp=NOW,
            bid=Decimal("0"),
            ask=Decimal("0"),
            last_traded_price=Decimal("0"),
            volume=0,
            open_interest=0,
            change_in_open_interest=0,
        )
        result = _size(
            _proposal(market_data=FuturesQuoteResponse(quote, is_synthetic=True))
        )

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.MARKET_PRICE_MISSING, _reason_codes(result))

    def test_missing_market_data_source_fails_closed(self):
        result = _size(_proposal(market_data_source=None))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.MARKET_DATA_SOURCE_MISSING, _reason_codes(result))

    def test_invalid_stop_placement_fails_closed(self):
        result = _size(_proposal(direction=SignalDirection.LONG, stop=Decimal("101")))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.INVALID_STOP_PLACEMENT, _reason_codes(result))

    def test_entry_price_off_contract_tick_fails_closed(self):
        result = _size(_proposal(entry=Decimal("100.01")))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.INVALID_PRICE_REFERENCE, _reason_codes(result))

    def test_entry_reference_missing_fails_closed(self):
        result = _size(_proposal(entry=None))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.ENTRY_REFERENCE_MISSING, _reason_codes(result))

    def test_invalid_lot_size_is_rejected_by_contract_model(self):
        with self.assertRaises(ValueError):
            _contract(lot_size=0)

    def test_non_positive_financial_inputs_are_rejected_or_fail_closed(self):
        with self.assertRaises(ValueError):
            _proposal(entry=Decimal("0"))
        with self.assertRaises(ValueError):
            _proposal(point_value=Decimal("-1"))
        with self.assertRaises(ValueError):
            _proposal(quantity=Decimal("1.5"))
        with self.assertRaises(ValueError):
            _account(equity=Decimal("-1"))
        result = _size(account=_account(equity=Decimal("0")))
        self.assertEqual(result.status, SizingStatus.NO_TRADE)

    def test_non_finite_financial_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            _proposal(entry=Decimal("NaN"))
        with self.assertRaises(ValueError):
            _proposal(point_value=Decimal("Infinity"))
        with self.assertRaises(ValueError):
            _account(equity=Decimal("Infinity"))

    def test_short_options_are_unsupported(self):
        contract = _contract(InstrumentType.OPT)
        result = _size(
            _proposal(
                instrument=InstrumentType.OPT,
                direction=SignalDirection.SHORT,
                contract=contract,
                entry=Decimal("12.50"),
                stop=Decimal("15.00"),
                point_value=None,
            )
        )

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.SHORT_OPTION_UNSUPPORTED, _reason_codes(result))

    def test_multi_leg_proposals_are_unsupported(self):
        result = _size(_proposal(structure=TradeStructure.MULTI_LEG))

        self.assertEqual(result.status, SizingStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.MULTI_LEG_UNSUPPORTED, _reason_codes(result))

    def test_decimal_precision_is_preserved_for_budget_and_unused_amount(self):
        contract = _contract(lot_size=1, tick_size=Decimal("0.01"))
        proposal = _proposal(
            contract=contract,
            entry=Decimal("100.00"),
            stop=Decimal("99.95"),
            point_value=Decimal("0.6"),
        )
        result = _size(
            proposal,
            account=_account(equity=Decimal("1")),
            policy=_policy(risk_fraction=Decimal("0.1")),
        )

        self.assertEqual(result.risk_per_contract, Decimal("0.0300"))
        self.assertEqual(result.risk_budget, Decimal("0.1"))
        self.assertEqual(result.selected_quantity, 3)
        self.assertEqual(result.total_estimated_risk, Decimal("0.090000"))
        self.assertEqual(result.unused_risk_budget, Decimal("0.010000"))

    def test_frozen_inputs_and_outputs_and_input_proposal_are_unchanged(self):
        proposal = _proposal(quantity=2)
        request = PositionSizingRequest(_policy(), _account(), proposal)
        result = size_position(request)

        self.assertEqual(proposal.quantity, 2)
        self.assertEqual(result.sized_proposal.quantity, 2)
        with self.assertRaises(FrozenInstanceError):
            request.proposal = _proposal()
        with self.assertRaises(FrozenInstanceError):
            result.selected_quantity = 3
        with self.assertRaises(FrozenInstanceError):
            result.sized_proposal.quantity = 3

    def test_repeated_calculations_are_deterministic(self):
        request = PositionSizingRequest(_policy(), _account(), _proposal())

        self.assertEqual(size_position(request), size_position(request))

    def test_sized_proposal_remains_subject_to_m81_evaluator(self):
        result = _size(_proposal())
        decision = evaluate_pre_trade_risk(
            _policy(),
            _account(),
            result.sized_proposal,
            evaluated_at=NOW,
            user_approved=True,
        )

        self.assertEqual(decision.status, RiskDecisionStatus.APPROVED)
        self.assertEqual(decision.estimated_risk_amount, result.risk_budget)
        self.assertLessEqual(decision.estimated_risk_amount, result.risk_budget)


if __name__ == "__main__":
    unittest.main()
