"""Deterministic pre-trade risk evaluation tests using synthetic inputs."""

from dataclasses import FrozenInstanceError, replace
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
from atlas.strategies.models import (
    SignalDirection,
    SignalEvent,
    SignalStatus,
    StrategySignal,
    WarmupStatus,
)


UTC = timezone.utc
NOW = datetime(2026, 11, 19, 10, 0, tzinfo=UTC)
_UNSET = object()


def _contract(instrument=InstrumentType.FUT, underlying=UnderlyingIndex.NIFTY):
    is_option = instrument is InstrumentType.OPT
    return DerivativeContract(
        exchange=Exchange.BSE if underlying is UnderlyingIndex.SENSEX else Exchange.NSE,
        underlying=underlying,
        instrument_type=instrument,
        expiry=date(2026, 12, 31),
        contract_id=f"{underlying.value}-{instrument.value}-20261231",
        lot_size=25,
        tick_size=Decimal("0.05"),
        strike=Decimal("25000") if is_option else None,
        option_type=OptionType.CE if is_option else None,
    )


def _signal(
    *,
    underlying=UnderlyingIndex.NIFTY,
    direction=SignalDirection.LONG,
    observation_at=NOW,
    evaluated_at=NOW,
    synthetic=False,
    status=SignalStatus.READY,
    incomplete=False,
):
    event = (
        SignalEvent.ENTER_SHORT
        if direction is SignalDirection.SHORT
        else SignalEvent.ENTER_LONG
    )
    no_trade = direction is SignalDirection.NO_TRADE
    return StrategySignal(
        strategy_name="fixture",
        strategy_version="1",
        underlying=underlying,
        input_symbol=underlying.value,
        timeframe=timedelta(minutes=5),
        observation_timestamp=observation_at,
        direction=direction,
        events=() if no_trade else (event,),
        status=SignalStatus.NO_TRADE if no_trade else status,
        warmup_status=WarmupStatus.SUFFICIENT,
        indicator_values=(),
        entry_reference_price=None if no_trade else 100.0,
        stop_reference_price=None if no_trade else 98.0,
        entry_conditions=("fixture entry",),
        exit_conditions=("fixture exit",),
        stop_loss_method="explicit fixture stop",
        input_requirements=("synthetic unit data",),
        reasons=("test proposal",),
        missing_data=(),
        invalid_conditions=(),
        data_source="synthetic-unit-fixture" if synthetic else "unit-fixture",
        is_synthetic=synthetic,
        used_incomplete_candle=incomplete,
        evaluation_at=evaluated_at,
    )


def _policy(**overrides):
    values = {
        "policy_id": "unit-policy",
        "version": "1",
        "max_risk_per_trade_fraction": Decimal("0.10"),
        "max_daily_loss_fraction": Decimal("0.10"),
        "max_account_drawdown_fraction": Decimal("0.20"),
        "max_open_positions": 3,
        "max_exposure_per_underlying": Decimal("1000000"),
        "allowed_underlyings": frozenset(UnderlyingIndex),
        "allowed_instrument_types": frozenset(InstrumentType),
        "allowed_horizons": frozenset(TradingHorizon),
        "allow_synthetic_data": True,
        "user_approval_required": True,
    }
    values.update(overrides)
    return RiskPolicy(**values)


def _snapshot(**overrides):
    values = {
        "current_equity": Decimal("100000"),
        "starting_daily_equity": Decimal("100000"),
        "peak_equity": Decimal("100000"),
        "daily_realised_pnl": Decimal("0"),
        "daily_unrealised_pnl": Decimal("0"),
        "open_position_count": 0,
        "exposure_by_underlying": {
            underlying: Decimal("0") for underlying in UnderlyingIndex
        },
        "snapshot_timestamp": NOW,
    }
    values.update(overrides)
    return AccountRiskSnapshot(**values)


def _trade(
    *,
    contract=None,
    direction=SignalDirection.LONG,
    signal=None,
    horizon=TradingHorizon.INTRADAY,
    structure=TradeStructure.SINGLE_LEG,
    market_data_source="unit-quote-fixture",
    entry=Decimal("100"),
    stop=Decimal("98"),
    target=None,
    quantity=2,
    point_value=Decimal("50"),
    market_data=_UNSET,
    assumptions=("point value supplied by test fixture",),
):
    selected_contract = _contract() if contract is None else contract
    if signal is None:
        signal = _signal(underlying=selected_contract.underlying, direction=direction)
    if market_data is _UNSET:
        synthetic = bool(signal.is_synthetic)
        if selected_contract.instrument_type is InstrumentType.FUT:
            quote = FuturesQuote(
                contract=selected_contract,
                timestamp=NOW,
                bid=Decimal("99.95"),
                ask=Decimal("100.00"),
                last_traded_price=entry,
                volume=100,
                open_interest=1000,
                change_in_open_interest=0,
            )
            market_data = FuturesQuoteResponse(quote, is_synthetic=synthetic)
        else:
            quote = OptionsQuote(
                contract=selected_contract,
                timestamp=NOW,
                bid=max(Decimal("0"), entry - selected_contract.tick_size),
                ask=entry,
                last_traded_price=entry,
                volume=100,
                open_interest=1000,
                change_in_open_interest=0,
            )
            market_data = OptionsQuoteResponse(quote, is_synthetic=synthetic)
    return ProposedTrade(
        contract=selected_contract,
        signal=signal,
        horizon=horizon,
        structure=structure,
        market_data=market_data,
        market_data_source=market_data_source,
        entry_reference_price=entry,
        stop_loss_reference_price=stop,
        target_reference_price=target,
        quantity=quantity,
        point_value=point_value,
        assumptions=assumptions,
    )


def _evaluate(policy=None, account=None, trade=None, *, evaluated_at=NOW, user_approved=True):
    return evaluate_pre_trade_risk(
        _policy() if policy is None else policy,
        _snapshot() if account is None else account,
        _trade() if trade is None else trade,
        evaluated_at=evaluated_at,
        user_approved=user_approved,
    )


def _codes(decision):
    return {reason.code for reason in decision.reasons}


class PreTradeRiskCalculationTests(unittest.TestCase):
    def test_valid_futures_risk_uses_hand_calculated_stop_formula(self):
        decision = _evaluate()

        self.assertEqual(decision.status, RiskDecisionStatus.APPROVED)
        self.assertEqual(decision.estimated_risk_amount, Decimal("5000"))
        self.assertEqual(decision.estimated_exposure_amount, Decimal("250000"))
        self.assertEqual(decision.risk_budget, Decimal("10000.00"))
        self.assertEqual(decision.remaining_daily_risk_allowance, Decimal("10000.00"))
        self.assertIn("abs(entry - stop)", decision.assumptions[-4])
        self.assertEqual(decision.data_source, "unit-quote-fixture")
        self.assertEqual(decision.signal_data_source, "unit-fixture")

    def test_valid_short_future_risk_uses_the_same_absolute_stop_distance(self):
        trade = _trade(
            direction=SignalDirection.SHORT,
            signal=replace(
                _signal(direction=SignalDirection.SHORT), stop_reference_price=102.0
            ),
            stop=Decimal("102"),
        )

        decision = _evaluate(trade=trade)

        self.assertEqual(decision.status, RiskDecisionStatus.APPROVED)
        self.assertEqual(decision.estimated_risk_amount, Decimal("5000"))

    def test_valid_long_option_uses_premium_paid_as_maximum_loss(self):
        trade = _trade(
            contract=_contract(InstrumentType.OPT),
            entry=Decimal("12.50"),
            stop=Decimal("10.00"),
            point_value=None,
        )

        decision = _evaluate(trade=trade)

        self.assertEqual(decision.status, RiskDecisionStatus.APPROVED)
        self.assertEqual(decision.estimated_risk_amount, Decimal("625.00"))
        self.assertEqual(decision.estimated_exposure_amount, Decimal("625.00"))
        self.assertIn("premium paid", decision.assumptions[-3])

    def test_short_option_is_rejected_as_unsupported_even_with_a_stop(self):
        trade = _trade(
            contract=_contract(InstrumentType.OPT),
            direction=SignalDirection.SHORT,
            entry=Decimal("12.50"),
            stop=Decimal("15.00"),
            point_value=None,
        )

        decision = _evaluate(trade=trade)

        self.assertEqual(decision.status, RiskDecisionStatus.REJECTED)
        self.assertIn(RiskReasonCode.SHORT_OPTION_UNSUPPORTED, _codes(decision))
        self.assertIsNone(decision.estimated_risk_amount)

    def test_future_risk_over_budget_is_rejected(self):
        trade = _trade(stop=Decimal("80"))

        decision = _evaluate(trade=trade)

        self.assertEqual(decision.estimated_risk_amount, Decimal("50000"))
        self.assertIn(RiskReasonCode.RISK_BUDGET_EXCEEDED, _codes(decision))

    def test_daily_loss_limit_reached_uses_realised_plus_unrealised_once(self):
        account = _snapshot(
            current_equity=Decimal("95000"),
            daily_realised_pnl=Decimal("-4000"),
            daily_unrealised_pnl=Decimal("-1000"),
        )
        policy = _policy(max_daily_loss_fraction=Decimal("0.05"))

        decision = _evaluate(policy, account)

        self.assertIn(RiskReasonCode.DAILY_LOSS_LIMIT_REACHED, _codes(decision))
        self.assertEqual(decision.remaining_daily_risk_allowance, Decimal("0.00"))
        self.assertEqual(decision.risk_budget, Decimal("9500.00"))

    def test_maximum_account_drawdown_reached_is_rejected(self):
        account = _snapshot(
            current_equity=Decimal("80000"),
            peak_equity=Decimal("100000"),
            daily_realised_pnl=Decimal("-20000"),
        )

        decision = _evaluate(account=account)

        self.assertIn(RiskReasonCode.MAXIMUM_DRAWDOWN_REACHED, _codes(decision))

    def test_open_position_limit_is_checked_before_adding_trade(self):
        account = _snapshot(
            open_position_count=2,
            exposure_by_underlying={
                underlying: Decimal("10") if underlying is UnderlyingIndex.NIFTY else Decimal("0")
                for underlying in UnderlyingIndex
            },
        )

        decision = _evaluate(_policy(max_open_positions=2), account)

        self.assertIn(RiskReasonCode.OPEN_POSITION_LIMIT_REACHED, _codes(decision))

    def test_existing_plus_proposed_underlying_exposure_is_limited(self):
        account = _snapshot(
            open_position_count=1,
            exposure_by_underlying={
                underlying: Decimal("800000") if underlying is UnderlyingIndex.NIFTY else Decimal("0")
                for underlying in UnderlyingIndex
            }
        )

        decision = _evaluate(
            _policy(max_exposure_per_underlying=Decimal("1000000")), account
        )

        self.assertIn(RiskReasonCode.UNDERLYING_EXPOSURE_LIMIT_EXCEEDED, _codes(decision))

    def test_disallowed_instrument_and_underlying_are_both_reported(self):
        trade = _trade(contract=_contract(InstrumentType.OPT, UnderlyingIndex.SENSEX))
        policy = _policy(
            allowed_underlyings=frozenset({UnderlyingIndex.NIFTY}),
            allowed_instrument_types=frozenset({InstrumentType.FUT}),
        )

        decision = _evaluate(policy, trade=trade)

        self.assertIn(RiskReasonCode.UNDERLYING_NOT_ALLOWED, _codes(decision))
        self.assertIn(RiskReasonCode.INSTRUMENT_NOT_ALLOWED, _codes(decision))

    def test_invalid_stop_placement_is_rejected(self):
        decision = _evaluate(trade=_trade(stop=Decimal("105")))

        self.assertIn(RiskReasonCode.INVALID_STOP_PLACEMENT, _codes(decision))
        self.assertIsNone(decision.estimated_risk_amount)

    def test_missing_market_data_fails_closed(self):
        decision = _evaluate(trade=_trade(market_data=None))

        self.assertEqual(decision.status, RiskDecisionStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.MARKET_DATA_MISSING, _codes(decision))

    def test_missing_market_data_remains_no_trade_and_keeps_risk_violations(self):
        decision = _evaluate(trade=_trade(stop=Decimal("80"), market_data=None))

        self.assertEqual(decision.status, RiskDecisionStatus.NO_TRADE)
        self.assertTrue(
            {
                RiskReasonCode.MARKET_DATA_MISSING,
                RiskReasonCode.RISK_BUDGET_EXCEEDED,
                RiskReasonCode.DAILY_RISK_ALLOWANCE_EXCEEDED,
            }.issubset(_codes(decision))
        )

    def test_missing_market_data_source_fails_closed(self):
        decision = _evaluate(trade=_trade(market_data_source=None))

        self.assertEqual(decision.status, RiskDecisionStatus.NO_TRADE)
        self.assertIsNone(decision.data_source)
        self.assertIn(RiskReasonCode.MARKET_DATA_SOURCE_MISSING, _codes(decision))

    def test_quote_without_any_market_price_fails_closed(self):
        trade = _trade()
        empty_quote = FuturesQuote(
            contract=trade.contract,
            timestamp=NOW,
            bid=None,
            ask=None,
            last_traded_price=None,
            volume=0,
            open_interest=0,
            change_in_open_interest=0,
        )
        trade = replace(
            trade,
            market_data=FuturesQuoteResponse(empty_quote, is_synthetic=False),
        )

        decision = _evaluate(trade=trade)

        self.assertEqual(decision.status, RiskDecisionStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.MARKET_PRICE_MISSING, _codes(decision))

    def test_synthetic_data_policy_is_enforced(self):
        trade = _trade(signal=_signal(synthetic=True))

        decision = _evaluate(_policy(allow_synthetic_data=False), trade=trade)

        self.assertEqual(decision.status, RiskDecisionStatus.REJECTED)
        self.assertIn(RiskReasonCode.SYNTHETIC_DATA_NOT_ALLOWED, _codes(decision))
        self.assertTrue(decision.is_synthetic)

    def test_quote_and_signal_synthetic_provenance_must_match(self):
        trade = _trade(signal=_signal(synthetic=True))
        real_quote = FuturesQuoteResponse(trade.market_data.quote, is_synthetic=False)
        trade = replace(trade, market_data=real_quote)

        decision = _evaluate(trade=trade)

        self.assertEqual(decision.status, RiskDecisionStatus.REJECTED)
        self.assertIn(RiskReasonCode.PROVENANCE_MISMATCH, _codes(decision))

    def test_disallowed_trading_horizon_is_rejected(self):
        policy = _policy(allowed_horizons=frozenset({TradingHorizon.SWING}))

        decision = _evaluate(policy=policy, trade=_trade(horizon=TradingHorizon.INTRADAY))

        self.assertIn(RiskReasonCode.HORIZON_NOT_ALLOWED, _codes(decision))

    def test_naive_timestamps_fail_closed(self):
        naive = datetime(2026, 11, 19, 10, 0)
        trade = _trade(
            signal=_signal(observation_at=naive, evaluated_at=naive),
        )

        decision = _evaluate(trade=trade)

        self.assertEqual(decision.status, RiskDecisionStatus.REJECTED)
        self.assertIn(RiskReasonCode.INVALID_TIMESTAMP, _codes(decision))

    def test_missing_account_equity_fails_closed(self):
        decision = _evaluate(account=_snapshot(current_equity=None))

        self.assertEqual(decision.status, RiskDecisionStatus.NO_TRADE)
        self.assertIsNone(decision.risk_budget)
        self.assertIn(RiskReasonCode.ACCOUNT_EQUITY_MISSING, _codes(decision))

    def test_missing_point_value_fails_closed_without_inventing_one(self):
        decision = _evaluate(trade=_trade(point_value=None))

        self.assertEqual(decision.status, RiskDecisionStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.POINT_VALUE_MISSING, _codes(decision))
        self.assertIsNone(decision.estimated_risk_amount)

    def test_non_finite_financial_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            _policy(max_risk_per_trade_fraction=Decimal("NaN"))
        with self.assertRaises(ValueError):
            _snapshot(current_equity=Decimal("Infinity"))
        with self.assertRaises(ValueError):
            _trade(entry=Decimal("NaN"))

    def test_numeric_policy_limits_are_bounded_and_non_negative(self):
        invalid_policies = (
            {"max_risk_per_trade_fraction": Decimal("-0.01")},
            {"max_daily_loss_fraction": Decimal("1.01")},
            {"max_account_drawdown_fraction": Decimal("Infinity")},
            {"max_open_positions": -1},
            {"max_exposure_per_underlying": Decimal("-1")},
        )
        for values in invalid_policies:
            with self.subTest(values=values), self.assertRaises(ValueError):
                _policy(**values)

    def test_repeated_evaluations_are_deterministic(self):
        first = _evaluate()
        second = _evaluate()

        self.assertEqual(first, second)

    def test_all_simultaneous_violations_are_returned(self):
        policy = _policy(
            max_risk_per_trade_fraction=Decimal("0.01"),
            max_open_positions=0,
            max_exposure_per_underlying=Decimal("1"),
            allowed_underlyings=frozenset(),
            allowed_instrument_types=frozenset(),
            allow_synthetic_data=False,
        )
        trade = _trade(signal=_signal(synthetic=True))

        decision = _evaluate(policy, _snapshot(), trade, user_approved=True)
        codes = _codes(decision)

        self.assertEqual(decision.status, RiskDecisionStatus.REJECTED)
        self.assertTrue(
            {
                RiskReasonCode.RISK_BUDGET_EXCEEDED,
                RiskReasonCode.OPEN_POSITION_LIMIT_REACHED,
                RiskReasonCode.UNDERLYING_EXPOSURE_LIMIT_EXCEEDED,
                RiskReasonCode.UNDERLYING_NOT_ALLOWED,
                RiskReasonCode.INSTRUMENT_NOT_ALLOWED,
                RiskReasonCode.SYNTHETIC_DATA_NOT_ALLOWED,
            }.issubset(codes)
        )

    def test_user_approval_policy_is_explicit(self):
        required = _evaluate(_policy(), user_approved=None)
        approved = _evaluate(_policy(), user_approved=True)
        rejected = _evaluate(_policy(), user_approved=False)

        self.assertEqual(required.status, RiskDecisionStatus.NO_TRADE)
        self.assertIn(RiskReasonCode.USER_APPROVAL_REQUIRED, _codes(required))
        self.assertEqual(approved.status, RiskDecisionStatus.APPROVED)
        self.assertTrue(approved.user_approval_required)
        self.assertEqual(rejected.status, RiskDecisionStatus.REJECTED)
        self.assertIn(RiskReasonCode.USER_REJECTED, _codes(rejected))
        with self.assertRaisesRegex(ValueError, "must be True"):
            _policy(user_approval_required=False)

    def test_multi_leg_proposals_are_explicitly_unsupported(self):
        decision = _evaluate(trade=_trade(structure=TradeStructure.MULTI_LEG))

        self.assertEqual(decision.status, RiskDecisionStatus.REJECTED)
        self.assertIn(RiskReasonCode.MULTI_LEG_UNSUPPORTED, _codes(decision))

    def test_input_objects_are_immutable_and_mapping_is_defensively_copied(self):
        original_exposure = {
            underlying: Decimal("0") for underlying in UnderlyingIndex
        }
        account = _snapshot(exposure_by_underlying=original_exposure)
        decision = _evaluate(account=account)
        original_exposure[UnderlyingIndex.NIFTY] = Decimal("999999")

        self.assertEqual(
            account.exposure_by_underlying[UnderlyingIndex.NIFTY], Decimal("0")
        )
        self.assertEqual(decision.status, RiskDecisionStatus.APPROVED)
        with self.assertRaises(FrozenInstanceError):
            account.current_equity = Decimal("1")

    def test_daily_profit_attribution_does_not_get_added_to_equity_twice(self):
        account = _snapshot(
            current_equity=Decimal("99000"),
            daily_realised_pnl=Decimal("1000"),
            daily_unrealised_pnl=Decimal("-2000"),
        )

        decision = _evaluate(account=account)

        self.assertEqual(decision.risk_budget, Decimal("9900.00"))
        self.assertEqual(decision.remaining_daily_risk_allowance, Decimal("9000.00"))


if __name__ == "__main__":
    unittest.main()
