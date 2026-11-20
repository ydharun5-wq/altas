"""Pure policy evaluation for a single proposed derivative trade."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from atlas.risk.calculations import (
    TradeRiskEstimate,
    calculate_combined_exposure,
    calculate_daily_pnl,
    calculate_drawdown,
    calculate_fractional_limit,
    calculate_remaining_daily_allowance,
    calculate_trade_risk,
)
from atlas.risk.models import (
    AccountRiskSnapshot,
    ProposedTrade,
    RiskCalculationError,
    RiskDecision,
    RiskDecisionStatus,
    RiskPolicy,
    RiskReason,
    RiskReasonCode,
)
from atlas.strategies.models import SignalDirection, SignalStatus


_NO_TRADE_CODES = frozenset(
    {
        RiskReasonCode.ACCOUNT_EQUITY_MISSING,
        RiskReasonCode.ACCOUNT_DAILY_BASELINE_MISSING,
        RiskReasonCode.ACCOUNT_PEAK_EQUITY_MISSING,
        RiskReasonCode.DAILY_PNL_MISSING,
        RiskReasonCode.OPEN_POSITION_COUNT_MISSING,
        RiskReasonCode.EXPOSURE_DATA_MISSING,
        RiskReasonCode.SNAPSHOT_TIMESTAMP_MISSING,
        RiskReasonCode.SIGNAL_TIMESTAMP_MISSING,
        RiskReasonCode.MARKET_DATA_MISSING,
        RiskReasonCode.MARKET_DATA_SOURCE_MISSING,
        RiskReasonCode.MARKET_PRICE_MISSING,
        RiskReasonCode.SIGNAL_NOT_READY,
        RiskReasonCode.INCOMPLETE_CANDLE,
        RiskReasonCode.NO_TRADE_SIGNAL,
        RiskReasonCode.SYNTHETIC_STATUS_MISSING,
        RiskReasonCode.ENTRY_REFERENCE_MISSING,
        RiskReasonCode.STOP_REFERENCE_MISSING,
        RiskReasonCode.QUANTITY_MISSING,
        RiskReasonCode.POINT_VALUE_MISSING,
        RiskReasonCode.USER_APPROVAL_REQUIRED,
    }
)


def evaluate_pre_trade_risk(
    policy: RiskPolicy,
    account: AccountRiskSnapshot,
    trade: ProposedTrade,
    *,
    evaluated_at: datetime | None,
    user_approved: bool | None,
) -> RiskDecision:
    """Return an auditable risk decision without changing account or trade state.

    ``APPROVED`` means the pre-trade risk checks passed under this policy. It
    never places an order or grants execution permission. Evaluation time and
    approval state are explicit so repeated calls with identical inputs are
    deterministic.
    """
    if not isinstance(policy, RiskPolicy):
        raise TypeError("policy must be a RiskPolicy")
    if not isinstance(account, AccountRiskSnapshot):
        raise TypeError("account must be an AccountRiskSnapshot")
    if not isinstance(trade, ProposedTrade):
        raise TypeError("trade must be a ProposedTrade")

    reasons: list[RiskReason] = []
    seen_codes: set[RiskReasonCode] = set()

    def add_reason(code: RiskReasonCode, explanation: str) -> None:
        if code not in seen_codes:
            reasons.append(RiskReason(code, explanation))
            seen_codes.add(code)

    valid_eval_time = _aware_datetime(evaluated_at)
    if not valid_eval_time:
        add_reason(
            RiskReasonCode.INVALID_TIMESTAMP,
            "Evaluation time must be a timezone-aware datetime.",
        )

    snapshot_at = account.snapshot_timestamp
    if snapshot_at is None:
        add_reason(
            RiskReasonCode.SNAPSHOT_TIMESTAMP_MISSING,
            "The account snapshot timestamp is required.",
        )
    elif not _aware_datetime(snapshot_at):
        add_reason(
            RiskReasonCode.INVALID_TIMESTAMP,
            "The account snapshot timestamp must be timezone-aware.",
        )
    elif valid_eval_time and snapshot_at > evaluated_at:
        add_reason(
            RiskReasonCode.FUTURE_ACCOUNT_SNAPSHOT,
            "The account snapshot cannot be later than the evaluation time.",
        )

    signal_at = trade.signal.observation_timestamp
    if signal_at is None:
        add_reason(
            RiskReasonCode.SIGNAL_TIMESTAMP_MISSING,
            "The strategy signal has no observation timestamp.",
        )
    elif not _aware_datetime(signal_at):
        add_reason(
            RiskReasonCode.INVALID_TIMESTAMP,
            "The strategy signal timestamp must be timezone-aware.",
        )
    elif valid_eval_time and signal_at > evaluated_at:
        add_reason(
            RiskReasonCode.SIGNAL_AFTER_EVALUATION,
            "The strategy signal cannot be later than the evaluation time.",
        )

    signal_evaluated_at = trade.signal.evaluation_at
    if not _aware_datetime(signal_evaluated_at):
        add_reason(
            RiskReasonCode.INVALID_TIMESTAMP,
            "The strategy evaluation timestamp must be timezone-aware.",
        )
    elif valid_eval_time and signal_evaluated_at > evaluated_at:
        add_reason(
            RiskReasonCode.SIGNAL_AFTER_EVALUATION,
            "The strategy evaluation cannot be later than the risk evaluation.",
        )

    market_data = trade.market_data
    if market_data is None:
        add_reason(
            RiskReasonCode.MARKET_DATA_MISSING,
            "A typed futures or options quote response is required for the proposed entry.",
        )
    else:
        quote = market_data.quote
        market_at = market_data.timestamp
        if quote.contract != trade.contract:
            add_reason(
                RiskReasonCode.QUOTE_CONTRACT_MISMATCH,
                "The market-data quote contract does not match the proposed contract.",
            )
        if (
            quote.bid is None
            and quote.ask is None
            and quote.last_traded_price is None
        ):
            add_reason(
                RiskReasonCode.MARKET_PRICE_MISSING,
                "The market-data quote contains no bid, ask, or last-traded price.",
            )
        if not _aware_datetime(market_at):
            add_reason(
                RiskReasonCode.INVALID_TIMESTAMP,
                "The market-data timestamp must be timezone-aware.",
            )
        elif valid_eval_time and market_at > evaluated_at:
            add_reason(
                RiskReasonCode.INVALID_TIMESTAMP,
                "Market data cannot be later than the risk evaluation time.",
            )

    if trade.market_data_source is None:
        add_reason(
            RiskReasonCode.MARKET_DATA_SOURCE_MISSING,
            "Market-data source identity is required for risk evaluation.",
        )

    synthetic_status = (
        trade.signal.is_synthetic
        if market_data is None
        else market_data.is_synthetic
    )
    if market_data is not None and trade.signal.is_synthetic is None:
        add_reason(
            RiskReasonCode.SYNTHETIC_STATUS_MISSING,
            "The strategy signal has no synthetic-data provenance status.",
        )
    elif (
        market_data is not None
        and trade.signal.is_synthetic is not None
        and trade.signal.is_synthetic is not market_data.is_synthetic
    ):
        add_reason(
            RiskReasonCode.PROVENANCE_MISMATCH,
            "Strategy-signal and quote synthetic-data provenance do not match.",
        )

    if trade.signal.status is not SignalStatus.READY:
        add_reason(
            RiskReasonCode.SIGNAL_NOT_READY,
            "Only a READY strategy signal can be evaluated as a trade proposal.",
        )
    if trade.signal.used_incomplete_candle:
        add_reason(
            RiskReasonCode.INCOMPLETE_CANDLE,
            "The strategy signal used an incomplete candle.",
        )
    if trade.signal.underlying is not trade.contract.underlying:
        add_reason(
            RiskReasonCode.UNDERLYING_MISMATCH,
            "The signal underlying does not match the derivative contract.",
        )
    if trade.contract.underlying not in policy.allowed_underlyings:
        add_reason(
            RiskReasonCode.UNDERLYING_NOT_ALLOWED,
            "The proposed underlying is not permitted by the risk policy.",
        )
    if trade.contract.instrument_type not in policy.allowed_instrument_types:
        add_reason(
            RiskReasonCode.INSTRUMENT_NOT_ALLOWED,
            "The proposed instrument type is not permitted by the risk policy.",
        )
    if trade.horizon not in policy.allowed_horizons:
        add_reason(
            RiskReasonCode.HORIZON_NOT_ALLOWED,
            "The proposed trading horizon is not permitted by the risk policy.",
        )

    if synthetic_status is None:
        add_reason(
            RiskReasonCode.SYNTHETIC_STATUS_MISSING,
            "Market-data synthetic status is required for risk evaluation.",
        )
    elif synthetic_status and not policy.allow_synthetic_data:
        add_reason(
            RiskReasonCode.SYNTHETIC_DATA_NOT_ALLOWED,
            "The risk policy does not permit synthetic market data.",
        )

    estimate: TradeRiskEstimate | None = None
    try:
        estimate = calculate_trade_risk(trade)
    except RiskCalculationError as exc:
        add_reason(exc.code, exc.explanation)

    risk_budget: Decimal | None = None
    current_equity = account.current_equity
    if current_equity is None:
        add_reason(
            RiskReasonCode.ACCOUNT_EQUITY_MISSING,
            "Current account equity is required to calculate the trade risk budget.",
        )
    else:
        if current_equity == 0:
            add_reason(
                RiskReasonCode.ACCOUNT_EQUITY_ZERO,
                "Current account equity is zero; no positive-risk trade is available.",
            )
        try:
            risk_budget = calculate_fractional_limit(
                current_equity,
                policy.max_risk_per_trade_fraction,
                "per-trade risk budget",
            )
        except RiskCalculationError as exc:
            add_reason(exc.code, exc.explanation)
    if estimate is not None and risk_budget is not None:
        if estimate.risk_amount > risk_budget:
            add_reason(
                RiskReasonCode.RISK_BUDGET_EXCEEDED,
                "Estimated trade risk exceeds the configured per-trade risk budget.",
            )

    remaining_daily_allowance: Decimal | None = None
    daily_limit: Decimal | None = None
    starting_daily_equity = account.starting_daily_equity
    if starting_daily_equity is None:
        add_reason(
            RiskReasonCode.ACCOUNT_DAILY_BASELINE_MISSING,
            "Starting daily equity is required to calculate the daily loss limit.",
        )
    elif starting_daily_equity == 0:
        add_reason(
            RiskReasonCode.ACCOUNT_DAILY_BASELINE_ZERO,
            "Starting daily equity must be positive to evaluate a daily loss limit.",
        )
    else:
        try:
            daily_limit = calculate_fractional_limit(
                starting_daily_equity,
                policy.max_daily_loss_fraction,
                "daily loss limit",
            )
        except RiskCalculationError as exc:
            add_reason(exc.code, exc.explanation)

    if account.daily_realised_pnl is None or account.daily_unrealised_pnl is None:
        add_reason(
            RiskReasonCode.DAILY_PNL_MISSING,
            "Both daily realised and unrealised P&L are required for the daily loss check.",
        )
        daily_pnl = None
    else:
        try:
            daily_pnl = calculate_daily_pnl(
                account.daily_realised_pnl, account.daily_unrealised_pnl
            )
        except RiskCalculationError as exc:
            add_reason(exc.code, exc.explanation)
            daily_pnl = None
    if daily_limit is not None and daily_pnl is not None:
        try:
            remaining_daily_allowance = calculate_remaining_daily_allowance(
                daily_limit, daily_pnl
            )
        except RiskCalculationError as exc:
            add_reason(exc.code, exc.explanation)
        if daily_pnl <= -daily_limit:
            add_reason(
                RiskReasonCode.DAILY_LOSS_LIMIT_REACHED,
                "Current daily realised plus unrealised loss has reached the configured limit.",
            )
        if estimate is not None and remaining_daily_allowance is not None:
            if estimate.risk_amount > remaining_daily_allowance:
                add_reason(
                    RiskReasonCode.DAILY_RISK_ALLOWANCE_EXCEEDED,
                    "Estimated trade risk exceeds the remaining daily loss allowance.",
                )

    peak_equity = account.peak_equity
    if peak_equity is None:
        add_reason(
            RiskReasonCode.ACCOUNT_PEAK_EQUITY_MISSING,
            "Peak account equity is required to evaluate maximum drawdown.",
        )
    elif peak_equity == 0:
        add_reason(
            RiskReasonCode.ACCOUNT_PEAK_EQUITY_ZERO,
            "Peak account equity must be positive to evaluate maximum drawdown.",
        )
    elif current_equity is not None:
        try:
            drawdown_limit = calculate_fractional_limit(
                peak_equity,
                policy.max_account_drawdown_fraction,
                "maximum account drawdown",
            )
            drawdown = calculate_drawdown(peak_equity, current_equity)
            if drawdown >= drawdown_limit:
                add_reason(
                    RiskReasonCode.MAXIMUM_DRAWDOWN_REACHED,
                    "Current account drawdown has reached the configured maximum.",
                )
        except RiskCalculationError as exc:
            add_reason(exc.code, exc.explanation)

    open_count = account.open_position_count
    if open_count is None:
        add_reason(
            RiskReasonCode.OPEN_POSITION_COUNT_MISSING,
            "Open-position count is required to evaluate the position limit.",
        )
    elif open_count + 1 > policy.max_open_positions:
        add_reason(
            RiskReasonCode.OPEN_POSITION_LIMIT_REACHED,
            "Adding the proposed trade would exceed the open-position limit.",
        )

    exposures = account.exposure_by_underlying
    if exposures is None:
        add_reason(
            RiskReasonCode.EXPOSURE_DATA_MISSING,
            "Explicit exposure for every supported underlying is required.",
        )
    elif estimate is not None:
        try:
            combined_exposure = calculate_combined_exposure(
                exposures[trade.contract.underlying], estimate.exposure_amount
            )
            if combined_exposure > policy.max_exposure_per_underlying:
                add_reason(
                    RiskReasonCode.UNDERLYING_EXPOSURE_LIMIT_EXCEEDED,
                    "Existing plus proposed exposure would exceed the underlying limit.",
                )
        except RiskCalculationError as exc:
            add_reason(exc.code, exc.explanation)

    if user_approved is not None and not isinstance(user_approved, bool):
        add_reason(
            RiskReasonCode.INVALID_APPROVAL_STATE,
            "user_approved must be True, False, or None.",
        )
        normalized_approval = None
    else:
        normalized_approval = user_approved
    if normalized_approval is False:
        add_reason(RiskReasonCode.USER_REJECTED, "The user rejected the proposed trade.")
    elif normalized_approval is None:
        add_reason(
            RiskReasonCode.USER_APPROVAL_REQUIRED,
            "Explicit user approval is required before this proposal can pass risk checks.",
        )

    reason_codes = {reason.code for reason in reasons}
    if not reasons:
        reasons.append(
            RiskReason(
                RiskReasonCode.RISK_CHECKS_PASSED,
                "The proposal passed configured pre-trade risk checks; no order was submitted.",
            )
        )
        status = RiskDecisionStatus.APPROVED
    elif RiskReasonCode.USER_REJECTED in reason_codes:
        status = RiskDecisionStatus.REJECTED
    elif reason_codes & _NO_TRADE_CODES:
        # Missing prerequisites or pending approval prevent complete clearance.
        # Keep all independently detected violations in ``reasons`` while
        # reporting the primary state as NO_TRADE.
        status = RiskDecisionStatus.NO_TRADE
    else:
        status = RiskDecisionStatus.REJECTED

    assumptions = trade.assumptions + (() if estimate is None else estimate.assumptions)
    return RiskDecision(
        status=status,
        reasons=tuple(reasons),
        estimated_risk_amount=None if estimate is None else estimate.risk_amount,
        estimated_exposure_amount=None if estimate is None else estimate.exposure_amount,
        risk_budget=risk_budget,
        remaining_daily_risk_allowance=remaining_daily_allowance,
        evaluated_at=evaluated_at if valid_eval_time else None,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        contract_id=trade.contract.contract_id,
        data_source=trade.market_data_source,
        signal_data_source=trade.signal.data_source,
        is_synthetic=synthetic_status,
        user_approval_required=policy.user_approval_required,
        user_approved=normalized_approval,
        assumptions=assumptions,
    )


def _aware_datetime(value: datetime | None) -> bool:
    if not isinstance(value, datetime):
        return False
    try:
        return value.utcoffset() is not None
    except (OverflowError, TypeError, ValueError):
        return False


__all__ = ["evaluate_pre_trade_risk"]
