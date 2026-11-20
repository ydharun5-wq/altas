"""Deterministic position sizing using the M8.1 per-trade risk rules."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from atlas.risk.calculations import (
    TradeRiskEstimate,
    calculate_fractional_limit,
    calculate_remaining_daily_allowance,
    calculate_trade_risk,
)
from atlas.risk.models import (
    AccountRiskSnapshot,
    ProposedTrade,
    RiskCalculationError,
    RiskPolicy,
    RiskReasonCode,
)
from atlas.sizing.models import (
    PositionSizingRequest,
    PositionSizingResult,
    SizingReason,
    SizingReasonCode,
    SizingStatus,
)


def size_position(request: PositionSizingRequest) -> PositionSizingResult:
    """Return the maximum risk-permitted whole-contract proposal.

    A caller-supplied ``proposal.quantity`` is an upper bound; when omitted,
    the maximum whole-contract quantity within the M8.1 per-trade risk budget
    is selected. A successful sizing result is not an approval or execution
    instruction. The resulting proposal must still pass
    :func:`atlas.risk.evaluate_pre_trade_risk`.

    Futures risk reuses the M8.1 absolute stop-distance, lot-size and explicit
    point-value formula. Long-option risk reuses M8.1 premium-at-risk. Fees,
    slippage, margin and affordability are not inferred here.
    """
    if not isinstance(request, PositionSizingRequest):
        raise TypeError("request must be a PositionSizingRequest")

    policy = request.policy
    account = request.account
    proposal = request.proposal
    reasons: list[SizingReason] = []
    seen_codes: set[RiskReasonCode | SizingReasonCode] = set()

    def add_reason(code: RiskReasonCode | SizingReasonCode, explanation: str) -> None:
        if code not in seen_codes:
            reasons.append(SizingReason(code, explanation))
            seen_codes.add(code)

    risk_budget: Decimal | None = None
    if account.current_equity is None:
        add_reason(
            RiskReasonCode.ACCOUNT_EQUITY_MISSING,
            "Current account equity is required to calculate the risk budget.",
        )
    elif account.current_equity == 0:
        add_reason(
            RiskReasonCode.ACCOUNT_EQUITY_ZERO,
            "Current account equity is zero; no positive-risk contract can be sized.",
        )
    else:
        try:
            risk_budget = calculate_fractional_limit(
                account.current_equity,
                policy.max_risk_per_trade_fraction,
                "per-trade risk budget",
            )
        except RiskCalculationError as exc:
            add_reason(exc.code, exc.explanation)

    market_data = proposal.market_data
    if market_data is None:
        add_reason(
            RiskReasonCode.MARKET_DATA_MISSING,
            "A typed quote response is required before sizing a proposal.",
        )
    else:
        quote = market_data.quote
        if quote.contract != proposal.contract:
            add_reason(
                RiskReasonCode.QUOTE_CONTRACT_MISMATCH,
                "The quote contract does not match the proposed contract.",
            )
        market_prices = (quote.bid, quote.ask, quote.last_traded_price)
        if not any(price is not None and price > 0 for price in market_prices):
            add_reason(
                RiskReasonCode.MARKET_PRICE_MISSING,
                "The quote contains no positive bid, ask, or last-traded price.",
            )
        if proposal.signal.is_synthetic is None:
            add_reason(
                RiskReasonCode.SYNTHETIC_STATUS_MISSING,
                "Strategy-signal synthetic-data provenance is required.",
            )
        elif proposal.signal.is_synthetic is not market_data.is_synthetic:
            add_reason(
                RiskReasonCode.PROVENANCE_MISMATCH,
                "Signal and quote synthetic-data provenance do not match.",
            )
    if proposal.market_data_source is None:
        add_reason(
            RiskReasonCode.MARKET_DATA_SOURCE_MISSING,
            "Market-data source identity is required before sizing.",
        )

    estimate: TradeRiskEstimate | None = None
    try:
        # Use M8.1 as the single authority for instrument risk formulas and
        # stop/price validation; set one contract solely to derive unit risk.
        estimate = calculate_trade_risk(replace(proposal, quantity=1))
    except RiskCalculationError as exc:
        add_reason(exc.code, exc.explanation)

    risk_per_contract = estimate.risk_amount if estimate is not None else None
    if risk_per_contract is not None and risk_per_contract <= 0:
        add_reason(
            SizingReasonCode.NUMERIC_RESULT_INVALID,
            "Per-contract risk must be positive to calculate whole-contract capacity.",
        )

    assumptions: tuple[str, ...] = ()
    if estimate is not None:
        assumptions = estimate.assumptions + (
            "The proposal's explicit entry reference is used; quote prices are not substituted.",
            "Whole contracts are sized by rounding down to the per-trade risk budget.",
            "A sized proposal is not approval; the M8.1 evaluator remains the final risk gate.",
            "Margin, capital affordability, fees, slippage, and execution risk are not calculated.",
        )
        if "premium paid" in estimate.formula:
            assumptions += (
                "A long-option stop is not treated as a guaranteed maximum-loss limit; maximum risk is premium paid.",
            )

    # Missing quote/source/provenance, account equity, or a valid one-contract
    # risk calculation blocks a proposed quantity. Preserve any known budget
    # and unit risk for diagnostics, while leaving capacity unavailable.
    if reasons:
        return _no_trade_result(
            request,
            reasons,
            risk_budget=risk_budget,
            risk_per_contract=risk_per_contract,
            assumptions=assumptions,
        )

    assert risk_budget is not None and risk_per_contract is not None
    try:
        maximum_quantity = _whole_contract_capacity(risk_budget, risk_per_contract)
    except (ArithmeticError, OverflowError, ValueError):
        add_reason(
            SizingReasonCode.NUMERIC_RESULT_INVALID,
            "Whole-contract capacity could not be represented safely.",
        )
        return _no_trade_result(
            request,
            reasons,
            risk_budget=risk_budget,
            risk_per_contract=risk_per_contract,
            assumptions=assumptions,
        )

    if maximum_quantity == 0:
        add_reason(
            SizingReasonCode.NO_CONTRACT_WITHIN_RISK_BUDGET,
            "One contract exceeds the configured per-trade risk budget.",
        )
        return _no_trade_result(
            request,
            reasons,
            risk_budget=risk_budget,
            risk_per_contract=risk_per_contract,
            maximum_quantity=0,
            total_risk=Decimal(0),
            unused_budget=risk_budget,
            assumptions=assumptions,
        )

    requested_quantity = proposal.quantity
    selected_quantity = maximum_quantity
    if requested_quantity is not None:
        selected_quantity = min(requested_quantity, maximum_quantity)
        if requested_quantity > maximum_quantity:
            add_reason(
                SizingReasonCode.REQUESTED_QUANTITY_CAPPED,
                "Requested contract quantity was reduced to the risk-permitted maximum.",
            )

    try:
        total_risk = calculate_fractional_limit(
            risk_per_contract,
            Decimal(selected_quantity),
            "total estimated position risk",
        )
        # M8.1 exposes exact finite addition for remaining allowance. Adding
        # the negative total risk is the exact Decimal subtraction needed here.
        unused_budget = calculate_remaining_daily_allowance(
            risk_budget, total_risk.copy_negate()
        )
        sized_proposal = replace(proposal, quantity=selected_quantity)
    except RiskCalculationError as exc:
        add_reason(exc.code, exc.explanation)
        return _no_trade_result(
            request,
            reasons,
            risk_budget=risk_budget,
            risk_per_contract=risk_per_contract,
            maximum_quantity=maximum_quantity,
            assumptions=assumptions,
        )

    if total_risk > risk_budget or unused_budget < 0:
        add_reason(
            SizingReasonCode.NUMERIC_RESULT_INVALID,
            "Calculated position risk exceeds the permitted budget.",
        )
        return _no_trade_result(
            request,
            reasons,
            risk_budget=risk_budget,
            risk_per_contract=risk_per_contract,
            maximum_quantity=maximum_quantity,
            assumptions=assumptions,
        )

    return PositionSizingResult(
        status=SizingStatus.SIZED,
        contract=proposal.contract,
        policy_id=policy.policy_id,
        policy_version=policy.version,
        requested_quantity=requested_quantity,
        risk_budget=risk_budget,
        risk_per_contract=risk_per_contract,
        maximum_permitted_contract_quantity=maximum_quantity,
        selected_quantity=selected_quantity,
        total_estimated_risk=total_risk,
        unused_risk_budget=unused_budget,
        sized_proposal=sized_proposal,
        reasons=tuple(reasons),
        assumptions=assumptions,
    )


def _whole_contract_capacity(risk_budget: Decimal, risk_per_contract: Decimal) -> int:
    """Return floor(risk_budget / risk_per_contract) using exact integers."""
    if risk_budget < 0 or risk_per_contract <= 0:
        raise ValueError("risk budget must be non-negative and contract risk positive")
    budget_numerator, budget_denominator = risk_budget.as_integer_ratio()
    risk_numerator, risk_denominator = risk_per_contract.as_integer_ratio()
    return (budget_numerator * risk_denominator) // (
        budget_denominator * risk_numerator
    )


def _no_trade_result(
    request: PositionSizingRequest,
    reasons: list[SizingReason],
    *,
    risk_budget: Decimal | None,
    risk_per_contract: Decimal | None,
    maximum_quantity: int | None = None,
    total_risk: Decimal | None = None,
    unused_budget: Decimal | None = None,
    assumptions: tuple[str, ...] = (),
) -> PositionSizingResult:
    return PositionSizingResult(
        status=SizingStatus.NO_TRADE,
        contract=request.proposal.contract,
        policy_id=request.policy.policy_id,
        policy_version=request.policy.version,
        requested_quantity=request.proposal.quantity,
        risk_budget=risk_budget,
        risk_per_contract=risk_per_contract,
        maximum_permitted_contract_quantity=maximum_quantity,
        selected_quantity=0,
        total_estimated_risk=total_risk,
        unused_risk_budget=unused_budget,
        sized_proposal=None,
        reasons=tuple(reasons),
        assumptions=assumptions,
    )


__all__ = ["size_position"]
