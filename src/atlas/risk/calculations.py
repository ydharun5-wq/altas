"""Pure Decimal calculations used by the pre-trade risk evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Decimal,
    DecimalException,
    MAX_EMAX,
    MIN_EMIN,
    localcontext,
)

from atlas.derivatives.models import InstrumentType
from atlas.strategies.models import SignalDirection, SignalStatus
from atlas.risk.models import (
    ProposedTrade,
    RiskCalculationError,
    RiskReasonCode,
)


@dataclass(frozen=True)
class TradeRiskEstimate:
    risk_amount: Decimal
    exposure_amount: Decimal
    formula: str
    assumptions: tuple[str, ...]


def calculate_trade_risk(trade: ProposedTrade) -> TradeRiskEstimate:
    """Calculate planned futures stop risk or long-option maximum premium loss.

    Futures exposure is entry notional multiplied by an explicitly supplied
    point value. Long option risk/exposure is premium paid times contracts and
    the contract lot size. No costs, margin, financing, or short-option payoff
    assumptions are introduced.
    """
    if not isinstance(trade, ProposedTrade):
        raise TypeError("trade must be a ProposedTrade")
    if trade.structure.value != "single_leg":
        raise RiskCalculationError(
            RiskReasonCode.MULTI_LEG_UNSUPPORTED,
            "Multi-leg and spread risk is unsupported by the current single-contract model.",
        )
    if trade.signal.direction is SignalDirection.NO_TRADE:
        raise RiskCalculationError(
            RiskReasonCode.NO_TRADE_SIGNAL, "The strategy signal does not propose a trade."
        )
    if trade.signal.status is not SignalStatus.READY:
        raise RiskCalculationError(
            RiskReasonCode.SIGNAL_NOT_READY,
            "The strategy signal is not in READY status.",
        )
    if trade.signal.underlying is not trade.contract.underlying:
        raise RiskCalculationError(
            RiskReasonCode.UNDERLYING_MISMATCH,
            "The signal underlying does not match the derivative contract.",
        )
    if trade.quantity is None:
        raise RiskCalculationError(
            RiskReasonCode.QUANTITY_MISSING,
            "Contract quantity is required to calculate risk.",
        )
    if trade.entry_reference_price is None:
        raise RiskCalculationError(
            RiskReasonCode.ENTRY_REFERENCE_MISSING,
            "An entry reference price is required to calculate risk.",
        )

    entry = trade.entry_reference_price
    _validate_tick_price(entry, trade.contract.tick_size, "entry")

    if trade.contract.instrument_type is InstrumentType.FUT:
        if trade.stop_loss_reference_price is None:
            raise RiskCalculationError(
                RiskReasonCode.STOP_REFERENCE_MISSING,
                "A futures stop-loss reference is required to calculate planned risk.",
            )
        if trade.point_value is None:
            raise RiskCalculationError(
                RiskReasonCode.POINT_VALUE_MISSING,
                "Futures point value must be supplied explicitly.",
            )
        stop = trade.stop_loss_reference_price
        _validate_tick_price(stop, trade.contract.tick_size, "stop")
        if not _stop_is_valid(trade.signal.direction, entry, stop):
            raise RiskCalculationError(
                RiskReasonCode.INVALID_STOP_PLACEMENT,
                "The stop-loss must be below a long entry or above a short entry.",
            )
        if trade.target_reference_price is not None:
            _validate_tick_price(trade.target_reference_price, trade.contract.tick_size, "target")
            if not _target_is_valid(
                trade.signal.direction, entry, trade.target_reference_price
            ):
                raise RiskCalculationError(
                    RiskReasonCode.INVALID_TARGET_PLACEMENT,
                    "The target must be above a long entry or below a short entry.",
                )
        distance = _finite_operation(
            "futures stop distance", lambda: abs(entry - stop), entry, stop
        )
        units = trade.quantity * trade.contract.lot_size
        risk = _finite_product(
            "futures stop risk", distance, Decimal(units), trade.point_value
        )
        exposure = _finite_product(
            "futures gross exposure", entry, Decimal(units), trade.point_value
        )
        return TradeRiskEstimate(
            risk_amount=risk,
            exposure_amount=exposure,
            formula="abs(entry - stop) * contract quantity * contract lot size * point value",
            assumptions=trade.assumptions
            + (
                "Formula: abs(entry - stop) * contract quantity * contract lot size * point value.",
                "Futures risk assumes the planned stop can be executed at its reference price.",
                "Futures gross exposure uses entry notional and the explicitly supplied point value.",
                "Fees, slippage, gaps, margin, financing, and taxes are excluded.",
            ),
        )

    if trade.signal.direction is SignalDirection.SHORT:
        raise RiskCalculationError(
            RiskReasonCode.SHORT_OPTION_UNSUPPORTED,
            "Short-option risk is unsupported; a stop does not cap theoretical loss.",
        )
    if trade.target_reference_price is not None:
        _validate_tick_price(trade.target_reference_price, trade.contract.tick_size, "target")
        if not _target_is_valid(
            SignalDirection.LONG, entry, trade.target_reference_price
        ):
            raise RiskCalculationError(
                RiskReasonCode.INVALID_TARGET_PLACEMENT,
                "A long-option target must be above the premium entry reference.",
            )
    if trade.stop_loss_reference_price is not None:
        _validate_tick_price(trade.stop_loss_reference_price, trade.contract.tick_size, "stop")
        if not _stop_is_valid(
            SignalDirection.LONG, entry, trade.stop_loss_reference_price
        ):
            raise RiskCalculationError(
                RiskReasonCode.INVALID_STOP_PLACEMENT,
                "A long-option stop reference must be below its premium entry.",
            )
    premium_risk = _finite_product(
        "long-option premium risk",
        entry,
        Decimal(trade.quantity * trade.contract.lot_size),
    )
    return TradeRiskEstimate(
        risk_amount=premium_risk,
        exposure_amount=premium_risk,
        formula="premium paid * contract quantity * contract lot size",
        assumptions=trade.assumptions
        + (
            "Formula: premium paid * contract quantity * contract lot size.",
            "Long-option maximum loss is limited to the premium paid.",
            "Long-option exposure is measured as premium deployed.",
            "Fees and execution costs are excluded.",
        ),
    )


def calculate_fractional_limit(amount: Decimal, fraction: Decimal, name: str) -> Decimal:
    """Return a deterministic, finite amount-times-fraction limit."""
    return _finite_product(name, amount, fraction)


def calculate_daily_pnl(realised: Decimal, unrealised: Decimal) -> Decimal:
    return _finite_operation(
        "daily P&L", lambda: realised + unrealised, realised, unrealised
    )


def calculate_remaining_daily_allowance(limit: Decimal, daily_pnl: Decimal) -> Decimal:
    """Loss room left is the daily loss limit offset by current day P&L."""
    return _finite_operation(
        "remaining daily risk allowance",
        lambda: limit + daily_pnl,
        limit,
        daily_pnl,
    )


def calculate_drawdown(peak_equity: Decimal, current_equity: Decimal) -> Decimal:
    return _finite_operation(
        "account drawdown",
        lambda: peak_equity - current_equity,
        peak_equity,
        current_equity,
    )


def calculate_combined_exposure(existing: Decimal, proposed: Decimal) -> Decimal:
    return _finite_operation(
        "combined underlying exposure", lambda: existing + proposed, existing, proposed
    )


def _validate_tick_price(price: Decimal, tick_size: Decimal, name: str) -> None:
    valid = price > 0 and _is_decimal_multiple(price, tick_size)
    if not valid:
        raise RiskCalculationError(
            RiskReasonCode.INVALID_PRICE_REFERENCE,
            f"The {name} price must be positive and an exact multiple of contract tick size.",
        )


def _stop_is_valid(direction: SignalDirection, entry: Decimal, stop: Decimal) -> bool:
    if direction is SignalDirection.LONG:
        return stop < entry
    if direction is SignalDirection.SHORT:
        return stop > entry
    return False


def _target_is_valid(direction: SignalDirection, entry: Decimal, target: Decimal) -> bool:
    if direction is SignalDirection.LONG:
        return target > entry
    if direction is SignalDirection.SHORT:
        return target < entry
    return False


def _finite_product(name: str, *values: Decimal) -> Decimal:
    try:
        with localcontext() as context:
            required_precision = max(
                80, sum(len(value.as_tuple().digits) for value in values) + 2
            )
            if required_precision > 10000:
                raise DecimalException("required precision exceeds safe limit")
            context.prec = required_precision
            context.Emax = MAX_EMAX
            context.Emin = MIN_EMIN
            result = Decimal(1)
            for value in values:
                result *= value
    except DecimalException as exc:
        raise RiskCalculationError(
            RiskReasonCode.NUMERIC_CALCULATION_FAILED,
            f"{name} could not be represented as a finite Decimal.",
        ) from exc
    if not result.is_finite():
        raise RiskCalculationError(
            RiskReasonCode.NUMERIC_CALCULATION_FAILED,
            f"{name} could not be represented as a finite Decimal.",
        )
    return result


def _finite_operation(name: str, operation, *operands: Decimal) -> Decimal:
    try:
        with localcontext() as context:
            context.prec = _exact_sum_precision(operands)
            context.Emax = MAX_EMAX
            context.Emin = MIN_EMIN
            result = operation()
    except DecimalException as exc:
        raise RiskCalculationError(
            RiskReasonCode.NUMERIC_CALCULATION_FAILED,
            f"{name} could not be represented as a finite Decimal.",
        ) from exc
    if not isinstance(result, Decimal) or not result.is_finite():
        raise RiskCalculationError(
            RiskReasonCode.NUMERIC_CALCULATION_FAILED,
            f"{name} could not be represented as a finite Decimal.",
        )
    return result


def _is_decimal_multiple(value: Decimal, increment: Decimal) -> bool:
    value_tuple = value.as_tuple()
    increment_tuple = increment.as_tuple()
    exponent = min(value_tuple.exponent, increment_tuple.exponent)
    value_scale = value_tuple.exponent - exponent
    increment_scale = increment_tuple.exponent - exponent
    if max(value_scale, increment_scale) > 10000:
        return False
    value_coefficient = int("".join(str(digit) for digit in value_tuple.digits) or "0")
    increment_coefficient = int(
        "".join(str(digit) for digit in increment_tuple.digits) or "0"
    )
    value_coefficient *= 10**value_scale
    increment_coefficient *= 10**increment_scale
    return value_coefficient % increment_coefficient == 0


def _exact_sum_precision(values: tuple[Decimal, ...]) -> int:
    least_exponent = min(value.as_tuple().exponent for value in values)
    greatest_adjusted_exponent = max(value.adjusted() for value in values)
    required = max(80, greatest_adjusted_exponent - least_exponent + 2)
    if required > 10000:
        raise DecimalException("required precision exceeds safe limit")
    return required


__all__ = [
    "TradeRiskEstimate",
    "calculate_combined_exposure",
    "calculate_daily_pnl",
    "calculate_drawdown",
    "calculate_fractional_limit",
    "calculate_remaining_daily_allowance",
    "calculate_trade_risk",
]
