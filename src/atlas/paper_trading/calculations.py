"""Finite Decimal calculations for the local paper execution ledger."""

from __future__ import annotations

from decimal import (
    Decimal,
    DecimalException,
    MAX_EMAX,
    MIN_EMIN,
    localcontext,
)

from atlas.derivatives.models import DerivativeContract, InstrumentType
from atlas.market_data.models import FuturesQuoteRequest, OptionsQuoteRequest
from atlas.market_data.provider import (
    MarketDataError,
    validate_futures_quote_response,
    validate_options_quote_response,
)
from atlas.risk.calculations import TradeRiskEstimate, calculate_trade_risk
from atlas.risk.models import ProposedTrade, RiskCalculationError
from atlas.paper_trading.models import (
    InvalidPaperQuoteError,
    PaperExecutionConfig,
    PaperFeeBasis,
    PaperFeeCharge,
    PaperFillSide,
    PaperPositionSide as PositionSide,
    PaperQuoteObservation,
)


_MAX_PRECISION = 10000


def calculate_risk_and_exposure_per_contract(
    proposal: ProposedTrade,
) -> TradeRiskEstimate:
    """Use M8.1's authoritative risk and exposure formulas for one contract."""
    from dataclasses import replace

    try:
        return calculate_trade_risk(replace(proposal, quantity=1))
    except RiskCalculationError:
        raise


def validate_quote_observation(
    contract: DerivativeContract,
    observation: PaperQuoteObservation,
    *,
    freshness_policy,
) -> PaperQuoteObservation:
    """Validate an explicit quote using the same validators as the provider boundary."""
    if not isinstance(contract, DerivativeContract):
        raise InvalidPaperQuoteError("contract must be a DerivativeContract")
    if not isinstance(observation, PaperQuoteObservation):
        raise InvalidPaperQuoteError("observation must be a PaperQuoteObservation")
    if observation.response.quote.contract != contract:
        raise InvalidPaperQuoteError("quote contract does not match the paper order")
    if not observation.is_synthetic and observation.approved is not True:
        raise InvalidPaperQuoteError("non-synthetic quote requires explicit approval")
    try:
        if contract.instrument_type is InstrumentType.FUT:
            request = FuturesQuoteRequest(contract, observation.as_of)
            validate_futures_quote_response(
                request,
                observation.response,
                freshness_policy=freshness_policy,
            )
        elif contract.instrument_type is InstrumentType.OPT:
            request = OptionsQuoteRequest(contract, observation.as_of)
            validate_options_quote_response(
                request,
                observation.response,
                freshness_policy=freshness_policy,
            )
        else:  # Defensive if the contract model expands in the future.
            raise InvalidPaperQuoteError("unsupported instrument type")
    except InvalidPaperQuoteError:
        raise
    except (MarketDataError, TypeError, ValueError) as exc:
        raise InvalidPaperQuoteError("quote failed contract, time, or freshness validation") from exc
    return observation


def executable_reference_price(
    observation: PaperQuoteObservation,
    side: PaperFillSide,
) -> Decimal:
    quote = observation.response.quote
    reference = quote.ask if side is PaperFillSide.BUY else quote.bid
    if reference is None or not isinstance(reference, Decimal) or reference <= 0:
        side_name = "ask" if side is PaperFillSide.BUY else "bid"
        raise InvalidPaperQuoteError(f"a positive executable {side_name} is required")
    return reference


def calculate_fill_price(
    reference_price: Decimal,
    side: PaperFillSide,
    slippage_bps: Decimal,
    tick_size: Decimal,
) -> Decimal:
    """Apply adverse bps slippage then round to the next adverse contract tick."""
    _require_positive("reference_price", reference_price)
    _require_positive("tick_size", tick_size)
    _require_nonnegative("slippage_bps", slippage_bps)
    if not isinstance(side, PaperFillSide):
        raise ValueError("side must be a PaperFillSide")
    if slippage_bps >= Decimal("10000"):
        raise ValueError("slippage_bps must be below 10000")

    bps_delta = slippage_bps if side is PaperFillSide.BUY else -slippage_bps
    factor = _finite_sum(
        "slippage factor",
        Decimal("1"),
        _finite_product("slippage bps factor", bps_delta, Decimal("0.0001")),
    )
    slipped = _finite_product("slipped reference price", reference_price, factor)
    rounded = round_price_to_tick(slipped, tick_size, side)
    if rounded <= 0:
        raise ValueError("slippage produces a non-positive executable fill")
    return rounded


def round_price_to_tick(
    price: Decimal,
    tick_size: Decimal,
    side: PaperFillSide,
) -> Decimal:
    """Round BUY upward and SELL downward to a valid, adverse tick."""
    _require_positive("price", price)
    _require_positive("tick_size", tick_size)
    if not isinstance(side, PaperFillSide):
        raise ValueError("side must be a PaperFillSide")
    price_num, price_den = price.as_integer_ratio()
    tick_num, tick_den = tick_size.as_integer_ratio()
    numerator = price_num * tick_den
    denominator = price_den * tick_num
    ticks, remainder = divmod(numerator, denominator)
    if side is PaperFillSide.BUY and remainder:
        ticks += 1
    if ticks <= 0:
        raise ValueError("price rounds to a non-positive tick")
    return _finite_product("rounded contract tick price", Decimal(ticks), tick_size)


def calculate_notional(
    contract: DerivativeContract,
    price: Decimal,
    quantity_contracts: int,
    *,
    point_value: Decimal | None,
) -> Decimal:
    """Return explicit account-currency notional used only as fee basis."""
    _require_positive("price", price)
    _require_positive_integer("quantity_contracts", quantity_contracts)
    units = Decimal(quantity_contracts * contract.lot_size)
    if contract.instrument_type is InstrumentType.FUT:
        if point_value is None:
            raise ValueError("futures point_value is required for notional")
        _require_positive("point_value", point_value)
        return _finite_product("futures notional", price, units, point_value)
    return _finite_product("option premium notional", price, units)


def calculate_fee_charges(
    execution: PaperExecutionConfig,
    notional: Decimal,
) -> tuple[PaperFeeCharge, ...]:
    """Calculate explicit per-fill fees in tuple order using Decimal arithmetic."""
    if not isinstance(execution, PaperExecutionConfig):
        raise ValueError("execution must be a PaperExecutionConfig")
    _require_nonnegative("notional", notional)
    charges: list[PaperFeeCharge] = []
    prior = Decimal(0)
    for component in execution.fee_components:
        if component.basis is PaperFeeBasis.PER_FILL:
            amount = component.amount
        elif component.basis is PaperFeeBasis.NOTIONAL_RATE:
            amount = _finite_product("notional rate fee", component.amount, notional)
        elif component.basis is PaperFeeBasis.PRIOR_CHARGES_RATE:
            amount = _finite_product("prior charges rate fee", component.amount, prior)
        else:
            raise ValueError("unsupported paper fee basis")
        if not amount.is_finite() or amount < 0:
            raise ValueError("fee rule produced a non-finite or negative amount")
        charges.append(PaperFeeCharge(component.name, amount))
        prior = _finite_sum("prior fee charges", prior, amount)
    return tuple(charges)


def total_fees(charges: tuple[PaperFeeCharge, ...]) -> Decimal:
    return _finite_sum("fee total", *(item.amount for item in charges))


def calculate_exposure(
    contract: DerivativeContract,
    entry_price: Decimal,
    quantity_contracts: int,
    *,
    point_value: Decimal | None,
) -> Decimal:
    return calculate_notional(
        contract,
        entry_price,
        quantity_contracts,
        point_value=point_value,
    )


def calculate_unrealized_pnl(
    contract: DerivativeContract,
    direction: PositionSide,
    quantity_contracts: int,
    average_entry_price: Decimal,
    mark_price: Decimal,
    *,
    point_value: Decimal | None,
) -> Decimal:
    _require_positive("average_entry_price", average_entry_price)
    _require_positive("mark_price", mark_price)
    _require_positive_integer("quantity_contracts", quantity_contracts)
    sign = Decimal(direction.sign)
    price_change = _finite_difference("unrealized price change", mark_price, average_entry_price)
    units = Decimal(quantity_contracts * contract.lot_size)
    if contract.instrument_type is InstrumentType.FUT:
        if point_value is None:
            raise ValueError("futures point_value is required")
        return _finite_product("futures unrealized P&L", sign, units, point_value, price_change)
    return _finite_product("long-option unrealized P&L", sign, units, price_change)


def allocate_proportional_amount(
    remaining_amount: Decimal,
    closed_quantity: int,
    remaining_quantity: int,
) -> tuple[Decimal, Decimal]:
    """Allocate fees across partial closes; the final close receives exact residual."""
    _require_nonnegative("remaining_amount", remaining_amount)
    _require_positive_integer("closed_quantity", closed_quantity)
    _require_positive_integer("remaining_quantity", remaining_quantity)
    if closed_quantity > remaining_quantity:
        raise ValueError("closed_quantity exceeds remaining_quantity")
    if closed_quantity == remaining_quantity:
        return remaining_amount, Decimal(0)
    try:
        with localcontext() as context:
            context.prec = _precision((remaining_amount,))
            context.Emax = MAX_EMAX
            context.Emin = MIN_EMIN
            allocation = remaining_amount * Decimal(closed_quantity) / Decimal(remaining_quantity)
    except DecimalException as exc:
        raise ValueError("fee allocation exceeds Decimal precision limits") from exc
    if not allocation.is_finite() or allocation < 0:
        raise ValueError("fee allocation is invalid")
    return allocation, _finite_difference("remaining fee balance", remaining_amount, allocation)


def finite_sum(name: str, *values: Decimal) -> Decimal:
    return _finite_sum(name, *values)


def finite_difference(name: str, left: Decimal, right: Decimal) -> Decimal:
    return _finite_difference(name, left, right)


def _finite_product(name: str, *values: Decimal) -> Decimal:
    if any(not isinstance(value, Decimal) or not value.is_finite() for value in values):
        raise ValueError(f"{name} operands must be finite Decimals")
    try:
        with localcontext() as context:
            context.prec = max(80, sum(len(value.as_tuple().digits) for value in values) + 2)
            if context.prec > _MAX_PRECISION:
                raise DecimalException("required precision exceeds safe limit")
            context.Emax = MAX_EMAX
            context.Emin = MIN_EMIN
            result = Decimal(1)
            for value in values:
                result *= value
    except DecimalException as exc:
        raise ValueError(f"{name} exceeds finite Decimal range") from exc
    if not result.is_finite():
        raise ValueError(f"{name} exceeds finite Decimal range")
    return result


def _finite_sum(name: str, *values: Decimal) -> Decimal:
    if any(not isinstance(value, Decimal) or not value.is_finite() for value in values):
        raise ValueError(f"{name} operands must be finite Decimals")
    if not values:
        return Decimal(0)
    try:
        with localcontext() as context:
            context.prec = _precision(values)
            context.Emax = MAX_EMAX
            context.Emin = MIN_EMIN
            result = sum(values, Decimal(0))
    except DecimalException as exc:
        raise ValueError(f"{name} exceeds finite Decimal range") from exc
    if not result.is_finite():
        raise ValueError(f"{name} exceeds finite Decimal range")
    return result


def _finite_difference(name: str, left: Decimal, right: Decimal) -> Decimal:
    if not isinstance(left, Decimal) or not left.is_finite():
        raise ValueError(f"{name} left operand must be finite")
    if not isinstance(right, Decimal) or not right.is_finite():
        raise ValueError(f"{name} right operand must be finite")
    try:
        with localcontext() as context:
            context.prec = _precision((left, right))
            context.Emax = MAX_EMAX
            context.Emin = MIN_EMIN
            result = left - right
    except DecimalException as exc:
        raise ValueError(f"{name} exceeds finite Decimal range") from exc
    if not result.is_finite():
        raise ValueError(f"{name} exceeds finite Decimal range")
    return result


def _precision(values: tuple[Decimal, ...]) -> int:
    least_exponent = min(value.as_tuple().exponent for value in values)
    greatest_adjusted = max(value.adjusted() for value in values)
    precision = max(80, greatest_adjusted - least_exponent + 2)
    if precision > _MAX_PRECISION:
        raise DecimalException("required precision exceeds safe limit")
    return precision


def _require_positive(name: str, value: Decimal) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ValueError(f"{name} must be a positive finite Decimal")


def _require_nonnegative(name: str, value: Decimal) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be a non-negative finite Decimal")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "allocate_proportional_amount",
    "calculate_exposure",
    "calculate_fee_charges",
    "calculate_fill_price",
    "calculate_notional",
    "calculate_risk_and_exposure_per_contract",
    "calculate_unrealized_pnl",
    "executable_reference_price",
    "finite_difference",
    "finite_product",
    "finite_sum",
    "round_price_to_tick",
    "total_fees",
    "validate_quote_observation",
]


finite_product = _finite_product
