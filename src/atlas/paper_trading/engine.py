"""In-memory paper-only order state machine; contains no transport or broker API."""

from __future__ import annotations

import logging
from dataclasses import replace
from decimal import Decimal

from atlas.derivatives.models import InstrumentType, UnderlyingIndex
from atlas.market_data.models import FuturesQuoteResponse, OptionsQuoteResponse
from atlas.risk import (
    AccountRiskSnapshot,
    ProposedTrade,
    RiskDecisionStatus,
    RiskReasonCode,
    TradeStructure,
    evaluate_pre_trade_risk,
)
from atlas.risk.calculations import calculate_fractional_limit
from atlas.sizing import PositionSizingRequest, SizingStatus, size_position
from atlas.strategies.models import SignalDirection
from atlas.paper_trading.calculations import (
    allocate_proportional_amount,
    calculate_exposure,
    calculate_fee_charges,
    calculate_fill_price,
    calculate_notional,
    calculate_risk_and_exposure_per_contract,
    calculate_unrealized_pnl,
    executable_reference_price,
    finite_difference,
    finite_sum,
    total_fees,
    validate_quote_observation,
)
from atlas.paper_trading.models import (
    DuplicateSubmissionError,
    InvalidPaperQuoteError,
    OrderStateError,
    PaperAccountConfig,
    PaperCloseOrderRequest,
    PaperExecutionConfig,
    PaperFeeCharge,
    PaperFill,
    PaperFillSide,
    PaperOpenOrderRequest,
    PaperOrder,
    PaperOrderAction,
    PaperOrderStatus,
    PaperPosition,
    PaperPositionSide as PositionSide,
    PaperPortfolioSnapshot,
    PaperQuoteObservation,
    PaperTradeHistory,
    PaperTradingError,
    validate_aware_datetime,
)


_ACTIVE_ORDER_STATUSES = frozenset(
    {PaperOrderStatus.OPEN, PaperOrderStatus.PARTIALLY_FILLED}
)
_LOGGER = logging.getLogger(__name__)


class PaperTradingEngine:
    """Deterministic, local execution simulator over supplied quote events.

    Open proposals are held pending until explicitly approved, sized again,
    and re-evaluated by M8.1 against a newly derived account snapshot. Fills
    require explicit whole-contract quantities and quote observations. No
    provider, socket, broker, credential, or order-submission interface exists.
    """

    def __init__(self, config: PaperAccountConfig) -> None:
        if not isinstance(config, PaperAccountConfig):
            raise TypeError("config must be a PaperAccountConfig")
        self._config = config
        self._cash = config.initial_cash
        self._gross_realized = Decimal(0)
        self._fees_paid = Decimal(0)
        self._positions: dict[str, PaperPosition] = {}
        self._orders: dict[str, PaperOrder] = {}
        self._order_terms: dict[str, tuple[object, ...]] = {}
        self._fills: dict[str, PaperFill] = {}
        self._fill_terms: dict[str, tuple[object, ...]] = {}
        self._trade_history: list[PaperTradeHistory] = []
        self._last_quote_timestamp: dict[str, object] = {}
        self._state_as_of = config.started_at
        self._peak_equity = config.initial_cash
        self._snapshot = self._build_snapshot(config.started_at)

    @property
    def snapshot(self) -> PaperPortfolioSnapshot:
        return self._snapshot

    def get_order(self, client_order_id: str) -> PaperOrder:
        try:
            return self._orders[client_order_id]
        except KeyError as exc:
            raise PaperTradingError("unknown paper order") from exc

    def get_fill(self, fill_id: str) -> PaperFill:
        try:
            return self._fills[fill_id]
        except KeyError as exc:
            raise PaperTradingError("unknown paper fill") from exc

    def submit_open_order(self, request: PaperOpenOrderRequest) -> PaperOrder:
        """Stage an opening proposal as non-executable pending user approval."""
        if not isinstance(request, PaperOpenOrderRequest):
            raise TypeError("request must be a PaperOpenOrderRequest")
        proposal = request.proposal
        terms = (
            PaperOrderAction.OPEN,
            proposal,
            request.quote_approved,
            request.submitted_at,
        )
        existing = self._orders.get(request.client_order_id)
        if existing is not None:
            if self._order_terms[request.client_order_id] == terms:
                return existing
            raise DuplicateSubmissionError("idempotency key was reused with different terms")
        if proposal.quantity is None:
            raise PaperTradingError("opening proposal must have an explicit sized quantity")
        if proposal.signal.direction is SignalDirection.NO_TRADE:
            raise PaperTradingError("NO_TRADE signals cannot become paper orders")
        if request.submitted_at < self._state_as_of:
            raise PaperTradingError("submission time precedes the current paper ledger")
        if proposal.signal.direction is SignalDirection.LONG:
            direction = PositionSide.LONG
        elif proposal.signal.direction is SignalDirection.SHORT:
            direction = PositionSide.SHORT
        else:
            raise PaperTradingError("unsupported signal direction")
        source = proposal.market_data_source
        synthetic = (
            None if proposal.market_data is None else proposal.market_data.is_synthetic
        )
        order = PaperOrder(
            client_order_id=request.client_order_id,
            action=PaperOrderAction.OPEN,
            contract=proposal.contract,
            direction=direction,
            requested_quantity=proposal.quantity,
            filled_quantity=0,
            status=PaperOrderStatus.PENDING_APPROVAL,
            submitted_at=request.submitted_at,
            updated_at=request.submitted_at,
            data_source=source,
            is_synthetic=synthetic,
            quote_approved=request.quote_approved,
            proposal=proposal,
            position_id=f"pos-{request.client_order_id}",
            risk_decision=None,
            risk_per_contract=None,
            exposure_per_contract=None,
            reserved_risk=Decimal(0),
            reserved_exposure=Decimal(0),
            reserved_cash=Decimal(0),
            limit_price=None,
            rejection_reason=None,
        )
        self._orders[request.client_order_id] = order
        self._order_terms[request.client_order_id] = terms
        self._state_as_of = request.submitted_at
        self._refresh_snapshot(request.submitted_at)
        _LOGGER.info(
            "paper order staged for approval: order_id=%s action=open",
            request.client_order_id,
        )
        return order

    def approve_open_order(
        self,
        client_order_id: str,
        *,
        user_approved: bool | None,
        evaluated_at,
    ) -> PaperOrder:
        """Approve only an exact, re-sized proposal with a fresh M8.1 pass."""
        order = self.get_order(client_order_id)
        if order.action is not PaperOrderAction.OPEN:
            raise OrderStateError("only opening orders use the M8.1 opening gate")
        if order.status is not PaperOrderStatus.PENDING_APPROVAL:
            raise OrderStateError("opening order is not pending approval")
        validate_aware_datetime("evaluated_at", evaluated_at)
        if evaluated_at < self._state_as_of or evaluated_at < order.submitted_at:
            raise PaperTradingError("approval time precedes the current ledger or submission")
        if user_approved is None:
            return order
        if not isinstance(user_approved, bool):
            raise ValueError("user_approved must be True, False, or None")
        if user_approved is False:
            return self._reject(order, "explicit user approval was denied", evaluated_at)

        proposal = order.proposal
        assert proposal is not None
        try:
            self._validate_proposal_quote(
                proposal,
                evaluated_at,
                quote_approved=order.quote_approved is True,
            )
        except (InvalidPaperQuoteError, ValueError) as exc:
            return self._reject(order, str(exc), evaluated_at)

        if proposal.market_data is None or proposal.market_data_source is None:
            return self._reject(order, "proposal requires quote data and source identity", evaluated_at)
        if proposal.structure is not TradeStructure.SINGLE_LEG:
            return self._reject(order, "multi-leg paper proposals are unsupported", evaluated_at)
        if (
            proposal.contract.instrument_type is InstrumentType.OPT
            and proposal.signal.direction is not SignalDirection.LONG
        ):
            return self._reject(order, "short-option openings are unsupported", evaluated_at)
        if any(
            position.contract == proposal.contract
            for position in self._positions.values()
        ) or any(
            other.action is PaperOrderAction.OPEN
            and other.contract == proposal.contract
            and other.status in _ACTIVE_ORDER_STATUSES
            for other in self._orders.values()
        ):
            return self._reject(order, "a position or active opening order already uses this contract", evaluated_at)

        if self._snapshot.equity <= 0:
            return self._reject(order, "portfolio equity must be positive", evaluated_at)
        account = self._make_risk_snapshot(evaluated_at)
        try:
            sizing = size_position(
                PositionSizingRequest(self._config.risk_policy, account, proposal)
            )
        except (TypeError, ValueError) as exc:
            return self._reject(order, f"position sizing failed: {exc}", evaluated_at)
        if sizing.status is not SizingStatus.SIZED or sizing.sized_proposal != proposal:
            codes = ", ".join(str(item.code.value) for item in sizing.reasons)
            return self._reject(
                order,
                "proposal is not correctly sized" + (f" ({codes})" if codes else ""),
                evaluated_at,
            )

        try:
            decision = evaluate_pre_trade_risk(
                self._config.risk_policy,
                account,
                proposal,
                evaluated_at=evaluated_at,
                user_approved=True,
            )
        except (TypeError, ValueError) as exc:
            return self._reject(order, f"M8.1 risk evaluation failed: {exc}", evaluated_at)
        if (
            decision.status is not RiskDecisionStatus.APPROVED
            or decision.user_approved is not True
            or decision.estimated_risk_amount is None
            or decision.estimated_exposure_amount is None
            or decision.risk_budget is None
        ):
            return self._reject(order, "fresh M8.1 risk decision was not APPROVED", evaluated_at, decision)

        try:
            unit_estimate = calculate_risk_and_exposure_per_contract(proposal)
            reserved_risk = _product(unit_estimate.risk_amount, Decimal(proposal.quantity))
            reserved_exposure = _product(unit_estimate.exposure_amount, Decimal(proposal.quantity))
            reserved_cash_per_contract = self._opening_cash_reserve_per_contract(proposal)
            reserved_cash = _product(reserved_cash_per_contract, Decimal(proposal.quantity))
            active_reserved_risk = finite_sum(
                "active order risk reservations",
                *(item.reserved_risk for item in self._orders.values()),
            )
            combined_reserved_risk = finite_sum(
                "combined order risk reservations", active_reserved_risk, reserved_risk
            )
        except (ValueError, ArithmeticError) as exc:
            return self._reject(order, f"reservation calculation failed: {exc}", evaluated_at, decision)
        if combined_reserved_risk > decision.risk_budget:
            return self._reject(
                order,
                "outstanding opening orders would exceed the shared per-trade risk reservation budget",
                evaluated_at,
                decision,
            )
        other_cash_reservations = self._active_reserved_cash(exclude=client_order_id)
        if finite_sum("cash plus reservations", self._cash, -other_cash_reservations) < reserved_cash:
            return self._reject(
                order,
                "opening order exceeds available paper cash after active cash reservations",
                evaluated_at,
                decision,
            )

        accepted = replace(
            order,
            status=PaperOrderStatus.OPEN,
            updated_at=evaluated_at,
            data_source=proposal.market_data_source,
            is_synthetic=proposal.market_data.is_synthetic,
            risk_decision=decision,
            risk_per_contract=unit_estimate.risk_amount,
            exposure_per_contract=unit_estimate.exposure_amount,
            reserved_risk=reserved_risk,
            reserved_exposure=reserved_exposure,
            reserved_cash=reserved_cash,
            rejection_reason=None,
        )
        self._orders[client_order_id] = accepted
        self._last_quote_timestamp[proposal.contract.contract_id] = max(
            self._last_quote_timestamp.get(proposal.contract.contract_id, proposal.market_data.timestamp),
            proposal.market_data.timestamp,
        )
        self._state_as_of = evaluated_at
        self._refresh_snapshot(evaluated_at)
        _LOGGER.info(
            "paper order approved after risk checks: order_id=%s action=open",
            client_order_id,
        )
        return accepted

    def submit_close_order(
        self,
        request: PaperCloseOrderRequest,
        *,
        user_approved: bool | None,
    ) -> PaperOrder:
        """Stage/approve an explicit position reduction; never opens a new side."""
        if not isinstance(request, PaperCloseOrderRequest):
            raise TypeError("request must be a PaperCloseOrderRequest")
        if user_approved is not None and not isinstance(user_approved, bool):
            raise ValueError("user_approved must be True, False, or None")
        terms = (
            PaperOrderAction.CLOSE,
            request.position_id,
            request.quantity,
            request.quote,
            request.submitted_at,
            user_approved,
        )
        existing = self._orders.get(request.client_order_id)
        if existing is not None:
            if self._order_terms[request.client_order_id] == terms:
                return existing
            raise DuplicateSubmissionError("idempotency key was reused with different terms")
        if request.submitted_at < self._state_as_of:
            raise PaperTradingError("submission time precedes the current paper ledger")
        if request.quote.as_of > request.submitted_at:
            raise PaperTradingError("close quote event must not follow order submission")
        position = self._positions.get(request.position_id)
        if position is None:
            raise PaperTradingError("close request must reference an open position")
        self._validate_close_request_quote(position, request.quote)
        if request.quantity > self._available_close_quantity(position.position_id):
            raise PaperTradingError("close quantity exceeds unreserved position quantity")
        status = (
            PaperOrderStatus.PENDING_APPROVAL
            if user_approved is None
            else PaperOrderStatus.OPEN
            if user_approved
            else PaperOrderStatus.REJECTED
        )
        order = self._make_close_order(
            client_order_id=request.client_order_id,
            action=PaperOrderAction.CLOSE,
            position=position,
            quantity=request.quantity,
            submitted_at=request.submitted_at,
            source=request.quote.data_source,
            is_synthetic=request.quote.is_synthetic,
            quote_approved=request.quote.approved,
            status=status,
            rejection_reason=("explicit user approval was denied" if user_approved is False else None),
        )
        self._orders[request.client_order_id] = order
        self._order_terms[request.client_order_id] = terms
        self._state_as_of = request.submitted_at
        self._remember_quote(request.quote)
        self._refresh_snapshot(request.submitted_at)
        _LOGGER.info(
            "paper close order staged: order_id=%s status=%s",
            request.client_order_id,
            status.value,
        )
        return order

    def approve_close_order(
        self,
        client_order_id: str,
        *,
        user_approved: bool | None,
        quote: PaperQuoteObservation,
        evaluated_at,
    ) -> PaperOrder:
        order = self.get_order(client_order_id)
        if order.action is not PaperOrderAction.CLOSE:
            raise OrderStateError("only manual close orders use this approval method")
        if order.status is not PaperOrderStatus.PENDING_APPROVAL:
            raise OrderStateError("close order is not pending approval")
        validate_aware_datetime("evaluated_at", evaluated_at)
        if not isinstance(quote, PaperQuoteObservation):
            raise TypeError("quote must be a PaperQuoteObservation")
        if evaluated_at < self._state_as_of:
            raise PaperTradingError("approval time precedes the current ledger")
        if quote.as_of > evaluated_at:
            raise PaperTradingError("close approval quote must not follow approval time")
        if user_approved is None:
            return order
        if not isinstance(user_approved, bool):
            raise ValueError("user_approved must be True, False, or None")
        if user_approved is False:
            return self._reject(order, "explicit user approval was denied", evaluated_at)
        position = self._positions.get(order.position_id)
        if position is None:
            return self._reject(order, "referenced position is no longer open", evaluated_at)
        try:
            self._validate_close_request_quote(position, quote)
            executable_reference_price(quote, self._close_fill_side(position.direction))
        except (InvalidPaperQuoteError, ValueError) as exc:
            return self._reject(order, str(exc), evaluated_at)
        if order.requested_quantity > self._available_close_quantity(position.position_id):
            return self._reject(order, "close quantity exceeds unreserved position quantity", evaluated_at)
        accepted = replace(
            order,
            status=PaperOrderStatus.OPEN,
            updated_at=evaluated_at,
            data_source=quote.data_source,
            is_synthetic=quote.is_synthetic,
            quote_approved=quote.approved,
            rejection_reason=None,
        )
        self._orders[client_order_id] = accepted
        self._state_as_of = evaluated_at
        self._remember_quote(quote)
        self._refresh_snapshot(evaluated_at)
        _LOGGER.info(
            "paper close order approved: order_id=%s",
            client_order_id,
        )
        return accepted

    def fill_order(
        self,
        client_order_id: str,
        *,
        fill_id: str,
        quantity: int,
        quote: PaperQuoteObservation,
    ) -> PaperFill:
        """Apply an explicit fill quantity against an explicit validated quote."""
        if not isinstance(fill_id, str) or not fill_id.strip() or any(c.isspace() for c in fill_id):
            raise ValueError("fill_id must be a non-empty identifier without whitespace")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("fill quantity must be a positive whole-contract integer")
        if not isinstance(quote, PaperQuoteObservation):
            raise TypeError("quote must be a PaperQuoteObservation")
        fill_terms = (client_order_id, quantity, quote)
        existing_fill = self._fills.get(fill_id)
        if existing_fill is not None:
            if self._fill_terms[fill_id] == fill_terms:
                return existing_fill
            raise DuplicateSubmissionError("fill id was reused with different terms")

        order = self.get_order(client_order_id)
        if order.status not in _ACTIVE_ORDER_STATUSES:
            raise OrderStateError("only OPEN or PARTIALLY_FILLED orders can receive fills")
        if quantity > order.remaining_quantity:
            raise PaperTradingError("fill quantity exceeds order remaining quantity")
        if quote.as_of < self._state_as_of:
            raise InvalidPaperQuoteError("quote event time precedes the paper ledger")
        try:
            validate_quote_observation(
                order.contract,
                quote,
                freshness_policy=self._config.execution.quote_freshness_policy,
            )
        except InvalidPaperQuoteError:
            raise
        previous_quote_at = self._last_quote_timestamp.get(order.contract.contract_id)
        if previous_quote_at is not None and quote.timestamp < previous_quote_at:
            raise InvalidPaperQuoteError("quote timestamp moved backwards for this contract")
        if order.is_synthetic is not quote.is_synthetic:
            raise InvalidPaperQuoteError("fill quote provenance differs from approved order")

        if order.action is PaperOrderAction.OPEN:
            return self._fill_open_order(order, fill_id, quantity, quote)
        return self._fill_close_order(order, fill_id, quantity, quote)

    def cancel_order(self, client_order_id: str, *, cancelled_at) -> PaperOrder:
        order = self.get_order(client_order_id)
        validate_aware_datetime("cancelled_at", cancelled_at)
        if order.status not in (
            PaperOrderStatus.PENDING_APPROVAL,
            PaperOrderStatus.OPEN,
            PaperOrderStatus.PARTIALLY_FILLED,
        ):
            raise OrderStateError("only pending or active orders can be cancelled")
        if cancelled_at < self._state_as_of or cancelled_at < order.updated_at:
            raise PaperTradingError("cancellation time precedes the current ledger")
        cancelled = replace(
            order,
            status=PaperOrderStatus.CANCELLED,
            updated_at=cancelled_at,
            reserved_risk=Decimal(0),
            reserved_exposure=Decimal(0),
            reserved_cash=Decimal(0),
        )
        self._orders[client_order_id] = cancelled
        self._state_as_of = cancelled_at
        self._refresh_snapshot(cancelled_at)
        _LOGGER.info("paper order cancelled: order_id=%s", client_order_id)
        return cancelled

    def reject_order(
        self,
        client_order_id: str,
        *,
        reason: str,
        rejected_at,
    ) -> PaperOrder:
        """Record an explicit paper rejection and release its remaining reserves."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("rejection reason must be a non-empty string")
        order = self.get_order(client_order_id)
        validate_aware_datetime("rejected_at", rejected_at)
        if order.status not in (
            PaperOrderStatus.PENDING_APPROVAL,
            PaperOrderStatus.OPEN,
            PaperOrderStatus.PARTIALLY_FILLED,
        ):
            raise OrderStateError("only pending or active orders can be rejected")
        if rejected_at < self._state_as_of or rejected_at < order.updated_at:
            raise PaperTradingError("rejection time precedes the current ledger")
        rejected = replace(
            order,
            status=PaperOrderStatus.REJECTED,
            updated_at=rejected_at,
            reserved_risk=Decimal(0),
            reserved_exposure=Decimal(0),
            reserved_cash=Decimal(0),
            rejection_reason=reason,
        )
        self._orders[client_order_id] = rejected
        self._state_as_of = rejected_at
        self._refresh_snapshot(rejected_at)
        _LOGGER.info("paper order rejected: order_id=%s", client_order_id)
        return rejected

    def mark_position(
        self,
        position_id: str,
        observation: PaperQuoteObservation,
    ) -> PaperPortfolioSnapshot:
        position = self._positions.get(position_id)
        if position is None:
            raise PaperTradingError("unknown open paper position")
        self._validate_position_quote(position, observation)
        mark_side = self._close_fill_side(position.direction)
        mark_price = executable_reference_price(observation, mark_side)
        updated = replace(
            position,
            updated_at=observation.as_of,
            mark_price=mark_price,
            mark_timestamp=observation.timestamp,
            data_sources=_append_source(position.data_sources, observation.data_source),
            all_real_quotes_approved=(
                position.all_real_quotes_approved
                and (observation.is_synthetic or observation.approved)
            ),
        )
        self._positions[position_id] = updated
        self._remember_quote(observation)
        self._state_as_of = observation.as_of
        return self._refresh_snapshot(observation.as_of)

    def check_protective_exit(
        self,
        position_id: str,
        observation: PaperQuoteObservation,
        *,
        client_order_id: str,
    ) -> PaperOrder | None:
        """Apply approved stop/target directives; generated orders can only reduce."""
        position = self._positions.get(position_id)
        if position is None:
            raise PaperTradingError("unknown open paper position")
        self._validate_position_quote(position, observation)
        close_side = self._close_fill_side(position.direction)
        trigger_price = executable_reference_price(observation, close_side)
        stop_hit = position.stop_price is not None and (
            trigger_price <= position.stop_price
            if position.direction is PositionSide.LONG
            else trigger_price >= position.stop_price
        )
        target_hit = position.target_price is not None and (
            trigger_price >= position.target_price
            if position.direction is PositionSide.LONG
            else trigger_price <= position.target_price
        )
        if not stop_hit and not target_hit:
            self.mark_position(position_id, observation)
            return None
        action = PaperOrderAction.STOP_LOSS if stop_hit else PaperOrderAction.TARGET
        if stop_hit and target_hit:
            action = PaperOrderAction.STOP_LOSS
        request_terms = (action, position_id, position.quantity, observation)
        existing = self._orders.get(client_order_id)
        if existing is not None:
            if self._order_terms[client_order_id] == request_terms:
                return existing
            raise DuplicateSubmissionError("idempotency key was reused with different terms")
        active_closes = [
            order for order in self._orders.values()
            if order.position_id == position_id
            and order.action is not PaperOrderAction.OPEN
            and order.status in _ACTIVE_ORDER_STATUSES
        ]
        for active_order in active_closes:
            self.cancel_order(
                active_order.client_order_id,
                cancelled_at=observation.as_of,
            )
        self.mark_position(position_id, observation)
        order = self._make_close_order(
            client_order_id=client_order_id,
            action=action,
            position=self._positions[position_id],
            quantity=position.quantity,
            submitted_at=observation.as_of,
            source=observation.data_source,
            is_synthetic=observation.is_synthetic,
            quote_approved=observation.approved,
            status=PaperOrderStatus.OPEN,
            limit_price=(position.target_price if action is PaperOrderAction.TARGET else None),
            rejection_reason=None,
        )
        self._orders[client_order_id] = order
        self._order_terms[client_order_id] = request_terms
        self._refresh_snapshot(observation.as_of)
        return order

    def _fill_open_order(
        self,
        order: PaperOrder,
        fill_id: str,
        quantity: int,
        quote: PaperQuoteObservation,
    ) -> PaperFill:
        proposal = order.proposal
        assert proposal is not None and proposal.entry_reference_price is not None
        self._require_two_sided_quote(quote)
        side = self._open_fill_side(order.direction)
        reference = executable_reference_price(quote, side)
        fill_price = calculate_fill_price(
            reference, side, self._config.execution.slippage_bps, order.contract.tick_size
        )
        if order.direction is PositionSide.LONG and fill_price > proposal.entry_reference_price:
            raise InvalidPaperQuoteError("buy fill would be worse than the approved entry reference")
        if order.direction is PositionSide.SHORT and fill_price < proposal.entry_reference_price:
            raise InvalidPaperQuoteError("sell fill would be worse than the approved entry reference")
        stop = proposal.stop_loss_reference_price
        if stop is not None and (
            (order.direction is PositionSide.LONG and stop >= fill_price)
            or (order.direction is PositionSide.SHORT and stop <= fill_price)
        ):
            raise InvalidPaperQuoteError("fill would breach the approved stop placement")

        notional = calculate_notional(
            order.contract,
            fill_price,
            quantity,
            point_value=proposal.point_value,
        )
        charges = calculate_fee_charges(self._config.execution, notional)
        fee_total = total_fees(charges)
        cash_debit = finite_sum(
            "opening cash debit",
            fee_total,
            notional if order.contract.instrument_type is InstrumentType.OPT else Decimal(0),
        )
        remaining_order = order.remaining_quantity - quantity
        other_reservations = self._active_reserved_cash(exclude=order.client_order_id)
        cash_after_debit = finite_difference("available cash", self._cash, other_reservations)
        post_fill_reserve = _product(
            self._opening_cash_reserve_per_contract(proposal), Decimal(remaining_order)
        )
        if finite_sum("cash commitment", cash_debit, post_fill_reserve) > cash_after_debit:
            raise PaperTradingError("fill and remaining paper cash reservation exceed available cash")

        position = self._positions.get(order.position_id)
        if position is None:
            mark_side = self._close_fill_side(order.direction)
            mark_price = executable_reference_price(quote, mark_side)
            position = PaperPosition(
                position_id=order.position_id,
                contract=order.contract,
                direction=order.direction,
                quantity=quantity,
                average_entry_price=fill_price,
                point_value=proposal.point_value,
                stop_price=proposal.stop_loss_reference_price,
                target_price=proposal.target_reference_price,
                opened_at=quote.timestamp,
                updated_at=quote.as_of,
                mark_price=mark_price,
                mark_timestamp=quote.timestamp,
                remaining_entry_fees=fee_total,
                realized_gross_pnl=Decimal(0),
                data_sources=(quote.data_source,),
                is_synthetic=quote.is_synthetic,
                all_real_quotes_approved=(quote.is_synthetic or quote.approved),
            )
        else:
            if position.direction is not order.direction or position.contract != order.contract:
                raise PaperTradingError("opening fill does not match the existing position")
            new_quantity = position.quantity + quantity
            average = _weighted_average(
                position.average_entry_price,
                position.quantity,
                fill_price,
                quantity,
            )
            position = replace(
                position,
                quantity=new_quantity,
                average_entry_price=average,
                updated_at=quote.as_of,
                mark_price=executable_reference_price(quote, self._close_fill_side(order.direction)),
                mark_timestamp=quote.timestamp,
                remaining_entry_fees=finite_sum(
                    "remaining entry fees", position.remaining_entry_fees, fee_total
                ),
                data_sources=_append_source(position.data_sources, quote.data_source),
                all_real_quotes_approved=(
                    position.all_real_quotes_approved
                    and (quote.is_synthetic or quote.approved)
                ),
            )

        self._cash = finite_difference("paper cash", self._cash, cash_debit)
        self._fees_paid = finite_sum("fees paid", self._fees_paid, fee_total)
        self._positions[order.position_id] = position
        self._last_quote_timestamp[order.contract.contract_id] = quote.timestamp
        fill = PaperFill(
            fill_id=fill_id,
            client_order_id=order.client_order_id,
            position_id=order.position_id,
            contract=order.contract,
            action=PaperOrderAction.OPEN,
            side=side,
            quantity=quantity,
            timestamp=quote.timestamp,
            reference_price=reference,
            fill_price=fill_price,
            notional=notional,
            fee_charges=charges,
            entry_fees_allocated=Decimal(0),
            gross_realized_pnl=Decimal(0),
            net_realized_pnl=finite_difference("opening fill net P&L", Decimal(0), fee_total),
            data_source=quote.data_source,
            is_synthetic=quote.is_synthetic,
            quote_approved=quote.approved,
        )
        self._fills[fill_id] = fill
        self._fill_terms[fill_id] = (order.client_order_id, quantity, quote)
        updated_order = self._updated_after_fill(order, quantity, quote.as_of)
        self._orders[order.client_order_id] = updated_order
        self._state_as_of = quote.as_of
        self._refresh_snapshot(quote.as_of)
        _LOGGER.info(
            "paper fill applied: fill_id=%s order_id=%s action=open quantity=%d",
            fill_id,
            order.client_order_id,
            quantity,
        )
        return fill

    def _fill_close_order(
        self,
        order: PaperOrder,
        fill_id: str,
        quantity: int,
        quote: PaperQuoteObservation,
    ) -> PaperFill:
        position = self._positions.get(order.position_id)
        if position is None:
            raise OrderStateError("position was fully closed before this fill")
        if quantity > position.quantity:
            raise PaperTradingError("close fill exceeds remaining open position quantity")
        self._validate_position_quote(position, quote)
        side = self._close_fill_side(position.direction)
        reference = executable_reference_price(quote, side)
        calculated_fill = calculate_fill_price(
            reference, side, self._config.execution.slippage_bps, order.contract.tick_size
        )
        fill_price = calculated_fill
        if order.action is PaperOrderAction.TARGET:
            if order.limit_price is None:
                raise OrderStateError("target close order is missing its limit price")
            meets_limit = (
                calculated_fill >= order.limit_price
                if side is PaperFillSide.SELL
                else calculated_fill <= order.limit_price
            )
            if not meets_limit:
                raise InvalidPaperQuoteError("adverse slippage would violate target limit")
            fill_price = order.limit_price

        notional = calculate_notional(
            order.contract,
            fill_price,
            quantity,
            point_value=position.point_value,
        )
        charges = calculate_fee_charges(self._config.execution, notional)
        exit_fees = total_fees(charges)
        gross = calculate_unrealized_pnl(
            position.contract,
            position.direction,
            quantity,
            position.average_entry_price,
            fill_price,
            point_value=position.point_value,
        )
        entry_fee_allocated, remaining_entry_fees = allocate_proportional_amount(
            position.remaining_entry_fees,
            quantity,
            position.quantity,
        )
        net = finite_difference(
            "net realized P&L",
            finite_difference("gross less entry fee", gross, entry_fee_allocated),
            exit_fees,
        )
        cash_change = (
            finite_difference("futures close proceeds", gross, exit_fees)
            if order.contract.instrument_type is InstrumentType.FUT
            else finite_difference("option close proceeds", notional, exit_fees)
        )
        remaining_position_quantity = position.quantity - quantity
        new_gross_realized = finite_sum("position realized gross P&L", position.realized_gross_pnl, gross)
        self._gross_realized = finite_sum("portfolio gross realized P&L", self._gross_realized, gross)
        self._fees_paid = finite_sum("fees paid", self._fees_paid, exit_fees)
        self._cash = finite_sum("paper cash", self._cash, cash_change)

        if remaining_position_quantity == 0:
            del self._positions[position.position_id]
        else:
            self._positions[position.position_id] = replace(
                position,
                quantity=remaining_position_quantity,
                updated_at=quote.as_of,
                mark_price=reference,
                mark_timestamp=quote.timestamp,
                remaining_entry_fees=remaining_entry_fees,
                realized_gross_pnl=new_gross_realized,
                data_sources=_append_source(position.data_sources, quote.data_source),
                all_real_quotes_approved=(
                    position.all_real_quotes_approved
                    and (quote.is_synthetic or quote.approved)
                ),
            )

        history = PaperTradeHistory(
            trade_id=len(self._trade_history) + 1,
            position_id=position.position_id,
            contract=position.contract,
            direction=position.direction,
            quantity_closed=quantity,
            opened_at=position.opened_at,
            closed_at=quote.timestamp,
            average_entry_price=position.average_entry_price,
            average_exit_price=fill_price,
            gross_pnl=gross,
            entry_fees=entry_fee_allocated,
            exit_fees=exit_fees,
            net_pnl=net,
            exit_reason=order.action,
            data_sources=_append_source(position.data_sources, quote.data_source),
            is_synthetic=position.is_synthetic,
            quote_approved=(
                position.all_real_quotes_approved
                and (quote.is_synthetic or quote.approved)
            ),
            position_fully_closed=remaining_position_quantity == 0,
        )
        self._trade_history.append(history)
        fill = PaperFill(
            fill_id=fill_id,
            client_order_id=order.client_order_id,
            position_id=order.position_id,
            contract=order.contract,
            action=order.action,
            side=side,
            quantity=quantity,
            timestamp=quote.timestamp,
            reference_price=reference,
            fill_price=fill_price,
            notional=notional,
            fee_charges=charges,
            entry_fees_allocated=entry_fee_allocated,
            gross_realized_pnl=gross,
            net_realized_pnl=net,
            data_source=quote.data_source,
            is_synthetic=quote.is_synthetic,
            quote_approved=quote.approved,
        )
        self._fills[fill_id] = fill
        self._fill_terms[fill_id] = (order.client_order_id, quantity, quote)
        self._last_quote_timestamp[order.contract.contract_id] = quote.timestamp
        updated_order = self._updated_after_fill(order, quantity, quote.as_of)
        self._orders[order.client_order_id] = updated_order
        self._state_as_of = quote.as_of
        self._refresh_snapshot(quote.as_of)
        _LOGGER.info(
            "paper fill applied: fill_id=%s order_id=%s action=close quantity=%d",
            fill_id,
            order.client_order_id,
            quantity,
        )
        return fill

    def _updated_after_fill(
        self,
        order: PaperOrder,
        quantity: int,
        updated_at,
    ) -> PaperOrder:
        filled = order.filled_quantity + quantity
        remaining = order.requested_quantity - filled
        status = PaperOrderStatus.FILLED if remaining == 0 else PaperOrderStatus.PARTIALLY_FILLED
        if order.action is PaperOrderAction.OPEN and status in _ACTIVE_ORDER_STATUSES:
            risk = _product(order.risk_per_contract, Decimal(remaining))
            exposure = _product(order.exposure_per_contract, Decimal(remaining))
            cash = _product(
                self._opening_cash_reserve_per_contract(order.proposal), Decimal(remaining)
            )
        else:
            risk = exposure = cash = Decimal(0)
        return replace(
            order,
            filled_quantity=filled,
            status=status,
            updated_at=updated_at,
            reserved_risk=risk,
            reserved_exposure=exposure,
            reserved_cash=cash,
        )

    def _validate_proposal_quote(
        self,
        proposal: ProposedTrade,
        evaluated_at,
        *,
        quote_approved: bool,
    ) -> None:
        if proposal.market_data is None or proposal.market_data_source is None:
            raise InvalidPaperQuoteError("proposal requires explicit quote and source identity")
        observation = PaperQuoteObservation(
            response=proposal.market_data,
            data_source=proposal.market_data_source,
            as_of=evaluated_at,
            approved=quote_approved,
        )
        validate_quote_observation(
            proposal.contract,
            observation,
            freshness_policy=self._config.execution.quote_freshness_policy,
        )
        self._require_two_sided_quote(observation)
        previous = self._last_quote_timestamp.get(proposal.contract.contract_id)
        if previous is not None and observation.timestamp < previous:
            raise InvalidPaperQuoteError("proposal quote timestamp moved backwards")

    def _validate_close_quote(
        self,
        position: PaperPosition,
        observation: PaperQuoteObservation,
    ) -> None:
        validate_quote_observation(
            position.contract,
            observation,
            freshness_policy=self._config.execution.quote_freshness_policy,
        )
        if observation.is_synthetic is not position.is_synthetic:
            raise InvalidPaperQuoteError("close quote provenance differs from position")

    def _validate_position_quote(
        self,
        position: PaperPosition,
        observation: PaperQuoteObservation,
    ) -> None:
        if observation.as_of < self._state_as_of:
            raise InvalidPaperQuoteError("quote event time precedes the paper ledger")
        self._validate_close_quote(position, observation)
        previous = self._last_quote_timestamp.get(position.contract.contract_id)
        if previous is not None and observation.timestamp < previous:
            raise InvalidPaperQuoteError("quote timestamp moved backwards for this contract")

    @staticmethod
    def _require_two_sided_quote(observation: PaperQuoteObservation) -> None:
        quote = observation.response.quote
        if quote.bid is None or quote.ask is None or quote.bid <= 0 or quote.ask <= 0:
            raise InvalidPaperQuoteError("opening fills require a positive two-sided bid and ask for mark-to-market")

    def _validate_close_request_quote(
        self,
        position: PaperPosition,
        observation: PaperQuoteObservation,
    ) -> None:
        self._validate_position_quote(position, observation)
        executable_reference_price(observation, self._close_fill_side(position.direction))

    def _make_close_order(
        self,
        *,
        client_order_id: str,
        action: PaperOrderAction,
        position: PaperPosition,
        quantity: int,
        submitted_at,
        source: str,
        is_synthetic: bool,
        quote_approved: bool,
        status: PaperOrderStatus,
        limit_price: Decimal | None = None,
        rejection_reason: str | None,
    ) -> PaperOrder:
        return PaperOrder(
            client_order_id=client_order_id,
            action=action,
            contract=position.contract,
            direction=position.direction,
            requested_quantity=quantity,
            filled_quantity=0,
            status=status,
            submitted_at=submitted_at,
            updated_at=submitted_at,
            data_source=source,
            is_synthetic=is_synthetic,
            quote_approved=quote_approved,
            proposal=None,
            position_id=position.position_id,
            risk_decision=None,
            risk_per_contract=None,
            exposure_per_contract=None,
            reserved_risk=Decimal(0),
            reserved_exposure=Decimal(0),
            reserved_cash=Decimal(0),
            limit_price=limit_price,
            rejection_reason=rejection_reason,
        )

    def _reject(
        self,
        order: PaperOrder,
        reason: str,
        rejected_at,
        risk_decision=None,
    ) -> PaperOrder:
        rejected = replace(
            order,
            status=PaperOrderStatus.REJECTED,
            updated_at=rejected_at,
            risk_decision=risk_decision,
            reserved_risk=Decimal(0),
            reserved_exposure=Decimal(0),
            reserved_cash=Decimal(0),
            rejection_reason=reason,
        )
        self._orders[order.client_order_id] = rejected
        self._state_as_of = rejected_at
        self._refresh_snapshot(rejected_at)
        _LOGGER.info(
            "paper order rejected after validation: order_id=%s action=%s",
            order.client_order_id,
            order.action.value,
        )
        return rejected

    def _opening_cash_reserve_per_contract(self, proposal: ProposedTrade | None) -> Decimal:
        if proposal is None or proposal.entry_reference_price is None:
            raise ValueError("opening order entry reference is required")
        notional = calculate_notional(
            proposal.contract,
            proposal.entry_reference_price,
            1,
            point_value=proposal.point_value,
        )
        fees = total_fees(calculate_fee_charges(self._config.execution, notional))
        return finite_sum(
            "opening cash reservation",
            fees,
            notional if proposal.contract.instrument_type is InstrumentType.OPT else Decimal(0),
        )

    def _active_reserved_cash(self, *, exclude: str | None = None) -> Decimal:
        return finite_sum(
            "active cash reservations",
            *(
                order.reserved_cash
                for order in self._orders.values()
                if order.client_order_id != exclude
                and order.status in _ACTIVE_ORDER_STATUSES
            ),
        )

    def _available_close_quantity(self, position_id: str) -> int:
        position = self._positions.get(position_id)
        if position is None:
            return 0
        reserved = sum(
            order.remaining_quantity
            for order in self._orders.values()
            if order.position_id == position_id
            and order.action is not PaperOrderAction.OPEN
            and order.status in _ACTIVE_ORDER_STATUSES
        )
        return max(0, position.quantity - reserved)

    def _active_reserved_risk(self) -> Decimal:
        return finite_sum(
            "active order risk reservations",
            *(order.reserved_risk for order in self._orders.values()),
        )

    def _remember_quote(self, observation: PaperQuoteObservation) -> None:
        contract_id = observation.response.quote.contract.contract_id
        prior = self._last_quote_timestamp.get(contract_id)
        if prior is not None and observation.timestamp < prior:
            raise InvalidPaperQuoteError("quote timestamp moved backwards for this contract")
        self._last_quote_timestamp[contract_id] = observation.timestamp

    def _make_risk_snapshot(self, evaluated_at) -> AccountRiskSnapshot:
        exposure: dict[UnderlyingIndex, Decimal] = {
            underlying: Decimal(0) for underlying in UnderlyingIndex
        }
        contract_slots: set[tuple[object, str]] = set()
        for position in self._positions.values():
            position_exposure = calculate_exposure(
                position.contract,
                position.average_entry_price,
                position.quantity,
                point_value=position.point_value,
            )
            exposure[position.contract.underlying] = finite_sum(
                "position exposure",
                exposure[position.contract.underlying],
                position_exposure,
            )
            contract_slots.add((position.contract.exchange, position.contract.contract_id))
        for order in self._orders.values():
            if order.action is not PaperOrderAction.OPEN or order.status not in _ACTIVE_ORDER_STATUSES:
                continue
            exposure[order.contract.underlying] = finite_sum(
                "reserved order exposure",
                exposure[order.contract.underlying],
                order.reserved_exposure,
            )
            contract_slots.add((order.contract.exchange, order.contract.contract_id))
        return AccountRiskSnapshot(
            current_equity=self._snapshot.equity,
            starting_daily_equity=self._config.initial_cash,
            peak_equity=self._snapshot.peak_equity,
            daily_realised_pnl=self._snapshot.realized_pnl,
            daily_unrealised_pnl=self._snapshot.unrealized_pnl,
            open_position_count=len(contract_slots),
            exposure_by_underlying=exposure,
            snapshot_timestamp=evaluated_at,
        )

    def _build_snapshot(self, as_of) -> PaperPortfolioSnapshot:
        unrealized_values: list[Decimal] = []
        option_mark_values: list[Decimal] = []
        for position in self._positions.values():
            if position.mark_price is None:
                raise PaperTradingError("an open position without an explicit mark cannot be valued")
            unrealized_values.append(
                calculate_unrealized_pnl(
                    position.contract,
                    position.direction,
                    position.quantity,
                    position.average_entry_price,
                    position.mark_price,
                    point_value=position.point_value,
                )
            )
            if position.contract.instrument_type is InstrumentType.OPT:
                option_mark_values.append(
                    calculate_notional(
                        position.contract,
                        position.mark_price,
                        position.quantity,
                        point_value=None,
                    )
                )
        unrealized = finite_sum("portfolio unrealized P&L", *unrealized_values)
        option_market_value = finite_sum("open option market value", *option_mark_values)
        futures_unrealized = finite_sum(
            "futures unrealized P&L",
            *(
                value
                for position, value in zip(self._positions.values(), unrealized_values)
                if position.contract.instrument_type is InstrumentType.FUT
            ),
        )
        equity = finite_sum("portfolio equity", self._cash, option_market_value, futures_unrealized)
        realized = finite_difference("portfolio realized P&L", self._gross_realized, self._fees_paid)
        peak = max(self._peak_equity, equity)
        reserved_risk = finite_sum(
            "portfolio reserved risk",
            *(order.reserved_risk for order in self._orders.values()),
        )
        reserved_exposure = finite_sum(
            "portfolio reserved exposure",
            *(order.reserved_exposure for order in self._orders.values()),
        )
        reserved_cash = self._active_reserved_cash()
        self._peak_equity = peak
        return PaperPortfolioSnapshot(
            initial_cash=self._config.initial_cash,
            execution_config=self._config.execution,
            cash=self._cash,
            equity=equity,
            realized_pnl=realized,
            gross_realized_pnl=self._gross_realized,
            unrealized_pnl=unrealized,
            fees_paid=self._fees_paid,
            reserved_risk=reserved_risk,
            reserved_exposure=reserved_exposure,
            reserved_cash=reserved_cash,
            peak_equity=peak,
            as_of=as_of,
            positions=tuple(self._positions.values()),
            orders=tuple(self._orders.values()),
            fills=tuple(self._fills.values()),
            trade_history=tuple(self._trade_history),
        )

    def _refresh_snapshot(self, as_of) -> PaperPortfolioSnapshot:
        self._snapshot = self._build_snapshot(as_of)
        return self._snapshot

    @staticmethod
    def _open_fill_side(direction: PositionSide) -> PaperFillSide:
        return PaperFillSide.BUY if direction is PositionSide.LONG else PaperFillSide.SELL

    @staticmethod
    def _close_fill_side(direction: PositionSide) -> PaperFillSide:
        return PaperFillSide.SELL if direction is PositionSide.LONG else PaperFillSide.BUY


def _product(left: Decimal | None, right: Decimal) -> Decimal:
    if left is None:
        raise ValueError("required Decimal value is unavailable")
    from atlas.paper_trading.calculations import finite_product

    return finite_product("paper reservation", left, right)


def _weighted_average(
    first_price: Decimal,
    first_quantity: int,
    second_price: Decimal,
    second_quantity: int,
) -> Decimal:
    from atlas.paper_trading.calculations import finite_product

    total_quantity = first_quantity + second_quantity
    total_cost = finite_sum(
        "weighted entry cost",
        finite_product("first entry cost", first_price, Decimal(first_quantity)),
        finite_product("second entry cost", second_price, Decimal(second_quantity)),
    )
    return _divide_by_integer(total_cost, total_quantity)


def _divide_by_integer(value: Decimal, divisor: int) -> Decimal:
    if isinstance(divisor, bool) or not isinstance(divisor, int) or divisor <= 0:
        raise ValueError("divisor must be a positive integer")
    from decimal import DecimalException, MAX_EMAX, MIN_EMIN, localcontext

    try:
        with localcontext() as context:
            context.prec = max(80, len(value.as_tuple().digits) + abs(value.adjusted()) + 2)
            context.Emax = MAX_EMAX
            context.Emin = MIN_EMIN
            result = value / Decimal(divisor)
    except DecimalException as exc:
        raise ValueError("weighted price exceeds Decimal precision limits") from exc
    if not result.is_finite():
        raise ValueError("weighted price is not finite")
    return result


def _append_source(sources: tuple[str, ...], source: str) -> tuple[str, ...]:
    return sources if source in sources else sources + (source,)


__all__ = ["PaperTradingEngine"]
