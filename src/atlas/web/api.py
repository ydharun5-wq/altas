"""Owner-authenticated application routes for research and paper workflows."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from atlas.backtesting import BacktestConfig, BacktestEngine, ExecutionAssumptions, FeeBasis, FeeComponent
from atlas.derivatives import (
    DerivativeContract, Exchange, FuturesQuote, InstrumentType, OptionType,
    OptionsQuote, UnderlyingIndex,
)
from atlas.market_data import CSVHistoricalMarketDataProvider, HistoricalOHLCVRequest, create_validated_market_data_provider
from atlas.market_data.models import FuturesQuoteResponse, OptionsQuoteResponse
from atlas.risk import (
    AccountRiskSnapshot, ProposedTrade, RiskPolicy, TradeStructure,
    TradingHorizon, evaluate_pre_trade_risk,
)
from atlas.sizing import PositionSizingRequest, SizingStatus, size_position
from atlas.strategies.models import (
    SignalDirection, SignalEvent, SignalStatus, StrategySignal, WarmupStatus,
)
from atlas.strategies.strategies import EMATrendStrategy, RSIMeanReversionStrategy
from atlas.web.database import get_db_session
from atlas.web.models import (
    AgentObservation, AgentRun, AuditEvent, BacktestResult, BacktestRun,
    Instrument, MarketDataMetadata, PaperOrder, Portfolio, Position,
    RiskDecisionRecord, RiskPolicyConfiguration, StrategyRecord, Trade,
    User, Watchlist, WatchlistItem,
)
from atlas.web.schemas import (
    AgentRunCreate, BacktestRequest, InstrumentCreate, PaperProposalCreate,
    PaperFillCreate, PaperCloseCreate, RiskPolicyUpdate, StrategyCreate, WatchlistCreate,
)
from atlas.web.security import AuthContext, current_auth, require_csrf
from atlas.web.serialization import to_jsonable

router = APIRouter(prefix="/api", tags=["atlas"])


def _audit(db: Session, user_id: int, action: str, resource: str, resource_id: str | None = None, details: dict | None = None) -> None:
    db.add(AuditEvent(actor_user_id=user_id, action=action, resource_type=resource, resource_id=resource_id, details_json=details or {}))


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db_session), auth: AuthContext = Depends(current_auth)) -> dict:
    return {
        "paper_only": True,
        "portfolio_count": db.scalar(select(func.count()).select_from(Portfolio).where(Portfolio.owner_id == auth.user.id)) or 0,
        "watchlist_count": db.scalar(select(func.count()).select_from(Watchlist).where(Watchlist.owner_id == auth.user.id)) or 0,
        "strategy_count": db.scalar(select(func.count()).select_from(StrategyRecord)) or 0,
        "backtest_count": db.scalar(select(func.count()).select_from(BacktestRun)) or 0,
        "pending_order_count": db.scalar(select(func.count()).select_from(PaperOrder).join(Portfolio).where(Portfolio.owner_id == auth.user.id, PaperOrder.status == "pending_approval")) or 0,
        "trade_count": db.scalar(select(func.count()).select_from(Trade).join(Portfolio).where(Portfolio.owner_id == auth.user.id)) or 0,
        "agent_execution": "disabled; tasks are stored as propose-only requests",
    }


@router.get("/market-data/status")
def market_data_status(request: Request) -> dict:
    settings = request.app.state.settings
    return {
        "provider": "local_csv" if settings.market_data_dir else None,
        "configured": bool(settings.market_data_dir and settings.market_data_source and settings.market_data_is_synthetic is not None),
        "source": settings.market_data_source,
        "is_synthetic": settings.market_data_is_synthetic,
        "network_market_data": False,
    }


@router.get("/watchlists")
def list_watchlists(db: Session = Depends(get_db_session), auth: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(Watchlist).where(Watchlist.owner_id == auth.user.id).order_by(Watchlist.id)).all()
    return [{"id": row.id, "name": row.name, "symbols": [item.symbol for item in row.items]} for row in rows]


@router.post("/watchlists", status_code=201)
def create_watchlist(payload: WatchlistCreate, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    row = Watchlist(owner_id=auth.user.id, name=payload.name.strip())
    db.add(row)
    try:
        db.flush()
        _audit(db, auth.user.id, "watchlist.create", "watchlist", str(row.id))
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="A watchlist with this name already exists") from exc
    return {"id": row.id, "name": row.name, "symbols": []}


@router.delete("/watchlists/{watchlist_id}", status_code=204)
def delete_watchlist(watchlist_id: int, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)):
    row = db.get(Watchlist, watchlist_id)
    if row is None or row.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    db.delete(row)
    _audit(db, auth.user.id, "watchlist.delete", "watchlist", str(watchlist_id))
    db.commit()


@router.get("/instruments")
def list_instruments(db: Session = Depends(get_db_session), _: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(Instrument).order_by(Instrument.symbol)).all()
    return [{"symbol": row.symbol, "display_name": row.display_name, "instrument_type": row.instrument_type, "underlying": row.underlying, "metadata": row.metadata_json} for row in rows]


@router.post("/instruments", status_code=201)
def create_instrument(payload: InstrumentCreate, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    row = Instrument(symbol=payload.symbol, display_name=payload.display_name, instrument_type=payload.instrument_type, underlying=payload.underlying, metadata_json=payload.metadata)
    db.add(row)
    try:
        db.flush()
        _audit(db, auth.user.id, "instrument.create", "instrument", row.symbol)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Instrument already exists") from exc
    return {"symbol": row.symbol, "display_name": row.display_name, "instrument_type": row.instrument_type, "underlying": row.underlying, "metadata": row.metadata_json}


@router.post("/watchlists/{watchlist_id}/items", status_code=201)
def add_watchlist_item(watchlist_id: int, payload: dict, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    symbol = payload.get("symbol") if isinstance(payload, dict) else None
    if not isinstance(symbol, str) or not symbol.strip():
        raise HTTPException(status_code=422, detail="symbol is required")
    watchlist = db.get(Watchlist, watchlist_id)
    if watchlist is None or watchlist.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Watchlist not found")
    if db.get(Instrument, symbol) is None:
        raise HTTPException(status_code=404, detail="Instrument not found")
    item = WatchlistItem(watchlist_id=watchlist_id, symbol=symbol)
    db.add(item)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Instrument is already on this watchlist") from exc
    _audit(db, auth.user.id, "watchlist.add_symbol", "watchlist", str(watchlist_id), {"symbol": symbol})
    db.commit()
    return {"id": item.id, "watchlist_id": watchlist_id, "symbol": symbol}


@router.get("/strategies/builtins")
def builtin_strategies(_: AuthContext = Depends(current_auth)) -> list[dict]:
    return [
        {"name": "ema_trend", "version": "1.0", "description": "EMA crossover research signals", "parameters": ["fast_period", "slow_period"]},
        {"name": "rsi_mean_reversion", "version": "1.0", "description": "RSI threshold-crossing research signals", "parameters": ["period", "thresholds"]},
    ]


@router.get("/strategies")
def list_strategies(db: Session = Depends(get_db_session), _: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(StrategyRecord).order_by(StrategyRecord.name, StrategyRecord.version)).all()
    return [{"id": r.id, "name": r.name, "version": r.version, "enabled": r.enabled, "config": r.config_json, "source": r.source} for r in rows]


@router.post("/strategies", status_code=201)
def create_strategy(payload: StrategyCreate, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    row = StrategyRecord(name=payload.name, version=payload.version, enabled=False, config_json=payload.config, source="owner")
    db.add(row)
    try:
        db.flush()
        _audit(db, auth.user.id, "strategy.save", "strategy", str(row.id))
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Strategy version already exists") from exc
    return {"id": row.id, "name": row.name, "version": row.version, "enabled": False, "config": row.config_json, "source": row.source}


@router.post("/backtests", status_code=201)
def run_backtest(payload: BacktestRequest, request: Request, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    settings = request.app.state.settings
    if not settings.market_data_dir or not settings.market_data_source or settings.market_data_is_synthetic is None:
        raise HTTPException(status_code=503, detail="Local CSV source and explicit provenance are not configured")
    if payload.end > datetime.now(timezone.utc):
        raise HTTPException(status_code=422, detail="Backtest range cannot include future timestamps")
    base = Path(settings.market_data_dir).resolve()
    csv_file = (base / f"{payload.symbol}.csv").resolve()
    if csv_file.parent != base:
        raise HTTPException(status_code=422, detail="Invalid symbol path")
    try:
        provider = create_validated_market_data_provider(CSVHistoricalMarketDataProvider(
            {payload.symbol: csv_file}, data_source=settings.market_data_source,
            is_synthetic=settings.market_data_is_synthetic,
        ))
        underlying = UnderlyingIndex(payload.underlying)
        timeframe = timedelta(seconds=payload.timeframe_seconds)
        if payload.strategy == "ema_trend":
            strategy = EMATrendStrategy(underlying, payload.symbol, timeframe, payload.fast_period, payload.slow_period)
        else:
            strategy = RSIMeanReversionStrategy(underlying, payload.symbol, timeframe, 14, 30.0, 70.0, 40.0, 60.0)
        config = BacktestConfig(
            initial_capital=payload.initial_capital, quantity=payload.quantity,
            unit_multiplier=payload.unit_multiplier,
            execution=ExecutionAssumptions(
                spread_bps=payload.spread_bps, slippage_bps=payload.slippage_bps,
                fee_components=(() if payload.fee_per_execution == 0 else (FeeComponent("per_execution", FeeBasis.PER_EXECUTION, payload.fee_per_execution),)),
            ),
            target_distance=None,
        )
        result = BacktestEngine(provider).run(
            strategy, HistoricalOHLCVRequest(payload.symbol, payload.start, payload.end),
            timeframe=timeframe, data_source=settings.market_data_source,
            config=config, last_bar_complete=payload.last_bar_complete,
        )
    except (ValueError, OSError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=f"Backtest input rejected: {exc}") from exc
    result_json = to_jsonable(result)
    run_id = str(uuid.uuid4())
    run = BacktestRun(
        id=run_id, symbol=payload.symbol, status="completed",
        data_source=settings.market_data_source, is_synthetic=settings.market_data_is_synthetic,
        config_json=to_jsonable(config), created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.flush()
    db.add(BacktestResult(run_id=run_id, metrics_json=result_json["metrics"], trades_json=result_json["trades"], equity_curve_json=result_json["equity_curve"]))
    db.add(MarketDataMetadata(
        symbol=payload.symbol, source=settings.market_data_source,
        is_synthetic=settings.market_data_is_synthetic,
        first_timestamp=payload.start, last_timestamp=payload.end,
        details={"kind": "local_csv", "backtest_id": run_id},
    ))
    _audit(db, auth.user.id, "backtest.run", "backtest", run_id, {"symbol": payload.symbol, "is_synthetic": settings.market_data_is_synthetic})
    db.commit()
    return {"id": run_id, "status": "completed", "result": result_json}


@router.get("/backtests")
def list_backtests(db: Session = Depends(get_db_session), _: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(BacktestRun).order_by(BacktestRun.created_at.desc()).limit(100)).all()
    return [{"id": r.id, "strategy_id": r.strategy_id, "symbol": r.symbol, "status": r.status, "data_source": r.data_source, "is_synthetic": r.is_synthetic, "created_at": r.created_at} for r in rows]


@router.post("/portfolios", status_code=201)
def create_portfolio(payload: dict, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    name, cash = payload.get("name"), payload.get("initial_cash")
    if not isinstance(name, str) or not name.strip() or len(name) > 100 or isinstance(cash, bool) or not isinstance(cash, (float, int)) or cash <= 0:
        raise HTTPException(status_code=422, detail="name and positive initial_cash are required")
    row = Portfolio(owner_id=auth.user.id, name=name.strip(), initial_cash=Decimal(str(cash)), trading_mode="paper")
    db.add(row)
    db.flush()
    _audit(db, auth.user.id, "portfolio.create", "portfolio", str(row.id))
    db.commit()
    return {"id": row.id, "name": row.name, "initial_cash": str(row.initial_cash), "trading_mode": "paper"}


@router.get("/portfolios")
def list_portfolios(db: Session = Depends(get_db_session), auth: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(Portfolio).where(Portfolio.owner_id == auth.user.id).order_by(Portfolio.id)).all()
    return [{"id": r.id, "name": r.name, "initial_cash": str(r.initial_cash), "trading_mode": r.trading_mode} for r in rows]


@router.put("/risk/policy")
def set_risk_policy(payload: RiskPolicyUpdate, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    valid = {
        "allowed_underlyings": {x.value for x in UnderlyingIndex},
        "allowed_instrument_types": {x.value for x in InstrumentType},
        "allowed_horizons": {x.value for x in TradingHorizon},
    }
    values = payload.model_dump(mode="json")
    for key, allowed in valid.items():
        if not values[key] or not set(values[key]).issubset(allowed):
            raise HTTPException(status_code=422, detail=f"Invalid {key}")
        values[key] = sorted(set(values[key]))
    values["user_approval_required"] = True
    values["policy_id"] = "owner-configured"
    values["version"] = "1"
    row = db.get(RiskPolicyConfiguration, 1)
    if row is None:
        row = RiskPolicyConfiguration(id=1, config_json=values)
        db.add(row)
    else:
        row.config_json = values
    _audit(db, auth.user.id, "risk.policy.update", "risk_policy", "1")
    db.commit()
    return {"configured": True, "policy": values, "user_approval_required": True}


@router.get("/risk/policy")
def get_risk_policy(db: Session = Depends(get_db_session), _: AuthContext = Depends(current_auth)) -> dict:
    row = db.get(RiskPolicyConfiguration, 1)
    return {"configured": row is not None, "policy": row.config_json if row else None}


def _build_risk_inputs(order: PaperOrder, db: Session, *, user_approved: bool):
    payload = PaperProposalCreate.model_validate(order.proposal_json)
    policy_row = db.get(RiskPolicyConfiguration, 1)
    if policy_row is None:
        raise HTTPException(status_code=409, detail="Risk policy has not been configured")
    values = policy_row.config_json
    policy = RiskPolicy(
        policy_id=values["policy_id"], version=values["version"],
        max_risk_per_trade_fraction=Decimal(values["max_risk_per_trade_fraction"]),
        max_daily_loss_fraction=Decimal(values["max_daily_loss_fraction"]),
        max_account_drawdown_fraction=Decimal(values["max_account_drawdown_fraction"]),
        max_open_positions=values["max_open_positions"],
        max_exposure_per_underlying=Decimal(values["max_exposure_per_underlying"]),
        allowed_underlyings=frozenset(UnderlyingIndex(v) for v in values["allowed_underlyings"]),
        allowed_instrument_types=frozenset(InstrumentType(v) for v in values["allowed_instrument_types"]),
        allowed_horizons=frozenset(TradingHorizon(v) for v in values["allowed_horizons"]),
        allow_synthetic_data=values["allow_synthetic_data"],
        user_approval_required=True,
    )
    p = payload
    contract = DerivativeContract(
        Exchange(p.contract.exchange), UnderlyingIndex(p.contract.underlying),
        InstrumentType(p.contract.instrument_type), p.contract.expiry,
        p.contract.contract_id, p.contract.lot_size, p.contract.tick_size,
        p.contract.strike, OptionType(p.contract.option_type) if p.contract.option_type else None,
    )
    q = p.quote
    quote_type = FuturesQuote if contract.instrument_type is InstrumentType.FUT else OptionsQuote
    quote_response_type = FuturesQuoteResponse if contract.instrument_type is InstrumentType.FUT else OptionsQuoteResponse
    quote = quote_type(contract, q.timestamp, q.bid, q.ask, q.last_traded_price, q.volume, q.open_interest, q.change_in_open_interest)
    market_data = quote_response_type(quote, p.is_synthetic)
    direction = SignalDirection(p.direction)
    event = SignalEvent.ENTER_LONG if direction is SignalDirection.LONG else SignalEvent.ENTER_SHORT
    signal = StrategySignal(
        strategy_name=p.strategy_name, strategy_version=p.strategy_version,
        underlying=contract.underlying, input_symbol=contract.underlying.value,
        timeframe=timedelta(seconds=p.timeframe_seconds),
        observation_timestamp=p.observation_timestamp, direction=direction,
        events=(event,), status=SignalStatus.READY, warmup_status=WarmupStatus.SUFFICIENT,
        indicator_values=(), entry_reference_price=float(p.entry_reference_price),
        stop_reference_price=float(p.stop_reference_price), entry_conditions=(p.rationale,),
        exit_conditions=("Close when the owner submits a paper close proposal.",),
        stop_loss_method="owner-supplied stop reference; separately risk checked",
        input_requirements=("owner-supplied contract and typed quote",), reasons=(p.rationale,),
        missing_data=(), invalid_conditions=(), data_source=p.data_source,
        is_synthetic=p.is_synthetic, used_incomplete_candle=False,
        evaluation_at=p.evaluation_at,
    )
    proposal = ProposedTrade(
        contract=contract, signal=signal, horizon=TradingHorizon(p.horizon),
        structure=TradeStructure.SINGLE_LEG, market_data=market_data,
        market_data_source=p.data_source,
        entry_reference_price=p.entry_reference_price,
        stop_loss_reference_price=p.stop_reference_price,
        target_reference_price=p.target_reference_price, quantity=p.quantity,
        point_value=p.point_value, assumptions=("Owner supplied quote; source provenance is asserted by the owner.",),
    )
    portfolio = db.get(Portfolio, order.portfolio_id)
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    positions = db.scalars(select(Position).where(Position.portfolio_id == portfolio.id)).all()
    if positions:
        raise HTTPException(status_code=409, detail="Risk evaluation with existing positions requires reconciled mark-to-market data")
    account = AccountRiskSnapshot(
        current_equity=Decimal(portfolio.initial_cash),
        starting_daily_equity=Decimal(portfolio.initial_cash),
        peak_equity=Decimal(portfolio.initial_cash), daily_realised_pnl=Decimal(0),
        daily_unrealised_pnl=Decimal(0), open_position_count=0,
        exposure_by_underlying={underlying: Decimal(0) for underlying in UnderlyingIndex},
        snapshot_timestamp=datetime.now(timezone.utc),
    )
    sizing = size_position(PositionSizingRequest(policy=policy, account=account, proposal=proposal))
    decision = evaluate_pre_trade_risk(policy, account, proposal, evaluated_at=datetime.now(timezone.utc), user_approved=user_approved)
    return decision, sizing


@router.post("/proposals", status_code=201)
def create_proposal(payload: PaperProposalCreate, request: Request, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    portfolio = db.get(Portfolio, payload.portfolio_id)
    if portfolio is None or portfolio.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    now = datetime.now(timezone.utc)
    quote_age = (now - payload.quote.timestamp.astimezone(timezone.utc)).total_seconds()
    if quote_age < 0 or quote_age > request.app.state.settings.quote_max_age_seconds:
        raise HTTPException(status_code=422, detail="Quote is in the future or older than the allowed freshness window")
    active_count = db.scalar(select(func.count()).select_from(PaperOrder).where(
        PaperOrder.portfolio_id == portfolio.id,
        PaperOrder.status.in_(("pending_approval", "approved")),
    )) or 0
    if active_count:
        raise HTTPException(status_code=409, detail="Resolve the existing pending or approved paper order first")
    row = PaperOrder(
        id=str(uuid.uuid4()), portfolio_id=portfolio.id,
        symbol=payload.contract.contract_id, side=payload.direction,
        quantity=payload.quantity, order_type="paper", status="pending_approval",
        limit_price=payload.entry_reference_price,
        proposal_json=payload.model_dump(mode="json"),
    )
    db.add(row)
    db.flush()
    try:
        decision, sizing = _build_risk_inputs(row, db, user_approved=False)
    except (TypeError, ValueError) as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=f"Proposal risk inputs rejected: {exc}") from exc
    details = {"status": decision.status.value, "decision": to_jsonable(decision), "sizing": to_jsonable(sizing)}
    row.risk_decision_json = details
    db.add(RiskDecisionRecord(order_id=row.id, status=details["status"], policy_id=decision.policy_id, policy_version=decision.policy_version, details_json=details))
    _audit(db, auth.user.id, "paper.proposal.create", "paper_order", row.id, {"symbol": row.symbol, "status": row.status})
    db.commit()
    return {"id": row.id, "status": row.status, "risk_decision": details, "paper_only": True}


@router.get("/orders")
def list_orders(db: Session = Depends(get_db_session), auth: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(PaperOrder).join(Portfolio).where(Portfolio.owner_id == auth.user.id).order_by(PaperOrder.created_at.desc()).limit(250)).all()
    return [{"id": r.id, "symbol": r.symbol, "side": r.side, "quantity": r.quantity, "status": r.status, "limit_price": str(r.limit_price) if r.limit_price is not None else None, "risk_decision": r.risk_decision_json, "created_at": r.created_at} for r in rows]


@router.post("/orders/{order_id}/approve")
def approve_order(order_id: str, request: Request, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    row = db.get(PaperOrder, order_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Paper order not found")
    portfolio = db.get(Portfolio, row.portfolio_id)
    if portfolio is None or portfolio.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Paper order not found")
    if row.status != "pending_approval":
        raise HTTPException(status_code=409, detail="Order is not pending approval")
    other_active = db.scalar(select(func.count()).select_from(PaperOrder).where(
        PaperOrder.portfolio_id == row.portfolio_id,
        PaperOrder.id != row.id,
        PaperOrder.status.in_(("pending_approval", "approved")),
    )) or 0
    if other_active:
        raise HTTPException(status_code=409, detail="Another paper order is pending or approved")
    proposal_input = PaperProposalCreate.model_validate(row.proposal_json)
    age = (datetime.now(timezone.utc) - proposal_input.quote.timestamp.astimezone(timezone.utc)).total_seconds()
    if age < 0 or age > request.app.state.settings.quote_max_age_seconds:
        raise HTTPException(status_code=409, detail="Quote is stale; submit a new proposal with a fresh quote")
    try:
        decision, sizing = _build_risk_inputs(row, db, user_approved=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Risk inputs rejected: {exc}") from exc
    sized_exactly = sizing.status is SizingStatus.SIZED and sizing.sized_proposal is not None and sizing.sized_proposal.quantity == row.quantity
    details = {"status": decision.status.value if sized_exactly else "rejected", "decision": to_jsonable(decision), "sizing": to_jsonable(sizing)}
    row.risk_decision_json = details
    db.add(RiskDecisionRecord(order_id=row.id, status=details["status"], policy_id=decision.policy_id, policy_version=decision.policy_version, details_json=details))
    if not sized_exactly or decision.status.value != "approved" or decision.user_approved is not True:
        row.status = "risk_rejected"
        _audit(db, auth.user.id, "paper.approval.risk_rejected", "paper_order", row.id)
        db.commit()
        raise HTTPException(status_code=409, detail={"message": "Risk review rejected this paper proposal", "decision": details})
    row.status = "approved"
    row.user_approved_at = datetime.now(timezone.utc)
    _audit(db, auth.user.id, "paper.approval.accepted", "paper_order", row.id, {"risk_status": decision.status.value})
    db.commit()
    return {"id": row.id, "status": row.status, "risk_decision": details, "paper_only": True}


@router.post("/orders/{order_id}/reject")
def reject_order(order_id: str, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    row = db.get(PaperOrder, order_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Paper order not found")
    portfolio = db.get(Portfolio, row.portfolio_id)
    if portfolio is None or portfolio.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Paper order not found")
    if row.status != "pending_approval":
        raise HTTPException(status_code=409, detail="Order is not pending approval")
    row.status = "cancelled"
    _audit(db, auth.user.id, "paper.proposal.reject", "paper_order", row.id)
    db.commit()
    return {"id": row.id, "status": row.status}


@router.post("/orders/{order_id}/fill", status_code=201)
def fill_approved_order(
    order_id: str,
    payload: PaperFillCreate,
    request: Request,
    db: Session = Depends(get_db_session),
    auth: AuthContext = Depends(require_csrf),
) -> dict:
    row = db.get(PaperOrder, order_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Paper order not found")
    portfolio = db.get(Portfolio, row.portfolio_id)
    if portfolio is None or portfolio.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Paper order not found")
    if row.status != "approved":
        raise HTTPException(status_code=409, detail="Only an approved paper order can be filled")

    now = datetime.now(timezone.utc)
    age = (now - payload.timestamp.astimezone(timezone.utc)).total_seconds()
    if age < 0 or age > request.app.state.settings.quote_max_age_seconds:
        raise HTTPException(status_code=422, detail="Fill quote is in the future or stale")
    proposal = PaperProposalCreate.model_validate(row.proposal_json)
    if payload.data_source != proposal.data_source or payload.is_synthetic != proposal.is_synthetic:
        raise HTTPException(status_code=422, detail="Fill provenance must match the approved proposal")

    fill_price = payload.ask if row.side == "long" else payload.bid
    trade = Trade(
        portfolio_id=row.portfolio_id,
        order_id=row.id,
        symbol=row.symbol,
        side=row.side,
        quantity=row.quantity,
        fill_price=fill_price,
        fee=payload.fee,
        trade_type="paper",
        filled_at=payload.timestamp,
    )
    db.add(trade)

    position = db.scalar(
        select(Position).where(Position.portfolio_id == row.portfolio_id, Position.symbol == row.symbol)
    )
    if position is None:
        position = Position(
            id=str(uuid.uuid4()),
            portfolio_id=row.portfolio_id,
            symbol=row.symbol,
            side=row.side,
            quantity=row.quantity,
            entry_price=fill_price,
            mark_price=fill_price,
            updated_at=payload.timestamp,
        )
        db.add(position)
    else:
        if position.side != row.side:
            raise HTTPException(status_code=409, detail="Cannot reverse an existing position through this paper fill")
        old_quantity = Decimal(position.quantity)
        total_quantity = old_quantity + Decimal(row.quantity)
        weighted = (
            Decimal(position.entry_price) * old_quantity + fill_price * Decimal(row.quantity)
        ) / total_quantity
        position.quantity = int(total_quantity)
        position.entry_price = weighted
        position.mark_price = fill_price
        position.updated_at = payload.timestamp

    row.status = "filled"
    row.limit_price = fill_price
    _audit(
        db,
        auth.user.id,
        "paper.order.filled",
        "paper_order",
        row.id,
        {
            "fill_price": str(fill_price),
            "quantity": row.quantity,
            "data_source": payload.data_source,
            "is_synthetic": payload.is_synthetic,
        },
    )
    db.commit()
    return {
        "id": trade.id,
        "order_id": row.id,
        "symbol": row.symbol,
        "side": row.side,
        "quantity": row.quantity,
        "fill_price": str(fill_price),
        "fee": str(payload.fee),
        "trade_type": "paper",
    }


@router.post("/positions/{position_id}/close", status_code=201)
def close_paper_position(
    position_id: str,
    payload: PaperCloseCreate,
    request: Request,
    db: Session = Depends(get_db_session),
    auth: AuthContext = Depends(require_csrf),
) -> dict:
    position = db.get(Position, position_id)
    if position is None:
        raise HTTPException(status_code=404, detail="Paper position not found")
    portfolio = db.get(Portfolio, position.portfolio_id)
    if portfolio is None or portfolio.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Paper position not found")
    if payload.quantity > position.quantity:
        raise HTTPException(status_code=422, detail="Close quantity exceeds open position quantity")

    now = datetime.now(timezone.utc)
    age = (now - payload.timestamp.astimezone(timezone.utc)).total_seconds()
    if age < 0 or age > request.app.state.settings.quote_max_age_seconds:
        raise HTTPException(status_code=422, detail="Close quote is in the future or stale")

    order = db.scalar(
        select(PaperOrder).where(
            PaperOrder.portfolio_id == position.portfolio_id,
            PaperOrder.symbol == position.symbol,
            PaperOrder.status == "filled",
        ).order_by(PaperOrder.created_at.desc())
    )
    if order is None:
        raise HTTPException(status_code=409, detail="No filled paper order is available for this position")

    fill_price = payload.bid if position.side == "long" else payload.ask
    close_side = "short" if position.side == "long" else "long"
    trade = Trade(
        portfolio_id=position.portfolio_id,
        order_id=order.id,
        symbol=position.symbol,
        side=close_side,
        quantity=payload.quantity,
        fill_price=fill_price,
        fee=payload.fee,
        trade_type="paper",
        filled_at=payload.timestamp,
    )
    db.add(trade)

    remaining = position.quantity - payload.quantity
    if remaining == 0:
        db.delete(position)
    else:
        position.quantity = remaining
        position.mark_price = fill_price
        position.updated_at = payload.timestamp

    _audit(
        db,
        auth.user.id,
        "paper.position.closed",
        "position",
        position_id,
        {
            "closed_quantity": payload.quantity,
            "fill_price": str(fill_price),
            "data_source": payload.data_source,
            "is_synthetic": payload.is_synthetic,
        },
    )
    db.commit()
    return {
        "id": trade.id,
        "position_id": position_id,
        "symbol": position.symbol,
        "close_side": close_side,
        "quantity": payload.quantity,
        "fill_price": str(fill_price),
        "fee": str(payload.fee),
        "trade_type": "paper",
    }


@router.get("/portfolio/{portfolio_id}")
def portfolio_detail(portfolio_id: int, db: Session = Depends(get_db_session), auth: AuthContext = Depends(current_auth)) -> dict:
    row = db.get(Portfolio, portfolio_id)
    if row is None or row.owner_id != auth.user.id:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    positions = db.scalars(select(Position).where(Position.portfolio_id == row.id)).all()
    orders = db.scalars(select(PaperOrder).where(PaperOrder.portfolio_id == row.id)).all()
    return {"id": row.id, "name": row.name, "initial_cash": str(row.initial_cash), "trading_mode": "paper", "positions": [{"symbol": p.symbol, "side": p.side, "quantity": p.quantity, "entry_price": str(p.entry_price), "mark_price": str(p.mark_price) if p.mark_price is not None else None} for p in positions], "orders": len(orders), "mark_to_market": "unavailable until an authorized quote adapter is configured"}


@router.get("/trades")
def list_trades(db: Session = Depends(get_db_session), auth: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(Trade).join(Portfolio).where(Portfolio.owner_id == auth.user.id).order_by(Trade.filled_at.desc()).limit(500)).all()
    return [{"id": r.id, "symbol": r.symbol, "side": r.side, "quantity": r.quantity, "fill_price": str(r.fill_price), "fee": str(r.fee), "trade_type": r.trade_type, "filled_at": r.filled_at} for r in rows]


@router.post("/agents/runs", status_code=202)
def create_agent_run(payload: AgentRunCreate, db: Session = Depends(get_db_session), auth: AuthContext = Depends(require_csrf)) -> dict:
    row = AgentRun(id=str(uuid.uuid4()), agent_name=payload.agent_name, status="queued", input_json=payload.input, output_json=None, permission_scope="propose_only", budget_units=payload.budget_units)
    db.add(row)
    _audit(db, auth.user.id, "agent.run.queued", "agent_run", row.id, {"permission_scope": "propose_only"})
    db.commit()
    return {"id": row.id, "status": row.status, "permission_scope": row.permission_scope, "execution_enabled": False}


@router.get("/agents/runs")
def list_agent_runs(db: Session = Depends(get_db_session), _: AuthContext = Depends(current_auth)) -> list[dict]:
    rows = db.scalars(select(AgentRun).order_by(AgentRun.created_at.desc()).limit(100)).all()
    return [{"id": r.id, "agent_name": r.agent_name, "status": r.status, "permission_scope": r.permission_scope, "budget_units": r.budget_units, "created_at": r.created_at} for r in rows]


@router.get("/agents/runs/{run_id}/observations")
def agent_observations(run_id: str, db: Session = Depends(get_db_session), _: AuthContext = Depends(current_auth)) -> list[dict]:
    if db.get(AgentRun, run_id) is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    rows = db.scalars(select(AgentObservation).where(AgentObservation.run_id == run_id).order_by(AgentObservation.id)).all()
    return [{"id": o.id, "type": o.observation_type, "payload": o.payload_json, "created_at": o.created_at} for o in rows]


@router.get("/audit")
def audit_events(limit: int = 100, db: Session = Depends(get_db_session), _: AuthContext = Depends(current_auth)) -> list[dict]:
    limit = max(1, min(limit, 500))
    rows = db.scalars(select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)).all()
    return [{"id": r.id, "actor_user_id": r.actor_user_id, "action": r.action, "resource_type": r.resource_type, "resource_id": r.resource_id, "details": r.details_json, "created_at": r.created_at} for r in rows]
