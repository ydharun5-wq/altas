"""Normalized SQLAlchemy schema for local owner data and paper records."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atlas.web.database import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("id = 1", name="ck_single_owner"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    sessions: Mapped[list["AuthSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    __table_args__ = (Index("ix_auth_sessions_expires", "expires_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    csrf_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user: Mapped[User] = relationship(back_populates="sessions")


class LoginAttempt(Base):
    __tablename__ = "login_attempts"
    __table_args__ = (Index("ix_login_attempts_updated", "updated_at"),)

    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class Watchlist(Base):
    __tablename__ = "watchlists"
    __table_args__ = (UniqueConstraint("owner_id", "name", name="uq_watchlist_owner_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    items: Mapped[list["WatchlistItem"]] = relationship(back_populates="watchlist", cascade="all, delete-orphan")


class Instrument(Base):
    __tablename__ = "instruments"

    symbol: Mapped[str] = mapped_column(String(80), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(160), nullable=False)
    instrument_type: Mapped[str] = mapped_column(String(24), nullable=False)
    underlying: Mapped[str | None] = mapped_column(String(80))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"
    __table_args__ = (UniqueConstraint("watchlist_id", "symbol", name="uq_watchlist_symbol"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    watchlist_id: Mapped[int] = mapped_column(ForeignKey("watchlists.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(ForeignKey("instruments.symbol"), nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    watchlist: Mapped[Watchlist] = relationship(back_populates="items")


class MarketDataMetadata(Base):
    __tablename__ = "market_data_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(80), index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(160), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    first_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    details: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)


class StrategyRecord(Base):
    __tablename__ = "strategies"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_strategy_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(40), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    source: Mapped[str] = mapped_column(String(80), default="manual", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategies.id"))
    symbol: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    data_source: Mapped[str] = mapped_column(String(160), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False)
    config_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("backtest_runs.id", ondelete="CASCADE"), unique=True, nullable=False)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    trades_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    equity_curve_json: Mapped[list] = mapped_column(JSON, default=list, nullable=False)


class Portfolio(Base):
    __tablename__ = "portfolios"
    __table_args__ = (
        CheckConstraint("initial_cash > 0", name="ck_portfolio_cash_positive"),
        CheckConstraint("trading_mode = 'paper'", name="ck_portfolio_paper_only"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    initial_cash: Mapped[float] = mapped_column(Numeric(20, 4), nullable=False)
    trading_mode: Mapped[str] = mapped_column(String(12), default="paper", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class Position(Base):
    __tablename__ = "positions"
    __table_args__ = (CheckConstraint("quantity > 0", name="ck_position_quantity_positive"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(80), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    mark_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class PaperOrder(Base):
    __tablename__ = "paper_orders"
    __table_args__ = (
        CheckConstraint("order_type = 'paper'", name="ck_order_paper_only"),
        CheckConstraint("quantity > 0", name="ck_order_quantity_positive"),
        CheckConstraint("status IN ('pending_approval', 'risk_rejected', 'approved', 'filled', 'cancelled')", name="ck_paper_order_status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(80), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    order_type: Mapped[str] = mapped_column(String(12), default="paper", nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="pending_approval", nullable=False)
    user_approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    limit_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    risk_decision_json: Mapped[dict | None] = mapped_column(JSON)
    proposal_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class Trade(Base):
    __tablename__ = "trades"
    __table_args__ = (CheckConstraint("trade_type = 'paper'", name="ck_trade_paper_only"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    order_id: Mapped[str] = mapped_column(ForeignKey("paper_orders.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(80), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    fill_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    fee: Mapped[float] = mapped_column(Numeric(20, 8), default=0, nullable=False)
    trade_type: Mapped[str] = mapped_column(String(12), default="paper", nullable=False)
    filled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class RiskDecisionRecord(Base):
    __tablename__ = "risk_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[str | None] = mapped_column(ForeignKey("paper_orders.id", ondelete="SET NULL"))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    policy_id: Mapped[str] = mapped_column(String(100), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(40), nullable=False)
    details_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class RiskPolicyConfiguration(Base):
    __tablename__ = "risk_policy_configuration"
    __table_args__ = (CheckConstraint("id = 1", name="ck_single_risk_policy"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (CheckConstraint("status IN ('queued', 'running', 'completed', 'failed', 'cancelled')", name="ck_agent_status"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", nullable=False)
    input_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    output_json: Mapped[dict | None] = mapped_column(JSON)
    permission_scope: Mapped[str] = mapped_column(String(32), default="propose_only", nullable=False)
    budget_units: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentObservation(Base):
    __tablename__ = "agent_observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False)
    observation_type: Mapped[str] = mapped_column(String(80), nullable=False)
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_created", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(80), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(100))
    details_json: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
