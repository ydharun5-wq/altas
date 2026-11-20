"""Validated JSON payloads for the owner-only ATLAS API."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BootstrapRequest(InputModel):
    bootstrap_token: SecretStr
    username: str = Field(min_length=3, max_length=80)
    password: SecretStr = Field(min_length=12, max_length=256)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        value = value.strip().lower()
        if not value or any(char.isspace() for char in value):
            raise ValueError("username must not contain whitespace")
        return value


class LoginRequest(InputModel):
    username: str = Field(min_length=1, max_length=80)
    password: SecretStr = Field(min_length=1, max_length=256)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip().lower()


class SessionResponse(BaseModel):
    authenticated: bool = True
    username: str
    csrf_token: str
    expires_at: datetime


class WatchlistCreate(InputModel):
    name: str = Field(min_length=1, max_length=100)


class InstrumentCreate(InputModel):
    symbol: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z0-9._-]+$")
    display_name: str = Field(min_length=1, max_length=160)
    instrument_type: str = Field(default="equity", pattern=r"^(equity|future|option|index|other)$")
    underlying: str | None = Field(default=None, max_length=80)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StrategyCreate(InputModel):
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=40)
    config: dict[str, Any] = Field(default_factory=dict)


class BacktestRequest(InputModel):
    strategy: str = Field(pattern=r"^(ema_trend|rsi_mean_reversion)$")
    symbol: str = Field(min_length=1, max_length=80, pattern=r"^[A-Z0-9._-]+$")
    underlying: str = Field(min_length=1, max_length=40)
    start: datetime
    end: datetime
    last_bar_complete: bool
    timeframe_seconds: int = Field(gt=0, le=604800)
    initial_capital: float = Field(gt=0, allow_inf_nan=False)
    quantity: int = Field(gt=0, le=1000000)
    unit_multiplier: float = Field(gt=0, allow_inf_nan=False)
    fast_period: int = Field(default=10, ge=2, le=500)
    slow_period: int = Field(default=30, ge=3, le=1000)
    spread_bps: float = Field(default=0, ge=0, allow_inf_nan=False)
    slippage_bps: float = Field(default=0, ge=0, allow_inf_nan=False)
    fee_per_execution: float = Field(default=0, ge=0, allow_inf_nan=False)

    @field_validator("start", "end")
    @classmethod
    def backtest_bounds_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("backtest bounds must be timezone-aware")
        return value


class DerivativeContractInput(InputModel):
    exchange: str = Field(pattern=r"^(NSE|BSE)$")
    underlying: str = Field(pattern=r"^(NIFTY|BANKNIFTY|SENSEX)$")
    instrument_type: str = Field(pattern=r"^(FUT|OPT)$")
    expiry: date
    contract_id: str = Field(min_length=1, max_length=120)
    lot_size: int = Field(gt=0, le=100000)
    tick_size: Decimal = Field(gt=0)
    strike: Decimal | None = Field(default=None, gt=0)
    option_type: str | None = Field(default=None, pattern=r"^(CE|PE)$")


class ProposalQuoteInput(InputModel):
    timestamp: datetime
    bid: Decimal | None = Field(default=None, gt=0)
    ask: Decimal | None = Field(default=None, gt=0)
    last_traded_price: Decimal | None = Field(default=None, gt=0)
    volume: int = Field(default=0, ge=0)
    open_interest: int = Field(default=0, ge=0)
    change_in_open_interest: int = 0

    @field_validator("timestamp")
    @classmethod
    def timestamp_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quote timestamp must be timezone-aware")
        return value


class PaperProposalCreate(InputModel):
    portfolio_id: int = Field(gt=0)
    contract: DerivativeContractInput
    quote: ProposalQuoteInput
    data_source: str = Field(min_length=1, max_length=160)
    is_synthetic: bool
    direction: str = Field(pattern=r"^(long|short)$")
    strategy_name: str = Field(min_length=1, max_length=120)
    strategy_version: str = Field(min_length=1, max_length=40)
    observation_timestamp: datetime
    evaluation_at: datetime
    timeframe_seconds: int = Field(gt=0, le=604800)
    entry_reference_price: Decimal = Field(gt=0)
    stop_reference_price: Decimal = Field(gt=0)
    target_reference_price: Decimal | None = Field(default=None, gt=0)
    quantity: int = Field(gt=0, le=1000000)
    point_value: Decimal = Field(gt=0)
    horizon: str = Field(pattern=r"^(intraday|swing)$")
    rationale: str = Field(min_length=1, max_length=2000)

    @field_validator("observation_timestamp", "evaluation_at")
    @classmethod
    def signal_timestamps_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("signal timestamps must be timezone-aware")
        return value


class RiskPolicyUpdate(InputModel):
    max_risk_per_trade_fraction: Decimal = Field(ge=0, le=1)
    max_daily_loss_fraction: Decimal = Field(ge=0, le=1)
    max_account_drawdown_fraction: Decimal = Field(ge=0, le=1)
    max_open_positions: int = Field(ge=0, le=10000)
    max_exposure_per_underlying: Decimal = Field(ge=0)
    allowed_underlyings: list[str]
    allowed_instrument_types: list[str]
    allowed_horizons: list[str]
    allow_synthetic_data: bool


class AgentRunCreate(InputModel):
    agent_name: str = Field(min_length=1, max_length=120)
    input: dict[str, Any]
    budget_units: int = Field(default=0, ge=0, le=100000)


class PaperFillCreate(InputModel):
    timestamp: datetime
    bid: Decimal = Field(gt=0)
    ask: Decimal = Field(gt=0)
    data_source: str = Field(min_length=1, max_length=160)
    is_synthetic: bool
    fee: Decimal = Field(default=Decimal("0"), ge=0)

    @field_validator("timestamp")
    @classmethod
    def timestamp_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fill timestamp must be timezone-aware")
        return value


class PaperCloseCreate(InputModel):
    quantity: int = Field(gt=0)
    timestamp: datetime
    bid: Decimal = Field(gt=0)
    ask: Decimal = Field(gt=0)
    data_source: str = Field(min_length=1, max_length=160)
    is_synthetic: bool
    fee: Decimal = Field(default=Decimal("0"), ge=0)

    @field_validator("timestamp")
    @classmethod
    def timestamp_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("close timestamp must be timezone-aware")
        return value

