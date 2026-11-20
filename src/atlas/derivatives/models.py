"""Immutable contract and market quote models for index derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum

from atlas.derivatives.validation import validate_contract_fields, validate_quote_fields


class Exchange(str, Enum):
    NSE = "NSE"
    BSE = "BSE"


class UnderlyingIndex(str, Enum):
    NIFTY = "NIFTY"
    BANKNIFTY = "BANKNIFTY"
    SENSEX = "SENSEX"


class InstrumentType(str, Enum):
    FUT = "FUT"
    OPT = "OPT"


class OptionType(str, Enum):
    CE = "CE"
    PE = "PE"


@dataclass(frozen=True)
class DerivativeContract:
    exchange: Exchange
    underlying: UnderlyingIndex
    instrument_type: InstrumentType
    expiry: date
    contract_id: str
    lot_size: int
    tick_size: Decimal
    strike: Decimal | None = None
    option_type: OptionType | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exchange, Exchange):
            raise ValueError("exchange must be an Exchange")
        if not isinstance(self.underlying, UnderlyingIndex):
            raise ValueError("underlying must be an UnderlyingIndex")
        if not isinstance(self.instrument_type, InstrumentType):
            raise ValueError("instrument_type must be an InstrumentType")
        if self.option_type is not None and not isinstance(self.option_type, OptionType):
            raise ValueError("option_type must be an OptionType")
        validate_contract_fields(
            exchange=self.exchange.value,
            underlying=self.underlying.value,
            instrument_type=self.instrument_type.value,
            expiry=self.expiry,
            contract_id=self.contract_id,
            lot_size=self.lot_size,
            tick_size=self.tick_size,
            strike=self.strike,
            option_type=self.option_type.value if self.option_type is not None else None,
        )


@dataclass(frozen=True)
class _MarketQuote:
    contract: DerivativeContract
    timestamp: datetime
    bid: Decimal | None
    ask: Decimal | None
    last_traded_price: Decimal | None
    volume: int
    open_interest: int
    change_in_open_interest: int

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        validate_quote_fields(
            timestamp=self.timestamp,
            bid=self.bid,
            ask=self.ask,
            last_traded_price=self.last_traded_price,
            volume=self.volume,
            open_interest=self.open_interest,
            change_in_open_interest=self.change_in_open_interest,
            tick_size=self.contract.tick_size,
        )


@dataclass(frozen=True)
class FuturesQuote(_MarketQuote):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.contract.instrument_type is not InstrumentType.FUT:
            raise ValueError("FuturesQuote requires a FUT contract")


@dataclass(frozen=True)
class OptionsQuote(_MarketQuote):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.contract.instrument_type is not InstrumentType.OPT:
            raise ValueError("OptionsQuote requires an OPT contract")


@dataclass(frozen=True)
class FuturesQuoteRequest:
    """Request a futures quote no later than ``as_of``."""

    contract: DerivativeContract
    as_of: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        if self.contract.instrument_type is not InstrumentType.FUT:
            raise ValueError("FuturesQuoteRequest requires a FUT contract")
        if not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")


@dataclass(frozen=True)
class OptionsQuoteRequest:
    """Request an options quote no later than ``as_of``."""

    contract: DerivativeContract
    as_of: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        if self.contract.instrument_type is not InstrumentType.OPT:
            raise ValueError("OptionsQuoteRequest requires an OPT contract")
        if not isinstance(self.as_of, datetime) or self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
