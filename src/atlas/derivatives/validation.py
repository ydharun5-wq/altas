"""Validation helpers for derivatives contracts and market quotes."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal


_EXCHANGES = {"NSE", "BSE"}
_UNDERLYING_EXCHANGES = {
    "NIFTY": "NSE",
    "BANKNIFTY": "NSE",
    "SENSEX": "BSE",
}
_INSTRUMENT_TYPES = {"FUT", "OPT"}
_OPTION_TYPES = {"CE", "PE"}


def validate_contract_fields(
    *,
    exchange: str,
    underlying: str,
    instrument_type: str,
    expiry: date,
    contract_id: str,
    lot_size: int,
    tick_size: Decimal,
    strike: Decimal | None,
    option_type: str | None,
) -> None:
    if exchange not in _EXCHANGES:
        raise ValueError("exchange must be NSE or BSE")
    if underlying not in _UNDERLYING_EXCHANGES:
        raise ValueError("underlying must be NIFTY, BANKNIFTY, or SENSEX")
    if instrument_type not in _INSTRUMENT_TYPES:
        raise ValueError("instrument_type must be FUT or OPT")
    if exchange != _UNDERLYING_EXCHANGES[underlying]:
        raise ValueError(f"{underlying} contracts must use the { _UNDERLYING_EXCHANGES[underlying] } exchange")
    if not isinstance(expiry, date) or isinstance(expiry, datetime):
        raise ValueError("expiry must be a date")
    if (
        not isinstance(contract_id, str)
        or not contract_id
        or any(character.isspace() for character in contract_id)
    ):
        raise ValueError("contract_id must be a non-empty identifier without whitespace")
    validate_positive_integer("lot_size", lot_size)
    validate_decimal("tick_size", tick_size, allow_zero=False)

    if instrument_type == "FUT":
        if strike is not None or option_type is not None:
            raise ValueError("futures must not have a strike or option_type")
    else:
        validate_decimal("strike", strike, allow_zero=False)
        if option_type not in _OPTION_TYPES:
            raise ValueError("options require option_type CE or PE")


def validate_quote_fields(
    *,
    timestamp: datetime,
    bid: Decimal | None,
    ask: Decimal | None,
    last_traded_price: Decimal | None,
    volume: int,
    open_interest: int,
    change_in_open_interest: int,
    tick_size: Decimal,
) -> None:
    if not isinstance(timestamp, datetime):
        raise ValueError("timestamp must be a datetime")
    try:
        offset = timestamp.utcoffset()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("timestamp must have valid timezone information") from exc
    if offset is None:
        raise ValueError("timestamp must be timezone-aware")
    validate_quote_price("bid", bid, tick_size)
    validate_quote_price("ask", ask, tick_size)
    validate_quote_price("last_traded_price", last_traded_price, tick_size)
    if bid is not None and ask is not None and bid > ask:
        raise ValueError("bid must not exceed ask")
    validate_nonnegative_integer("volume", volume)
    validate_nonnegative_integer("open_interest", open_interest)
    validate_integer("change_in_open_interest", change_in_open_interest)


def validate_quote_price(
    name: str,
    value: Decimal | None,
    tick_size: Decimal,
) -> None:
    if value is None:
        return
    validate_decimal(name, value, allow_zero=True)
    if value % tick_size != 0:
        raise ValueError(f"{name} must be an exact multiple of contract tick_size")


def validate_decimal(name: str, value: Decimal | None, *, allow_zero: bool) -> None:
    if not isinstance(value, Decimal):
        raise ValueError(f"{name} must be a Decimal")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    if value < 0 or (not allow_zero and value == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def validate_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
