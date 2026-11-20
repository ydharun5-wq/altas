"""Deterministic descriptive analytics for paired European option quotes.

This module aggregates supplied CE/PE quote data. It does not infer missing
contracts, OI, volume, market prices, or strike intervals, and its comparison
text is descriptive only, never a trading signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from math import isfinite
from atlas.derivatives.models import (
    InstrumentType,
    OptionType,
    OptionsQuote,
    UnderlyingIndex,
)


class RatioStatus(str, Enum):
    """Whether a PCR has a finite denominator and can be reported."""

    AVAILABLE = "available"
    ZERO_DENOMINATOR = "zero_denominator"


@dataclass(frozen=True)
class PCRResult:
    """A put-call ratio and explicit status when its denominator is zero."""

    value: float | None
    status: RatioStatus

    def __post_init__(self) -> None:
        if not isinstance(self.status, RatioStatus):
            raise ValueError("status must be a RatioStatus")
        if self.status is RatioStatus.ZERO_DENOMINATOR:
            if self.value is not None:
                raise ValueError("zero-denominator PCR must not have a value")
        elif (
            isinstance(self.value, bool)
            or not isinstance(self.value, float)
            or not isfinite(self.value)
            or self.value < 0
        ):
            raise ValueError("available PCR must have a finite non-negative value")


@dataclass(frozen=True)
class OptionStrike:
    """One strike with its call and put quotes from the same contract series."""

    strike: Decimal
    call_quote: OptionsQuote
    put_quote: OptionsQuote

    def __post_init__(self) -> None:
        if (
            not isinstance(self.strike, Decimal)
            or not self.strike.is_finite()
            or self.strike <= 0
        ):
            raise ValueError("strike must be a finite positive Decimal")
        if not isinstance(self.call_quote, OptionsQuote):
            raise ValueError("call_quote must be an OptionsQuote")
        if not isinstance(self.put_quote, OptionsQuote):
            raise ValueError("put_quote must be an OptionsQuote")

        call_contract = self.call_quote.contract
        put_contract = self.put_quote.contract
        for contract, expected_type, name in (
            (call_contract, OptionType.CE, "call_quote"),
            (put_contract, OptionType.PE, "put_quote"),
        ):
            if contract.instrument_type is not InstrumentType.OPT:
                raise ValueError(f"{name} must reference an options contract")
            if contract.option_type is not expected_type:
                raise ValueError(f"{name} must reference a {expected_type.value} contract")
            if contract.strike != self.strike:
                raise ValueError(f"{name} contract strike does not match row strike")

        if (
            call_contract.exchange is not put_contract.exchange
            or call_contract.underlying is not put_contract.underlying
            or call_contract.expiry != put_contract.expiry
        ):
            raise ValueError("call and put contracts must share exchange, underlying, and expiry")


@dataclass(frozen=True)
class OptionChain:
    """Immutable chain snapshot with explicit source and synthetic-data status.

    Every listed strike must contain both CE and PE quotes. Each quote may
    still have missing bid, ask, or last-traded-price values; those remain
    ``None``. The quote models require OI and volume values, so absent quotes
    are rejected rather than counted as zero. Quote timestamps may differ,
    but none may be later than ``observed_at``. No maximum age is assumed.
    """

    underlying: UnderlyingIndex
    expiry: date
    observed_at: datetime
    data_source: str
    is_synthetic: bool
    strikes: tuple[OptionStrike, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.underlying, UnderlyingIndex):
            raise ValueError("underlying must be an UnderlyingIndex")
        if not isinstance(self.expiry, date) or isinstance(self.expiry, datetime):
            raise ValueError("expiry must be a date")
        _validate_aware_timestamp("observed_at", self.observed_at)
        if not isinstance(self.data_source, str) or not self.data_source.strip():
            raise ValueError("data_source must be a non-empty string")
        if not isinstance(self.is_synthetic, bool):
            raise ValueError("is_synthetic must be a bool")

        if isinstance(self.strikes, (str, bytes, bytearray)):
            raise ValueError("strikes must be an iterable of OptionStrike rows")
        try:
            rows = tuple(self.strikes)
        except TypeError as exc:
            raise ValueError("strikes must be an iterable of OptionStrike rows") from exc
        if not rows:
            raise ValueError("option chain must contain at least one strike")

        seen_strikes: set[Decimal] = set()
        seen_contract_ids: set[str] = set()
        for row in rows:
            if not isinstance(row, OptionStrike):
                raise ValueError("strikes must contain OptionStrike rows")
            if row.strike in seen_strikes:
                raise ValueError(f"duplicate option-chain strike: {row.strike}")
            seen_strikes.add(row.strike)

            for quote in (row.call_quote, row.put_quote):
                contract = quote.contract
                if contract.underlying is not self.underlying:
                    raise ValueError("option contract underlying does not match chain")
                if contract.expiry != self.expiry:
                    raise ValueError("option contract expiry does not match chain")
                if contract.contract_id in seen_contract_ids:
                    raise ValueError("duplicate option contract identifier in chain")
                seen_contract_ids.add(contract.contract_id)
                _validate_aware_timestamp("quote timestamp", quote.timestamp)
                try:
                    quote_is_future = quote.timestamp > self.observed_at
                except TypeError as exc:
                    raise ValueError(
                        "quote timestamp must be comparable to observed_at"
                    ) from exc
                if quote_is_future:
                    raise ValueError("quote timestamp must not be later than observed_at")

        object.__setattr__(self, "strikes", tuple(sorted(rows, key=lambda row: row.strike)))


@dataclass(frozen=True)
class OptionSideAnalytics:
    """Observed metrics for one option side; prices remain optional."""

    open_interest: int
    change_in_open_interest: int
    volume: int
    bid: Decimal | None
    ask: Decimal | None
    last_traded_price: Decimal | None


@dataclass(frozen=True)
class StrikeAnalytics:
    """CE/PE metrics and arithmetic comparisons for one strike.

    ``descriptive_observation`` explicitly describes the supplied data only;
    it is not a support/resistance claim, forecast, or trading signal.
    """

    strike: Decimal
    call: OptionSideAnalytics
    put: OptionSideAnalytics
    call_minus_put_open_interest: int
    call_minus_put_change_in_open_interest: int
    call_minus_put_volume: int
    descriptive_observation: str


@dataclass(frozen=True)
class OptionChainAnalytics:
    """Aggregate and strike-level descriptive metrics for one chain snapshot."""

    underlying: UnderlyingIndex
    expiry: date
    observed_at: datetime
    data_source: str
    is_synthetic: bool
    total_call_open_interest: int
    total_put_open_interest: int
    total_call_change_in_open_interest: int
    total_put_change_in_open_interest: int
    total_call_volume: int
    total_put_volume: int
    oi_pcr: PCRResult
    volume_pcr: PCRResult
    strikes: tuple[StrikeAnalytics, ...]


def analyze_option_chain(chain: OptionChain) -> OptionChainAnalytics:
    """Return OI, change-in-OI, volume, PCR, and strike comparisons.

    PCR is put total divided by call total. A zero call-side denominator is
    represented by ``value=None`` and ``RatioStatus.ZERO_DENOMINATOR``.
    """
    if not isinstance(chain, OptionChain):
        raise ValueError("chain must be an OptionChain")

    strike_summaries: list[StrikeAnalytics] = []
    call_oi = put_oi = 0
    call_change = put_change = 0
    call_volume = put_volume = 0

    for row in chain.strikes:
        call_quote = row.call_quote
        put_quote = row.put_quote
        call_metrics = OptionSideAnalytics(
            open_interest=call_quote.open_interest,
            change_in_open_interest=call_quote.change_in_open_interest,
            volume=call_quote.volume,
            bid=call_quote.bid,
            ask=call_quote.ask,
            last_traded_price=call_quote.last_traded_price,
        )
        put_metrics = OptionSideAnalytics(
            open_interest=put_quote.open_interest,
            change_in_open_interest=put_quote.change_in_open_interest,
            volume=put_quote.volume,
            bid=put_quote.bid,
            ask=put_quote.ask,
            last_traded_price=put_quote.last_traded_price,
        )

        call_oi += call_metrics.open_interest
        put_oi += put_metrics.open_interest
        call_change += call_metrics.change_in_open_interest
        put_change += put_metrics.change_in_open_interest
        call_volume += call_metrics.volume
        put_volume += put_metrics.volume

        oi_difference = call_metrics.open_interest - put_metrics.open_interest
        if oi_difference > 0:
            observation = (
                "Descriptive comparison only; not a trading signal: "
                f"CE open interest exceeds PE open interest by {oi_difference}."
            )
        elif oi_difference < 0:
            observation = (
                "Descriptive comparison only; not a trading signal: "
                f"PE open interest exceeds CE open interest by {-oi_difference}."
            )
        else:
            observation = (
                "Descriptive comparison only; not a trading signal: "
                "CE and PE open interest are equal."
            )

        strike_summaries.append(
            StrikeAnalytics(
                strike=row.strike,
                call=call_metrics,
                put=put_metrics,
                call_minus_put_open_interest=oi_difference,
                call_minus_put_change_in_open_interest=(
                    call_metrics.change_in_open_interest
                    - put_metrics.change_in_open_interest
                ),
                call_minus_put_volume=call_metrics.volume - put_metrics.volume,
                descriptive_observation=observation,
            )
        )

    return OptionChainAnalytics(
        underlying=chain.underlying,
        expiry=chain.expiry,
        observed_at=chain.observed_at,
        data_source=chain.data_source,
        is_synthetic=chain.is_synthetic,
        total_call_open_interest=call_oi,
        total_put_open_interest=put_oi,
        total_call_change_in_open_interest=call_change,
        total_put_change_in_open_interest=put_change,
        total_call_volume=call_volume,
        total_put_volume=put_volume,
        oi_pcr=_pcr(put_oi, call_oi),
        volume_pcr=_pcr(put_volume, call_volume),
        strikes=tuple(strike_summaries),
    )


def _pcr(put_total: int, call_total: int) -> PCRResult:
    if call_total == 0:
        return PCRResult(value=None, status=RatioStatus.ZERO_DENOMINATOR)
    try:
        ratio = put_total / call_total
    except OverflowError as exc:
        raise ValueError("PCR exceeds the supported numeric range") from exc
    if not isfinite(ratio):
        raise ValueError("PCR exceeds the supported numeric range")
    if put_total > 0 and ratio == 0.0:
        raise ValueError("PCR is below the supported numeric range")
    return PCRResult(value=ratio, status=RatioStatus.AVAILABLE)


def _validate_aware_timestamp(name: str, value: datetime) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    try:
        offset = value.utcoffset()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have valid timezone information") from exc
    if offset is None:
        raise ValueError(f"{name} must be timezone-aware")


__all__ = [
    "OptionChain",
    "OptionChainAnalytics",
    "OptionSideAnalytics",
    "OptionStrike",
    "PCRResult",
    "RatioStatus",
    "StrikeAnalytics",
    "analyze_option_chain",
]
