"""Load and validate local OHLCV CSV data."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


REQUIRED_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


class DataValidationError(ValueError):
    """Raised when market data is malformed or internally inconsistent."""


@dataclass(frozen=True)
class OHLCVBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


def read_ohlcv_csv(path: str | Path) -> list[OHLCVBar]:
    """Read OHLCV bars from a CSV file, rejecting invalid data."""
    csv_path = Path(path)
    try:
        source = csv_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise DataValidationError(f"Cannot read OHLCV CSV '{csv_path}': {exc}") from exc

    with source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise DataValidationError("CSV is empty or has no header row")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise DataValidationError("CSV contains duplicate column names")

        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise DataValidationError(f"CSV is missing required columns: {', '.join(missing)}")

        bars: list[OHLCVBar] = []
        seen: set[tuple[str, datetime]] = set()
        last_timestamp: dict[str, datetime] = {}

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row[column] is None or not row[column].strip() for column in REQUIRED_COLUMNS):
                raise DataValidationError(f"Line {line_number}: missing or extra field")
            try:
                timestamp = datetime.fromisoformat(row["timestamp"].strip())
                symbol = row["symbol"].strip().upper()
                open_price = float(row["open"])
                high = float(row["high"])
                low = float(row["low"])
                close = float(row["close"])
                raw_volume = Decimal(row["volume"].strip())
            except (InvalidOperation, ValueError, OverflowError) as exc:
                raise DataValidationError(f"Line {line_number}: invalid timestamp or numeric value") from exc

            prices = (open_price, high, low, close)
            if not symbol:
                raise DataValidationError(f"Line {line_number}: symbol must not be empty")
            if not all(math.isfinite(value) and value > 0 for value in prices):
                raise DataValidationError(f"Line {line_number}: prices must be finite and positive")
            if (
                not raw_volume.is_finite()
                or raw_volume < 0
                or raw_volume != raw_volume.to_integral_value()
            ):
                raise DataValidationError(f"Line {line_number}: volume must be a non-negative integer")
            if high < max(open_price, low, close) or low > min(open_price, high, close):
                raise DataValidationError(f"Line {line_number}: inconsistent OHLC prices")

            key = (symbol, timestamp)
            if key in seen:
                raise DataValidationError(f"Line {line_number}: duplicate timestamp for {symbol}")
            if symbol in last_timestamp:
                try:
                    is_not_increasing = timestamp <= last_timestamp[symbol]
                except TypeError as exc:
                    raise DataValidationError(
                        f"Line {line_number}: timestamps for {symbol} must use comparable timezones"
                    ) from exc
                if is_not_increasing:
                    raise DataValidationError(f"Line {line_number}: timestamps for {symbol} must increase")

            seen.add(key)
            last_timestamp[symbol] = timestamp
            bars.append(
                OHLCVBar(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=int(raw_volume),
                )
            )

    if not bars:
        raise DataValidationError("CSV contains no OHLCV data rows")
    return bars
