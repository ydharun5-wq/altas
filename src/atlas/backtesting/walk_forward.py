"""Date-separated walk-forward windows, without a parameter-selection API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from atlas.market_data.models import HistoricalOHLCVRequest


@dataclass(frozen=True)
class WalkForwardSplit:
    """One non-overlapping training interval followed by a test interval.

    The training bars may be supplied as indicator warm-up for a test run.
    BacktestEngine begins signal evaluation at test_start and does not evaluate
    or trade the training interval.
    """

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        for name in ("train_start", "train_end", "test_start", "test_end"):
            if not isinstance(getattr(self, name), datetime):
                raise ValueError(f"{name} must be a datetime")
        try:
            valid_training = self.train_start <= self.train_end
            separated = self.train_end < self.test_start
            valid_test = self.test_start <= self.test_end
        except TypeError as exc:
            raise ValueError("walk-forward timestamps must be comparable") from exc
        if not valid_training:
            raise ValueError("train_start must not follow train_end")
        if not separated:
            raise ValueError("training and test intervals must not overlap")
        if not valid_test:
            raise ValueError("test_start must not follow test_end")

    def request(self, symbol: str) -> HistoricalOHLCVRequest:
        """Request training history plus test bars for warm-up and evaluation."""
        return HistoricalOHLCVRequest(
            symbol=symbol,
            start=self.train_start,
            end=self.test_end,
        )


@dataclass(frozen=True)
class WalkForwardPlan:
    """Validated collection of chronological, non-overlapping test windows.

    No test results or parameter-selection callbacks are accepted by this
    model. Training intervals are also rejected if they intersect any test
    interval in this plan, so a test window cannot become another fold's
    training sample through this interface.
    """

    splits: tuple[WalkForwardSplit, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.splits, tuple) or not self.splits:
            raise ValueError("splits must be a non-empty tuple")
        if any(not isinstance(split, WalkForwardSplit) for split in self.splits):
            raise ValueError("splits must contain WalkForwardSplit values")
        for previous, current in zip(self.splits, self.splits[1:]):
            try:
                ordered = current.test_start > previous.test_end
            except TypeError as exc:
                raise ValueError("walk-forward windows must use comparable timestamps") from exc
            if not ordered:
                raise ValueError("test windows must be chronological and non-overlapping")

        for train_split in self.splits:
            for test_split in self.splits:
                if _intervals_overlap(
                    train_split.train_start,
                    train_split.train_end,
                    test_split.test_start,
                    test_split.test_end,
                ):
                    raise ValueError(
                        "no test interval may overlap any training interval in the plan"
                    )


def _intervals_overlap(
    first_start: datetime,
    first_end: datetime,
    second_start: datetime,
    second_end: datetime,
) -> bool:
    try:
        return first_start <= second_end and second_start <= first_end
    except TypeError as exc:
        raise ValueError("walk-forward intervals must use comparable timestamps") from exc


__all__ = ["WalkForwardPlan", "WalkForwardSplit"]
