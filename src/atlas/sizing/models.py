"""Immutable request and result types for risk-budget position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from atlas.derivatives.models import DerivativeContract
from atlas.derivatives.validation import validate_nonnegative_integer
from atlas.risk.models import (
    AccountRiskSnapshot,
    ProposedTrade,
    RiskPolicy,
    RiskReasonCode,
)


class SizingStatus(str, Enum):
    SIZED = "sized"
    NO_TRADE = "no_trade"


class SizingReasonCode(str, Enum):
    NO_CONTRACT_WITHIN_RISK_BUDGET = "no_contract_within_risk_budget"
    REQUESTED_QUANTITY_CAPPED = "requested_quantity_capped"
    NUMERIC_RESULT_INVALID = "numeric_result_invalid"


SizingReasonCodeValue = RiskReasonCode | SizingReasonCode


@dataclass(frozen=True)
class PositionSizingRequest:
    """Inputs already represented by M8.1 immutable risk models.

    If ``proposal.quantity`` is supplied, it is treated as the caller's
    requested upper bound and is reduced when it exceeds risk capacity. If it
    is ``None``, the maximum risk-permitted quantity is selected.
    """

    policy: RiskPolicy
    account: AccountRiskSnapshot
    proposal: ProposedTrade

    def __post_init__(self) -> None:
        if not isinstance(self.policy, RiskPolicy):
            raise ValueError("policy must be a RiskPolicy")
        if not isinstance(self.account, AccountRiskSnapshot):
            raise ValueError("account must be an AccountRiskSnapshot")
        if not isinstance(self.proposal, ProposedTrade):
            raise ValueError("proposal must be a ProposedTrade")


@dataclass(frozen=True)
class SizingReason:
    code: SizingReasonCodeValue
    explanation: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, (RiskReasonCode, SizingReasonCode)):
            raise ValueError("code must be a sizing or risk reason code")
        if not isinstance(self.explanation, str) or not self.explanation.strip():
            raise ValueError("explanation must be a non-empty string")


@dataclass(frozen=True)
class PositionSizingResult:
    """Sizing output, never an approval or execution instruction."""

    status: SizingStatus
    contract: DerivativeContract
    policy_id: str
    policy_version: str
    requested_quantity: int | None
    risk_budget: Decimal | None
    risk_per_contract: Decimal | None
    maximum_permitted_contract_quantity: int | None
    selected_quantity: int
    total_estimated_risk: Decimal | None
    unused_risk_budget: Decimal | None
    sized_proposal: ProposedTrade | None
    reasons: tuple[SizingReason, ...]
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, SizingStatus):
            raise ValueError("status must be a SizingStatus")
        if not isinstance(self.contract, DerivativeContract):
            raise ValueError("contract must be a DerivativeContract")
        for name in ("policy_id", "policy_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.requested_quantity is not None:
            validate_nonnegative_integer("requested_quantity", self.requested_quantity)
            if self.requested_quantity == 0:
                raise ValueError("requested_quantity must be positive or None")
        if self.maximum_permitted_contract_quantity is not None:
            validate_nonnegative_integer(
                "maximum_permitted_contract_quantity",
                self.maximum_permitted_contract_quantity,
            )
        validate_nonnegative_integer("selected_quantity", self.selected_quantity)
        for name in (
            "risk_budget",
            "risk_per_contract",
            "total_estimated_risk",
            "unused_risk_budget",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, Decimal)
                or not value.is_finite()
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative Decimal or None")
        if self.sized_proposal is not None:
            if not isinstance(self.sized_proposal, ProposedTrade):
                raise ValueError("sized_proposal must be a ProposedTrade or None")
            if self.sized_proposal.contract != self.contract:
                raise ValueError("sized proposal must preserve contract identity")
            if self.sized_proposal.quantity != self.selected_quantity:
                raise ValueError("sized proposal quantity must match selected_quantity")
        if self.status is SizingStatus.SIZED:
            if self.selected_quantity <= 0 or self.sized_proposal is None:
                raise ValueError("SIZED results require a positive quantity and proposal")
            if self.total_estimated_risk is None or self.risk_budget is None:
                raise ValueError("SIZED results require risk and budget values")
            if self.risk_per_contract is None or self.risk_per_contract <= 0:
                raise ValueError("SIZED results require positive per-contract risk")
            if self.maximum_permitted_contract_quantity is None:
                raise ValueError("SIZED results require a calculated maximum quantity")
            if self.selected_quantity > self.maximum_permitted_contract_quantity:
                raise ValueError("selected quantity must not exceed the calculated maximum")
            if self.total_estimated_risk > self.risk_budget:
                raise ValueError("sized risk must not exceed the risk budget")
        else:
            if self.selected_quantity != 0 or self.sized_proposal is not None:
                raise ValueError("NO_TRADE results must not contain a sized proposal")
            if not self.reasons:
                raise ValueError("NO_TRADE results require at least one reason")
        if not isinstance(self.reasons, tuple) or any(
            not isinstance(item, SizingReason) for item in self.reasons
        ):
            raise ValueError("reasons must be a tuple of SizingReason values")
        if not isinstance(self.assumptions, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.assumptions
        ):
            raise ValueError("assumptions must be a tuple of non-empty strings")


__all__ = [
    "PositionSizingRequest",
    "PositionSizingResult",
    "SizingReason",
    "SizingReasonCode",
    "SizingReasonCodeValue",
    "SizingStatus",
]
