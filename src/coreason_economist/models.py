# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Decision(str, Enum):
    """Decision made by the Budget Authority."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"


class Budget(BaseModel):
    """
    Represents the budget constraints or costs in three currencies.
    """

    financial: float = Field(0.0, description="Hard dollar limit or cost (USD)", ge=0.0)
    latency_ms: float = Field(0.0, description="Time budget or duration in milliseconds", ge=0.0)
    token_volume: int = Field(0, description="Context window limit or token count", ge=0)

    model_config = ConfigDict(frozen=True)


class BudgetVariance(BaseModel):
    """
    Represents the difference between two budgets (Actual - Estimated).
    Values can be negative (under budget).
    """

    financial_delta: float = Field(..., description="Actual - Estimated financial cost")
    latency_ms_delta: float = Field(..., description="Actual - Estimated latency")
    token_volume_delta: int = Field(..., description="Actual - Estimated token volume")

    model_config = ConfigDict(frozen=True)


class RequestPayload(BaseModel):
    """
    Payload for requesting execution permission or pricing.
    """

    model_name: str = Field(..., description="Name of the model to use (e.g., 'gpt-4')")
    prompt: str = Field(..., description="Input prompt text")
    estimated_output_tokens: Optional[int] = Field(None, description="Estimated number of output tokens", ge=0)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="List of tool calls if any")
    max_budget: Optional[Budget] = Field(None, description="Maximum budget for this specific request")
    difficulty_score: Optional[float] = Field(
        None, description="Caller-provided difficulty score (0.0 to 1.0)", ge=0.0, le=1.0
    )
    agent_count: int = Field(1, description="Number of agents participating", ge=1)
    rounds: int = Field(1, description="Number of rounds of execution", ge=1)
    quality_warning: Optional[str] = Field(
        None, description="Warning if the request was downgraded or modified to fit budget"
    )
    soft_limit_threshold: float = Field(
        0.8, description="Usage threshold (0.0 to 1.0) to trigger a warning", ge=0.0, le=1.0
    )


class EconomicTrace(BaseModel):
    """
    Log object for every transaction.
    Includes computed efficiency metrics for dashboard observability.
    """

    estimated_cost: Budget = Field(..., description="Estimated cost before execution")
    actual_cost: Optional[Budget] = Field(None, description="Actual cost after execution")
    decision: Decision = Field(..., description="Decision made by the authority")
    voc_score: Optional[float] = Field(None, description="Value of Computation score", ge=0.0, le=1.0)
    model_used: str = Field(..., description="The model actually used")
    reason: Optional[str] = Field(None, description="Reason for the decision (e.g., 'BudgetExhausted')")
    suggested_alternative: Optional[RequestPayload] = Field(
        None, description="Alternative configuration suggested by Arbitrageur"
    )
    input_tokens: int = Field(..., description="Number of input tokens used for estimation", ge=0)
    budget_warning: bool = Field(False, description="True if budget soft limit was exceeded")
    warning_message: Optional[str] = Field(None, description="Details of soft limit warning")

    @property
    def _effective_cost(self) -> Budget:
        """Internal helper to get the cost used for metrics (Actual > Estimated)."""
        return self.actual_cost if self.actual_cost else self.estimated_cost

    @computed_field(return_type=float)  # type: ignore[misc]
    @property
    def tokens_per_dollar(self) -> float:
        """
        Calculates financial efficiency: Total Tokens / Financial Cost.
        """
        cost = self._effective_cost
        if cost.financial <= 0:
            return 0.0
        return float(cost.token_volume) / cost.financial

    @computed_field(return_type=float)  # type: ignore[misc]
    @property
    def tokens_per_second(self) -> float:
        """
        Calculates speed efficiency: Total Tokens / (Latency ms / 1000).
        """
        cost = self._effective_cost
        if cost.latency_ms <= 0:
            return 0.0
        seconds = cost.latency_ms / 1000.0
        return float(cost.token_volume) / seconds

    @computed_field(return_type=float)  # type: ignore[misc]
    @property
    def latency_per_token(self) -> float:
        """
        Calculates latency per token: Latency ms / Total Tokens.
        """
        cost = self._effective_cost
        if cost.token_volume <= 0:
            return 0.0
        return cost.latency_ms / float(cost.token_volume)

    @computed_field(return_type=float)  # type: ignore[misc]
    @property
    def cost_per_insight(self) -> float:
        """
        Exposes the raw financial cost as 'Cost per Insight' for dashboarding.
        Returns the financial cost of the transaction (Actual > Estimated).
        """
        return self._effective_cost.financial


class AuthResult(BaseModel):
    """
    Result returned by the Budget Authority.
    """

    allowed: bool = Field(..., description="Whether the request is allowed")
    warning: bool = Field(False, description="Whether a soft limit warning was triggered")
    message: Optional[str] = Field(None, description="Warning or error message")

    model_config = ConfigDict(frozen=True)


class CalibrationResult(BaseModel):
    """
    Result of the reconciliation process, including budget variance and heuristic updates.
    """

    variance: BudgetVariance = Field(..., description="Difference between actual and estimated budget")
    observed_multiplier: float = Field(..., description="Actual output/input ratio observed", ge=0.0)
    recommended_multiplier: float = Field(..., description="Recommended value for heuristic multiplier", ge=0.0)

    model_config = ConfigDict(frozen=True)


class VOCDecision(str, Enum):
    """Decision made by the VOC Engine."""

    CONTINUE = "CONTINUE"
    STOP = "STOP"


class ReasoningTrace(BaseModel):
    """
    Represents a chronological sequence of reasoning steps or outputs.
    Used by the VOC Engine to detect diminishing returns.
    """

    steps: List[str] = Field(..., description="List of text outputs from previous reasoning steps")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional context about the trace")


class VOCResult(BaseModel):
    """
    Result of a Value of Computation evaluation.
    """

    decision: VOCDecision = Field(..., description="Recommendation to stop or continue")
    score: float = Field(..., description="Calculated similarity or utility score (0.0 to 1.0)", ge=0.0, le=1.0)
    reason: str = Field(..., description="Explanation for the decision")

    model_config = ConfigDict(frozen=True)


class AuthorizeRequest(BaseModel):
    """Payload for budget authorization."""

    project_id: str = Field(..., description="Unique identifier for the project")
    estimated_cost: float = Field(..., gt=0.0, description="Estimated financial cost to reserve")


class AuthorizeResponse(BaseModel):
    """Response for budget authorization."""

    authorized: bool = Field(..., description="Whether the budget was authorized")
    transaction_id: Optional[str] = Field(None, description="Transaction ID if authorized")
    message: Optional[str] = Field(None, description="Error message if not authorized")


class CommitRequest(BaseModel):
    """Payload for budget settlement."""

    project_id: str = Field(..., description="Unique identifier for the project")
    estimated_cost: float = Field(..., description="Original estimated cost reserved")
    actual_cost: float = Field(..., description="Actual cost incurred")


class VocAnalyzeRequest(BaseModel):
    """Payload for VoC analysis."""

    task_complexity: float = Field(..., ge=0.0, le=1.0, description="Complexity of the task")
    current_uncertainty: float = Field(..., ge=0.0, le=1.0, description="Current uncertainty or ambiguity")


class VocAnalyzeResponse(BaseModel):
    """Response for VoC analysis."""

    should_execute: bool = Field(..., description="Recommendation to execute or not")
    max_allowable_cost: float = Field(..., description="Maximum budget allowed based on VoC")
