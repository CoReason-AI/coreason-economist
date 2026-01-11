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

from pydantic import BaseModel, ConfigDict, Field


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


class RequestPayload(BaseModel):
    """
    Payload for requesting execution permission or pricing.
    """

    model_name: str = Field(..., description="Name of the model to use (e.g., 'gpt-4')")
    prompt: str = Field(..., description="Input prompt text")
    estimated_output_tokens: Optional[int] = Field(None, description="Estimated number of output tokens", ge=0)
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="List of tool calls if any")
    max_budget: Optional[Budget] = Field(None, description="Maximum budget for this specific request")


class EconomicTrace(BaseModel):
    """
    Log object for every transaction.
    """

    estimated_cost: Budget = Field(..., description="Estimated cost before execution")
    actual_cost: Optional[Budget] = Field(None, description="Actual cost after execution")
    decision: Decision = Field(..., description="Decision made by the authority")
    voc_score: Optional[float] = Field(None, description="Value of Computation score", ge=0.0, le=1.0)
    model_used: str = Field(..., description="The model actually used")
    reason: Optional[str] = Field(None, description="Reason for the decision (e.g., 'BudgetExhausted')")


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
