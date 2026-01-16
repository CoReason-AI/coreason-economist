# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


class ModelRate(BaseModel):
    """
    Rate card for a specific model.
    Prices are in USD per 1,000 tokens.
    Latency is in milliseconds per output token.
    """

    input_cost_per_1k: float = Field(..., description="Cost per 1,000 input tokens", ge=0.0)
    output_cost_per_1k: float = Field(..., description="Cost per 1,000 output tokens", ge=0.0)
    latency_ms_per_output_token: float = Field(..., description="Estimated latency per output token generated", ge=0.0)

    model_config = ConfigDict(frozen=True)


class ToolRate(BaseModel):
    """
    Rate card for a specific tool.
    Prices are in USD per call.
    """

    cost_per_call: float = Field(..., description="Cost per single execution of the tool", ge=0.0)

    model_config = ConfigDict(frozen=True)


# Default rates (approximate as of late 2024/early 2025)
DEFAULT_MODEL_RATES: Dict[str, ModelRate] = {
    "gpt-4o": ModelRate(
        input_cost_per_1k=0.005,  # $5.00 / 1M
        output_cost_per_1k=0.015,  # $15.00 / 1M
        latency_ms_per_output_token=12.0,  # ~80 tokens/sec -> 12.5ms
    ),
    "gpt-4o-mini": ModelRate(
        input_cost_per_1k=0.00015,  # $0.15 / 1M
        output_cost_per_1k=0.0006,  # $0.60 / 1M
        latency_ms_per_output_token=8.0,  # ~125 tokens/sec -> 8ms
    ),
    "claude-3-5-sonnet": ModelRate(
        input_cost_per_1k=0.003,  # $3.00 / 1M
        output_cost_per_1k=0.015,  # $15.00 / 1M
        latency_ms_per_output_token=15.0,  # Slower than GPT-4o
    ),
    "llama-3.1-70b": ModelRate(
        input_cost_per_1k=0.00088,  # $0.88 / 1M (Based on Together AI pricing)
        output_cost_per_1k=0.00088,  # $0.88 / 1M (Based on Together AI pricing)
        latency_ms_per_output_token=10.0,  # Fast inference
    ),
}

# Default tool rates
DEFAULT_TOOL_RATES: Dict[str, ToolRate] = {
    "web_search": ToolRate(cost_per_call=0.01),  # Premium search API
    "calculator": ToolRate(cost_per_call=0.0),  # Local computation
    "database_query": ToolRate(cost_per_call=0.005),  # Database access cost
}
