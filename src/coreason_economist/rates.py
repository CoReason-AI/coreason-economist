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
    """

    input_cost_per_1k: float = Field(..., description="Cost per 1,000 input tokens", ge=0.0)
    output_cost_per_1k: float = Field(..., description="Cost per 1,000 output tokens", ge=0.0)

    model_config = ConfigDict(frozen=True)


# Default rates (approximate as of late 2024/early 2025)
DEFAULT_MODEL_RATES: Dict[str, ModelRate] = {
    "gpt-4o": ModelRate(
        input_cost_per_1k=0.005,  # $5.00 / 1M
        output_cost_per_1k=0.015,  # $15.00 / 1M
    ),
    "gpt-4o-mini": ModelRate(
        input_cost_per_1k=0.00015,  # $0.15 / 1M
        output_cost_per_1k=0.0006,  # $0.60 / 1M
    ),
    "claude-3-5-sonnet": ModelRate(
        input_cost_per_1k=0.003,  # $3.00 / 1M
        output_cost_per_1k=0.015,  # $15.00 / 1M
    ),
}
