# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Dict, Optional

from coreason_economist.models import Budget
from coreason_economist.rates import DEFAULT_MODEL_RATES, ModelRate


class Pricer:
    """
    The Estimator: Calculates the cost of an action before it happens.
    """

    def __init__(self, rates: Optional[Dict[str, ModelRate]] = None) -> None:
        """
        Initialize the Pricer with a rate registry.
        If no rates are provided, uses the default registry.
        """
        self.rates = rates if rates is not None else DEFAULT_MODEL_RATES

    def estimate_financial_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the financial cost for a given model and token counts.
        Raises ValueError if model is unknown.
        """
        if model_name not in self.rates:
            raise ValueError(f"Unknown model: {model_name}")

        rate = self.rates[model_name]

        input_cost = (input_tokens / 1000.0) * rate.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * rate.output_cost_per_1k

        return input_cost + output_cost

    def estimate_request_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> Budget:
        """
        Creates a Budget object representing the estimated cost.
        """
        cost = self.estimate_financial_cost(model_name, input_tokens, output_tokens)
        total_tokens = input_tokens + output_tokens

        return Budget(
            financial=cost,
            token_volume=total_tokens,
            # Latency is not calculated in this iteration
            latency_ms=0.0,
        )
