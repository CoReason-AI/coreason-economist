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

    def __init__(
        self,
        rates: Optional[Dict[str, ModelRate]] = None,
        heuristic_multiplier: float = 0.2,
    ) -> None:
        """
        Initialize the Pricer with a rate registry and heuristic settings.
        If no rates are provided, uses the default registry.
        """
        self.rates = rates if rates is not None else DEFAULT_MODEL_RATES
        self.heuristic_multiplier = heuristic_multiplier

    def estimate_financial_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the financial cost for a given model and token counts.
        Raises ValueError if model is unknown or if token counts are negative.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        if model_name not in self.rates:
            raise ValueError(f"Unknown model: {model_name}")

        rate = self.rates[model_name]

        input_cost = (input_tokens / 1000.0) * rate.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * rate.output_cost_per_1k

        return input_cost + output_cost

    def estimate_latency_ms(self, model_name: str, output_tokens: int) -> float:
        """
        Estimates latency based on output token count.
        """
        if output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        if model_name not in self.rates:
            raise ValueError(f"Unknown model: {model_name}")

        rate = self.rates[model_name]
        return float(output_tokens) * rate.latency_ms_per_output_token

    def estimate_request_cost(self, model_name: str, input_tokens: int, output_tokens: Optional[int] = None) -> Budget:
        """
        Creates a Budget object representing the estimated cost.
        If output_tokens is None, uses heuristics to estimate it.
        """
        if input_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        if output_tokens is None:
            # Heuristic: output is a fraction of input
            estimated_output = int(input_tokens * self.heuristic_multiplier)
            # Ensure at least 1 token if input > 0 to be safe, or 0 if input is 0
            if input_tokens > 0 and estimated_output == 0:
                estimated_output = 1
            output_tokens = estimated_output
        elif output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        financial_cost = self.estimate_financial_cost(model_name, input_tokens, output_tokens)
        latency_cost = self.estimate_latency_ms(model_name, output_tokens)
        total_tokens = input_tokens + output_tokens

        return Budget(
            financial=financial_cost,
            token_volume=total_tokens,
            latency_ms=latency_cost,
        )
