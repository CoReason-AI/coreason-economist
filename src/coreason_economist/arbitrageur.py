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

from coreason_economist.models import RequestPayload
from coreason_economist.rates import DEFAULT_MODEL_RATES, ModelRate


class Arbitrageur:
    """
    The Optimizer: Recommends cheaper alternatives based on difficulty.
    """

    def __init__(self, rates: Optional[Dict[str, ModelRate]] = None, threshold: float = 0.5) -> None:
        """
        Initialize the Arbitrageur.

        Args:
            rates: Registry of model rates. Defaults to DEFAULT_MODEL_RATES.
            threshold: Difficulty score threshold below which to suggest downgrades.
                       Default is 0.5.
        """
        self.rates = rates if rates is not None else DEFAULT_MODEL_RATES
        self.threshold = threshold

    def recommend_alternative(self, request: RequestPayload) -> Optional[RequestPayload]:
        """
        Analyzes the request and recommends a cheaper model if appropriate.

        Returns:
            A new RequestPayload with the modified model_name, or None if no
            recommendation is made.
        """
        # If no difficulty score is provided, we assume it's hard / unknown, so do nothing.
        if request.difficulty_score is None:
            return None

        # If difficulty is high enough, we trust the caller's choice.
        if request.difficulty_score >= self.threshold:
            return None

        # If the requested model is unknown, we can't compare prices.
        if request.model_name not in self.rates:
            return None

        current_rate = self.rates[request.model_name]
        current_cost_index = current_rate.input_cost_per_1k + current_rate.output_cost_per_1k

        # Find the cheapest available model in the registry
        # We define "cheapest" by the sum of input and output rates for simplicity.
        # Ideally, we might weight this by the specific prompt length, but this is a heuristic.
        sorted_models = sorted(
            self.rates.items(),
            key=lambda item: item[1].input_cost_per_1k + item[1].output_cost_per_1k,
        )

        # Get the absolute cheapest model
        cheapest_name, cheapest_rate = sorted_models[0]
        cheapest_cost_index = cheapest_rate.input_cost_per_1k + cheapest_rate.output_cost_per_1k

        # If the cheapest model is significantly cheaper than current, recommend it.
        # "Significantly" here just means strictly less expensive.
        if cheapest_cost_index < current_cost_index:
            # Create a new payload with the updated model name
            return request.model_copy(update={"model_name": cheapest_name})

        return None
