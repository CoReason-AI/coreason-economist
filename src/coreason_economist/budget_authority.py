# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Optional

from coreason_economist.exceptions import BudgetExhaustedError
from coreason_economist.models import RequestPayload
from coreason_economist.pricer import Pricer


class BudgetAuthority:
    """
    The Controller: Enforces limits set by the parent application.
    """

    def __init__(self, pricer: Optional[Pricer] = None) -> None:
        """
        Initialize with a Pricer instance.
        If no pricer is provided, creates a default one.
        """
        self.pricer = pricer if pricer is not None else Pricer()

    def allow_execution(self, request: RequestPayload) -> bool:
        """
        Determines if the request is within the budget limits.
        Raises BudgetExhaustedError if limits are exceeded.
        Returns True if authorized.
        """
        # If no budget constraints are defined, we allow execution (unlimited)
        if request.max_budget is None:
            return True

        # Estimate the cost of the request
        # We estimate input tokens using char/4 heuristic as Pricer requires integer inputs
        estimated_cost = self.pricer.estimate_request_cost(
            model_name=request.model_name,
            input_tokens=len(request.prompt) // 4,
            output_tokens=request.estimated_output_tokens,
            tool_calls=request.tool_calls,
            agent_count=request.agent_count,
            rounds=request.rounds,
        )

        max_budget = request.max_budget

        # Check Financial Budget
        # We treat 0 as "no budget allocated", so if max is 0 and cost > 0, it fails.
        if estimated_cost.financial > max_budget.financial:
            raise BudgetExhaustedError(
                message=(
                    f"Financial budget exceeded: "
                    f"estimated ${estimated_cost.financial:.4f} > limit ${max_budget.financial:.4f}"
                ),
                limit_type="financial",
                limit_value=max_budget.financial,
                estimated_value=estimated_cost.financial,
            )

        # Check Latency Budget
        if estimated_cost.latency_ms > max_budget.latency_ms:
            raise BudgetExhaustedError(
                message=(
                    f"Latency budget exceeded: "
                    f"estimated {estimated_cost.latency_ms:.1f}ms > limit {max_budget.latency_ms:.1f}ms"
                ),
                limit_type="latency_ms",
                limit_value=max_budget.latency_ms,
                estimated_value=estimated_cost.latency_ms,
            )

        # Check Token Volume Budget
        if estimated_cost.token_volume > max_budget.token_volume:
            raise BudgetExhaustedError(
                message=(
                    f"Token volume budget exceeded: "
                    f"estimated {estimated_cost.token_volume} > limit {max_budget.token_volume}"
                ),
                limit_type="token_volume",
                limit_value=float(max_budget.token_volume),
                estimated_value=float(estimated_cost.token_volume),
            )

        return True
