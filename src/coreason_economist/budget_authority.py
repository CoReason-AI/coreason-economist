# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import List, Optional

from coreason_economist.exceptions import BudgetExhaustedError
from coreason_economist.models import AuthResult, RequestPayload
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

    def allow_execution(self, request: RequestPayload) -> AuthResult:
        """
        Determines if the request is within the budget limits.
        Raises BudgetExhaustedError if limits are exceeded.
        Returns AuthResult with allowed=True and potential warnings.
        """
        # If no budget constraints are defined, we allow execution (unlimited)
        if request.max_budget is None:
            return AuthResult(allowed=True, warning=False, message=None)

        # Estimate the cost of the request
        estimated_cost = self.pricer.estimate_request_cost(
            model_name=request.model_name,
            input_tokens=len(request.prompt) // 4,
            output_tokens=request.estimated_output_tokens,
            tool_calls=request.tool_calls,
            agent_count=request.agent_count,
            rounds=request.rounds,
        )

        max_budget = request.max_budget
        warnings: List[str] = []

        # Helper to check limits and generate warnings
        def check_limit(est_val: float, limit_val: float, name: str, unit: str = "") -> None:
            # 1. Check Hard Limit (Strict: limit=0 implies nothing allowed if cost > 0)
            if est_val > limit_val:
                raise BudgetExhaustedError(
                    message=(f"{name} budget exceeded: estimated {est_val}{unit} > limit {limit_val}{unit}"),
                    limit_type=name.lower(),
                    limit_value=limit_val,
                    estimated_value=est_val,
                )

            # 2. Check Soft Limit Warning (Only if limit > 0)
            if limit_val > 0:
                ratio = est_val / limit_val
                if ratio > request.soft_limit_threshold:
                    pct = ratio * 100
                    warnings.append(f"{name} budget at {pct:.1f}% ({est_val}{unit}/{limit_val}{unit})")

        # Check Financial Budget
        check_limit(estimated_cost.financial, max_budget.financial, "Financial", "$")

        # Check Latency Budget
        check_limit(estimated_cost.latency_ms, max_budget.latency_ms, "Latency", "ms")

        # Check Token Volume Budget
        check_limit(float(estimated_cost.token_volume), float(max_budget.token_volume), "Token volume")

        if warnings:
            return AuthResult(
                allowed=True,
                warning=True,
                message="; ".join(warnings),
            )

        return AuthResult(allowed=True, warning=False, message=None)
