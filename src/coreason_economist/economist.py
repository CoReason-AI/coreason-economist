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

from coreason_economist.budget_authority import BudgetAuthority
from coreason_economist.exceptions import BudgetExhaustedError
from coreason_economist.models import Decision, EconomicTrace, RequestPayload
from coreason_economist.pricer import Pricer


class Economist:
    """
    The Cognitive "CFO" and Optimization Engine.
    Orchestrates the components (Pricer, BudgetAuthority, etc.) to optimize
    resource allocation and enforce budgets.
    """

    def __init__(
        self,
        pricer: Optional[Pricer] = None,
        budget_authority: Optional[BudgetAuthority] = None,
    ) -> None:
        """
        Initialize the Economist.

        Args:
            pricer: Instance of Pricer. If None, creates a default one.
            budget_authority: Instance of BudgetAuthority. If None, creates a default one.
        """
        self.pricer = pricer if pricer is not None else Pricer()
        self.budget_authority = (
            budget_authority if budget_authority is not None else BudgetAuthority(pricer=self.pricer)
        )

    def check_execution(self, request: RequestPayload) -> EconomicTrace:
        """
        Evaluates a request against the budget and returns an execution decision.

        1. Estimates the cost of the request.
        2. Checks if the estimated cost is within the budget.
        3. Returns an EconomicTrace with the decision (APPROVED/REJECTED).
        """
        # 1. Estimate Cost
        # We estimate input tokens using char/4 heuristic as Pricer requires integer inputs
        # (Same logic as BudgetAuthority, but we need the estimate for the trace)
        estimated_cost = self.pricer.estimate_request_cost(
            model_name=request.model_name,
            input_tokens=len(request.prompt) // 4,
            output_tokens=request.estimated_output_tokens,
            tool_calls=request.tool_calls,
            agent_count=request.agent_count,
            rounds=request.rounds,
        )

        try:
            # 2. Check Budget
            self.budget_authority.allow_execution(request)

            # If no exception, it's approved
            return EconomicTrace(
                estimated_cost=estimated_cost,
                decision=Decision.APPROVED,
                model_used=request.model_name,
                reason="Budget check passed.",
            )

        except BudgetExhaustedError as e:
            # 3. Handle Rejection
            return EconomicTrace(
                estimated_cost=estimated_cost,
                decision=Decision.REJECTED,
                model_used=request.model_name,
                reason=str(e),
            )
