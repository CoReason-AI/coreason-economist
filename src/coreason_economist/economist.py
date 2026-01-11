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

from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.budget_authority import BudgetAuthority
from coreason_economist.calibration import calculate_budget_variance
from coreason_economist.exceptions import BudgetExhaustedError
from coreason_economist.models import (
    Budget,
    CalibrationResult,
    Decision,
    EconomicTrace,
    RequestPayload,
)
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
        arbitrageur: Optional[Arbitrageur] = None,
    ) -> None:
        """
        Initialize the Economist.

        Args:
            pricer: Instance of Pricer. If None, creates a default one.
            budget_authority: Instance of BudgetAuthority. If None, creates a default one.
            arbitrageur: Instance of Arbitrageur. If None, creates a default one.
        """
        self.pricer = pricer if pricer is not None else Pricer()
        self.budget_authority = (
            budget_authority if budget_authority is not None else BudgetAuthority(pricer=self.pricer)
        )
        self.arbitrageur = arbitrageur if arbitrageur is not None else Arbitrageur()

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
        input_tokens_est = len(request.prompt) // 4
        estimated_cost = self.pricer.estimate_request_cost(
            model_name=request.model_name,
            input_tokens=input_tokens_est,
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
                input_tokens=input_tokens_est,
            )

        except BudgetExhaustedError as e:
            # 3. Handle Rejection
            suggestion = self.arbitrageur.recommend_alternative(request)

            return EconomicTrace(
                estimated_cost=estimated_cost,
                decision=Decision.REJECTED,
                model_used=request.model_name,
                reason=str(e),
                suggested_alternative=suggestion,
                input_tokens=input_tokens_est,
            )

    def reconcile(self, trace: EconomicTrace, actual_cost: Budget) -> CalibrationResult:
        """
        Reconciles the estimated cost with the actual cost.
        Calculates budget variance and recommends heuristic updates for the Pricer.

        Args:
            trace: The original EconomicTrace containing the estimate and input token count.
            actual_cost: The actual budget consumed (provided by the execution engine).

        Returns:
            CalibrationResult containing variance and recommended heuristic multiplier.
        """
        variance = calculate_budget_variance(trace.estimated_cost, actual_cost)

        # Calculate observed multiplier (Output / Input)
        # Note: actual_cost.token_volume is Total Tokens (Input + Output).
        # We assume input tokens were approx what we estimated (or passed in trace).
        # So actual_output = actual_total - input_tokens.
        # If input_tokens is 0, we can't calculate a multiplier properly, default to 0.

        input_tokens = trace.input_tokens
        if input_tokens > 0:
            actual_output_tokens = actual_cost.token_volume - input_tokens
            # Ensure non-negative (e.g. if actual was somehow less than input due to token counting diffs)
            actual_output_tokens = max(0, actual_output_tokens)
            observed_multiplier = actual_output_tokens / input_tokens
        else:
            observed_multiplier = 0.0

        # Since we are stateless, we return the observed multiplier as the recommendation
        # for this specific transaction type. The caller can aggregate/smooth this.
        return CalibrationResult(
            variance=variance,
            observed_multiplier=observed_multiplier,
            recommended_multiplier=observed_multiplier,
        )
