# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Any, Dict, Optional

from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer
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
        self.pricer = Pricer(rates=self.rates)

    def _is_budget_exceeded(self, cost: Budget, limit: Budget) -> bool:
        """
        Check if cost exceeds limit.
        Treats 0 as 'unlimited' (ignore) to determine if we should trigger budget-fitting.
        """
        if limit.financial > 0 and cost.financial > limit.financial:
            return True
        if limit.latency_ms > 0 and cost.latency_ms > limit.latency_ms:
            return True
        if limit.token_volume > 0 and cost.token_volume > limit.token_volume:
            return True
        return False

    def _is_within_limits(self, cost: Budget, limit: Budget) -> bool:
        """
        Check if cost fits within non-zero limits.
        Treats 0 as 'unlimited' for the purpose of finding a valid alternative,
        to avoid blocking suggestions due to uninitialized budget fields.
        """
        if limit.financial > 0 and cost.financial > limit.financial:
            return False
        if limit.latency_ms > 0 and cost.latency_ms > limit.latency_ms:
            return False
        if limit.token_volume > 0 and cost.token_volume > limit.token_volume:
            return False
        return True

    def recommend_alternative(self, request: RequestPayload) -> Optional[RequestPayload]:
        """
        Analyzes the request and recommends a cheaper model if appropriate.

        Returns:
            A new RequestPayload with the modified model_name, or None if no
            recommendation is made.
        """
        # If the requested model is unknown, we can't estimate prices or compare.
        if request.model_name not in self.rates:
            return None

        # Calculate current cost
        current_cost = self.pricer.estimate_request_cost(
            model_name=request.model_name,
            input_tokens=len(request.prompt) // 4,
            output_tokens=request.estimated_output_tokens,
            tool_calls=request.tool_calls,
            agent_count=request.agent_count,
            rounds=request.rounds,
        )

        budget_exceeded = False
        if request.max_budget:
            budget_exceeded = self._is_budget_exceeded(current_cost, request.max_budget)

        updates: Dict[str, Any] = {}
        quality_warning: Optional[str] = None
        suggestion_found = False

        # STRATEGY 1: Budget Fitting (High Difficulty / "Hard Stop" Fallback)
        # If budget is exceeded, we attempt to find a cheaper way.
        if budget_exceeded and request.max_budget:
            # First try reducing topology with the REQUESTED model.
            # Priority: Reduce Rounds FIRST, then Agent Count.
            # Maximize Agents > Maximize Rounds.
            for a in range(request.agent_count, 0, -1):
                for r in range(request.rounds, 0, -1):
                    # Skip the original config since we know it failed
                    if a == request.agent_count and r == request.rounds:
                        continue

                    cost = self.pricer.estimate_request_cost(
                        model_name=request.model_name,
                        input_tokens=len(request.prompt) // 4,
                        output_tokens=request.estimated_output_tokens,
                        tool_calls=request.tool_calls,
                        agent_count=a,
                        rounds=r,
                    )

                    if self._is_within_limits(cost, request.max_budget):
                        updates["agent_count"] = a
                        updates["rounds"] = r
                        suggestion_found = True
                        if request.difficulty_score and request.difficulty_score >= self.threshold:
                            quality_warning = f"Reduced topology to {a} agents, {r} rounds to fit budget."
                        break
                if suggestion_found:
                    break

            if not suggestion_found:
                # Topology reduction wasn't enough (or wasn't possible), try cheapest model + reduced topology
                # We default to single-shot (1A, 1R) for the cheapest model fallback
                sorted_models = sorted(
                    self.rates.items(),
                    key=lambda item: item[1].input_cost_per_1k + item[1].output_cost_per_1k,
                )
                cheapest_name, _ = sorted_models[0]

                # Check if cheapest fits with single-shot
                cheapest_cost = self.pricer.estimate_request_cost(
                    model_name=cheapest_name,
                    input_tokens=len(request.prompt) // 4,
                    output_tokens=request.estimated_output_tokens,
                    tool_calls=request.tool_calls,
                    agent_count=1,
                    rounds=1,
                )

                if self._is_within_limits(cheapest_cost, request.max_budget):
                    updates["agent_count"] = 1
                    updates["rounds"] = 1
                    updates["model_name"] = cheapest_name
                    suggestion_found = True
                    if request.difficulty_score and request.difficulty_score >= self.threshold:
                        quality_warning = f"Downgraded to {cheapest_name} and single-shot to fit budget."
                    else:
                        quality_warning = f"Downgraded to {cheapest_name} to fit budget."

        # STRATEGY 2: Standard Arbitrage (Low Difficulty Optimization)
        # Logic: If task is easy, optimize for savings, even if budget is not strictly exceeded
        # (or if we already found a fitting solution but can optimize further).
        if request.difficulty_score is not None and request.difficulty_score < self.threshold:
            # Determine baseline for comparison.
            # If we already have updates, we should compare against the updated config?
            # Actually, standard arbitrage logic compares original request model vs cheapest.
            # If we already downgraded model in Strategy 1, 'updates' has it.
            # If we kept original model in Strategy 1, 'updates' might just have topology.

            current_model = updates.get("model_name", request.model_name)
            current_rate = self.rates[current_model]
            current_cost_index = current_rate.input_cost_per_1k + current_rate.output_cost_per_1k

            sorted_models = sorted(
                self.rates.items(),
                key=lambda item: item[1].input_cost_per_1k + item[1].output_cost_per_1k,
            )
            cheapest_name, cheapest_rate = sorted_models[0]
            cheapest_cost_index = cheapest_rate.input_cost_per_1k + cheapest_rate.output_cost_per_1k

            # 1. Topology Reduction (if not already done)
            # Strategy 1 might have done this.
            current_agent_count = updates.get("agent_count", request.agent_count)
            current_rounds = updates.get("rounds", request.rounds)

            if current_agent_count > 1 or current_rounds > 1:
                updates["agent_count"] = 1
                updates["rounds"] = 1

            # 2. Model Downgrade
            if cheapest_cost_index < current_cost_index:
                updates["model_name"] = cheapest_name

        if updates:
            if quality_warning:
                updates["quality_warning"] = quality_warning
            return request.model_copy(update=updates)

        return None
