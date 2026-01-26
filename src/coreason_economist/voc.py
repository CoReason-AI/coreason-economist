# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

import difflib
from typing import Optional

from coreason_economist.models import Budget, ReasoningTrace, VOCDecision, VOCResult


class VOCEngine:
    """
    The Value of Computation (VOC) Engine.
    Acts as the "Stop Button" logic based on diminishing returns.
    """

    def __init__(self, default_threshold: float = 0.95) -> None:
        """
        Initialize the VOC Engine.

        Args:
            default_threshold: The similarity threshold (0.0 to 1.0) above which
                               we consider the outputs to have converged (diminishing returns).
                               Default is 0.95 (95% similar).
        """
        self.default_threshold = default_threshold

    def _calculate_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculates the similarity between two text strings using a lightweight heuristic.
        Uses difflib.SequenceMatcher (Ratcliff-Obershelp algorithm).

        Note:
            - This is a lexical similarity check, not semantic. It may not detect
              semantic convergence where different words mean the same thing.
            - Performance: O(N*M) complexity. May be slow for very large strings (e.g., >100k chars).
        """
        if not text_a and not text_b:
            return 1.0
        if not text_a or not text_b:
            return 0.0

        return difflib.SequenceMatcher(None, text_a, text_b).ratio()

    def _is_budget_critical(self, remaining: Budget, total: Budget, critical_threshold: float = 0.2) -> bool:
        """
        Checks if any budget dimension is critically low (below the threshold percentage).
        """
        # Financial
        if total.financial > 0:
            if remaining.financial / total.financial < critical_threshold:
                return True

        # Latency
        if total.latency_ms > 0:
            if remaining.latency_ms / total.latency_ms < critical_threshold:
                return True

        # Token Volume
        if total.token_volume > 0:
            if remaining.token_volume / total.token_volume < critical_threshold:
                return True

        return False

    def evaluate(
        self,
        trace: ReasoningTrace,
        threshold: Optional[float] = None,
        remaining_budget: Optional[Budget] = None,
        total_budget: Optional[Budget] = None,
    ) -> VOCResult:
        """
        Evaluates the reasoning trace to decide whether to continue computation.

        Strategy:
        - If there are fewer than 2 steps, we cannot compare, so we CONTINUE.
        - Compare the last step (N) with the previous step (N-1).
        - If similarity > threshold, we STOP (Diminishing returns).

        Opportunity Cost Logic:
        - If remaining_budget is critically low (< 20% of total_budget), we effectively
          increase the "cost" of continuing by lowering the similarity threshold.
          This makes the system "impatient" and more likely to STOP to save resources.

        Args:
            trace: The history of reasoning steps.
            threshold: Optional override for the similarity threshold.
            remaining_budget: The remaining budget for the request.
            total_budget: The total allocated budget for the request.

        Returns:
            VOCResult with decision, score, and reason.
        """
        base_thresh = threshold if threshold is not None else self.default_threshold
        effective_thresh = base_thresh
        opportunity_cost_active = False

        # Check Opportunity Cost (Budget Constraint)
        if remaining_budget is not None and total_budget is not None:
            if self._is_budget_critical(remaining_budget, total_budget):
                # Reduce threshold by 10% (e.g., 0.95 -> 0.855)
                # This means we accept "less similarity" (more difference) as "good enough" to stop.
                # Or rather: we stop even if they are only 85% similar, because we can't afford to refine further.
                effective_thresh = base_thresh * 0.9
                opportunity_cost_active = True

        if len(trace.steps) < 2:
            return VOCResult(
                decision=VOCDecision.CONTINUE,
                score=0.0,
                reason="Insufficient history to determine diminishing returns.",
            )

        last_step = trace.steps[-1]
        prev_step = trace.steps[-2]

        similarity = self._calculate_similarity(prev_step, last_step)

        if similarity >= effective_thresh:
            reason = f"Diminishing returns detected. Similarity {similarity:.4f} >= threshold {effective_thresh:.4f}."
            if opportunity_cost_active:
                reason += " (Threshold lowered due to critical budget - Opportunity Cost)."
            return VOCResult(
                decision=VOCDecision.STOP,
                score=similarity,
                reason=reason,
            )

        return VOCResult(
            decision=VOCDecision.CONTINUE,
            score=similarity,
            reason=f"Significant change detected. Similarity {similarity:.4f} < threshold {effective_thresh:.4f}.",
        )

    def assess_viability(self, task_complexity: float, current_uncertainty: float) -> tuple[bool, float]:
        """
        Assess if the task is worth executing based on complexity and uncertainty.
        Returns (should_execute, max_allowable_cost).
        """
        # Heuristic: High uncertainty requires higher potential value (or budget) to resolve.
        # But extremely high uncertainty + high complexity might be a money sink.

        # Base allowable cost for a standard task
        base_budget = 0.10

        # Complexity multiplier: More complex tasks need more budget
        # range: 1.0 to 2.0
        complexity_multiplier = 1.0 + task_complexity

        # Uncertainty penalty/bonus?
        # If uncertainty is low, we are confident, budget is predictable.
        # If uncertainty is high, we might need more rounds (higher budget) OR we might abort.
        # Let's assume we allow more budget to reduce uncertainty, up to a point.
        uncertainty_multiplier = 1.0 + (current_uncertainty * 0.5)

        max_allowable_cost = base_budget * complexity_multiplier * uncertainty_multiplier

        # Decision threshold
        # If complexity > 0.9 AND uncertainty > 0.9, it's too risky/expensive.
        should_execute = not (task_complexity > 0.9 and current_uncertainty > 0.9)

        return should_execute, max_allowable_cost
