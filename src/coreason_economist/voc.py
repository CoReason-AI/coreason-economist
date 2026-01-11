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

from coreason_economist.models import ReasoningTrace, VOCDecision, VOCResult


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

    def evaluate(self, trace: ReasoningTrace, threshold: Optional[float] = None) -> VOCResult:
        """
        Evaluates the reasoning trace to decide whether to continue computation.

        Strategy:
        - If there are fewer than 2 steps, we cannot compare, so we CONTINUE.
        - Compare the last step (N) with the previous step (N-1).
        - If similarity > threshold, we STOP (Diminishing returns).

        Args:
            trace: The history of reasoning steps.
            threshold: Optional override for the similarity threshold.

        Returns:
            VOCResult with decision, score, and reason.
        """
        thresh = threshold if threshold is not None else self.default_threshold

        if len(trace.steps) < 2:
            return VOCResult(
                decision=VOCDecision.CONTINUE,
                score=0.0,
                reason="Insufficient history to determine diminishing returns.",
            )

        last_step = trace.steps[-1]
        prev_step = trace.steps[-2]

        similarity = self._calculate_similarity(prev_step, last_step)

        if similarity >= thresh:
            return VOCResult(
                decision=VOCDecision.STOP,
                score=similarity,
                reason=f"Diminishing returns detected. Similarity {similarity:.4f} >= threshold {thresh:.4f}.",
            )

        return VOCResult(
            decision=VOCDecision.CONTINUE,
            score=similarity,
            reason=f"Significant change detected. Similarity {similarity:.4f} < threshold {thresh:.4f}.",
        )
