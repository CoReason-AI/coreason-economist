# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

import math

from coreason_economist.models import Budget, Decision, EconomicTrace


class TestEconomicTraceComplexObservability:
    """
    Complex scenarios and edge cases for EconomicTrace observability metrics.
    """

    def test_metric_consistency(self) -> None:
        """
        Verify the mathematical relationship between tokens_per_second and latency_per_token.
        Relationship: latency_per_token (ms/token) = 1000 / tokens_per_second (tokens/s)
        """
        budget = Budget(financial=1.0, latency_ms=500.0, token_volume=1000)
        trace = EconomicTrace(
            estimated_cost=budget,
            actual_cost=budget,
            decision=Decision.APPROVED,
            model_used="gpt-4",
            input_tokens=500,
        )

        tps = trace.tokens_per_second  # 1000 / 0.5 = 2000
        lpt = trace.latency_per_token  # 500 / 1000 = 0.5

        assert tps > 0
        assert lpt > 0

        # 2000 * 0.5 = 1000
        assert math.isclose(tps * lpt, 1000.0, rel_tol=1e-9)

    def test_dynamic_updates(self) -> None:
        """
        Verify that computed fields update dynamically if actual_cost is set after initialization.
        """
        estimated = Budget(financial=1.0, latency_ms=1000.0, token_volume=1000)
        trace = EconomicTrace(
            estimated_cost=estimated,
            actual_cost=None,
            decision=Decision.APPROVED,
            model_used="gpt-4",
            input_tokens=500,
        )

        # Initially uses estimated
        assert trace.tokens_per_dollar == 1000.0  # 1000 / 1.0

        # Update actual cost (cheaper and faster)
        new_actual = Budget(financial=0.5, latency_ms=500.0, token_volume=1000)
        trace.actual_cost = new_actual

        # Should now use actual
        assert trace.tokens_per_dollar == 2000.0  # 1000 / 0.5
        assert trace.tokens_per_second == 2000.0  # 1000 / 0.5s

    def test_large_scale_metrics(self) -> None:
        """
        Test metrics with very large values (e.g., billions of tokens) to ensure float stability.
        """
        # 1 Billion tokens, $1000 cost, 1000 seconds (1M ms)
        large_budget = Budget(financial=1000.0, latency_ms=1_000_000.0, token_volume=1_000_000_000)
        trace = EconomicTrace(
            estimated_cost=large_budget,
            actual_cost=large_budget,
            decision=Decision.APPROVED,
            model_used="future-model",
            input_tokens=1000,
        )

        # 1B / 1000 = 1,000,000 tokens/$
        assert trace.tokens_per_dollar == 1_000_000.0

        # 1B / 1000s = 1,000,000 tokens/s
        assert trace.tokens_per_second == 1_000_000.0

        # 1M ms / 1B tokens = 0.001 ms/token
        assert trace.latency_per_token == 0.001

    def test_rejected_opportunity_metrics(self) -> None:
        """
        Verify that a REJECTED trace still provides metrics based on the ESTIMATED cost.
        This allows for 'Lost Opportunity' analysis (e.g., "We rejected a request that would have been very efficient").
        """
        # Very efficient but expensive request (e.g., massive batch job)
        estimated = Budget(
            financial=100.0,  # Expensive!
            latency_ms=1000.0,  # Fast!
            token_volume=1_000_000,  # Huge volume!
        )

        # Rejected because $100 > limit
        trace = EconomicTrace(
            estimated_cost=estimated,
            actual_cost=None,
            decision=Decision.REJECTED,
            reason="BudgetExhausted",
            model_used="gpt-4",
            input_tokens=1000,
        )

        # Should still calculate potential efficiency
        assert trace.tokens_per_dollar == 10_000.0  # 1M / 100
        assert trace.tokens_per_second == 1_000_000.0  # 1M / 1s

        # Dashboard can read this and say "We blocked a job that was running at 1M tokens/sec"
