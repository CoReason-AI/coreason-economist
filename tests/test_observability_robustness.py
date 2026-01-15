# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.models import Budget, Decision, EconomicTrace


def test_observability_zero_values() -> None:
    """
    Test that efficiency metrics handle zero values gracefully (no division by zero).
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=0.0, latency_ms=0.0, token_volume=0),
        actual_cost=Budget(financial=0.0, latency_ms=0.0, token_volume=0),
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=100,
    )

    # All efficiency metrics should be 0.0, not raise Error or return Infinity
    assert trace.tokens_per_dollar == 0.0
    assert trace.tokens_per_second == 0.0
    assert trace.latency_per_token == 0.0
    assert trace.cost_per_insight == 0.0


def test_observability_precedence() -> None:
    """
    Test that actual_cost takes precedence over estimated_cost for metrics.
    """
    est = Budget(financial=10.0, latency_ms=1000.0, token_volume=100)
    act = Budget(financial=20.0, latency_ms=2000.0, token_volume=200)

    trace = EconomicTrace(
        estimated_cost=est, actual_cost=act, decision=Decision.APPROVED, model_used="test-model", input_tokens=100
    )

    # Cost per insight should follow actual ($20) not estimated ($10)
    assert trace.cost_per_insight == 20.0

    # Tokens per dollar: 200 tokens / $20 = 10.0
    # If it used estimated: 100 / 10 = 10.0 (coincidence)
    # Let's change values to be distinct
    # Est: $10, 100 tok -> 10 t/$
    # Act: $50, 200 tok -> 4 t/$

    est = Budget(financial=10.0, latency_ms=1000.0, token_volume=100)
    act = Budget(financial=50.0, latency_ms=2000.0, token_volume=200)

    trace_mixed = EconomicTrace(
        estimated_cost=est, actual_cost=act, decision=Decision.APPROVED, model_used="test-model", input_tokens=100
    )

    assert trace_mixed.tokens_per_dollar == 4.0  # (200 / 50)
    assert trace_mixed.cost_per_insight == 50.0


def test_observability_fallback() -> None:
    """
    Test that metrics fallback to estimated_cost if actual_cost is None.
    """
    est = Budget(financial=10.0, latency_ms=1000.0, token_volume=100)

    trace = EconomicTrace(
        estimated_cost=est, actual_cost=None, decision=Decision.APPROVED, model_used="test-model", input_tokens=100
    )

    assert trace.cost_per_insight == 10.0
    assert trace.tokens_per_dollar == 10.0  # 100 / 10
