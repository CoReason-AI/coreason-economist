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


def test_observability_small_floats() -> None:
    """
    Test metrics calculation with extremely small float values (e.g. 1e-10).
    Ensures no underflow crashes or division by zero errors if values approach zero.
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=1e-10, token_volume=1000, latency_ms=1e-5),
        decision=Decision.APPROVED,
        model_used="micro-model",
        input_tokens=100,
    )

    # Financial: 1e-10
    # Tokens: 1000
    # T/D = 1000 / 1e-10 = 1e13
    assert trace.tokens_per_dollar > 1e12

    # Latency: 1e-5 ms
    # T/S = 1000 / (1e-5 / 1000) = 1000 / 1e-8 = 1e11
    assert trace.tokens_per_second > 1e10


def test_observability_missing_actual_cost() -> None:
    """
    Verify behavior when actual_cost is None.
    Should fall back to estimated_cost.
    """
    est = Budget(financial=10.0, token_volume=100, latency_ms=1000.0)
    trace = EconomicTrace(
        estimated_cost=est, actual_cost=None, decision=Decision.APPROVED, model_used="gpt-4", input_tokens=50
    )

    # Check effective cost used
    assert trace._effective_cost == est

    # Metrics based on estimate
    # 100 / 10 = 10.0
    assert abs(trace.tokens_per_dollar - 10.0) < 1e-9


def test_observability_large_integers() -> None:
    """
    Test with large token counts (e.g. full context window usage or batch processing).
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=100.0, token_volume=10_000_000, latency_ms=1000.0),
        decision=Decision.APPROVED,
        model_used="batch-model",
        input_tokens=5_000_000,
    )

    # 10M tokens / $100 = 100,000 T/$
    assert trace.tokens_per_dollar == 100_000.0


def test_observability_zero_latency_positive_tokens() -> None:
    """
    Edge case: Infinite speed? (Latency = 0, Tokens > 0).
    Should return 0.0 (or handle gracefully) to avoid ZeroDivisionError in tokens_per_second.

    The implementation returns 0.0 if latency <= 0.
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=1.0, token_volume=100, latency_ms=0.0),
        decision=Decision.APPROVED,
        model_used="magic-model",
        input_tokens=50,
    )

    assert trace.tokens_per_second == 0.0


def test_observability_zero_financial() -> None:
    """
    Edge case: Free model (Financial = 0).
    Should return 0.0 for tokens_per_dollar.
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=0.0, token_volume=100, latency_ms=100.0),
        decision=Decision.APPROVED,
        model_used="free-model",
        input_tokens=50,
    )

    assert trace.tokens_per_dollar == 0.0
