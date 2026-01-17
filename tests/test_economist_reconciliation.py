# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.economist import Economist
from coreason_economist.models import Budget, Decision, EconomicTrace


def test_reconcile_exact_match() -> None:
    """Test reconciliation when estimated matches actual."""
    economist = Economist()

    estimated_budget = Budget(financial=1.0, latency_ms=100.0, token_volume=100)
    input_tokens = 50
    # Pricer heuristic is output = input * 0.2 = 10.
    # Total volume = 60.
    # But here we are manually creating the trace, so we define the baseline.

    trace = EconomicTrace(
        estimated_cost=estimated_budget,
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=input_tokens,
    )

    actual_budget = Budget(financial=1.0, latency_ms=100.0, token_volume=100)
    # Output tokens = 100 - 50 = 50.
    # Observed multiplier = 50 / 50 = 1.0

    result = economist.reconcile(trace, actual_budget)

    assert result.variance.financial_delta == 0.0
    assert result.variance.latency_ms_delta == 0.0
    assert result.variance.token_volume_delta == 0.0
    assert result.observed_multiplier == 1.0  # (100-50)/50 = 1.0
    assert result.recommended_multiplier == 1.0


def test_reconcile_variance() -> None:
    """Test reconciliation with variance."""
    economist = Economist()

    estimated_budget = Budget(financial=1.0, latency_ms=100.0, token_volume=100)
    input_tokens = 50

    trace = EconomicTrace(
        estimated_cost=estimated_budget,
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=input_tokens,
    )

    # Actual cost higher
    actual_budget = Budget(financial=1.5, latency_ms=150.0, token_volume=150)
    # Actual output = 150 - 50 = 100
    # Observed multiplier = 100 / 50 = 2.0

    result = economist.reconcile(trace, actual_budget)

    assert result.variance.financial_delta == 0.5
    assert result.variance.latency_ms_delta == 50.0
    assert result.variance.token_volume_delta == 50
    assert result.observed_multiplier == 2.0
    assert result.recommended_multiplier == 2.0


def test_reconcile_zero_input() -> None:
    """Test reconciliation with zero input tokens."""
    economist = Economist()
    estimated_budget = Budget(financial=0.0, latency_ms=0.0, token_volume=0)
    input_tokens = 0

    trace = EconomicTrace(
        estimated_cost=estimated_budget,
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=input_tokens,
    )

    actual_budget = Budget(financial=0.0, latency_ms=0.0, token_volume=10)
    # Actual output = 10 - 0 = 10.
    # Multiplier would be div/0, so we expect 0.0

    result = economist.reconcile(trace, actual_budget)

    assert result.observed_multiplier == 0.0
    assert result.recommended_multiplier == 0.0
