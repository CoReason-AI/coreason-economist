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


def test_reconcile_actual_less_than_input() -> None:
    """
    Test scenario where reported actual total tokens are less than the input tokens
    used for estimation. This shouldn't happen in reality unless input tokens were
    over-counted or the model failed to generate, but the system should be robust.
    """
    economist = Economist()
    input_tokens = 100
    estimated_budget = Budget(financial=1.0, token_volume=120)  # Est 20 output

    trace = EconomicTrace(
        estimated_cost=estimated_budget,
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=input_tokens,
    )

    # Actual reports only 90 tokens total (less than 100 input)
    actual_budget = Budget(financial=0.8, token_volume=90)

    result = economist.reconcile(trace, actual_budget)

    # Variance: 90 - 120 = -30
    assert result.variance.token_volume_delta == -30

    # Observed multiplier: max(0, 90 - 100) / 100 = 0.0
    assert result.observed_multiplier == 0.0
    assert result.recommended_multiplier == 0.0


def test_reconcile_large_values() -> None:
    """Test reconciliation with very large numbers to ensure no overflow/precision crashes."""
    economist = Economist()
    input_tokens = 1_000_000
    estimated_budget = Budget(financial=1000.0, token_volume=1_200_000)

    trace = EconomicTrace(
        estimated_cost=estimated_budget,
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=input_tokens,
    )

    actual_budget = Budget(financial=1000.0, token_volume=1_200_000)

    result = economist.reconcile(trace, actual_budget)

    assert result.variance.financial_delta == 0.0
    assert result.variance.token_volume_delta == 0
    # (1.2M - 1.0M) / 1.0M = 0.2
    assert abs(result.observed_multiplier - 0.2) < 1e-9


def test_reconcile_floating_point_precision() -> None:
    """Test variance calculation with small float differences."""
    economist = Economist()
    input_tokens = 10

    # Cost 0.0001
    estimated_budget = Budget(financial=0.0001, token_volume=12)

    trace = EconomicTrace(
        estimated_cost=estimated_budget,
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=input_tokens,
    )

    # Actual 0.0002
    actual_budget = Budget(financial=0.0002, token_volume=12)

    result = economist.reconcile(trace, actual_budget)

    # Delta should be 0.0001
    assert abs(result.variance.financial_delta - 0.0001) < 1e-9


def test_reconcile_partial_budgets() -> None:
    """Test reconciliation when some budget fields are zero (unused currencies)."""
    economist = Economist()
    input_tokens = 50

    # Only tracking token volume, financial/latency are 0
    estimated_budget = Budget(financial=0.0, latency_ms=0.0, token_volume=60)

    trace = EconomicTrace(
        estimated_cost=estimated_budget,
        decision=Decision.APPROVED,
        model_used="test-model",
        input_tokens=input_tokens,
    )

    actual_budget = Budget(financial=0.0, latency_ms=0.0, token_volume=70)

    result = economist.reconcile(trace, actual_budget)

    assert result.variance.financial_delta == 0.0
    assert result.variance.latency_ms_delta == 0.0
    assert result.variance.token_volume_delta == 10

    # Output: 70 - 50 = 20. Multiplier: 20/50 = 0.4
    assert result.observed_multiplier == 0.4
