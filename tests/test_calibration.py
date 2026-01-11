# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.calibration import calculate_budget_variance
from coreason_economist.models import Budget, BudgetVariance


def test_calculate_budget_variance_exact() -> None:
    """Test variance when actual equals estimated (zero variance)."""
    budget = Budget(financial=1.0, latency_ms=100.0, token_volume=100)
    variance = calculate_budget_variance(budget, budget)

    assert variance.financial_delta == 0.0
    assert variance.latency_ms_delta == 0.0
    assert variance.token_volume_delta == 0


def test_calculate_budget_variance_over_budget() -> None:
    """Test variance when actual exceeds estimated (positive variance)."""
    est = Budget(financial=1.0, latency_ms=100.0, token_volume=100)
    act = Budget(financial=1.5, latency_ms=150.0, token_volume=150)

    variance = calculate_budget_variance(est, act)

    assert variance.financial_delta == 0.5
    assert variance.latency_ms_delta == 50.0
    assert variance.token_volume_delta == 50


def test_calculate_budget_variance_under_budget() -> None:
    """Test variance when actual is less than estimated (negative variance)."""
    est = Budget(financial=1.0, latency_ms=100.0, token_volume=100)
    act = Budget(financial=0.8, latency_ms=80.0, token_volume=80)

    variance = calculate_budget_variance(est, act)

    # Floating point comparison
    assert abs(variance.financial_delta - (-0.2)) < 1e-9
    assert variance.latency_ms_delta == -20.0
    assert variance.token_volume_delta == -20


def test_budget_variance_model() -> None:
    """Test the BudgetVariance model."""
    var = BudgetVariance(financial_delta=-0.5, latency_ms_delta=100.0, token_volume_delta=0)
    assert var.financial_delta == -0.5
    assert var.latency_ms_delta == 100.0
    assert var.token_volume_delta == 0
