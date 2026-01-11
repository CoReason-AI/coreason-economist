# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.models import Budget, BudgetVariance


def calculate_budget_variance(estimated: Budget, actual: Budget) -> BudgetVariance:
    """
    Calculates the variance between the estimated and actual budget.
    Variance = Actual - Estimated.

    Args:
        estimated: The estimated budget.
        actual: The actual budget consumed.

    Returns:
        BudgetVariance object containing the deltas.
    """
    return BudgetVariance(
        financial_delta=actual.financial - estimated.financial,
        latency_ms_delta=actual.latency_ms - estimated.latency_ms,
        token_volume_delta=actual.token_volume - estimated.token_volume,
    )
