# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import List

from coreason_economist.economist import Economist
from coreason_economist.models import Budget, Decision, EconomicTrace


def test_complex_reconciliation_batch() -> None:
    """
    Simulate a batch processing scenario where we reconcile multiple traces
    with varying degrees of accuracy and check the aggregate recommendation.
    """
    economist = Economist()

    # Batch of 3 transactions
    # 1. Exact match (Input 100, Est Output 20, Actual Output 20) -> Mult 0.2
    # 2. Overestimate (Input 100, Est Output 20, Actual Output 10) -> Mult 0.1
    # 3. Underestimate (Input 100, Est Output 20, Actual Output 40) -> Mult 0.4

    input_tokens = 100
    est_output = 20
    est_total = input_tokens + est_output  # 120

    traces: List[EconomicTrace] = []
    actuals: List[Budget] = []

    # 1. Exact
    traces.append(
        EconomicTrace(
            estimated_cost=Budget(token_volume=est_total),
            decision=Decision.APPROVED,
            model_used="model-A",
            input_tokens=input_tokens,
        )
    )
    actuals.append(Budget(token_volume=120))

    # 2. Overestimate (Actual used less)
    traces.append(
        EconomicTrace(
            estimated_cost=Budget(token_volume=est_total),
            decision=Decision.APPROVED,
            model_used="model-A",
            input_tokens=input_tokens,
        )
    )
    actuals.append(Budget(token_volume=110))  # 10 output

    # 3. Underestimate (Actual used more)
    traces.append(
        EconomicTrace(
            estimated_cost=Budget(token_volume=est_total),
            decision=Decision.APPROVED,
            model_used="model-A",
            input_tokens=input_tokens,
        )
    )
    actuals.append(Budget(token_volume=140))  # 40 output

    results = []
    for t, a in zip(traces, actuals, strict=False):
        results.append(economist.reconcile(t, a))

    # verify individual results
    assert results[0].observed_multiplier == 0.2
    assert results[1].observed_multiplier == 0.1
    assert results[2].observed_multiplier == 0.4

    # Verify we can calculate an average from the results (Caller's responsibility, but simulating it)
    avg_multiplier = sum(r.observed_multiplier for r in results) / len(results)
    # (0.2 + 0.1 + 0.4) / 3 = 0.7 / 3 = 0.2333...
    assert abs(avg_multiplier - 0.233333) < 1e-5
