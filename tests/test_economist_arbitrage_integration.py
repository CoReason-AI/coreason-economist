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
from coreason_economist.models import Budget, Decision, RequestPayload


def test_economist_rejection_without_alternative_high_difficulty() -> None:
    """
    Verifies that if difficulty is high, an alternative IS suggested
    if it fits the budget (Budget Fitting Mode), correcting previous assumptions.
    """
    economist = Economist()

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 1000,
        estimated_output_tokens=200,
        max_budget=Budget(financial=0.001),
        difficulty_score=0.9,  # High difficulty
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    # New Expectation: Arbitrageur should suggest a downgrade with warning
    assert trace.suggested_alternative is not None
    assert trace.suggested_alternative.quality_warning is not None
    assert "Downgraded" in trace.suggested_alternative.quality_warning


def test_economist_rejection_topology_reduction() -> None:
    """
    Verifies that multi-agent request is reduced to single agent
    if budget is blown and difficulty is low.
    """
    economist = Economist()

    # Multi-agent request that blows budget
    request = RequestPayload(
        model_name="gpt-4o-mini",  # Already cheap model
        prompt="Test",
        agent_count=5,
        rounds=3,
        max_budget=Budget(financial=0.0001),  # Very low budget
        difficulty_score=0.1,
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert trace.suggested_alternative is not None
    assert trace.suggested_alternative.agent_count == 1
    assert trace.suggested_alternative.rounds == 1
