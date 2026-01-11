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


def test_economist_rejection_with_alternative() -> None:
    """
    Story A: The "Hard Stop" (Budget Enforcement) with fallback.
    Verifies that when a budget is exhausted, the Economist returns REJECTED
    and provides a suggested alternative if one exists.
    """
    economist = Economist()

    # Request using expensive model (gpt-4o) with low budget
    # Input: 1000 chars (~250 tokens), Output: 200 tokens
    # Cost: 250/1000 * 0.005 + 200/1000 * 0.015 = 0.00125 + 0.003 = 0.00425
    # Budget: 0.001 (should fail)
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 1000,
        estimated_output_tokens=200,
        max_budget=Budget(financial=0.001),
        difficulty_score=0.2,  # Low difficulty, eligible for downgrade
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert "Financial budget exceeded" in str(trace.reason)
    assert trace.suggested_alternative is not None

    # Verify the suggestion is cheaper/different
    suggestion = trace.suggested_alternative
    assert suggestion.model_name != "gpt-4o"
    # Based on default rates, should suggest gpt-4o-mini
    assert suggestion.model_name == "gpt-4o-mini"


def test_economist_rejection_without_alternative_high_difficulty() -> None:
    """
    Verifies that if difficulty is high, no alternative is suggested
    even if rejected.
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
    assert trace.suggested_alternative is None


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

    suggestion = trace.suggested_alternative
    # Should reduce to 1 agent, 1 round
    assert suggestion.agent_count == 1
    assert suggestion.rounds == 1
