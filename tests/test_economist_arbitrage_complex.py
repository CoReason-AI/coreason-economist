# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Any, Dict, List

import pytest
from coreason_economist.economist import Economist
from coreason_economist.models import Budget, Decision, RequestPayload


def test_complex_combined_downgrade() -> None:
    """
    Complex Scenario: "The Desperate Downgrade".
    User requests an expensive setup (Council of 5 GPT-4o agents) with a tiny budget.
    Expectation: Arbitrageur suggests BOTH topology reduction (to 1 agent) AND model downgrade (to GPT-4o-mini).
    """
    economist = Economist()

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Describe the universe.",
        agent_count=5,
        rounds=3,
        max_budget=Budget(financial=0.0001),  # Extremely low budget
        difficulty_score=0.1,  # Low difficulty
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert trace.suggested_alternative is not None

    suggestion = trace.suggested_alternative
    # Verify Model Downgrade
    assert suggestion.model_name == "gpt-4o-mini"
    # Verify Topology Reduction
    assert suggestion.agent_count == 1
    assert suggestion.rounds == 1


def test_edge_case_already_cheapest() -> None:
    """
    Edge Case: User is already using the cheapest configuration but still exceeds budget.
    Expectation: REJECTED, but suggested_alternative is None (cannot optimize further).
    """
    economist = Economist()

    request = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="A" * 1000,
        estimated_output_tokens=1000,
        agent_count=1,
        rounds=1,
        max_budget=Budget(financial=0.0000001),  # Impossible budget
        difficulty_score=0.1,
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert trace.suggested_alternative is None


def test_edge_case_latency_failure_downgrade() -> None:
    """
    Edge Case: Financial budget is fine, but Latency budget is exhausted.
    Expectation: Economist rejects and suggests a cheaper (and typically faster) model.
    """
    economist = Economist()

    # GPT-4o latency ~12ms/token. 100 tokens -> 1200ms.
    # Budget 500ms.
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Quick answer",
        estimated_output_tokens=100,
        max_budget=Budget(latency_ms=500.0, financial=100.0),  # Unlimited money, strict time
        difficulty_score=0.2,
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert "Latency budget exceeded" in str(trace.reason)
    assert trace.suggested_alternative is not None
    assert trace.suggested_alternative.model_name == "gpt-4o-mini"


def test_edge_case_tools_blow_budget() -> None:
    """
    Edge Case: Tool calls are the primary cost driver.
    Expectation: Arbitrageur still suggests downgrading the model/topology,
    even though it cannot strip the tools. The user is responsible for removing tools if needed,
    but the Economist does its best to lower the *other* costs.
    """
    economist = Economist()

    # Expensive tool call ($0.01) + Expensive Model
    # Budget $0.005
    tool_calls: List[Dict[str, Any]] = [{"function": {"name": "web_search"}}]  # Cost $0.01

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Search for X",
        estimated_output_tokens=50,
        tool_calls=tool_calls,
        max_budget=Budget(financial=0.005),
        difficulty_score=0.2,
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert "Financial budget exceeded" in str(trace.reason)

    # It should still suggest downgrading the model to save *some* money
    assert trace.suggested_alternative is not None
    assert trace.suggested_alternative.model_name == "gpt-4o-mini"
    # Tools should remain in the suggestion
    assert trace.suggested_alternative.tool_calls == tool_calls


def test_edge_case_unknown_model_no_suggestion() -> None:
    """
    Edge Case: Request uses an unknown model.
    Expectation: Pricer raises ValueError before BudgetCheck.
    Economist propagates the error (it does NOT catch ValueError, only BudgetExhaustedError).
    """
    economist = Economist()

    request = RequestPayload(
        model_name="gpt-5-future",
        prompt="Test",
        max_budget=Budget(financial=1.0),
    )

    with pytest.raises(ValueError, match="Unknown model"):
        economist.check_execution(request)
