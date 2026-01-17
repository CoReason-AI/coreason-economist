# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Dict

import pytest
from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.budget_authority import BudgetAuthority
from coreason_economist.exceptions import BudgetExhaustedError
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer
from coreason_economist.rates import ModelRate, ToolRate


@pytest.fixture  # type: ignore
def mock_rates() -> Dict[str, ModelRate]:
    return {
        "gpt-4": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=2.0, latency_ms_per_output_token=10.0),
        "gpt-4-clone": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=2.0, latency_ms_per_output_token=10.0),
        "cheap": ModelRate(input_cost_per_1k=0.1, output_cost_per_1k=0.2, latency_ms_per_output_token=5.0),
    }


@pytest.fixture  # type: ignore
def mock_tool_rates() -> Dict[str, ToolRate]:
    return {
        "search": ToolRate(cost_per_call=0.5),
    }


def test_arbitrageur_boundary_condition(mock_rates: Dict[str, ModelRate]) -> None:
    """
    Test Arbitrageur behavior when difficulty_score exactly equals threshold.
    It should NOT recommend a change (trust the caller at threshold).
    """
    # Refactor: Inject Pricer instead of passing rates directly
    pricer = Pricer(rates=mock_rates)
    arb = Arbitrageur(pricer=pricer, threshold=0.5)
    # Score 0.5 == Threshold 0.5 -> Should be treated as "hard enough" -> Return None
    payload = RequestPayload(model_name="gpt-4", prompt="test", difficulty_score=0.5)

    assert arb.recommend_alternative(payload) is None

    # Score 0.499 < Threshold 0.5 -> Should recommend
    payload_low = RequestPayload(model_name="gpt-4", prompt="test", difficulty_score=0.499)
    res = arb.recommend_alternative(payload_low)
    assert res is not None
    assert res.model_name == "cheap"


def test_arbitrageur_equal_cost(mock_rates: Dict[str, ModelRate]) -> None:
    """
    Test that Arbitrageur does not recommend a switch if the 'cheapest' model
    costs exactly the same as the current model.
    """
    # Assume we are on 'gpt-4' and 'gpt-4-clone' is the "cheapest" (first in sort if equal)
    # or if we are on 'gpt-4-clone' and 'gpt-4' is same price.

    # Let's use a registry where the "cheapest" is the current one.
    # Actually, the logic finds the absolute cheapest.
    # If I am on "gpt-4" ($3 total index), and "gpt-4-clone" is also $3.
    # If "cheap" was NOT in the list, then "gpt-4" is tied for cheapest.

    rates_equal = {
        "model-a": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=1.0),
        "model-b": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=1.0),
    }
    pricer_equal = Pricer(rates=rates_equal)
    arb = Arbitrageur(pricer=pricer_equal, threshold=0.5)

    # Current is model-a. Cheapest is model-a (or b).
    # Cost index is equal. Logic says `if cheapest < current`. 2.0 < 2.0 is False.
    # So it should return None.
    payload = RequestPayload(model_name="model-a", prompt="test", difficulty_score=0.1)
    assert arb.recommend_alternative(payload) is None


def test_pricer_council_math_verification(
    mock_rates: Dict[str, ModelRate], mock_tool_rates: Dict[str, ToolRate]
) -> None:
    """
    Explicitly verify the formula: (Model + Tool) * Agents * Rounds.
    """
    pricer = Pricer(rates=mock_rates, tool_rates=mock_tool_rates)

    # Inputs
    input_tokens = 1000  # Cost: 1.0 * 1 = $1.0
    output_tokens = 1000  # Cost: 2.0 * 1 = $2.0
    # Model Unit Cost = $3.0

    # Tools
    tools = [{"name": "search"}]  # Cost: $0.5

    # Total Unit Financial Cost = $3.5

    # Config
    agents = 5
    rounds = 3

    # Expected Financial: 3.5 * 5 * 3 = 17.5 * 3 = 52.5

    budget = pricer.estimate_request_cost(
        "gpt-4",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tools,
        agent_count=agents,
        rounds=rounds,
    )

    assert abs(budget.financial - 52.5) < 1e-9


def test_pricer_latency_independence(mock_rates: Dict[str, ModelRate]) -> None:
    """
    Verify that increasing agent_count does NOT increase latency (Parallel Assumption),
    but increasing rounds DOES.
    """
    pricer = Pricer(rates=mock_rates)
    input_t = 100
    output_t = 100
    # Unit Latency: 100 * 10ms = 1000ms

    # Case 1: 1 Agent, 1 Round -> 1000ms
    b1 = pricer.estimate_request_cost("gpt-4", input_t, output_tokens=output_t, agent_count=1, rounds=1)
    assert b1.latency_ms == 1000.0

    # Case 2: 10 Agents, 1 Round -> Still 1000ms (Parallel)
    b2 = pricer.estimate_request_cost("gpt-4", input_t, output_tokens=output_t, agent_count=10, rounds=1)
    assert b2.latency_ms == 1000.0

    # Case 3: 1 Agent, 5 Rounds -> 5000ms (Sequential)
    b3 = pricer.estimate_request_cost("gpt-4", input_t, output_tokens=output_t, agent_count=1, rounds=5)
    assert b3.latency_ms == 5000.0

    # Case 4: 10 Agents, 5 Rounds -> 5000ms
    b4 = pricer.estimate_request_cost("gpt-4", input_t, output_tokens=output_t, agent_count=10, rounds=5)
    assert b4.latency_ms == 5000.0


def test_integrated_expensive_council_rejection(mock_rates: Dict[str, ModelRate]) -> None:
    """
    Scenario: A request fits within budget for a single agent, but fails when scaled
    to a Council (multi-agent/multi-round).
    """
    pricer = Pricer(rates=mock_rates)
    authority = BudgetAuthority(pricer=pricer)

    # Unit Cost: 1000 in ($1), 1000 out ($2) = $3.0
    # Budget: $10.0

    max_budget = Budget(financial=10.0, latency_ms=100000.0, token_volume=100000)

    # Single Agent Request ($3.0 < $10.0) -> Should Pass
    req_single = RequestPayload(
        model_name="gpt-4",
        prompt="A" * 4000,  # Approx 1000 tokens
        estimated_output_tokens=1000,
        max_budget=max_budget,
        agent_count=1,
        rounds=1,
    )
    assert authority.allow_execution(req_single).allowed is True

    # Council Request (5 agents) -> $3.0 * 5 = $15.0 > $10.0 -> Should Fail
    req_council = RequestPayload(
        model_name="gpt-4",
        prompt="A" * 4000,
        estimated_output_tokens=1000,
        max_budget=max_budget,
        agent_count=5,
        rounds=1,
    )

    with pytest.raises(BudgetExhaustedError) as exc:
        authority.allow_execution(req_council)

    assert exc.value.limit_type == "financial"
    assert exc.value.estimated_value == 15.0
