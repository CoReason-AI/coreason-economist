# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

import pytest
from coreason_economist.budget_authority import BudgetAuthority
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer


@pytest.fixture  # type: ignore
def budget_authority() -> BudgetAuthority:
    return BudgetAuthority(pricer=Pricer())


def test_boundary_exact_match(budget_authority: BudgetAuthority) -> None:
    """
    Edge Case: Usage is EXACTLY the threshold.
    Expectation: Should NOT warn (strict greater than).
    """
    # Cost: 1000 input ($0.005) + 0 output = $0.005
    # Budget: $0.010. Usage = 50%.
    # Set threshold = 0.5
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=0,
        max_budget=Budget(financial=0.010, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=0.5,
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True
    assert result.warning is False, "Should not warn if usage == threshold"


def test_boundary_just_above(budget_authority: BudgetAuthority) -> None:
    """
    Edge Case: Usage is slightly above threshold.
    Expectation: Should warn.
    """
    # Usage 50% ($0.005 / $0.010)
    # Threshold 0.499
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=0,
        max_budget=Budget(financial=0.010, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=0.499,
    )
    result = budget_authority.allow_execution(req)
    assert result.warning is True


def test_threshold_zero(budget_authority: BudgetAuthority) -> None:
    """
    Edge Case: Threshold is 0.0.
    Expectation: Any non-zero usage triggers warning.
    """
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=0,
        max_budget=Budget(financial=1.0, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=0.0,
    )
    result = budget_authority.allow_execution(req)
    assert result.warning is True
    assert "Financial budget at 0.5%" in str(result.message)


def test_threshold_one(budget_authority: BudgetAuthority) -> None:
    """
    Edge Case: Threshold is 1.0.
    Expectation: Should effectively disable warnings (unless >100% which is error).
    """
    # Usage 99%
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=0,
        # Cost $0.005. Budget $0.00505 (approx 99%)
        max_budget=Budget(financial=0.00505, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=1.0,
    )
    result = budget_authority.allow_execution(req)
    assert result.warning is False


def test_multi_agent_scaling_trigger(budget_authority: BudgetAuthority) -> None:
    """
    Complex Scenario: Single agent is fine (20% usage), but 4 agents (80%) trigger warning.
    """
    # 1 Agent cost: $0.005
    # Budget: $0.024
    # 1 Agent: 0.005 / 0.024 = ~20.8% -> No warning (default 0.8)
    # 4 Agents: 0.020 / 0.024 = ~83.3% -> Warning

    budget = Budget(financial=0.024, latency_ms=100000, token_volume=100000)

    # Case 1: 1 Agent
    req_1 = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=0,
        max_budget=budget,
        agent_count=1,
    )
    res_1 = budget_authority.allow_execution(req_1)
    assert res_1.warning is False

    # Case 2: 4 Agents
    req_4 = req_1.model_copy(update={"agent_count": 4})
    res_4 = budget_authority.allow_execution(req_4)
    assert res_4.warning is True
    assert "Financial budget at 83.3%" in str(res_4.message)


def test_tiny_budget_precision(budget_authority: BudgetAuthority) -> None:
    """
    Edge Case: Very small budget numbers to check floating point stability.
    """
    # Cost: 0.005 (1k tokens)
    # Budget: 0.006
    # Ratio: 0.8333...

    # Let's try extremely small numbers if we can simulate them,
    # but Pricer is bound by model rates.
    # We can use a custom Pricer/Rates if needed, but let's stick to standard flow.

    # Let's test "barely under budget"
    # Budget = 0.0050000001
    # Cost = 0.005
    # Ratio ~99.999%
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=0,
        max_budget=Budget(financial=0.005000001, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=0.99,
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True
    assert result.warning is True
    assert "Financial budget at 100.0%" in str(result.message) or "99.9%" in str(result.message)
