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


def test_impossible_budget() -> None:
    """
    Edge Case: Budget is so low that absolutely nothing fits.
    Arbitrageur should return None.
    """
    economist = Economist()

    # Budget: $0.00000001 (Tiny)
    # Cheapest model (mini) 1 token ~ 0.00000015 (still > budget)
    request = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="A",  # 1 token
        estimated_output_tokens=1,
        max_budget=Budget(financial=1e-9),  # 1 nanodollar
    )

    trace = economist.check_execution(request)
    assert trace.decision == Decision.REJECTED
    assert trace.suggested_alternative is None


def test_tool_cost_dominance() -> None:
    """
    Edge Case: Tool cost is the main driver and exceeds budget on its own.
    Arbitrageur cannot strip tools, so it fails to find an alternative.
    """
    economist = Economist()

    # Tool cost: $0.01
    # Budget: $0.005
    tool_calls = [{"name": "web_search"}]

    request = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Search",
        estimated_output_tokens=10,
        tool_calls=tool_calls,
        max_budget=Budget(financial=0.005),
    )

    trace = economist.check_execution(request)
    assert trace.decision == Decision.REJECTED
    assert "Financial budget exceeded" in str(trace.reason)
    assert trace.suggested_alternative is None


def test_latency_driven_downgrade() -> None:
    """
    Edge Case: Financial budget is fine, but Latency constraint fails.
    Switching to a faster (and cheaper) model fixes it.
    """
    economist = Economist()

    # GPT-4o: 12ms/token. 100 tokens -> 1200ms.
    # GPT-4o-mini: 8ms/token. 100 tokens -> 800ms.
    # Budget: 1000ms.
    # Financial Budget: Unlimited (or high enough).

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello",
        estimated_output_tokens=100,
        max_budget=Budget(financial=1.0, latency_ms=1000.0),
        difficulty_score=0.9,  # High difficulty, so we need "Budget Fitting" logic
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert "Latency budget exceeded" in str(trace.reason)

    suggestion = trace.suggested_alternative
    assert suggestion is not None
    assert suggestion.model_name == "gpt-4o-mini"
    assert "Downgraded" in suggestion.quality_warning


def test_llama_optimization() -> None:
    """
    Complex Scenario: User requests Llama 3.1 70B for a Low Difficulty task.
    Arbitrageur optimizes to GPT-4o-mini because it is cheaper.
    Llama: $0.88/$0.88
    Mini: $0.15/$0.60
    """
    economist = Economist()

    # We use a constrained budget to force rejection first, triggering the Arbitrageur.
    # Or rely on the "Low Difficulty" optimization path if we implement proactive optimization.
    # Currently, optimization is triggered on rejection.
    # So we set a budget that Llama fails but Mini passes.

    # Llama Cost (1k/1k): ~$0.00176.
    # Mini Cost: ~$0.00075.
    # Budget: $0.001.

    request = RequestPayload(
        model_name="llama-3.1-70b",
        prompt="A" * 4000,  # 1k
        estimated_output_tokens=1000,
        max_budget=Budget(financial=0.001),
        difficulty_score=0.9,  # High difficulty, force fit
    )

    trace = economist.check_execution(request)
    assert trace.decision == Decision.REJECTED
    assert trace.suggested_alternative is not None
    assert trace.suggested_alternative.model_name == "gpt-4o-mini"


def test_triple_constraint_squeeze() -> None:
    """
    Complex Scenario: "Triple Constraint Squeeze".
    Fails Financial, Latency, AND Token Volume (context limit).
    Downgrade/Topology Reduction fixes all three.
    """
    economist = Economist()

    # Request: 5 Agents, 3 Rounds.
    # Tokens: 1000 in, 1000 out per agent/round.
    # Total Tokens: 2000 * 15 = 30,000.
    # Latency: 12ms * 1000 * 3 = 36,000ms (36s).
    # Cost: High.

    # Budget:
    # Token Volume: 5,000 (Request 30k -> Fail)
    # Latency: 5,000ms (Request 36k -> Fail)
    # Financial: Low.

    budget = Budget(financial=0.1, latency_ms=5000.0, token_volume=5000)

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 4000,
        estimated_output_tokens=1000,
        agent_count=5,
        rounds=3,
        max_budget=budget,
        difficulty_score=0.9,
    )

    trace = economist.check_execution(request)
    assert trace.decision == Decision.REJECTED

    suggestion = trace.suggested_alternative
    # Phase 1: Budget is Impossible (Latency 5s vs 12s min).
    assert suggestion is None

    # Phase 2: Feasible Budget.
    # Latency Budget: 9000ms.
    # 4o: 12s (Fail).
    # Mini: 8s (Pass).

    budget_feasible = Budget(financial=0.1, latency_ms=9000.0, token_volume=5000)

    request.max_budget = budget_feasible
    trace = economist.check_execution(request)

    suggestion = trace.suggested_alternative
    assert suggestion is not None
    assert suggestion.model_name == "gpt-4o-mini"
    assert suggestion.agent_count == 1
    assert suggestion.rounds == 1
