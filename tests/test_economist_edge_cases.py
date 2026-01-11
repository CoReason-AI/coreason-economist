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
from coreason_economist.economist import Economist
from coreason_economist.models import Budget, Decision, RequestPayload


def test_economist_council_rejection() -> None:
    """
    Edge Case: Request passes for single agent but fails when scaled to a council.
    Demonstrates Multi-Agent Cost Scaling logic in integration.
    """
    economist = Economist()

    # Cheap request: 10 in, 10 out.
    # gpt-4o-mini rates:
    # In: 0.00015 / 1k -> ~0.0000015 for 10
    # Out: 0.0006 / 1k -> ~0.000006 for 10
    # Total Unit ~ 0.0000075 USD

    budget = Budget(financial=0.0001, latency_ms=5000, token_volume=1000)

    # 1. Single Agent (Cost ~0.0000075 < 0.0001) -> APPROVED
    req_single = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello " * 2,  # ~10 chars
        estimated_output_tokens=10,
        max_budget=budget,
        agent_count=1,
        rounds=1,
    )
    trace_single = economist.check_execution(req_single)
    assert trace_single.decision == Decision.APPROVED

    # 2. Council (20 agents) -> Cost ~0.00015 > 0.0001 -> REJECTED
    req_council = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello " * 2,
        estimated_output_tokens=10,
        max_budget=budget,
        agent_count=20,
        rounds=1,
    )
    trace_council = economist.check_execution(req_council)
    assert trace_council.decision == Decision.REJECTED
    assert "Financial budget exceeded" in str(trace_council.reason)


def test_economist_tool_cost_rejection() -> None:
    """
    Edge Case: Request passes without tools, but fails when expensive tool is added.
    Demonstrates Tool Pricing integration.
    """
    economist = Economist()
    # Budget just enough for model inference but not tool
    # gpt-4o-mini Unit ~0.0000075
    # Tool "web_search" cost: $0.01 (DEFAULT_TOOL_RATES)

    budget = Budget(financial=0.005, latency_ms=5000, token_volume=1000)

    # 1. No Tools -> Cost ~0.0000075 < 0.005 -> APPROVED
    req_no_tool = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello",
        estimated_output_tokens=10,
        max_budget=budget,
    )
    trace_no_tool = economist.check_execution(req_no_tool)
    assert trace_no_tool.decision == Decision.APPROVED

    # 2. With Tool -> Cost ~0.0100075 > 0.005 -> REJECTED
    req_with_tool = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello",
        estimated_output_tokens=10,
        max_budget=budget,
        tool_calls=[{"name": "web_search"}],
    )
    trace_with_tool = economist.check_execution(req_with_tool)
    assert trace_with_tool.decision == Decision.REJECTED
    assert "Financial budget exceeded" in str(trace_with_tool.reason)


def test_economist_zero_budget_strictness() -> None:
    """
    Edge Case: Budget is strictly 0.0. Even the cheapest request should fail.
    """
    economist = Economist()
    budget = Budget(financial=0.0, latency_ms=5000, token_volume=1000)

    # Even 1 token costs > 0
    req = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="H",
        estimated_output_tokens=1,
        max_budget=budget,
    )

    trace = economist.check_execution(req)
    assert trace.decision == Decision.REJECTED
    assert "Financial budget exceeded" in str(trace.reason)


def test_economist_latency_accumulation() -> None:
    """
    Edge Case: Latency accumulates with rounds, causing rejection.
    Verify parallel execution assumption (agents don't sum latency) vs sequential rounds.
    """
    economist = Economist()
    # gpt-4o-mini latency: 8ms per output token.
    # 100 output tokens -> 800ms.
    budget = Budget(financial=10.0, latency_ms=1000.0, token_volume=100000)

    # 1. 10 Agents, 1 Round -> Latency max(800ms) = 800ms < 1000ms -> APPROVED
    req_parallel = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello",
        estimated_output_tokens=100,
        max_budget=budget,
        agent_count=10,
        rounds=1,
    )
    trace_parallel = economist.check_execution(req_parallel)
    assert trace_parallel.decision == Decision.APPROVED
    assert trace_parallel.estimated_cost.latency_ms == 800.0

    # 2. 1 Agent, 2 Rounds -> Latency 800ms * 2 = 1600ms > 1000ms -> REJECTED
    req_sequential = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello",
        estimated_output_tokens=100,
        max_budget=budget,
        agent_count=1,
        rounds=2,
    )
    trace_sequential = economist.check_execution(req_sequential)
    assert trace_sequential.decision == Decision.REJECTED
    assert "Latency budget exceeded" in str(trace_sequential.reason)


def test_economist_unknown_model_error() -> None:
    """
    Edge Case: Requesting an unknown model should raise ValueError,
    NOT return a REJECTED trace (unless we decide to handle it, but currently we don't).
    """
    economist = Economist()
    req = RequestPayload(
        model_name="unknown-model-xyz",
        prompt="Hello",
        estimated_output_tokens=10,
        max_budget=Budget(financial=1.0, latency_ms=5000, token_volume=1000),
    )

    with pytest.raises(ValueError, match="Unknown model"):
        economist.check_execution(req)
