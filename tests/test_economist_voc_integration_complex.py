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
from coreason_economist.models import Budget, Decision, ReasoningTrace, RequestPayload, VOCDecision


def test_opportunity_cost_end_to_end() -> None:
    """
    Complex Scenario: Verify that passing critical budget values through Economist
    correctly triggers the internal logic of the real VOCEngine (Opportunity Cost).

    We compare two identical traces.
    One with ample budget -> Should CONTINUE (similarity below default 0.95).
    One with critical budget -> Should STOP (threshold lowered, so similarity is now 'enough').
    """
    economist = Economist()  # Uses real default VOCEngine (threshold=0.95)

    # Create two steps that are very similar but not identical (e.g. >90% but <95%)
    # "The quick brown fox jumps over the lazy dog" (43 chars)
    # "The quick brown fox jumps over the lazy cat" (43 chars)
    # Difference is 'dog' vs 'cat'.
    # difflib.SequenceMatcher ratio:
    # M = 40 (matches) * 2 / 86 = 80/86 ~= 0.93
    step1 = "The quick brown fox jumps over the lazy dog."
    step2 = "The quick brown fox jumps over the lazy cat."
    trace = ReasoningTrace(steps=[step1, step2])

    # Case A: Ample Budget (50% remaining)
    # Critical threshold is 20%. 50% > 20%, so no modifier.
    # Similarity ~0.93 < 0.95 (default). -> CONTINUE.
    budget_total = Budget(financial=1.0, latency_ms=1000, token_volume=1000)
    budget_ample = Budget(financial=0.5, latency_ms=500, token_volume=500)

    result_ample = economist.should_continue(trace=trace, remaining_budget=budget_ample, total_budget=budget_total)

    assert result_ample.decision == VOCDecision.CONTINUE
    assert "Significant change detected" in result_ample.reason

    # Case B: Critical Budget (10% remaining)
    # 10% < 20%. Modifier applied (threshold * 0.9).
    # New effective threshold = 0.95 * 0.9 = 0.855.
    # Similarity ~0.93 >= 0.855. -> STOP.
    budget_critical = Budget(financial=0.1, latency_ms=100, token_volume=100)

    result_critical = economist.should_continue(
        trace=trace, remaining_budget=budget_critical, total_budget=budget_total
    )

    assert result_critical.decision == VOCDecision.STOP
    assert "Diminishing returns detected" in result_critical.reason
    assert "Opportunity Cost" in result_critical.reason


def test_workflow_interleaved_responsibilities() -> None:
    """
    Complex Scenario: Verify that Economist can handle interleaved calls to
    check_execution (Budget Authority) and should_continue (VOC Engine)
    simulating a real agent loop.
    """
    economist = Economist()

    # Defined budgets
    # Request budget: $1.00
    req_budget = Budget(financial=1.0, latency_ms=10000, token_volume=10000)

    # Step 1: Check Execution
    payload1 = RequestPayload(model_name="gpt-4o-mini", prompt="Step 1 prompt", max_budget=req_budget)
    trace1 = economist.check_execution(payload1)
    assert trace1.decision == Decision.APPROVED

    # Step 1: Execution (simulated) -> Output "Hello"
    reasoning_trace = ReasoningTrace(steps=["Hello"])

    # Step 1: VOC Check
    voc1 = economist.should_continue(trace=reasoning_trace)
    assert voc1.decision == VOCDecision.CONTINUE  # Not enough history

    # Step 2: Check Execution
    # Assume previous cost consumed some budget, but we are stateless here regarding cumulative,
    # unless we manage it externally. The test just checks the method calls work.
    payload2 = RequestPayload(model_name="gpt-4o-mini", prompt="Step 2 prompt", max_budget=req_budget)
    trace2 = economist.check_execution(payload2)
    assert trace2.decision == Decision.APPROVED

    # Step 2: Execution -> Output "Hello World"
    # Similarity "Hello" vs "Hello World".
    # "Hello" (5) vs "Hello World" (11). Matches: "Hello" (5).
    # Ratio = 2*5 / (5+11) = 10/16 = 0.625.
    reasoning_trace.steps.append("Hello World")

    voc2 = economist.should_continue(trace=reasoning_trace)
    assert voc2.decision == VOCDecision.CONTINUE

    # Step 3: Execution -> Output "Hello World" (Convergence)
    reasoning_trace.steps.append("Hello World")

    voc3 = economist.should_continue(trace=reasoning_trace)
    assert voc3.decision == VOCDecision.STOP
    assert "Diminishing returns" in voc3.reason
