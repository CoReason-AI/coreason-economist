# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.models import Budget, Decision, EconomicTrace, RequestPayload


def test_budget_creation() -> None:
    """Test creating a Budget object."""
    budget = Budget(financial=0.1, latency_ms=100.0, token_volume=1000)
    assert budget.financial == 0.1
    assert budget.latency_ms == 100.0
    assert budget.token_volume == 1000


def test_budget_defaults() -> None:
    """Test Budget default values."""
    budget = Budget()
    assert budget.financial == 0.0
    assert budget.latency_ms == 0.0
    assert budget.token_volume == 0


def test_request_payload_creation() -> None:
    """Test creating a RequestPayload."""
    payload = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello",
        agent_count=2,
        rounds=3,
        quality_warning="This is a warning",
    )
    assert payload.model_name == "gpt-4o"
    assert payload.agent_count == 2
    assert payload.rounds == 3
    assert payload.quality_warning == "This is a warning"


def test_request_payload_defaults() -> None:
    """Test RequestPayload defaults."""
    payload = RequestPayload(model_name="gpt-4o", prompt="Hello")
    assert payload.agent_count == 1
    assert payload.rounds == 1
    assert payload.estimated_output_tokens is None
    assert payload.quality_warning is None


def test_economic_trace_creation() -> None:
    """Test creating an EconomicTrace."""
    budget = Budget(financial=0.1)
    trace = EconomicTrace(
        estimated_cost=budget,
        decision=Decision.APPROVED,
        model_used="gpt-4o",
        input_tokens=100,
    )
    assert trace.estimated_cost == budget
    assert trace.decision == Decision.APPROVED
    assert trace.model_used == "gpt-4o"
    assert trace.input_tokens == 100


def test_economic_trace_computed_fields() -> None:
    """Test that computed fields are calculated correctly and serialized."""
    # Scenario 1: Basic case with actual cost
    actual_budget = Budget(financial=1.0, latency_ms=2000.0, token_volume=1000)
    trace = EconomicTrace(
        estimated_cost=Budget(financial=0.5),  # Different estimate
        actual_cost=actual_budget,
        decision=Decision.APPROVED,
        model_used="gpt-4o",
        input_tokens=100,
    )

    # tokens_per_dollar = 1000 / 1.0 = 1000.0
    assert trace.tokens_per_dollar == 1000.0

    # tokens_per_second = 1000 / (2000ms / 1000) = 1000 / 2.0 = 500.0
    assert trace.tokens_per_second == 500.0

    # latency_per_token = 2000ms / 1000 = 2.0 ms/token
    assert trace.latency_per_token == 2.0

    # Check serialization
    trace_json = trace.model_dump(mode="json")
    assert "tokens_per_dollar" in trace_json
    assert "tokens_per_second" in trace_json
    assert "latency_per_token" in trace_json
    assert trace_json["tokens_per_dollar"] == 1000.0
    assert trace_json["tokens_per_second"] == 500.0
    assert trace_json["latency_per_token"] == 2.0


def test_economic_trace_computed_fields_fallback() -> None:
    """Test that computed fields fall back to estimated cost if actual is None."""
    est_budget = Budget(financial=2.0, latency_ms=4000.0, token_volume=2000)
    trace = EconomicTrace(
        estimated_cost=est_budget,
        actual_cost=None,
        decision=Decision.APPROVED,
        model_used="gpt-4o",
        input_tokens=100,
    )

    # tokens_per_dollar = 2000 / 2.0 = 1000.0
    assert trace.tokens_per_dollar == 1000.0

    # tokens_per_second = 2000 / 4.0 = 500.0
    assert trace.tokens_per_second == 500.0

    # latency_per_token = 4000 / 2000 = 2.0
    assert trace.latency_per_token == 2.0


def test_economic_trace_computed_fields_zero_values() -> None:
    """Test division by zero handling in computed fields."""
    zero_budget = Budget(financial=0.0, latency_ms=0.0, token_volume=0)
    trace = EconomicTrace(
        estimated_cost=zero_budget,
        decision=Decision.APPROVED,
        model_used="gpt-4o",
        input_tokens=100,
    )

    # Should handle division by zero safely
    assert trace.tokens_per_dollar == 0.0
    assert trace.tokens_per_second == 0.0
    assert trace.latency_per_token == 0.0

    # Test only financial zero
    partial_budget = Budget(financial=0.0, latency_ms=1000.0, token_volume=100)
    trace_partial = EconomicTrace(
        estimated_cost=partial_budget,
        decision=Decision.APPROVED,
        model_used="gpt-4o",
        input_tokens=100,
    )
    assert trace_partial.tokens_per_dollar == 0.0
    assert trace_partial.tokens_per_second == 100.0  # 100 / 1.0
    assert trace_partial.latency_per_token == 10.0  # 1000 / 100
