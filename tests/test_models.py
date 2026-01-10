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
from coreason_economist.models import Budget, Decision, EconomicTrace, RequestPayload
from pydantic import ValidationError


def test_budget_creation() -> None:
    """Test creating a Budget with valid values."""
    budget = Budget(financial=0.50, latency_ms=5000, token_volume=128000)
    assert budget.financial == 0.50
    assert budget.latency_ms == 5000.0
    assert budget.token_volume == 128000


def test_budget_defaults() -> None:
    """Test Budget defaults."""
    budget = Budget()
    assert budget.financial == 0.0
    assert budget.latency_ms == 0.0
    assert budget.token_volume == 0


def test_budget_validation_negative() -> None:
    """Test that Budget rejects negative values."""
    with pytest.raises(ValidationError):
        Budget(financial=-1.0)

    with pytest.raises(ValidationError):
        Budget(latency_ms=-100)

    with pytest.raises(ValidationError):
        Budget(token_volume=-1)


def test_budget_immutability() -> None:
    """Test that Budget is immutable."""
    budget = Budget(financial=10.0)
    with pytest.raises(ValidationError):
        # Type ignore used because mypy knows it's frozen and would flag this statically
        budget.financial = 20.0  # type: ignore


def test_budget_equality() -> None:
    """Test value equality for Budget objects."""
    b1 = Budget(financial=10.0, latency_ms=100.0, token_volume=1000)
    b2 = Budget(financial=10.0, latency_ms=100.0, token_volume=1000)
    b3 = Budget(financial=20.0, latency_ms=100.0, token_volume=1000)

    assert b1 == b2
    assert b1 != b3


def test_request_payload_creation() -> None:
    """Test creating a RequestPayload."""
    payload = RequestPayload(model_name="gpt-4", prompt="Hello world", estimated_output_tokens=100)
    assert payload.model_name == "gpt-4"
    assert payload.prompt == "Hello world"
    assert payload.estimated_output_tokens == 100


def test_request_payload_validation_missing_fields() -> None:
    """Test that RequestPayload requires mandatory fields."""
    with pytest.raises(ValidationError):
        # Missing prompt
        RequestPayload(model_name="gpt-4")  # type: ignore

    with pytest.raises(ValidationError):
        # Missing model_name
        RequestPayload(prompt="Hello")  # type: ignore


def test_request_payload_with_budget() -> None:
    """Test RequestPayload with a nested Budget."""
    budget = Budget(financial=1.0)
    payload = RequestPayload(model_name="gpt-4", prompt="test", max_budget=budget)
    assert payload.max_budget is not None
    assert payload.max_budget.financial == 1.0


def test_request_payload_complex_tool_calls() -> None:
    """Test RequestPayload with complex nested tool calls."""
    tool_calls = [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "San Francisco, CA", "unit": "celsius", "nested": {"key": [1, 2, 3]}},
            },
        }
    ]
    payload = RequestPayload(model_name="gpt-4", prompt="Check weather", tool_calls=tool_calls)
    assert payload.tool_calls is not None
    assert payload.tool_calls[0]["function"]["arguments"]["nested"]["key"] == [1, 2, 3]


def test_unicode_handling() -> None:
    """Test handling of Unicode characters."""
    prompt = "Hello 🌍! 你好!"
    model = "gpt-4-🚀"
    payload = RequestPayload(model_name=model, prompt=prompt)

    assert payload.prompt == prompt
    assert payload.model_name == model

    # Ensure serialization preserves unicode
    json_str = payload.model_dump_json()
    assert "🌍" in json_str or "\\ud83c\\udf0d" in json_str  # Depends on json dumper settings


def test_extreme_values() -> None:
    """Test handling of large numbers."""
    large_financial = 1e9  # 1 billion dollars
    large_tokens = 10**9   # 1 billion tokens

    budget = Budget(financial=large_financial, token_volume=large_tokens)
    assert budget.financial == large_financial
    assert budget.token_volume == large_tokens


def test_economic_trace_creation() -> None:
    """Test creating an EconomicTrace."""
    est_budget = Budget(financial=0.1)
    act_budget = Budget(financial=0.08)

    trace = EconomicTrace(
        estimated_cost=est_budget,
        actual_cost=act_budget,
        decision=Decision.APPROVED,
        voc_score=0.9,
        model_used="gpt-4",
        reason="Good to go",
    )

    assert trace.decision == Decision.APPROVED
    assert trace.voc_score == 0.9
    assert trace.estimated_cost.financial == 0.1
    assert trace.actual_cost is not None
    assert trace.actual_cost.financial == 0.08


def test_economic_trace_validation() -> None:
    """Test EconomicTrace validation."""
    est_budget = Budget(financial=0.1)

    # voc_score must be <= 1.0
    with pytest.raises(ValidationError):
        EconomicTrace(estimated_cost=est_budget, decision=Decision.APPROVED, voc_score=1.5, model_used="gpt-4")

    # voc_score must be >= 0.0
    with pytest.raises(ValidationError):
        EconomicTrace(estimated_cost=est_budget, decision=Decision.APPROVED, voc_score=-0.1, model_used="gpt-4")


def test_voc_score_boundaries() -> None:
    """Test boundary values for VOC score."""
    est_budget = Budget(financial=0.1)

    # Test 0.0
    trace_zero = EconomicTrace(estimated_cost=est_budget, decision=Decision.REJECTED, voc_score=0.0, model_used="gpt-4")
    assert trace_zero.voc_score == 0.0

    # Test 1.0
    trace_one = EconomicTrace(estimated_cost=est_budget, decision=Decision.APPROVED, voc_score=1.0, model_used="gpt-4")
    assert trace_one.voc_score == 1.0


def test_json_roundtrip() -> None:
    """Test JSON serialization and deserialization."""
    budget = Budget(financial=10.5, latency_ms=100.0, token_volume=500)
    original_trace = EconomicTrace(
        estimated_cost=budget,
        decision=Decision.MODIFIED,
        voc_score=0.75,
        model_used="gpt-3.5-turbo",
        reason="Cost optimized",
    )

    # Serialize
    json_str = original_trace.model_dump_json()

    # Deserialize
    restored_trace = EconomicTrace.model_validate_json(json_str)

    assert restored_trace == original_trace
    assert restored_trace.estimated_cost.financial == 10.5
    assert restored_trace.decision == Decision.MODIFIED
