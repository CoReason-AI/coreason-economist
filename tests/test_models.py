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
    )
    assert trace.estimated_cost == budget
    assert trace.decision == Decision.APPROVED
    assert trace.model_used == "gpt-4o"
