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
from pydantic import ValidationError
from coreason_economist.models import Budget, RequestPayload, EconomicTrace, Decision


def test_budget_creation():
    """Test creating a Budget with valid values."""
    budget = Budget(financial=0.50, latency_ms=5000, token_volume=128000)
    assert budget.financial == 0.50
    assert budget.latency_ms == 5000.0
    assert budget.token_volume == 128000


def test_budget_defaults():
    """Test Budget defaults."""
    budget = Budget()
    assert budget.financial == 0.0
    assert budget.latency_ms == 0.0
    assert budget.token_volume == 0


def test_budget_validation_negative():
    """Test that Budget rejects negative values."""
    with pytest.raises(ValidationError):
        Budget(financial=-1.0)

    with pytest.raises(ValidationError):
        Budget(latency_ms=-100)

    with pytest.raises(ValidationError):
        Budget(token_volume=-1)


def test_request_payload_creation():
    """Test creating a RequestPayload."""
    payload = RequestPayload(
        model_name="gpt-4",
        prompt="Hello world",
        estimated_output_tokens=100
    )
    assert payload.model_name == "gpt-4"
    assert payload.prompt == "Hello world"
    assert payload.estimated_output_tokens == 100


def test_request_payload_with_budget():
    """Test RequestPayload with a nested Budget."""
    budget = Budget(financial=1.0)
    payload = RequestPayload(
        model_name="gpt-4",
        prompt="test",
        max_budget=budget
    )
    assert payload.max_budget.financial == 1.0


def test_economic_trace_creation():
    """Test creating an EconomicTrace."""
    est_budget = Budget(financial=0.1)
    act_budget = Budget(financial=0.08)

    trace = EconomicTrace(
        estimated_cost=est_budget,
        actual_cost=act_budget,
        decision=Decision.APPROVED,
        voc_score=0.9,
        model_used="gpt-4",
        reason="Good to go"
    )

    assert trace.decision == Decision.APPROVED
    assert trace.voc_score == 0.9
    assert trace.estimated_cost.financial == 0.1
    assert trace.actual_cost.financial == 0.08


def test_economic_trace_validation():
    """Test EconomicTrace validation."""
    est_budget = Budget(financial=0.1)

    # voc_score must be <= 1.0
    with pytest.raises(ValidationError):
        EconomicTrace(
            estimated_cost=est_budget,
            decision=Decision.APPROVED,
            voc_score=1.5,
            model_used="gpt-4"
        )
