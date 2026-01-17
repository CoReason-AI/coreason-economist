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


def test_economist_check_execution_approved() -> None:
    """Test that check_execution returns APPROVED when within budget."""
    economist = Economist()

    # Cheap request
    request = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello",
        estimated_output_tokens=10,
        max_budget=Budget(financial=1.0, latency_ms=5000, token_volume=1000),
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.APPROVED
    assert trace.model_used == "gpt-4o-mini"
    assert trace.estimated_cost.financial < 1.0
    assert trace.reason == "Budget check passed."


def test_economist_check_execution_rejected_financial() -> None:
    """Test that check_execution returns REJECTED when financial budget exceeded."""
    economist = Economist()

    # Request that will exceed very low budget
    # Using gpt-4o which is in the default rates
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A very long prompt " * 100,
        estimated_output_tokens=1000,
        max_budget=Budget(financial=0.0000001, latency_ms=5000, token_volume=100000),
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert trace.model_used == "gpt-4o"
    assert trace.reason is not None
    assert "Financial budget exceeded" in trace.reason


def test_economist_check_execution_rejected_latency() -> None:
    """Test that check_execution returns REJECTED when latency budget exceeded."""
    economist = Economist()

    # Request that will likely exceed low latency budget
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello",
        estimated_output_tokens=100,
        max_budget=Budget(financial=10.0, latency_ms=1.0, token_volume=100000),
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert trace.reason is not None
    assert "Latency budget exceeded" in trace.reason


def test_economist_check_execution_rejected_token_volume() -> None:
    """Test that check_execution returns REJECTED when token volume budget exceeded."""
    economist = Economist()

    # Request that will exceed token volume budget
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello " * 100,
        estimated_output_tokens=1000,
        max_budget=Budget(financial=100.0, latency_ms=100000.0, token_volume=50),
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.REJECTED
    assert trace.reason is not None
    assert "Token volume budget exceeded" in trace.reason


def test_economist_check_execution_no_budget() -> None:
    """Test that check_execution approves when no budget limit is set."""
    economist = Economist()

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello",
        estimated_output_tokens=100,
        max_budget=None,  # No limits
    )

    trace = economist.check_execution(request)

    assert trace.decision == Decision.APPROVED
