# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from unittest.mock import Mock

import pytest
from coreason_economist.budget_authority import BudgetAuthority
from coreason_economist.exceptions import BudgetExhaustedError
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer


@pytest.fixture  # type: ignore[misc]
def mock_pricer() -> Mock:
    pricer = Mock(spec=Pricer)
    return pricer


@pytest.fixture  # type: ignore[misc]
def authority(mock_pricer: Mock) -> BudgetAuthority:
    return BudgetAuthority(pricer=mock_pricer)


def test_allow_execution_no_budget_limit(authority: BudgetAuthority) -> None:
    """Test that requests with no max_budget are allowed."""
    request = RequestPayload(model_name="gpt-4o", prompt="Hello", max_budget=None)
    assert authority.allow_execution(request) is True


def test_allow_execution_within_limits(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """Test that requests within all budget limits are allowed."""
    mock_pricer.estimate_request_cost.return_value = Budget(financial=0.01, latency_ms=100.0, token_volume=100)

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello world",
        max_budget=Budget(financial=0.05, latency_ms=500.0, token_volume=1000),
    )

    assert authority.allow_execution(request) is True
    mock_pricer.estimate_request_cost.assert_called_once()


def test_reject_financial_limit(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """Test rejection when financial cost exceeds budget."""
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.10,  # Exceeds 0.05
        latency_ms=100.0,
        token_volume=100,
    )

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Expensive query",
        max_budget=Budget(financial=0.05, latency_ms=500.0, token_volume=1000),
    )

    with pytest.raises(BudgetExhaustedError) as exc_info:
        authority.allow_execution(request)

    assert exc_info.value.limit_type == "financial"
    assert exc_info.value.limit_value == 0.05
    assert exc_info.value.estimated_value == 0.10
    assert "Financial budget exceeded" in str(exc_info.value)


def test_reject_latency_limit(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """Test rejection when latency exceeds budget."""
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.01,
        latency_ms=1000.0,  # Exceeds 500
        token_volume=100,
    )

    request = RequestPayload(
        model_name="gpt-4o", prompt="Slow query", max_budget=Budget(financial=0.05, latency_ms=500.0, token_volume=1000)
    )

    with pytest.raises(BudgetExhaustedError) as exc_info:
        authority.allow_execution(request)

    assert exc_info.value.limit_type == "latency_ms"
    assert exc_info.value.limit_value == 500.0
    assert exc_info.value.estimated_value == 1000.0
    assert "Latency budget exceeded" in str(exc_info.value)


def test_reject_token_volume_limit(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """Test rejection when token volume exceeds budget."""
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.01,
        latency_ms=100.0,
        token_volume=2000,  # Exceeds 1000
    )

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Big context",
        max_budget=Budget(financial=0.05, latency_ms=500.0, token_volume=1000),
    )

    with pytest.raises(BudgetExhaustedError) as exc_info:
        authority.allow_execution(request)

    assert exc_info.value.limit_type == "token_volume"
    assert exc_info.value.limit_value == 1000
    assert exc_info.value.estimated_value == 2000
    assert "Token volume budget exceeded" in str(exc_info.value)


def test_pricer_integration() -> None:
    """Integration test with the real Pricer (no mocks)."""
    authority = BudgetAuthority()  # uses real default Pricer

    request = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello world",
        max_budget=Budget(
            financial=1.0,  # Plenty
            latency_ms=1000.0,  # Plenty
            token_volume=1000,  # Plenty
        ),
    )

    assert authority.allow_execution(request) is True

    # Test failure with real pricer
    request_tiny = RequestPayload(
        model_name="gpt-4o-mini",
        prompt="Hello world",
        max_budget=Budget(
            financial=0.0,  # Zero budget
            latency_ms=0.0,
            token_volume=0,
        ),
    )

    with pytest.raises(BudgetExhaustedError):
        authority.allow_execution(request_tiny)


# --- Edge Case Tests ---


def test_boundary_exact_match(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """Test edge case where cost equals budget limit exactly (should pass)."""
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=1.00,
        latency_ms=100.0,
        token_volume=100,
    )

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Test",
        max_budget=Budget(
            financial=1.00,  # Exact match
            latency_ms=100.0,  # Exact match
            token_volume=100,  # Exact match
        ),
    )

    # Should not raise
    assert authority.allow_execution(request) is True


def test_boundary_epsilon_fail(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """Test edge case where cost is slightly above budget (should fail)."""
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=1.0000001,
        latency_ms=100.0,
        token_volume=100,
    )

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Test",
        max_budget=Budget(financial=1.00, latency_ms=100.0, token_volume=100),
    )

    with pytest.raises(BudgetExhaustedError) as exc:
        authority.allow_execution(request)
    assert exc.value.limit_type == "financial"


def test_zero_budget_zero_cost(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """Test zero budget allows zero cost."""
    mock_pricer.estimate_request_cost.return_value = Budget(financial=0.0, latency_ms=0.0, token_volume=0)

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="",
        max_budget=Budget(financial=0.0, latency_ms=0.0, token_volume=0),
    )

    assert authority.allow_execution(request) is True


def test_failure_precedence(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """
    Verify precedence: Financial > Latency > Token Volume.
    If multiple limits are exceeded, the first one checked should raise.
    """
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=10.0,  # Exceeds 1.0
        latency_ms=1000.0,  # Exceeds 100.0
        token_volume=1000,  # Exceeds 100
    )

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Test",
        max_budget=Budget(financial=1.0, latency_ms=100.0, token_volume=100),
    )

    with pytest.raises(BudgetExhaustedError) as exc:
        authority.allow_execution(request)

    # Expect financial error first
    assert exc.value.limit_type == "financial"


def test_simulated_workflow_drain(authority: BudgetAuthority, mock_pricer: Mock) -> None:
    """
    Complex scenario: Simulating a client making requests and draining budget.
    """
    # Cost is fixed per request for this simulation
    cost_per_req = 0.40
    mock_pricer.estimate_request_cost.return_value = Budget(financial=cost_per_req, latency_ms=10.0, token_volume=10)

    total_budget = 1.00
    remaining_budget = total_budget

    # Request 1: 0.40 cost, 1.00 budget. OK.
    req1 = RequestPayload(
        model_name="gpt-4o",
        prompt="Req1",
        max_budget=Budget(financial=remaining_budget, latency_ms=100, token_volume=100),
    )
    assert authority.allow_execution(req1) is True
    remaining_budget -= cost_per_req  # 0.60 left

    # Request 2: 0.40 cost, 0.60 budget. OK.
    req2 = RequestPayload(
        model_name="gpt-4o",
        prompt="Req2",
        max_budget=Budget(financial=remaining_budget, latency_ms=100, token_volume=100),
    )
    assert authority.allow_execution(req2) is True
    remaining_budget -= cost_per_req  # 0.20 left

    # Request 3: 0.40 cost, 0.20 budget. FAIL.
    req3 = RequestPayload(
        model_name="gpt-4o",
        prompt="Req3",
        max_budget=Budget(financial=remaining_budget, latency_ms=100, token_volume=100),
    )

    with pytest.raises(BudgetExhaustedError) as exc:
        authority.allow_execution(req3)

    assert exc.value.limit_type == "financial"
    assert exc.value.limit_value == pytest.approx(0.20)
    assert exc.value.estimated_value == 0.40
