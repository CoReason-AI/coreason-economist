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
from coreason_economist.economist import Economist
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer


@pytest.fixture  # type: ignore
def budget_authority() -> BudgetAuthority:
    return BudgetAuthority(pricer=Pricer())


@pytest.fixture  # type: ignore
def economist() -> Economist:
    return Economist()


def test_no_warning_when_usage_low(budget_authority: BudgetAuthority) -> None:
    """Ensure no warning is triggered when usage is well below threshold."""
    # Cost is very low, budget is high
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="short",
        estimated_output_tokens=10,
        max_budget=Budget(financial=10.0, latency_ms=10000, token_volume=10000),
        soft_limit_threshold=0.8,
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True
    assert result.warning is False
    assert result.message is None


def test_warning_at_soft_limit_financial(budget_authority: BudgetAuthority) -> None:
    """Test warning when financial usage > 80% but <= 100%."""
    # 1000 tokens input for gpt-4o = $0.005
    # Let's set budget to $0.006. Usage = 0.005/0.006 = ~83%
    # We must ensure other budgets are sufficient.
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,  # ~1000 tokens
        estimated_output_tokens=0,
        max_budget=Budget(financial=0.006, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=0.8,
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True
    assert result.warning is True
    assert "Financial budget at 83.3%" in str(result.message)


def test_warning_at_soft_limit_token_volume(budget_authority: BudgetAuthority) -> None:
    """Test warning when token volume usage > 80%."""
    # Usage 900, Limit 1000 -> 90%
    # Financial cost > 0, so must set financial budget high.
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 3600,  # ~900 tokens
        estimated_output_tokens=0,
        max_budget=Budget(financial=10.0, latency_ms=100000, token_volume=1000),
        soft_limit_threshold=0.8,
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True
    assert result.warning is True
    assert "Token volume budget at 90.0%" in str(result.message)


def test_multiple_warnings(budget_authority: BudgetAuthority) -> None:
    """Test that multiple warnings are aggregated."""
    # Financial ~83% ($0.005 / $0.006)
    # Token Volume 90% (1000 / 1111) approx
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,  # ~1000 tokens input ($0.005)
        estimated_output_tokens=0,
        max_budget=Budget(financial=0.006, latency_ms=100000, token_volume=1111),
        soft_limit_threshold=0.8,
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True
    assert result.warning is True
    assert "Financial budget at" in str(result.message)
    assert "Token volume budget at" in str(result.message)


def test_custom_threshold(budget_authority: BudgetAuthority) -> None:
    """Test that the threshold can be customized via RequestPayload."""
    # Usage ~50% ($0.005 / $0.010)
    # Default 0.8 -> No Warning
    # Custom 0.4 -> Warning
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=0,
        max_budget=Budget(financial=0.010, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=0.4,
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True
    assert result.warning is True
    assert "Financial budget at 50.0%" in str(result.message)


def test_economist_propagates_warning(economist: Economist) -> None:
    """Ensure Economist puts the warning into the trace."""
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,  # ~1000 tokens ($0.005)
        estimated_output_tokens=0,
        max_budget=Budget(financial=0.006, latency_ms=100000, token_volume=100000),
        soft_limit_threshold=0.8,
    )
    trace = economist.check_execution(req)
    assert trace.decision.value == "APPROVED"
    assert trace.budget_warning is True
    assert trace.warning_message is not None
    assert "Financial budget at" in trace.warning_message
