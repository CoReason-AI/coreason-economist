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
from coreason_economist.exceptions import BudgetExhaustedError
from coreason_economist.models import AuthResult, Budget, RequestPayload
from coreason_economist.pricer import Pricer


@pytest.fixture  # type: ignore
def budget_authority() -> BudgetAuthority:
    return BudgetAuthority(pricer=Pricer())


def test_allow_execution_no_limits(budget_authority: BudgetAuthority) -> None:
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello world",
        max_budget=None,  # No limit
    )
    result = budget_authority.allow_execution(req)
    assert isinstance(result, AuthResult)
    assert result.allowed is True
    assert result.warning is False


def test_allow_execution_within_limits(budget_authority: BudgetAuthority) -> None:
    # Estimate: ~1000 input tokens -> cost $0.005, latency ~200ms
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=10,
        max_budget=Budget(financial=1.0, latency_ms=5000, token_volume=10000),
    )
    result = budget_authority.allow_execution(req)
    assert result.allowed is True


def test_allow_execution_financial_exceeded(budget_authority: BudgetAuthority) -> None:
    # 100k tokens input -> $0.50 input cost
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 400000,
        estimated_output_tokens=10,
        max_budget=Budget(financial=0.10, latency_ms=5000, token_volume=1_000_000),
    )
    with pytest.raises(BudgetExhaustedError) as excinfo:
        budget_authority.allow_execution(req)
    assert "Financial budget exceeded" in str(excinfo.value)


def test_allow_execution_latency_exceeded(budget_authority: BudgetAuthority) -> None:
    # 1000 output tokens -> 12000ms latency (12ms/token)
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a",
        estimated_output_tokens=1000,
        max_budget=Budget(financial=10.0, latency_ms=500, token_volume=10000),
    )
    with pytest.raises(BudgetExhaustedError) as excinfo:
        budget_authority.allow_execution(req)
    assert "Latency budget exceeded" in str(excinfo.value)


def test_allow_execution_token_volume_exceeded(budget_authority: BudgetAuthority) -> None:
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,  # 1000 tokens
        estimated_output_tokens=100,
        max_budget=Budget(financial=10.0, latency_ms=50000, token_volume=500),
    )
    with pytest.raises(BudgetExhaustedError) as excinfo:
        budget_authority.allow_execution(req)
    assert "Token volume budget exceeded" in str(excinfo.value)


def test_zero_limits_strict(budget_authority: BudgetAuthority) -> None:
    # Limits set to 0 are STRICT (no budget).
    # If cost > 0, it should fail.
    req = RequestPayload(
        model_name="gpt-4o",
        prompt="a" * 4000,
        estimated_output_tokens=10,
        max_budget=Budget(financial=0.0, latency_ms=0.0, token_volume=0),
    )
    with pytest.raises(BudgetExhaustedError):
        budget_authority.allow_execution(req)
