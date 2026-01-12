# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from unittest.mock import MagicMock

import pytest
from coreason_economist.budget_authority import BudgetAuthority
from coreason_economist.economist import Economist
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer


def test_soft_limit_warning_scenario_90_percent_usage() -> None:
    """
    Verifies the specific requirement on BudgetAuthority directly:
    "if a request consumes >80% of the remaining budget ... return APPROVED with a specific warning_flag"

    Scenario:
    - Budget: $0.10
    - Estimated Cost: $0.09
    - Usage: 90% (> 80%)
    - Expectation: Allowed=True, Warning=True, Message contains "90.0%"
    """
    # 1. Setup Pricer Mock to return exactly $0.09 cost
    mock_pricer = MagicMock(spec=Pricer)
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.09,
        latency_ms=100.0,
        token_volume=100
    )

    # 2. Setup BudgetAuthority
    authority = BudgetAuthority(pricer=mock_pricer)

    # 3. Create Request with $0.10 budget and default 0.8 threshold
    request = RequestPayload(
        model_name="mock-model",
        prompt="test prompt",
        max_budget=Budget(
            financial=0.10,
            latency_ms=10000.0, # Plenty of latency budget
            token_volume=10000  # Plenty of token budget
        ),
        soft_limit_threshold=0.8
    )

    # 4. Execute
    result = authority.allow_execution(request)

    # 5. Verify
    assert result.allowed is True, "Request should be allowed as it is within budget ($0.09 < $0.10)"
    assert result.warning is True, "Warning should be triggered as usage (90%) > threshold (80%)"
    assert "Financial budget at 90.0%" in str(result.message), \
        f"Warning message should accurately report usage. Got: {result.message}"


def test_soft_limit_boundary_80_percent_usage() -> None:
    """
    Verifies that exactly 80% usage does NOT trigger a warning (strictly > 80%).

    Scenario:
    - Budget: $0.10
    - Estimated Cost: $0.08
    - Usage: 80%
    - Expectation: Allowed=True, Warning=False
    """
    mock_pricer = MagicMock(spec=Pricer)
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.08,
        latency_ms=100.0,
        token_volume=100
    )

    authority = BudgetAuthority(pricer=mock_pricer)

    request = RequestPayload(
        model_name="mock-model",
        prompt="test prompt",
        max_budget=Budget(
            financial=0.10,
            latency_ms=10000.0,
            token_volume=10000
        ),
        soft_limit_threshold=0.8
    )

    result = authority.allow_execution(request)

    assert result.allowed is True
    assert result.warning is False, "Should not warn at exactly 80% usage"


def test_soft_limit_just_above_threshold() -> None:
    """
    Verifies that just above 80% triggers the warning.

    Scenario:
    - Budget: $0.10
    - Estimated Cost: $0.08001
    - Usage: ~80.01%
    - Expectation: Allowed=True, Warning=True
    """
    mock_pricer = MagicMock(spec=Pricer)
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.08001,
        latency_ms=100.0,
        token_volume=100
    )

    authority = BudgetAuthority(pricer=mock_pricer)

    request = RequestPayload(
        model_name="mock-model",
        prompt="test prompt",
        max_budget=Budget(
            financial=0.10,
            latency_ms=10000.0,
            token_volume=10000
        ),
        soft_limit_threshold=0.8
    )

    result = authority.allow_execution(request)

    assert result.allowed is True
    assert result.warning is True


def test_economist_soft_limit_trace_mapping() -> None:
    """
    Verifies that the Economist correctly maps the AuthResult warning to the EconomicTrace.
    This ensures that the `budget_warning` and `warning_message` fields are populated correctly.
    """
    mock_pricer = MagicMock(spec=Pricer)
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.09,
        latency_ms=100.0,
        token_volume=100
    )

    # Economist uses its own internal components if not provided, but we can inject them.
    # However, Economist.__init__ takes optional components.
    # We want to use our mocked pricer.
    economist = Economist(pricer=mock_pricer)

    request = RequestPayload(
        model_name="mock-model",
        prompt="test prompt",
        max_budget=Budget(
            financial=0.10,
            latency_ms=10000.0,
            token_volume=10000
        ),
        soft_limit_threshold=0.8
    )

    trace = economist.check_execution(request)

    assert trace.decision.value == "APPROVED"
    assert trace.budget_warning is True, "EconomicTrace.budget_warning should be True"
    assert trace.warning_message is not None
    assert "Financial budget at 90.0%" in trace.warning_message
