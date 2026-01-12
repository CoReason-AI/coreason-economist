# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from coreason_economist.budget_authority import BudgetAuthority
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer


@pytest.fixture  # type: ignore
def mock_pricer() -> MagicMock:
    return MagicMock(spec=Pricer)


@pytest.fixture  # type: ignore
def authority(mock_pricer: MagicMock) -> BudgetAuthority:
    return BudgetAuthority(pricer=mock_pricer)


def test_soft_limit_threshold_zero(authority: BudgetAuthority, mock_pricer: MagicMock) -> None:
    """
    Edge Case: Threshold = 0.0.
    Any usage > 0 should trigger a warning.
    """
    mock_pricer.estimate_request_cost.return_value = Budget(financial=0.01, latency_ms=10.0, token_volume=10)

    request = RequestPayload(
        model_name="mock-model",
        prompt="test",
        max_budget=Budget(financial=100.0, latency_ms=1000.0, token_volume=1000),
        soft_limit_threshold=0.0,
    )

    result = authority.allow_execution(request)

    assert result.allowed is True
    assert result.warning is True
    assert "Financial budget at 0.0%" in str(result.message) or "0.1%" in str(result.message)


def test_soft_limit_threshold_one(authority: BudgetAuthority, mock_pricer: MagicMock) -> None:
    """
    Edge Case: Threshold = 1.0.
    Usage < 100% should NOT warn.
    Usage > 100% triggers BudgetExhaustedError (Hard Limit).
    So warnings are effectively disabled for allowed requests.
    """
    # 99.9% usage
    mock_pricer.estimate_request_cost.return_value = Budget(financial=0.999, latency_ms=10.0, token_volume=10)

    request = RequestPayload(
        model_name="mock-model",
        prompt="test",
        max_budget=Budget(financial=1.0, latency_ms=1000.0, token_volume=1000),
        soft_limit_threshold=1.0,
    )

    result = authority.allow_execution(request)

    assert result.allowed is True
    assert result.warning is False, "Should not warn if usage < threshold (0.999 < 1.0)"


def test_soft_limit_mixed_currencies(authority: BudgetAuthority, mock_pricer: MagicMock) -> None:
    """
    Complex Scenario: Mixed Currencies.
    - Financial: 90% (Warn)
    - Latency: 20% (No Warn)
    - Token: 85% (Warn)
    """
    mock_pricer.estimate_request_cost.return_value = Budget(
        financial=0.9,  # 90% of 1.0
        latency_ms=200.0,  # 20% of 1000
        token_volume=850,  # 85% of 1000
    )

    request = RequestPayload(
        model_name="mock-model",
        prompt="test",
        max_budget=Budget(financial=1.0, latency_ms=1000.0, token_volume=1000),
        soft_limit_threshold=0.8,
    )

    result = authority.allow_execution(request)

    assert result.allowed is True
    assert result.warning is True
    msg = str(result.message)
    assert "Financial budget at 90.0%" in msg
    assert "Token volume budget at 85.0%" in msg
    assert "Latency" not in msg, "Should not warn about Latency (20% < 80%)"


def test_soft_limit_multi_agent_scaling(authority: BudgetAuthority, mock_pricer: MagicMock) -> None:
    """
    Complex Scenario: Multi-Agent Scaling.
    Verify that increasing agent count pushes the request into warning territory.
    """

    # Setup mock to return cost * agent_count (simulating pricer logic roughly)
    def side_effect(
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        tool_calls: list[Dict[str, Any]],
        agent_count: int,
        rounds: int,
    ) -> Budget:
        # Base cost $0.02 per agent
        return Budget(financial=0.02 * agent_count, latency_ms=100.0, token_volume=100)

    mock_pricer.estimate_request_cost.side_effect = side_effect

    # Budget $0.10. Threshold 0.8 ($0.08).

    # Case A: 3 Agents -> $0.06 (60%). No Warning.
    req_3 = RequestPayload(
        model_name="mock-model",
        prompt="test",
        agent_count=3,
        max_budget=Budget(financial=0.10, latency_ms=1000.0, token_volume=1000),
        soft_limit_threshold=0.8,
    )
    res_3 = authority.allow_execution(req_3)
    assert res_3.allowed is True
    assert res_3.warning is False

    # Case B: 5 Agents -> $0.10 (100%). Warning!
    req_5 = RequestPayload(
        model_name="mock-model",
        prompt="test",
        agent_count=5,
        max_budget=Budget(financial=0.10, latency_ms=1000.0, token_volume=1000),
        soft_limit_threshold=0.8,
    )
    res_5 = authority.allow_execution(req_5)
    assert res_5.allowed is True
    assert res_5.warning is True
    assert "Financial budget at 100.0%" in str(res_5.message)
