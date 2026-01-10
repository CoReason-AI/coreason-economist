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
from coreason_economist.models import Budget
from coreason_economist.pricer import Pricer
from coreason_economist.rates import ModelRate


def test_pricer_init_default() -> None:
    """Test Pricer initialization with default rates."""
    pricer = Pricer()
    assert "gpt-4o" in pricer.rates
    assert "gpt-4o-mini" in pricer.rates


def test_pricer_init_custom() -> None:
    """Test Pricer initialization with custom rates."""
    custom_rates = {"custom-model": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=2.0)}
    pricer = Pricer(rates=custom_rates)
    assert "custom-model" in pricer.rates
    assert "gpt-4o" not in pricer.rates


def test_estimate_financial_cost_gpt4o() -> None:
    """Test cost calculation for GPT-4o."""
    pricer = Pricer()
    # GPT-4o: Input $0.005/1k, Output $0.015/1k
    # 1000 input, 1000 output -> 0.005 + 0.015 = 0.02
    cost = pricer.estimate_financial_cost("gpt-4o", 1000, 1000)
    assert cost == pytest.approx(0.02)


def test_estimate_financial_cost_gpt4o_mini() -> None:
    """Test cost calculation for GPT-4o-mini."""
    pricer = Pricer()
    # GPT-4o-mini: Input $0.00015/1k, Output $0.0006/1k
    # 1M input -> $0.15
    # 1M output -> $0.60
    cost = pricer.estimate_financial_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.75)


def test_estimate_financial_cost_zero_tokens() -> None:
    """Test cost calculation with zero tokens."""
    pricer = Pricer()
    cost = pricer.estimate_financial_cost("gpt-4o", 0, 0)
    assert cost == 0.0


def test_estimate_financial_cost_unknown_model() -> None:
    """Test that unknown models raise ValueError."""
    pricer = Pricer()
    with pytest.raises(ValueError, match="Unknown model: unknown-model"):
        pricer.estimate_financial_cost("unknown-model", 100, 100)


def test_estimate_request_cost() -> None:
    """Test returning a Budget object."""
    pricer = Pricer()
    budget = pricer.estimate_request_cost("gpt-4o", 1000, 1000)

    assert isinstance(budget, Budget)
    assert budget.financial == pytest.approx(0.02)
    assert budget.token_volume == 2000
    assert budget.latency_ms == 0.0
