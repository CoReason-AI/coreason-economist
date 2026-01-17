# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.rates import DEFAULT_MODEL_RATES, ToolRate


def test_tool_rate_model() -> None:
    """
    Test that ToolRate model is initialized correctly.
    """
    rate = ToolRate(cost_per_call=0.05)
    assert rate.cost_per_call == 0.05


def test_default_rates_contain_required_models() -> None:
    """
    Verify that the default rate registry contains all required models.
    """
    assert "gpt-4o" in DEFAULT_MODEL_RATES
    assert "gpt-4o-mini" in DEFAULT_MODEL_RATES
    assert "claude-3-5-sonnet" in DEFAULT_MODEL_RATES
    assert "llama-3.1-70b" in DEFAULT_MODEL_RATES


def test_llama_3_1_70b_pricing() -> None:
    """
    Verify the specific pricing values for Llama 3.1 70B.
    """
    rate = DEFAULT_MODEL_RATES["llama-3.1-70b"]
    assert rate.input_cost_per_1k == 0.00088
    assert rate.output_cost_per_1k == 0.00088
    assert rate.latency_ms_per_output_token == 10.0
