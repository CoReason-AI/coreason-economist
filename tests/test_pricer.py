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
from coreason_economist.rates import ModelRate, ToolRate


def test_pricer_init_default() -> None:
    """Test Pricer initialization with default rates."""
    pricer = Pricer()
    assert "gpt-4o" in pricer.rates
    assert "gpt-4o-mini" in pricer.rates
    assert "web_search" in pricer.tool_rates
    assert pricer.heuristic_multiplier == 0.2


def test_pricer_init_custom() -> None:
    """Test Pricer initialization with custom rates."""
    custom_rates = {
        "custom-model": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=2.0, latency_ms_per_output_token=10.0)
    }
    custom_tool_rates = {"custom-tool": ToolRate(cost_per_call=5.0)}
    pricer = Pricer(rates=custom_rates, tool_rates=custom_tool_rates, heuristic_multiplier=0.5)
    assert "custom-model" in pricer.rates
    assert "gpt-4o" not in pricer.rates
    assert "custom-tool" in pricer.tool_rates
    assert pricer.heuristic_multiplier == 0.5


def test_estimate_financial_cost_gpt4o() -> None:
    """Test cost calculation for GPT-4o."""
    pricer = Pricer()
    # GPT-4o: Input $0.005/1k, Output $0.015/1k
    # 1000 input, 1000 output -> 0.005 + 0.015 = 0.02
    cost = pricer.estimate_financial_cost("gpt-4o", 1000, 1000)
    assert cost == pytest.approx(0.02)


def test_estimate_request_cost_explicit_output() -> None:
    """Test returning a Budget object with explicit output tokens."""
    pricer = Pricer()
    budget = pricer.estimate_request_cost("gpt-4o", 1000, 1000)

    assert isinstance(budget, Budget)
    assert budget.financial == pytest.approx(0.02)
    assert budget.token_volume == 2000
    # Latency: 1000 output * 12ms = 12000ms
    assert budget.latency_ms == 12000.0


def test_estimate_request_cost_heuristic() -> None:
    """Test heuristic estimation when output_tokens is None."""
    pricer = Pricer(heuristic_multiplier=0.2)
    # 1000 input -> 200 output estimated
    budget = pricer.estimate_request_cost("gpt-4o", 1000)

    assert budget.token_volume == 1200
    # Financial: (1000 * 0.005/1k) + (200 * 0.015/1k) = 0.005 + 0.003 = 0.008
    assert budget.financial == pytest.approx(0.008)
    # Latency: 200 output * 12ms = 2400ms
    assert budget.latency_ms == 2400.0


def test_estimate_request_cost_heuristic_zero_input() -> None:
    """Test heuristic with zero input."""
    pricer = Pricer()
    budget = pricer.estimate_request_cost("gpt-4o", 0)
    assert budget.token_volume == 0
    assert budget.financial == 0.0
    assert budget.latency_ms == 0.0


def test_estimate_request_cost_heuristic_small_input() -> None:
    """Test heuristic ensures at least 1 output token for non-zero input."""
    pricer = Pricer(heuristic_multiplier=0.001)  # Very small multiplier
    # 10 input -> 0.01 output -> 0 int. Should enforce 1 if input > 0
    budget = pricer.estimate_request_cost("gpt-4o", 10)

    assert budget.token_volume == 11  # 10 input + 1 estimated output
    assert budget.latency_ms > 0


def test_estimate_financial_cost_negative_tokens() -> None:
    """Test that negative token counts raise ValueError."""
    pricer = Pricer()
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        pricer.estimate_financial_cost("gpt-4o", -10, 10)


def test_estimate_request_cost_negative_input() -> None:
    """Test negative input for request cost."""
    pricer = Pricer()
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        pricer.estimate_request_cost("gpt-4o", -10)


def test_estimate_request_cost_negative_output() -> None:
    """Test negative output for request cost."""
    pricer = Pricer()
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        pricer.estimate_request_cost("gpt-4o", 10, -5)


def test_estimate_latency_ms() -> None:
    """Test latency estimation directly."""
    pricer = Pricer()
    latency = pricer.estimate_latency_ms("gpt-4o", 100)
    assert latency == 1200.0  # 100 * 12ms


def test_estimate_latency_negative_tokens() -> None:
    """Test latency estimation with negative tokens."""
    pricer = Pricer()
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        pricer.estimate_latency_ms("gpt-4o", -10)


def test_estimate_latency_unknown_model() -> None:
    """Test latency for unknown model."""
    pricer = Pricer()
    with pytest.raises(ValueError):
        pricer.estimate_latency_ms("unknown", 100)


def test_estimate_financial_cost_free_model() -> None:
    """Test cost calculation for a free model."""
    free_rates = {
        "free-model": ModelRate(input_cost_per_1k=0.0, output_cost_per_1k=0.0, latency_ms_per_output_token=1.0)
    }
    pricer = Pricer(rates=free_rates)
    cost = pricer.estimate_financial_cost("free-model", 10000, 10000)
    assert cost == 0.0


def test_unknown_model_in_estimate_financial_cost() -> None:
    """Test that unknown models raise ValueError in estimate_financial_cost."""
    pricer = Pricer()
    with pytest.raises(ValueError, match="Unknown model: unknown"):
        pricer.estimate_financial_cost("unknown", 100, 100)


def test_estimate_tools_cost_no_tools() -> None:
    """Test tool cost estimation with no tools."""
    pricer = Pricer()
    assert pricer.estimate_tools_cost(None) == 0.0
    assert pricer.estimate_tools_cost([]) == 0.0


def test_estimate_tools_cost_simple_format() -> None:
    """Test tool cost estimation with simple {'name': ...} format."""
    pricer = Pricer()
    # web_search cost is 0.01
    tool_calls = [{"name": "web_search"}, {"name": "web_search"}]
    cost = pricer.estimate_tools_cost(tool_calls)
    assert cost == pytest.approx(0.02)


def test_estimate_tools_cost_openai_format() -> None:
    """Test tool cost estimation with OpenAI {'function': {'name': ...}} format."""
    pricer = Pricer()
    # web_search cost is 0.01
    tool_calls = [{"function": {"name": "web_search", "arguments": "{}"}}]
    cost = pricer.estimate_tools_cost(tool_calls)
    assert cost == pytest.approx(0.01)


def test_estimate_tools_cost_mixed_unknown() -> None:
    """Test tool cost estimation with unknown tools."""
    pricer = Pricer()
    # web_search (0.01) + unknown (0.0) = 0.01
    tool_calls = [{"name": "web_search"}, {"name": "unknown_tool"}]
    cost = pricer.estimate_tools_cost(tool_calls)
    assert cost == pytest.approx(0.01)


def test_estimate_tools_cost_malformed() -> None:
    """Test tool cost estimation with malformed tool call."""
    pricer = Pricer()
    tool_calls = [{"no_name": "whatsoever"}]
    # Should log warning and cost 0.0
    cost = pricer.estimate_tools_cost(tool_calls)
    assert cost == 0.0


def test_estimate_request_cost_with_tools() -> None:
    """Test estimate_request_cost includes tool costs."""
    pricer = Pricer()
    # GPT-4o: 1000 in, 1000 out -> $0.02
    # Tools: 2 web_search -> $0.02
    # Total: $0.04
    tool_calls = [{"name": "web_search"}, {"name": "web_search"}]
    budget = pricer.estimate_request_cost("gpt-4o", 1000, 1000, tool_calls=tool_calls)
    assert budget.financial == pytest.approx(0.04)
    assert budget.token_volume == 2000


def test_estimate_tools_cost_high_volume() -> None:
    """Test high volume of tool calls to ensure accurate summation."""
    pricer = Pricer()
    # web_search cost is 0.01
    count = 1000
    tool_calls = [{"name": "web_search"} for _ in range(count)]
    cost = pricer.estimate_tools_cost(tool_calls)
    assert cost == pytest.approx(0.01 * count)


def test_estimate_tools_cost_zero_cost() -> None:
    """Test that zero-cost tools (like 'calculator') don't add to cost."""
    pricer = Pricer()
    # calculator is $0.0 in defaults
    tool_calls = [{"name": "calculator"} for _ in range(50)]
    cost = pricer.estimate_tools_cost(tool_calls)
    assert cost == 0.0


def test_tool_calls_malformed_structure() -> None:
    """Test robust handling of weirdly nested structures."""
    pricer = Pricer()
    tool_calls = [
        {"function": "not_a_dict"},  # Malformed function key
        {"function": {}},  # Missing name inside function
        {},  # Empty dict
    ]
    cost = pricer.estimate_tools_cost(tool_calls)  # type: ignore[arg-type]
    assert cost == 0.0
