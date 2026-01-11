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

import pytest
from coreason_economist.pricer import Pricer
from coreason_economist.rates import ModelRate, ToolRate
from loguru import logger


@pytest.fixture  # type: ignore
def mock_rates() -> Dict[str, ModelRate]:
    return {
        "gpt-4": ModelRate(input_cost_per_1k=0.03, output_cost_per_1k=0.06, latency_ms_per_output_token=10.0),
        "cheap-model": ModelRate(
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
            latency_ms_per_output_token=5.0,
        ),
    }


@pytest.fixture  # type: ignore
def mock_tool_rates() -> Dict[str, ToolRate]:
    return {
        "search": ToolRate(cost_per_call=0.01),
        "calc": ToolRate(cost_per_call=0.0),
    }


def test_pricer_initialization() -> None:
    pricer = Pricer()
    assert pricer.rates is not None
    assert pricer.tool_rates is not None
    assert pricer.heuristic_multiplier == 0.2


def test_estimate_financial_cost(mock_rates: Dict[str, ModelRate]) -> None:
    pricer = Pricer(rates=mock_rates)
    # 1000 input tokens = $0.03
    # 1000 output tokens = $0.06
    # Total = $0.09
    cost = pricer.estimate_financial_cost("gpt-4", 1000, 1000)
    assert cost == 0.09


def test_estimate_financial_cost_invalid_model(mock_rates: Dict[str, ModelRate]) -> None:
    pricer = Pricer(rates=mock_rates)
    with pytest.raises(ValueError):
        pricer.estimate_financial_cost("unknown", 100, 100)


def test_estimate_financial_cost_negative_tokens(mock_rates: Dict[str, ModelRate]) -> None:
    pricer = Pricer(rates=mock_rates)
    with pytest.raises(ValueError):
        pricer.estimate_financial_cost("gpt-4", -1, 100)


def test_estimate_tools_cost(mock_tool_rates: Dict[str, ToolRate]) -> None:
    pricer = Pricer(tool_rates=mock_tool_rates)
    calls: Any = [
        {"name": "search"},  # $0.01
        {"name": "calc"},  # $0.0
        {"name": "unknown"},  # $0.0 (warning)
    ]
    cost = pricer.estimate_tools_cost(calls)
    assert cost == 0.01


def test_estimate_tools_cost_openai_format(mock_tool_rates: Dict[str, ToolRate]) -> None:
    pricer = Pricer(tool_rates=mock_tool_rates)
    calls: Any = [
        {"function": {"name": "search", "arguments": "{}"}},  # $0.01
    ]
    cost = pricer.estimate_tools_cost(calls)
    assert cost == 0.01


def test_estimate_tools_cost_empty() -> None:
    pricer = Pricer()
    assert pricer.estimate_tools_cost(None) == 0.0
    assert pricer.estimate_tools_cost([]) == 0.0


def test_estimate_tools_cost_unknown_tool_logging(mock_tool_rates: Dict[str, ToolRate], caplog: Any) -> None:
    """Test that unknown tools generate a warning."""
    # Since we are using loguru, we need to make sure pytest-caplog captures it.
    # Usually pytest-caplog captures std logging. Loguru intercepts std logging.
    # If the app uses loguru logger directly, we need to sink it to caplog handler.
    # However, loguru provides a `caplog` fixture integration if properly configured or mocked.
    # A simpler way is to use a custom sink for the test.
    messages = []
    logger.add(lambda msg: messages.append(msg))

    pricer = Pricer(tool_rates=mock_tool_rates)
    calls: Any = [{"name": "mystery_tool"}]
    cost = pricer.estimate_tools_cost(calls)
    assert cost == 0.0
    assert any("Unknown tool: mystery_tool" in str(m) for m in messages)


def test_estimate_tools_cost_malformed_call(mock_tool_rates: Dict[str, ToolRate], caplog: Any) -> None:
    """Test that malformed tool calls generate a warning."""
    messages = []
    logger.add(lambda msg: messages.append(msg))

    pricer = Pricer(tool_rates=mock_tool_rates)
    calls: Any = [{"invalid": "format"}]
    cost = pricer.estimate_tools_cost(calls)
    assert cost == 0.0
    assert any("Could not determine tool name" in str(m) for m in messages)


def test_estimate_latency_ms(mock_rates: Dict[str, ModelRate]) -> None:
    pricer = Pricer(rates=mock_rates)
    # 100 tokens * 10ms/token = 1000ms
    latency = pricer.estimate_latency_ms("gpt-4", 100)
    assert latency == 1000.0


def test_estimate_latency_ms_negative_output(mock_rates: Dict[str, ModelRate]) -> None:
    """Test negative output tokens in estimate_latency_ms."""
    pricer = Pricer(rates=mock_rates)
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        pricer.estimate_latency_ms("gpt-4", -1)


def test_estimate_latency_ms_unknown_model(mock_rates: Dict[str, ModelRate]) -> None:
    """Test unknown model in estimate_latency_ms."""
    pricer = Pricer(rates=mock_rates)
    with pytest.raises(ValueError, match="Unknown model"):
        pricer.estimate_latency_ms("unknown", 100)


def test_estimate_request_cost_simple(mock_rates: Dict[str, ModelRate]) -> None:
    pricer = Pricer(rates=mock_rates, heuristic_multiplier=1.0)
    # Input 1000, output 1000 (heuristic 1.0)
    # Financial: 0.03 + 0.06 = 0.09
    # Latency: 1000 * 10 = 10000ms
    # Tokens: 2000
    budget = pricer.estimate_request_cost("gpt-4", 1000)
    assert budget.financial == 0.09
    assert budget.latency_ms == 10000.0
    assert budget.token_volume == 2000


def test_estimate_request_cost_with_overrides(mock_rates: Dict[str, ModelRate]) -> None:
    pricer = Pricer(rates=mock_rates)
    # Explicit output tokens = 500
    # Financial: (1000/1000)*0.03 + (500/1000)*0.06 = 0.03 + 0.03 = 0.06
    # Latency: 500 * 10 = 5000ms
    budget = pricer.estimate_request_cost("gpt-4", 1000, output_tokens=500)
    assert budget.financial == 0.06
    assert budget.latency_ms == 5000.0


def test_estimate_request_cost_negative_input(mock_rates: Dict[str, ModelRate]) -> None:
    """Test negative input tokens in estimate_request_cost."""
    pricer = Pricer(rates=mock_rates)
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        pricer.estimate_request_cost("gpt-4", -10)


def test_estimate_request_cost_negative_output_override(mock_rates: Dict[str, ModelRate]) -> None:
    """Test that negative output tokens raise ValueError."""
    pricer = Pricer(rates=mock_rates)
    with pytest.raises(ValueError, match="Token counts cannot be negative"):
        pricer.estimate_request_cost("gpt-4", 100, output_tokens=-5)


def test_estimate_request_cost_with_tools(
    mock_rates: Dict[str, ModelRate], mock_tool_rates: Dict[str, ToolRate]
) -> None:
    pricer = Pricer(rates=mock_rates, tool_rates=mock_tool_rates)
    tools: Any = [{"name": "search"}]  # $0.01
    # Model cost (1k in/out, same as simple test): $0.09
    # Total: $0.10
    budget = pricer.estimate_request_cost("gpt-4", 1000, output_tokens=1000, tool_calls=tools)
    assert abs(budget.financial - 0.10) < 1e-9


def test_estimate_request_cost_heuristic_edge_case(mock_rates: Dict[str, ModelRate]) -> None:
    pricer = Pricer(rates=mock_rates, heuristic_multiplier=0.1)
    # Input 1 -> Output 0.1 -> int(0) -> enforced 1
    budget = pricer.estimate_request_cost("gpt-4", 1)
    # Should calculate for 1 output token
    assert budget.token_volume == 2  # 1 in + 1 out


def test_estimate_request_cost_heuristic_zero_input(mock_rates: Dict[str, ModelRate]) -> None:
    """Test heuristic when input is 0."""
    pricer = Pricer(rates=mock_rates)
    budget = pricer.estimate_request_cost("gpt-4", 0)
    # 0 input -> 0 output
    assert budget.token_volume == 0
    assert budget.financial == 0.0


def test_estimate_request_cost_multi_agent(mock_rates: Dict[str, ModelRate]) -> None:
    """Test cost estimation with multiple agents."""
    pricer = Pricer(rates=mock_rates)
    # Base cost for 1 agent:
    # In: 1000, Out: 1000
    # Fin: 0.03 + 0.06 = 0.09
    # Latency: 1000 * 10 = 10000
    # Tokens: 2000

    # 5 agents:
    # Fin: 0.09 * 5 = 0.45
    # Tokens: 2000 * 5 = 10000
    # Latency: 10000 (Parallel execution)
    budget = pricer.estimate_request_cost("gpt-4", 1000, output_tokens=1000, agent_count=5)

    assert abs(budget.financial - 0.45) < 1e-9
    assert budget.token_volume == 10000
    assert budget.latency_ms == 10000.0


def test_estimate_request_cost_multi_round(mock_rates: Dict[str, ModelRate]) -> None:
    """Test cost estimation with multiple rounds."""
    pricer = Pricer(rates=mock_rates)
    # Base cost 1 agent: $0.09, 2000 tokens, 10000ms

    # 3 rounds (1 agent):
    # Fin: 0.09 * 3 = 0.27
    # Tokens: 2000 * 3 = 6000
    # Latency: 10000 * 3 = 30000 (Sequential execution)
    budget = pricer.estimate_request_cost("gpt-4", 1000, output_tokens=1000, rounds=3)

    assert abs(budget.financial - 0.27) < 1e-9
    assert budget.token_volume == 6000
    assert budget.latency_ms == 30000.0


def test_estimate_request_cost_council_scenario(mock_rates: Dict[str, ModelRate]) -> None:
    """Test cost estimation for Story B: 5 agents, 3 rounds."""
    pricer = Pricer(rates=mock_rates)
    # Base cost 1 agent: $0.09, 2000 tokens, 10000ms

    # 5 agents, 3 rounds:
    # Fin: 0.09 * 5 * 3 = 1.35
    # Tokens: 2000 * 5 * 3 = 30000
    # Latency: 10000 * 3 = 30000 (Parallel agents, sequential rounds)
    budget = pricer.estimate_request_cost("gpt-4", 1000, output_tokens=1000, agent_count=5, rounds=3)

    assert abs(budget.financial - 1.35) < 1e-9
    assert budget.token_volume == 30000
    assert budget.latency_ms == 30000.0


def test_invalid_agent_round_counts(mock_rates: Dict[str, ModelRate]) -> None:
    """Test that invalid agent/round counts raise ValueError."""
    pricer = Pricer(rates=mock_rates)

    with pytest.raises(ValueError, match="Agent count"):
        pricer.estimate_request_cost("gpt-4", 100, agent_count=0)

    with pytest.raises(ValueError, match="Rounds"):
        pricer.estimate_request_cost("gpt-4", 100, rounds=0)
