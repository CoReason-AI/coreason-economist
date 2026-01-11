# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Any, Dict, List, Optional

from coreason_economist.models import Budget
from coreason_economist.rates import DEFAULT_MODEL_RATES, DEFAULT_TOOL_RATES, ModelRate, ToolRate
from coreason_economist.utils.logger import logger


class Pricer:
    """
    The Estimator: Calculates the cost of an action before it happens.
    """

    def __init__(
        self,
        rates: Optional[Dict[str, ModelRate]] = None,
        tool_rates: Optional[Dict[str, ToolRate]] = None,
        heuristic_multiplier: float = 0.2,
    ) -> None:
        """
        Initialize the Pricer with a rate registry and heuristic settings.
        If no rates are provided, uses the default registry.
        """
        self.rates = rates if rates is not None else DEFAULT_MODEL_RATES
        self.tool_rates = tool_rates if tool_rates is not None else DEFAULT_TOOL_RATES
        self.heuristic_multiplier = heuristic_multiplier

    def estimate_financial_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the financial cost for a given model and token counts.
        Raises ValueError if model is unknown or if token counts are negative.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        if model_name not in self.rates:
            raise ValueError(f"Unknown model: {model_name}")

        rate = self.rates[model_name]

        input_cost = (input_tokens / 1000.0) * rate.input_cost_per_1k
        output_cost = (output_tokens / 1000.0) * rate.output_cost_per_1k

        return input_cost + output_cost

    def estimate_tools_cost(self, tool_calls: Optional[List[Dict[str, Any]]]) -> float:
        """
        Calculates the total cost of all tool calls in the list.
        Supports direct {"name": "tool"} and OpenAI-style {"function": {"name": "tool"}} formats.
        """
        if not tool_calls:
            return 0.0

        total_tool_cost = 0.0
        for call in tool_calls:
            tool_name = None

            # Try to get name from "name" key
            if "name" in call:
                tool_name = call["name"]
            # Try to get name from "function" -> "name" key (OpenAI style)
            elif "function" in call and isinstance(call["function"], dict) and "name" in call["function"]:
                tool_name = call["function"]["name"]

            if tool_name:
                if tool_name in self.tool_rates:
                    total_tool_cost += self.tool_rates[tool_name].cost_per_call
                else:
                    logger.warning(f"Unknown tool: {tool_name}. Assuming cost $0.0.")
            else:
                logger.warning("Could not determine tool name from call. Assuming cost $0.0.")

        return total_tool_cost

    def estimate_latency_ms(self, model_name: str, output_tokens: int) -> float:
        """
        Estimates latency based on output token count.
        """
        if output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        if model_name not in self.rates:
            raise ValueError(f"Unknown model: {model_name}")

        rate = self.rates[model_name]
        return float(output_tokens) * rate.latency_ms_per_output_token

    def estimate_request_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: Optional[int] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        agent_count: int = 1,
        rounds: int = 1,
    ) -> Budget:
        """
        Creates a Budget object representing the estimated cost.
        If output_tokens is None, uses heuristics to estimate it.
        Includes estimated cost of tool calls if provided.

        Args:
            model_name: Name of the model.
            input_tokens: Number of input tokens per agent.
            output_tokens: Number of output tokens per agent per round.
            tool_calls: List of tool calls per agent per round.
            agent_count: Number of agents participating (parallel execution assumed).
            rounds: Number of sequential rounds.
        """
        if input_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        if agent_count < 1:
            raise ValueError("Agent count must be at least 1")
        if rounds < 1:
            raise ValueError("Rounds must be at least 1")

        if output_tokens is None:
            # Heuristic: output is a fraction of input
            estimated_output = int(input_tokens * self.heuristic_multiplier)
            # Ensure at least 1 token if input > 0 to be safe, or 0 if input is 0
            if input_tokens > 0 and estimated_output == 0:
                estimated_output = 1
            output_tokens = estimated_output
        elif output_tokens < 0:
            raise ValueError("Token counts cannot be negative")

        # Calculate unit costs (per agent, per round)
        financial_cost_unit = self.estimate_financial_cost(model_name, input_tokens, output_tokens)
        tools_cost_unit = self.estimate_tools_cost(tool_calls)
        latency_cost_unit = self.estimate_latency_ms(model_name, output_tokens)
        token_volume_unit = input_tokens + output_tokens

        # Scale by agent count and rounds
        # Financial: every agent in every round costs money
        total_financial = (financial_cost_unit + tools_cost_unit) * agent_count * rounds

        # Token Volume: every agent in every round consumes context
        total_token_volume = token_volume_unit * agent_count * rounds

        # Latency:
        # Assumption: Agents within a round run in parallel, so latency is max of one agent.
        # Rounds are sequential, so latencies sum up.
        # total_latency = unit_latency * rounds
        total_latency = latency_cost_unit * rounds

        return Budget(
            financial=total_financial,
            token_volume=total_token_volume,
            latency_ms=total_latency,
        )
