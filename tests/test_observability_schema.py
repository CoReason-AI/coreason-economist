# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

import json

from coreason_economist.models import Budget, Decision, EconomicTrace


class TestObservabilitySchema:
    """
    Strict schema validation for observability requirements.
    Ensures that computed efficiency metrics are present in the serialized output
    consumed by the dashboard (maco UI).
    """

    def test_json_serialization_includes_metrics(self) -> None:
        """
        Verify that efficiency metrics are included in the JSON dump.
        This is a critical requirement for the dashboard integration.
        """
        estimated = Budget(financial=0.10, latency_ms=1000.0, token_volume=1000)
        actual = Budget(financial=0.05, latency_ms=500.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated,
            actual_cost=actual,
            decision=Decision.APPROVED,
            model_used="test-model",
            input_tokens=500,
        )

        # 1. Check model_dump() dictionary
        data_dict = trace.model_dump()
        assert "tokens_per_dollar" in data_dict
        assert "tokens_per_second" in data_dict
        assert "latency_per_token" in data_dict
        assert "cost_per_insight" in data_dict

        # Verify values in dict match properties
        assert data_dict["tokens_per_dollar"] == trace.tokens_per_dollar
        assert data_dict["tokens_per_second"] == trace.tokens_per_second
        assert data_dict["latency_per_token"] == trace.latency_per_token
        assert data_dict["cost_per_insight"] == trace.cost_per_insight

        # 2. Check model_dump_json() string
        json_str = trace.model_dump_json()
        data_loaded = json.loads(json_str)

        assert "tokens_per_dollar" in data_loaded
        assert "tokens_per_second" in data_loaded
        assert "latency_per_token" in data_loaded
        assert "cost_per_insight" in data_loaded

        # Verify values in JSON match properties
        assert data_loaded["tokens_per_dollar"] == trace.tokens_per_dollar
        assert data_loaded["tokens_per_second"] == trace.tokens_per_second
        assert data_loaded["latency_per_token"] == trace.latency_per_token
        assert data_loaded["cost_per_insight"] == trace.cost_per_insight

    def test_latency_per_token_definition(self) -> None:
        """
        Strictly verify the definition of latency_per_token.
        Requirement: ms per token.
        """
        # 1000ms latency, 2000 tokens -> 0.5 ms/token
        budget = Budget(financial=1.0, latency_ms=1000.0, token_volume=2000)
        trace = EconomicTrace(
            estimated_cost=budget, decision=Decision.APPROVED, model_used="test-model", input_tokens=1000
        )

        expected = 1000.0 / 2000.0  # 0.5
        assert trace.latency_per_token == expected
        assert trace.model_dump()["latency_per_token"] == expected
