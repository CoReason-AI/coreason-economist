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


def test_observability_computed_fields_serialization() -> None:
    """
    Verify that computed fields like latency_per_token are included in the
    serialized output of EconomicTrace.
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=1.0, token_volume=1000, latency_ms=1000),
        actual_cost=Budget(financial=1.0, token_volume=2000, latency_ms=2000),
        decision=Decision.APPROVED,
        model_used="gpt-4",
        input_tokens=100,
    )

    # Serialize to JSON
    json_output = trace.model_dump_json()
    data = json.loads(json_output)

    # Check for presence of computed fields
    assert "tokens_per_dollar" in data
    assert "tokens_per_second" in data
    assert "latency_per_token" in data
    assert "cost_per_insight" in data

    # Verify calculation logic (based on _effective_cost, which is actual_cost here)
    # actual_cost: financial=1.0, token_volume=2000, latency_ms=2000

    # tokens_per_dollar = 2000 / 1.0 = 2000.0
    assert data["tokens_per_dollar"] == 2000.0

    # tokens_per_second = 2000 / (2000/1000) = 2000 / 2.0 = 1000.0
    assert data["tokens_per_second"] == 1000.0

    # latency_per_token = 2000 / 2000 = 1.0
    # Must use TOTAL volume (2000), not just output or input
    assert data["latency_per_token"] == 1.0

    # cost_per_insight = 1.0
    assert data["cost_per_insight"] == 1.0


def test_observability_computed_fields_zero_division() -> None:
    """
    Verify that computed fields handle zero values gracefully (return 0.0).
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=0.0, token_volume=0, latency_ms=0),
        decision=Decision.APPROVED,
        model_used="gpt-4",
        input_tokens=0,
    )

    json_output = trace.model_dump_json()
    data = json.loads(json_output)

    assert data["tokens_per_dollar"] == 0.0
    assert data["tokens_per_second"] == 0.0
    assert data["latency_per_token"] == 0.0
    assert data["cost_per_insight"] == 0.0
