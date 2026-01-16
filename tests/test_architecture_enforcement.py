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

from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.models import Budget, Decision, EconomicTrace
from coreason_economist.pricer import Pricer
from coreason_economist.rates import ModelRate


def test_architecture_split_brain_prevention() -> None:
    """
    Verify that Arbitrageur strictly mirrors Pricer updates (Dynamic Pricing)
    and that removing 'rates' from Arbitrageur init prevents split-brain.
    """
    # Create a Pricer
    p = Pricer()
    # Create Arbitrageur with this Pricer
    a = Arbitrageur(pricer=p)

    # Assert they share state
    assert a.rates is p.rates

    # Modify Pricer
    new_rate = ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=1.0)
    p.rates["dynamic-model"] = new_rate

    # Assert Arbitrageur sees it
    assert "dynamic-model" in a.rates
    assert a.rates["dynamic-model"] == new_rate


def test_observability_schema_metrics() -> None:
    """
    Verify EconomicTrace computed fields calculation and serialization.
    """
    # 1. Standard Case
    trace = EconomicTrace(
        estimated_cost=Budget(financial=10.0, token_volume=1000, latency_ms=1000.0),
        decision=Decision.APPROVED,
        model_used="gpt-4",
        input_tokens=500,
    )

    # Calculation Checks
    # tokens_per_dollar = 1000 / 10.0 = 100.0
    assert abs(trace.tokens_per_dollar - 100.0) < 1e-9

    # tokens_per_second = 1000 / (1000ms / 1000) = 1000 / 1.0 = 1000.0
    assert abs(trace.tokens_per_second - 1000.0) < 1e-9

    # latency_per_token = 1000ms / 1000 tokens = 1.0 ms/token
    assert abs(trace.latency_per_token - 1.0) < 1e-9

    # cost_per_insight = financial = 10.0
    assert abs(trace.cost_per_insight - 10.0) < 1e-9

    # Serialization Check
    data = json.loads(trace.model_dump_json())
    assert data["tokens_per_dollar"] == 100.0
    assert data["tokens_per_second"] == 1000.0
    assert data["latency_per_token"] == 1.0
    assert data["cost_per_insight"] == 10.0


def test_observability_schema_edge_cases_zero() -> None:
    """
    Verify EconomicTrace handles division-by-zero (returns 0.0).
    """
    trace = EconomicTrace(
        estimated_cost=Budget(financial=0.0, token_volume=0, latency_ms=0.0),
        decision=Decision.APPROVED,
        model_used="free-model",
        input_tokens=0,
    )

    # Should not raise ZeroDivisionError
    assert trace.tokens_per_dollar == 0.0
    assert trace.tokens_per_second == 0.0
    assert trace.latency_per_token == 0.0
    assert trace.cost_per_insight == 0.0

    data = json.loads(trace.model_dump_json())
    assert data["tokens_per_dollar"] == 0.0
