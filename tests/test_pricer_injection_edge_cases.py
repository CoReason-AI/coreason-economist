# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.economist import Economist
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer
from coreason_economist.rates import ModelRate


def test_arbitrageur_pricer_precedence() -> None:
    """
    Test that if both 'rates' and 'pricer' are passed to Arbitrageur,
    'pricer' takes precedence.
    """
    # 1. Setup rates
    rates_ignored = {
        "model-A": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=10.0)
    }

    rates_used = {
        "model-B": ModelRate(input_cost_per_1k=0.5, output_cost_per_1k=0.5, latency_ms_per_output_token=5.0)
    }
    pricer = Pricer(rates=rates_used)

    # 2. Initialize Arbitrageur
    arb = Arbitrageur(rates=rates_ignored, pricer=pricer)

    # 3. Verify
    # The arbitrageur.rates should point to pricer.rates (model-B), not rates_ignored (model-A)
    assert "model-B" in arb.rates
    assert "model-A" not in arb.rates
    assert arb.pricer is pricer


def test_economist_propagates_pricer() -> None:
    """
    Test that Economist initializes its default Arbitrageur with the same Pricer instance.
    """
    # 1. Setup
    pricer = Pricer()
    economist = Economist(pricer=pricer)

    # 2. Verify
    assert economist.arbitrageur.pricer is pricer
    assert economist.arbitrageur.rates is pricer.rates


def test_dynamic_rate_updates_complex_scenario() -> None:
    """
    Complex Scenario: Verify that dynamic updates to the Pricer's rates
    are immediately reflected in the Arbitrageur's decision making.

    This simulates the "Split-Brain" fix verification.
    """
    # 1. Setup
    # Expensive model: $1.00 total
    # Cheap model: $0.01 total
    model_expensive = "model-expensive"
    model_cheap = "model-cheap"

    initial_rates = {
        model_expensive: ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=10.0),
        model_cheap: ModelRate(input_cost_per_1k=0.01, output_cost_per_1k=0.01, latency_ms_per_output_token=10.0)
    }

    pricer = Pricer(rates=initial_rates)
    economist = Economist(pricer=pricer)

    # Request for expensive model with low budget ($0.05)
    # Should trigger arbitrage to 'model-cheap'
    request = RequestPayload(
        model_name=model_expensive,
        prompt="A" * 400, # 100 tokens
        estimated_output_tokens=100,
        max_budget=Budget(financial=0.05, latency_ms=10000, token_volume=10000)
    )

    # 2. First Execution: Expect Rejection + Suggestion (Cheap)
    trace1 = economist.check_execution(request)
    assert trace1.decision == "REJECTED"
    assert trace1.suggested_alternative is not None
    assert trace1.suggested_alternative.model_name == model_cheap

    # 3. MODIFY RATES
    # Make 'model-cheap' super expensive ($10.00), so it no longer fits the budget.
    # Arbitrageur should now fail to find a suggestion (or find nothing).
    pricer.rates[model_cheap] = ModelRate(
        input_cost_per_1k=10.0, output_cost_per_1k=10.0, latency_ms_per_output_token=10.0
    )

    # 4. Second Execution: Expect Rejection + NO Suggestion
    # (Because the cheap model is now too expensive)
    trace2 = economist.check_execution(request)
    assert trace2.decision == "REJECTED"
    assert trace2.suggested_alternative is None

    # 5. RESTORE RATES (Verify it works back)
    pricer.rates[model_cheap] = ModelRate(
        input_cost_per_1k=0.01, output_cost_per_1k=0.01, latency_ms_per_output_token=10.0
    )
    trace3 = economist.check_execution(request)
    assert trace3.decision == "REJECTED"
    assert trace3.suggested_alternative is not None
    assert trace3.suggested_alternative.model_name == model_cheap
