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


def test_arbitrageur_dynamic_flash_sale_scenario() -> None:
    """
    Complex Scenario: "Flash Sale".

    1. A user requests a high-quality model (Expensive).
    2. Budget is tight, so it gets rejected (no arbitrage found because all high-quality models are expensive).
    3. SUDDENLY, a "Flash Sale" occurs (rates drop significantly).
    4. The user retries the SAME request.
    5. It should now be APPROVED or Arbitrage should find a solution that was previously impossible.
    """
    # Setup
    model_name = "gpt-flash"
    pricer = Pricer(
        rates={model_name: ModelRate(input_cost_per_1k=10.0, output_cost_per_1k=10.0, latency_ms_per_output_token=10.0)}
    )
    economist = Economist(pricer=pricer)

    # Very tight budget ($1.00)
    # Note: Must set other budgets high to avoid rejection on those fronts (0.0 = 0 allowed)
    # Request ~1k tokens -> $10 + $10 = $20.00 cost.
    request = RequestPayload(
        model_name=model_name,
        prompt="A" * 4000,
        estimated_output_tokens=1000,
        max_budget=Budget(financial=1.00, latency_ms=1e9, token_volume=100000),
        difficulty_score=0.9,  # High difficulty, so we need this model or equivalent
    )

    # Step 1: Execute -> REJECTED
    # Arbitrageur checks "gpt-flash" (too expensive).
    # It tries to find alternative?
    # If "gpt-flash" is the only model, it returns None.

    trace1 = economist.check_execution(request)
    assert trace1.decision == "REJECTED"
    # No alternative because only 1 model exists and it's too expensive
    if trace1.suggested_alternative:
        # If it suggests topology reduction, that's fine, but let's assume even 1 agent is too expensive ($20 > $1)
        pass

    # Step 2: FLASH SALE! Prices drop by 99%
    # New Cost: $0.10 + $0.10 = $0.20 < $1.00
    pricer.rates[model_name] = ModelRate(
        input_cost_per_1k=0.1, output_cost_per_1k=0.1, latency_ms_per_output_token=10.0
    )

    # Step 3: Retry
    trace2 = economist.check_execution(request)

    # Should be APPROVED now because estimated cost ($0.20) < Budget ($1.00)
    assert trace2.decision == "APPROVED", f"Expected APPROVED, got {trace2.decision}. Reason: {trace2.reason}"
    assert trace2.model_used == model_name


def test_arbitrageur_dynamic_price_hike_mid_process() -> None:
    """
    Complex Scenario: Price Hike.

    1. Arbitrageur recommends a cheap model.
    2. Before the user can use it, the price spikes.
    3. User asks again (or validates suggestion).
    4. Suggestion should change or become invalid.
    """
    cheap_model = "cheap-v1"
    expensive_model = "expensive-v1"

    pricer = Pricer(
        rates={
            cheap_model: ModelRate(input_cost_per_1k=0.1, output_cost_per_1k=0.1, latency_ms_per_output_token=10),
            expensive_model: ModelRate(input_cost_per_1k=10.0, output_cost_per_1k=10.0, latency_ms_per_output_token=10),
        }
    )

    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    # Request for expensive model, low difficulty -> Should suggest cheap
    req = RequestPayload(model_name=expensive_model, prompt="hi", difficulty_score=0.1)

    sugg1 = arb.recommend_alternative(req)
    assert sugg1 is not None
    assert sugg1.model_name == cheap_model

    # PRICE HIKE: Cheap model becomes more expensive than expensive model
    pricer.rates[cheap_model] = ModelRate(
        input_cost_per_1k=100.0, output_cost_per_1k=100.0, latency_ms_per_output_token=10
    )

    # Retry recommendation
    # Now "expensive-v1" ($20) is cheaper than "cheap-v1" ($200)
    # So it should probably NOT recommend switching to "cheap-v1".
    # Or it might find NO cheaper alternative.

    sugg2 = arb.recommend_alternative(req)

    # If original request was expensive-v1, and it is now the cheapest (relative to cheap-v1),
    # Arbitrageur should return None (keep original).
    assert sugg2 is None
