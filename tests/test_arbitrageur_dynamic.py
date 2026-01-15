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
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer
from coreason_economist.rates import ModelRate


def test_arbitrageur_dynamic_price_spike() -> None:
    """
    Test that Arbitrageur immediately reacts to a sudden price increase in the Pricer.
    """
    # 1. Setup Pricer and Arbitrageur with default rates
    pricer = Pricer()
    # Initially gpt-4o is affordable in our budget
    # gpt-4o: $0.005 input / $0.015 output

    arb = Arbitrageur(pricer=pricer)

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 4000,  # 1000 tokens
        estimated_output_tokens=1000,
        max_budget=Budget(financial=0.03, latency_ms=50000, token_volume=10000),  # 0.03 is enough for normal price
        difficulty_score=1.0,  # High difficulty, so only downgrade if "Hard Stop" budget exhausted
    )

    # 2. Check that normally it would NOT recommend (returns None means original is fine, or no cheaper option found)
    # Actually, recommend_alternative only returns a result if budget is exceeded OR optimization possible.
    # Let's check if budget is exceeded first.
    # 1k in * 0.005 + 1k out * 0.015 = $0.02. Budget is 0.03.
    # So it fits. recommendation should be None (or maybe optimization if diff < threshold, but diff is 1.0)

    suggestion = arb.recommend_alternative(request)
    assert suggestion is None, "Should not suggest alternative when budget fits and difficulty is high"

    # 3. SPIKE THE PRICE!
    # Update the rate in the Pricer instance
    pricer.rates["gpt-4o"] = ModelRate(
        input_cost_per_1k=1.0,  # Massive spike
        output_cost_per_1k=1.0,
        latency_ms_per_output_token=10.0,
    )

    # 4. Verify Arbitrageur sees the new price and now suggests a downgrade/topology change
    # Cost is now ~$2.00, budget is $0.03. Exceeded!
    suggestion_after_spike = arb.recommend_alternative(request)

    assert suggestion_after_spike is not None
    assert suggestion_after_spike.quality_warning is not None
    # It should have either reduced topology (not possible as it's 1 agent 1 round default)
    # or switched model to cheapest (gpt-4o-mini usually)
    assert suggestion_after_spike.model_name != "gpt-4o"
    assert (
        "gpt-4o-mini" in suggestion_after_spike.model_name
        or "cheapest" in suggestion_after_spike.quality_warning.lower()
    )


def test_arbitrageur_dynamic_rate_swap() -> None:
    """
    Test that Arbitrageur works correctly when the entire rates dictionary is replaced.
    """
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer)

    request = RequestPayload(
        model_name="old-model",
        prompt="test",
        estimated_output_tokens=10,
        max_budget=Budget(financial=1.0, latency_ms=1000, token_volume=1000),
    )

    # Initially "old-model" doesn't exist in defaults, so it returns None
    assert arb.recommend_alternative(request) is None

    # Swap rates to a new dict containing ONLY "old-model" and a cheap "new-model"
    # Make old-model extremely expensive to guarantee budget exhaustion ($1000/1k tokens)
    new_rates = {
        "old-model": ModelRate(input_cost_per_1k=1000.0, output_cost_per_1k=1000.0, latency_ms_per_output_token=10),
        "new-model": ModelRate(input_cost_per_1k=0.01, output_cost_per_1k=0.01, latency_ms_per_output_token=10),
    }
    pricer.rates = new_rates

    # Now verify Arbitrageur sees these new rates
    assert "old-model" in arb.rates

    # Re-run recommendation. old-model is expensive ($10/1k). Budget $1.0.
    # It should recommend new-model.
    suggestion = arb.recommend_alternative(request)
    assert suggestion is not None
    assert suggestion.model_name == "new-model"
