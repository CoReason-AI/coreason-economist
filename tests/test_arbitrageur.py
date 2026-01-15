# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from typing import Dict

from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.models import RequestPayload
from coreason_economist.pricer import Pricer
from coreason_economist.rates import ModelRate


def test_arbitrageur_initialization() -> None:
    """Test initializing Arbitrageur with defaults and overrides."""
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer)
    assert arb.threshold == 0.5
    assert "gpt-4o" in arb.rates

    arb_custom = Arbitrageur(pricer=pricer, threshold=0.8)
    assert arb_custom.threshold == 0.8


def test_recommend_alternative_no_difficulty() -> None:
    """Test that it returns None if difficulty_score is missing."""
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer)
    payload = RequestPayload(model_name="gpt-4o", prompt="test")
    assert payload.difficulty_score is None

    recommendation = arb.recommend_alternative(payload)
    assert recommendation is None


def test_recommend_alternative_high_difficulty() -> None:
    """Test that it returns None if difficulty is above threshold."""
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer, threshold=0.5)
    payload = RequestPayload(model_name="gpt-4o", prompt="test", difficulty_score=0.6)

    recommendation = arb.recommend_alternative(payload)
    assert recommendation is None


def test_recommend_alternative_success() -> None:
    """Test that it recommends a cheaper model when appropriate."""
    # Custom rates to ensure predictable order
    rates: Dict[str, ModelRate] = {
        "expensive": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=10),
        "cheap": ModelRate(input_cost_per_1k=0.1, output_cost_per_1k=0.1, latency_ms_per_output_token=10),
    }
    pricer = Pricer(rates=rates)
    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    payload = RequestPayload(model_name="expensive", prompt="test", difficulty_score=0.2)

    recommendation = arb.recommend_alternative(payload)

    assert recommendation is not None
    assert recommendation.model_name == "cheap"
    # Ensure other fields are preserved
    assert recommendation.prompt == "test"
    assert recommendation.difficulty_score == 0.2


def test_recommend_alternative_already_cheapest() -> None:
    """Test that it returns None if already on the cheapest model."""
    rates: Dict[str, ModelRate] = {
        "expensive": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=10),
        "cheap": ModelRate(input_cost_per_1k=0.1, output_cost_per_1k=0.1, latency_ms_per_output_token=10),
    }
    pricer = Pricer(rates=rates)
    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    payload = RequestPayload(model_name="cheap", prompt="test", difficulty_score=0.2)

    recommendation = arb.recommend_alternative(payload)
    assert recommendation is None


def test_recommend_alternative_unknown_model() -> None:
    """Test that it ignores unknown models."""
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer)
    payload = RequestPayload(model_name="unknown-model-xyz", prompt="test", difficulty_score=0.1)

    recommendation = arb.recommend_alternative(payload)
    assert recommendation is None


def test_recommend_topology_reduction() -> None:
    """Test that Arbitrageur recommends reducing topology for low difficulty tasks."""
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    # Complex topology, low difficulty
    payload = RequestPayload(
        model_name="gpt-4o",
        prompt="simple question",
        difficulty_score=0.2,
        agent_count=5,
        rounds=3,
    )

    recommendation = arb.recommend_alternative(payload)

    assert recommendation is not None
    assert recommendation.agent_count == 1
    assert recommendation.rounds == 1
    # It might also change the model, but primarily we want topology reduction
    assert recommendation.difficulty_score == 0.2


def test_recommend_topology_reduction_mixed() -> None:
    """Test that Arbitrageur handles both model and topology opportunities."""
    # Custom rates
    rates: Dict[str, ModelRate] = {
        "expensive": ModelRate(input_cost_per_1k=1.0, output_cost_per_1k=1.0, latency_ms_per_output_token=10),
        "cheap": ModelRate(input_cost_per_1k=0.1, output_cost_per_1k=0.1, latency_ms_per_output_token=10),
    }
    pricer = Pricer(rates=rates)
    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    # Expensive model AND complex topology
    payload = RequestPayload(
        model_name="expensive",
        prompt="simple question",
        difficulty_score=0.2,
        agent_count=3,
        rounds=2,
    )

    recommendation = arb.recommend_alternative(payload)

    assert recommendation is not None
    # We expect topology reduction as a priority or alongside model change.
    # Let's assert it definitely does topology reduction.
    assert recommendation.agent_count == 1
    assert recommendation.rounds == 1
    assert recommendation.model_name == "cheap"


def test_no_topology_change_needed() -> None:
    """Test that it doesn't change topology if already simple."""
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    payload = RequestPayload(
        model_name="gpt-4o",
        prompt="simple question",
        difficulty_score=0.2,
        agent_count=1,
        rounds=1,
    )

    # It might still recommend a model change if gpt-4o is not the cheapest
    # But it shouldn't change agent_count/rounds (they are already 1)

    recommendation = arb.recommend_alternative(payload)

    if recommendation:
        assert recommendation.agent_count == 1
        assert recommendation.rounds == 1


def test_partial_topology_reduction() -> None:
    """
    Test scenario: agent_count=1, rounds=5.
    Expect: rounds reduced to 1.
    """
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer, threshold=0.5)
    payload = RequestPayload(model_name="gpt-4o", prompt="test", difficulty_score=0.2, agent_count=1, rounds=5)

    rec = arb.recommend_alternative(payload)
    assert rec is not None
    assert rec.rounds == 1
    assert rec.agent_count == 1
    # Check that model change also happened if gpt-4o is not cheapest
    # gpt-4o-mini is cheapest in default rates.
    assert rec.model_name == "gpt-4o-mini"


def test_topology_reduction_keep_model() -> None:
    """
    Test scenario: Already on cheapest model, but complex topology.
    Expect: Topology reduction, model name unchanged.
    """
    # Assuming gpt-4o-mini is cheapest in default rates
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer, threshold=0.5)
    payload = RequestPayload(model_name="gpt-4o-mini", prompt="test", difficulty_score=0.2, agent_count=5, rounds=1)

    rec = arb.recommend_alternative(payload)
    assert rec is not None
    assert rec.agent_count == 1
    assert rec.rounds == 1
    assert rec.model_name == "gpt-4o-mini"


def test_boundary_difficulty() -> None:
    """
    Test boundary condition for difficulty score.
    Threshold = 0.5.
    0.5 -> No change (None).
    0.499999 -> Change.
    """
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    # At threshold
    p1 = RequestPayload(model_name="gpt-4o", prompt="test", difficulty_score=0.5)
    assert arb.recommend_alternative(p1) is None

    # Just below threshold
    p2 = RequestPayload(model_name="gpt-4o", prompt="test", difficulty_score=0.499999)
    assert arb.recommend_alternative(p2) is not None


def test_empty_rates_graceful_handling() -> None:
    """
    Test that empty rates registry does not crash the Arbitrageur.
    It should return None because it can't find 'cheapest'.
    """
    pricer = Pricer(rates={})
    arb = Arbitrageur(pricer=pricer)
    payload = RequestPayload(model_name="gpt-4o", prompt="test", difficulty_score=0.1)

    # Case 1: Model not in empty rates -> Returns None immediately
    res = arb.recommend_alternative(payload)
    assert res is None

    # Case 2: Topology change needed
    # Even if rates empty, if model check fails, it returns None.
    # Wait, if model check fails, it returns None BEFORE topology check?
    # Let's check logic order in source.

    # Code:
    # if request.model_name not in self.rates: return None
    # ...
    # Logic for recommendation: (Topology check is later?)

    # No, I put it AFTER the model check?
    # Let me check the source again.
