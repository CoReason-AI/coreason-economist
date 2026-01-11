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
from coreason_economist.rates import ModelRate


def test_arbitrageur_initialization() -> None:
    """Test initializing Arbitrageur with defaults and overrides."""
    arb = Arbitrageur()
    assert arb.threshold == 0.5
    assert "gpt-4o" in arb.rates

    arb_custom = Arbitrageur(threshold=0.8)
    assert arb_custom.threshold == 0.8


def test_recommend_alternative_no_difficulty() -> None:
    """Test that it returns None if difficulty_score is missing."""
    arb = Arbitrageur()
    payload = RequestPayload(model_name="gpt-4o", prompt="test")
    assert payload.difficulty_score is None

    recommendation = arb.recommend_alternative(payload)
    assert recommendation is None


def test_recommend_alternative_high_difficulty() -> None:
    """Test that it returns None if difficulty is above threshold."""
    arb = Arbitrageur(threshold=0.5)
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
    arb = Arbitrageur(rates=rates, threshold=0.5)

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
    arb = Arbitrageur(rates=rates, threshold=0.5)

    payload = RequestPayload(model_name="cheap", prompt="test", difficulty_score=0.2)

    recommendation = arb.recommend_alternative(payload)
    assert recommendation is None


def test_recommend_alternative_unknown_model() -> None:
    """Test that it ignores unknown models."""
    arb = Arbitrageur()
    payload = RequestPayload(model_name="unknown-model-xyz", prompt="test", difficulty_score=0.1)

    recommendation = arb.recommend_alternative(payload)
    assert recommendation is None
