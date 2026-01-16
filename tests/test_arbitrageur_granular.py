# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

import pytest
from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer

# Rates for gpt-4o:
# Input: $0.005 / 1k
# Output: $0.015 / 1k
# Prompt: 4000 chars -> 1000 tokens input.
# Est Output: 200 tokens.
# Cost per agent-round:
# Input: 1 * 0.005 = 0.005
# Output: 0.2 * 0.015 = 0.003
# Total: $0.008


@pytest.fixture  # type: ignore
def arbitrageur() -> Arbitrageur:
    return Arbitrageur(pricer=Pricer())


def test_granular_reduction_rounds_priority(arbitrageur: Arbitrageur) -> None:
    """
    Request: 5 Agents, 3 Rounds.
    Cost: 5 * 3 * 0.008 = $0.12.
    Budget: $0.041 (Fits 5 Agents, 1 Round = $0.04).
    Expectation: Reduce Rounds to 1, keep Agents at 5.
    """
    prompt = "a" * 4000  # ~1000 tokens
    request = RequestPayload(
        model_name="gpt-4o",
        prompt=prompt,
        agent_count=5,
        rounds=3,
        max_budget=Budget(financial=0.041),
        difficulty_score=0.9,  # High difficulty, prefer current model
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    assert suggestion.model_name == "gpt-4o"
    assert suggestion.agent_count == 5
    assert suggestion.rounds == 1
    assert suggestion.quality_warning is not None
    assert "Reduced topology" in suggestion.quality_warning


def test_granular_reduction_agents_secondary(arbitrageur: Arbitrageur) -> None:
    """
    Request: 5 Agents, 3 Rounds.
    Cost: 5 * 3 * 0.008 = $0.12.
    Budget: $0.025 (Fits 3 Agents, 1 Round = $0.024).
    Expectation: Reduce Rounds to 1, Reduce Agents to 3.
    """
    prompt = "a" * 4000  # ~1000 tokens
    request = RequestPayload(
        model_name="gpt-4o",
        prompt=prompt,
        agent_count=5,
        rounds=3,
        max_budget=Budget(financial=0.025),
        difficulty_score=0.9,  # High difficulty
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    assert suggestion.model_name == "gpt-4o"
    assert suggestion.agent_count == 3
    assert suggestion.rounds == 1


def test_granular_reduction_fails_if_too_small(arbitrageur: Arbitrageur) -> None:
    """
    Request: 5 Agents, 3 Rounds.
    Budget: $0.001 (Too small even for 1A 1R = $0.008).
    Expectation: Try to find cheaper model (Strategy 1 fallback).
    Cheapest is gpt-4o-mini ($0.00015 + $0.0006).
    1000 in, 200 out.
    In: 0.00015
    Out: 0.2 * 0.0006 = 0.00012
    Total: 0.00027.
    Budget $0.001 covers it.
    """
    prompt = "a" * 4000
    request = RequestPayload(
        model_name="gpt-4o",
        prompt=prompt,
        agent_count=5,
        rounds=3,
        max_budget=Budget(financial=0.001),
        difficulty_score=0.9,
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    # Should switch model because gpt-4o single shot is too expensive
    assert suggestion.model_name == "gpt-4o-mini"
    assert suggestion.agent_count == 1
    assert suggestion.rounds == 1
