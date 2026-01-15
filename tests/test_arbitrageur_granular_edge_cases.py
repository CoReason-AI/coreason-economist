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
# Latency: 12ms / output token.
#
# Setup: 4000 chars input (~1000 tokens). 200 output tokens.
# Cost per agent-round:
#   Financial: (1 * 0.005) + (0.2 * 0.015) = 0.005 + 0.003 = $0.008
#   Latency: 200 * 12ms = 2400ms (2.4s) per round.
#   Token Vol: 1000 + 200 = 1200 tokens per agent-round.


@pytest.fixture  # type: ignore
def arbitrageur() -> Arbitrageur:
    return Arbitrageur(pricer=Pricer())


def test_exact_budget_match(arbitrageur: Arbitrageur) -> None:
    """
    Test that if budget exactly matches a reduced topology, it is picked.
    Target: 2 Agents, 2 Rounds.
    Cost: 2 * 2 * 0.008 = $0.032.
    """
    prompt = "a" * 4000
    request = RequestPayload(
        model_name="gpt-4o",
        prompt=prompt,
        agent_count=5,
        rounds=5,
        max_budget=Budget(financial=0.032),  # Exact match for 2A, 2R (or 4A, 1R)
        # 4A, 1R = 4 * 1 * 0.008 = 0.032.
        # Priority: Reduce Rounds first.
        # So we prefer 4A, 1R over 2A, 2R?
        # Yes.
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    # We expect 4A, 1R because we iterate Agents (5->1) then Rounds (5->1).
    # 5A: fails.
    # 4A:
    #   5R fail
    #   ...
    #   1R fits (0.032).
    # So we expect 4 Agents, 1 Round.
    assert suggestion.agent_count == 4
    assert suggestion.rounds == 1


def test_latency_dominates_rounds(arbitrageur: Arbitrageur) -> None:
    """
    Financial allows many rounds. Latency allows few.
    Constraint: Latency < 3000ms.
    1 Round = 2400ms.
    2 Rounds = 4800ms.
    So Max Rounds = 1.
    Financial allows 10 agents easily.
    """
    prompt = "a" * 4000
    request = RequestPayload(
        model_name="gpt-4o",
        prompt=prompt,
        agent_count=5,
        rounds=3,
        difficulty_score=0.9,  # High difficulty to trigger warning
        max_budget=Budget(
            financial=1.0,  # Plenty
            latency_ms=3000.0,  # Stricter: Only fits 1 round (2400ms)
        ),
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    # Agents should stay at 5 because financial is high.
    assert suggestion.agent_count == 5
    # Rounds must drop to 1.
    assert suggestion.rounds == 1
    assert suggestion.quality_warning is not None
    assert "Reduced topology" in suggestion.quality_warning


def test_token_volume_dominates_agents(arbitrageur: Arbitrageur) -> None:
    """
    Financial allows many agents. Token Volume allows few.
    Per agent-round: 1200 tokens.
    Limit: 4000 tokens.
    Request: 5 Agents, 1 Round. (6000 tokens).
    Max Agents: 3 (3600 tokens).
    """
    prompt = "a" * 4000
    request = RequestPayload(
        model_name="gpt-4o",
        prompt=prompt,
        agent_count=5,
        rounds=1,
        max_budget=Budget(
            financial=1.0,
            token_volume=4000,
        ),
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    # Rounds stay 1.
    assert suggestion.rounds == 1
    # Agents reduce to 3.
    assert suggestion.agent_count == 3


def test_large_scale_reduction(arbitrageur: Arbitrageur) -> None:
    """
    Ensure logic handles larger numbers reasonably fast.
    Request: 50 Agents, 10 Rounds.
    Budget allows: 2 Agents, 2 Rounds.
    """
    prompt = "a" * 4000
    # Cost per unit: $0.008.
    # Target: 2A * 2R = 4 units = $0.032.
    # 4A * 1R = 4 units = $0.032.
    # We prefer 4A, 1R.
    # Let's verify we get 4A, 1R.

    request = RequestPayload(
        model_name="gpt-4o",
        prompt=prompt,
        agent_count=50,
        rounds=10,
        max_budget=Budget(financial=0.0321),  # Slightly over 0.032 to account for floats
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    assert suggestion.agent_count == 4
    assert suggestion.rounds == 1
