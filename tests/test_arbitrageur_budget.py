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


def test_budget_fitting_high_difficulty() -> None:
    """
    Test that Arbitrageur fits budget even for high difficulty tasks.
    """
    arbitrageur = Arbitrageur()

    # Request: High cost, high difficulty
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 4000,  # ~1k tokens
        estimated_output_tokens=1000,
        agent_count=5,  # Expensive topology
        rounds=3,
        difficulty_score=0.9,  # High difficulty
        max_budget=Budget(financial=0.01),  # Very tight budget
    )

    # Normal execution would cost ~ $0.02 * 15 = $0.30 (approx)
    # Budget is $0.01.

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    # Should reduce topology
    assert suggestion.agent_count == 1
    assert suggestion.rounds == 1
    # Should likely downgrade model too, as GPT-4o 1k/1k single shot is ~0.02 which is > 0.01
    assert suggestion.model_name == "gpt-4o-mini"  # or cheapest
    assert suggestion.quality_warning is not None
    assert "Downgraded" in suggestion.quality_warning


def test_budget_fitting_topology_only_high_difficulty() -> None:
    """
    Test that Arbitrageur only reduces topology if that's enough,
    even for high difficulty, and sets appropriate warning.
    This hits the coverage gap for "Reduced to single-shot...".
    """
    arbitrageur = Arbitrageur()

    # Request: Moderate cost, High difficulty
    # 100 in, 100 out. Per agent: ~0.002.
    # 10 agents: 0.02.
    # Budget: 0.01.
    # 5 agents fit (0.01).
    # Prior logic: Reduced to 1 agent.
    # New logic: Maximizes utility -> reduces to 5 agents.

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 400,  # ~100 tokens
        estimated_output_tokens=100,
        agent_count=10,
        rounds=1,
        difficulty_score=0.9,  # High
        max_budget=Budget(financial=0.01),
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    assert suggestion.agent_count == 5  # Reduced to fit budget (5 * 0.002 = 0.01)
    assert suggestion.model_name == "gpt-4o"  # Kept same model
    assert suggestion.quality_warning is not None
    assert "Reduced topology to" in suggestion.quality_warning


def test_budget_fitting_topology_no_difficulty_score() -> None:
    """
    Test that Arbitrageur reduces topology if needed even if difficulty is unknown,
    but does NOT set the specific high-difficulty warning.
    """
    arbitrageur = Arbitrageur()

    # Request: Moderate cost, Unknown difficulty
    # 10 agents -> 0.02. Budget -> 0.01.
    # New logic: Reduces to 5 agents.

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 400,
        estimated_output_tokens=100,
        agent_count=10,
        rounds=1,
        difficulty_score=None,  # Unknown
        max_budget=Budget(financial=0.01),
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    assert suggestion.agent_count == 5
    assert suggestion.model_name == "gpt-4o"
    # Should NOT have the specific high-difficulty warning
    assert suggestion.quality_warning is None


def test_budget_fitting_topology_low_difficulty() -> None:
    """
    Test that Arbitrageur reduces topology if needed for low difficulty,
    and does NOT set warning (because low difficulty implies it's fine).
    AND it should proceed to optimize model (Strategy 2) because difficulty is low.
    """
    arbitrageur = Arbitrageur()

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 400,
        estimated_output_tokens=100,
        agent_count=10,
        rounds=1,
        difficulty_score=0.1,  # Low
        max_budget=Budget(financial=0.01),
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    # Strategy 1 fits budget at agent_count=5
    # Strategy 2 then kicks in.
    # It sets agent_count=1 (topology optimization for low difficulty)
    # Then downgrades model.
    assert suggestion.agent_count == 1
    # Strategy 2 should kick in and downgrade to mini because it's cheaper and safe (low difficulty)
    assert suggestion.model_name == "gpt-4o-mini"
    assert suggestion.quality_warning is None


def test_budget_fitting_model_downgrade_low_difficulty() -> None:
    """
    Test that Arbitrageur downgrades model if topology reduction isn't enough,
    even for low difficulty.
    This hits the coverage gap for line 149 (else block).
    """
    arbitrageur = Arbitrageur()

    # Request: gpt-4o, 1 agent.
    # Cost: ~0.002.
    # Budget: 0.0005.
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 400,
        estimated_output_tokens=100,
        agent_count=1,  # Already min topology
        rounds=1,
        difficulty_score=0.1,  # Low
        max_budget=Budget(financial=0.0005),
    )

    suggestion = arbitrageur.recommend_alternative(request)

    assert suggestion is not None
    assert suggestion.model_name == "gpt-4o-mini"
    # Should have warning because we forced a downgrade due to budget, even though difficulty is low?
    # Logic says yes: "Downgraded ... to fit budget."
    assert suggestion.quality_warning is not None
    assert "Downgraded to gpt-4o-mini to fit budget" in suggestion.quality_warning


def test_no_change_if_budget_ok_high_difficulty() -> None:
    """
    Test that no changes are made if budget is respected and difficulty is high.
    """
    arbitrageur = Arbitrageur()

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="Hello",
        difficulty_score=0.9,
        max_budget=Budget(financial=100.0),
    )

    suggestion = arbitrageur.recommend_alternative(request)
    assert suggestion is None


def test_quality_warning_content() -> None:
    """Test that quality warning message is descriptive."""
    arbitrageur = Arbitrageur()

    request = RequestPayload(
        model_name="gpt-4o",
        prompt="A" * 4000,
        estimated_output_tokens=1000,
        agent_count=5,
        rounds=5,
        difficulty_score=0.9,
        max_budget=Budget(financial=0.001),  # Extremely tight
    )

    suggestion = arbitrageur.recommend_alternative(request)
    assert suggestion is not None
    assert suggestion.quality_warning is not None
    assert "fit budget" in suggestion.quality_warning
