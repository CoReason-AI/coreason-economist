from unittest.mock import MagicMock

from coreason_economist.economist import Economist
from coreason_economist.models import Budget, ReasoningTrace, VOCDecision, VOCResult
from coreason_economist.voc import VOCEngine


def test_economist_init_with_voc_engine() -> None:
    """Test that Economist can be initialized with a VOCEngine."""
    voc_engine = VOCEngine()
    economist = Economist(voc_engine=voc_engine)
    assert economist.voc_engine is voc_engine


def test_economist_init_default_voc_engine() -> None:
    """Test that Economist creates a default VOCEngine if none is provided."""
    economist = Economist()
    assert isinstance(economist.voc_engine, VOCEngine)


def test_should_continue_delegation() -> None:
    """Test that should_continue delegates to voc_engine.evaluate."""
    # Setup
    mock_voc = MagicMock(spec=VOCEngine)
    expected_result = VOCResult(decision=VOCDecision.CONTINUE, score=0.5, reason="Test reason")
    mock_voc.evaluate.return_value = expected_result

    economist = Economist(voc_engine=mock_voc)

    # Inputs
    trace = ReasoningTrace(steps=["step1", "step2"])
    remaining_budget = Budget(financial=0.5, latency_ms=1000, token_volume=1000)
    total_budget = Budget(financial=1.0, latency_ms=2000, token_volume=2000)
    threshold = 0.85

    # Execute
    result = economist.should_continue(
        trace=trace, threshold=threshold, remaining_budget=remaining_budget, total_budget=total_budget
    )

    # Verify
    mock_voc.evaluate.assert_called_once_with(
        trace=trace, threshold=threshold, remaining_budget=remaining_budget, total_budget=total_budget
    )
    assert result == expected_result
