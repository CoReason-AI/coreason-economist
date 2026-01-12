from unittest.mock import MagicMock

import pytest
from coreason_economist.economist import Economist
from coreason_economist.models import ReasoningTrace, VOCDecision, VOCResult
from coreason_economist.voc import VOCEngine


def test_should_continue_exception_propagation() -> None:
    """Test that exceptions from VOCEngine bubble up through Economist."""
    mock_voc = MagicMock(spec=VOCEngine)
    mock_voc.evaluate.side_effect = ValueError("VOC calculation failed")

    economist = Economist(voc_engine=mock_voc)
    trace = ReasoningTrace(steps=["step1"])

    with pytest.raises(ValueError, match="VOC calculation failed"):
        economist.should_continue(trace=trace)


def test_should_continue_empty_trace() -> None:
    """Test delegation with an empty trace."""
    # Use real VOCEngine to verify it handles empty trace gracefully when called via Economist
    economist = Economist()
    trace = ReasoningTrace(steps=[])

    result = economist.should_continue(trace=trace)

    # VOCEngine returns CONTINUE for < 2 steps
    assert result.decision == VOCDecision.CONTINUE
    assert "Insufficient history" in result.reason


def test_should_continue_single_step_trace() -> None:
    """Test delegation with a single step trace."""
    economist = Economist()
    trace = ReasoningTrace(steps=["Just one step"])

    result = economist.should_continue(trace=trace)

    # VOCEngine returns CONTINUE for < 2 steps
    assert result.decision == VOCDecision.CONTINUE
    assert "Insufficient history" in result.reason


def test_should_continue_explicit_none_args() -> None:
    """Test passing explicit None to optional arguments."""
    mock_voc = MagicMock(spec=VOCEngine)
    expected = VOCResult(decision=VOCDecision.CONTINUE, score=0.0, reason="ok")
    mock_voc.evaluate.return_value = expected

    economist = Economist(voc_engine=mock_voc)
    trace = ReasoningTrace(steps=["a", "b"])

    # Pass None explicitly
    result = economist.should_continue(trace=trace, threshold=None, remaining_budget=None, total_budget=None)

    assert result == expected
    mock_voc.evaluate.assert_called_once_with(trace=trace, threshold=None, remaining_budget=None, total_budget=None)
