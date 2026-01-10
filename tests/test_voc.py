# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist


from coreason_economist.models import ReasoningTrace, VOCDecision
from coreason_economist.voc import VOCEngine


class TestVOCEngine:
    def test_initialization(self) -> None:
        engine = VOCEngine(default_threshold=0.8)
        assert engine.default_threshold == 0.8

        default_engine = VOCEngine()
        assert default_engine.default_threshold == 0.95

    def test_similarity_calculation(self) -> None:
        engine = VOCEngine()

        # Exact match
        assert engine._calculate_similarity("hello world", "hello world") == 1.0

        # Completely different
        assert engine._calculate_similarity("abc", "xyz") == 0.0

        # Partial match
        # "apple" vs "apply" -> 4 common chars out of 10 total length?
        # difflib ratio is 2*M / T. M=4 ("appl"), T=10. 8/10 = 0.8.
        assert 0.7 < engine._calculate_similarity("apple", "apply") < 0.9

    def test_insufficient_history(self) -> None:
        engine = VOCEngine()

        # Empty trace
        trace = ReasoningTrace(steps=[])
        result = engine.evaluate(trace)
        assert result.decision == VOCDecision.CONTINUE
        assert "Insufficient history" in result.reason

        # Single step trace
        trace = ReasoningTrace(steps=["Just starting"])
        result = engine.evaluate(trace)
        assert result.decision == VOCDecision.CONTINUE
        assert "Insufficient history" in result.reason

    def test_diminishing_returns_stop(self) -> None:
        engine = VOCEngine(default_threshold=0.9)

        # Two very similar steps
        step1 = "The answer is 42 because it is the meaning of life."
        step2 = "The answer is 42 because it is the meaning of life."

        trace = ReasoningTrace(steps=[step1, step2])
        result = engine.evaluate(trace)

        assert result.decision == VOCDecision.STOP
        assert result.score == 1.0
        assert "Diminishing returns" in result.reason

    def test_significant_change_continue(self) -> None:
        engine = VOCEngine(default_threshold=0.9)

        step1 = "I think the answer might be 10."
        step2 = "Upon further review, the answer is definitely 42."

        trace = ReasoningTrace(steps=[step1, step2])
        result = engine.evaluate(trace)

        assert result.decision == VOCDecision.CONTINUE
        assert result.score < 0.9
        assert "Significant change" in result.reason

    def test_threshold_override(self) -> None:
        engine = VOCEngine(default_threshold=0.95)

        step1 = "abcde"
        step2 = "abcdf"
        # similarity is 0.8

        trace = ReasoningTrace(steps=[step1, step2])

        # With default 0.95, it should CONTINUE (0.8 < 0.95)
        result_default = engine.evaluate(trace)
        assert result_default.decision == VOCDecision.CONTINUE

        # With override 0.5, it should STOP (0.8 > 0.5)
        result_override = engine.evaluate(trace, threshold=0.5)
        assert result_override.decision == VOCDecision.STOP

    def test_many_steps_only_checks_last_two(self) -> None:
        engine = VOCEngine()

        # steps: [A, B, A, B]
        # Compare last two: A vs B (different) -> Continue
        # Even though A was repeated before, we only look at immediate convergence in this iteration.

        stepA = "Apple"
        stepB = "Banana"

        trace = ReasoningTrace(steps=[stepA, stepB, stepA, stepB])
        result = engine.evaluate(trace)

        assert result.decision == VOCDecision.CONTINUE
