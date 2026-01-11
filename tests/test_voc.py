# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist


import json

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

    def test_similarity_edge_cases(self) -> None:
        """Test empty string edge cases for 100% coverage."""
        engine = VOCEngine()

        # Both empty -> 1.0 similarity (identical)
        assert engine._calculate_similarity("", "") == 1.0

        # One empty -> 0.0 similarity
        assert engine._calculate_similarity("", "abc") == 0.0
        assert engine._calculate_similarity("abc", "") == 0.0

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

    def test_whitespace_sensitivity(self) -> None:
        """Test how the engine handles whitespace differences."""
        engine = VOCEngine()

        text_a = "The quick brown fox"
        text_b = "The quick brown fox "  # Trailing space

        # Expect high similarity but not 1.0
        similarity = engine._calculate_similarity(text_a, text_b)
        assert 0.9 < similarity < 1.0

        # With significant whitespace change
        text_c = "The     quick     brown     fox"
        similarity_c = engine._calculate_similarity(text_a, text_c)
        assert similarity_c < 0.9  # Should be lower

    def test_case_sensitivity(self) -> None:
        """Test case sensitivity."""
        engine = VOCEngine()

        text_a = "STOP"
        text_b = "stop"

        # Expect low similarity due to case difference in short string
        # S != s, T != t, etc.
        similarity = engine._calculate_similarity(text_a, text_b)
        assert similarity == 0.0

        text_long_a = "The answer is definitely forty-two."
        text_long_b = "The answer is definitely FORTY-TWO."
        # Longer string shares more common chars, so similarity is non-zero but < 1.0
        similarity_long = engine._calculate_similarity(text_long_a, text_long_b)
        assert 0.5 < similarity_long < 1.0

    def test_json_structure_sensitivity(self) -> None:
        """
        Test that JSON with reordered keys is treated as different.
        This confirms the engine is lexical, not semantic.
        """
        engine = VOCEngine()

        obj_a = {"name": "Alice", "age": 30}
        obj_b = {"age": 30, "name": "Alice"}

        text_a = json.dumps(obj_a)
        text_b = json.dumps(obj_b)

        # Lexically these strings are quite different:
        # '{"name": "Alice", "age": 30}'
        # '{"age": 30, "name": "Alice"}'
        similarity = engine._calculate_similarity(text_a, text_b)

        # They share content, so > 0, but reordering lowers the score significantly
        assert similarity < 0.95
        assert similarity > 0.4

    def test_large_payloads(self) -> None:
        """
        Test performance/correctness with large inputs.
        Simulating a scenario with ~50k characters.
        """
        engine = VOCEngine()

        # Create a large base string
        base_chunk = "The quick brown fox jumps over the lazy dog. " * 1000  # ~45k chars
        text_a = base_chunk
        text_b = base_chunk + "And then it slept."

        # Should be very similar
        similarity = engine._calculate_similarity(text_a, text_b)
        assert similarity > 0.99
        assert similarity < 1.0

        trace = ReasoningTrace(steps=[text_a, text_b])
        result = engine.evaluate(trace)

        # With default threshold 0.95, this should STOP
        assert result.decision == VOCDecision.STOP
        assert "Diminishing returns" in result.reason

    def test_code_block_similarity(self) -> None:
        """Test similarity on code blocks with comment changes."""
        engine = VOCEngine()

        code_a = """
        def add(a, b):
            # Adds two numbers
            return a + b
        """

        code_b = """
        def add(a, b):
            # This function adds two integers
            return a + b
        """

        similarity = engine._calculate_similarity(code_a, code_b)

        # High similarity expected, but definitely not 1.0
        assert 0.8 < similarity < 0.95

        # Should likely CONTINUE if threshold is 0.95
        trace = ReasoningTrace(steps=[code_a, code_b])
        result = engine.evaluate(trace, threshold=0.95)
        assert result.decision == VOCDecision.CONTINUE
