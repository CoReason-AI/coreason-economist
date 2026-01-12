import pytest
from coreason_economist.models import Budget, ReasoningTrace, VOCDecision
from coreason_economist.voc import VOCEngine


class TestVOCEdgeCases:
    """
    Tests for edge cases and complex scenarios in VOCEngine Opportunity Cost logic.
    """

    @pytest.fixture  # type: ignore
    def voc_engine(self) -> VOCEngine:
        return VOCEngine(default_threshold=0.90)

    def test_zero_total_budget_ignored(self, voc_engine: VOCEngine) -> None:
        """
        Test that if total budget is 0 (unlimited/unset), it does NOT trigger critical budget logic
        and does NOT cause ZeroDivisionError.
        """
        # Similarity 0.85 (below default 0.90)
        # Trace: "0123456789" (10) vs "01234567" (8) -> Sim ~0.88
        trace = ReasoningTrace(steps=["0123456789", "01234567"])

        # Total is 0, Remaining is 0 (or anything)
        total = Budget(financial=0.0, latency_ms=0.0, token_volume=0)
        remaining = Budget(financial=0.0, latency_ms=0.0, token_volume=0)

        # Should behave as normal (CONTINUE because 0.88 < 0.90)
        # If it triggered critical, it would lower threshold to 0.81 and STOP.
        res = voc_engine.evaluate(trace, remaining_budget=remaining, total_budget=total)

        assert res.decision == VOCDecision.CONTINUE
        assert "Opportunity Cost" not in res.reason

    def test_zero_remaining_budget(self, voc_engine: VOCEngine) -> None:
        """
        Test that zero remaining budget (fully exhausted) triggers critical state.
        0 < 0.2 * Total
        """
        trace = ReasoningTrace(steps=["0123456789", "01234567"])  # Sim ~0.88

        total = Budget(financial=10.0)
        # Budget model enforces ge=0, so we test 0.0 (exhausted)
        remaining = Budget(financial=0.0)

        # Should trigger critical -> threshold 0.81 -> STOP
        res = voc_engine.evaluate(trace, remaining_budget=remaining, total_budget=total)

        assert res.decision == VOCDecision.STOP
        assert "Opportunity Cost" in res.reason

    def test_boundary_condition_exact_threshold(self, voc_engine: VOCEngine) -> None:
        """
        Test boundary conditions around the 20% threshold.
        """
        trace = ReasoningTrace(steps=["0123456789", "01234567"])  # Sim ~0.88
        total = Budget(financial=100.0)

        # Case 1: Exactly 20.0 remaining (20%)
        # Logic is remaining / total < 0.2
        # 20/100 = 0.2. 0.2 < 0.2 is False. NOT critical.
        remaining_exact = Budget(financial=20.0)
        res_exact = voc_engine.evaluate(trace, remaining_budget=remaining_exact, total_budget=total)
        assert res_exact.decision == VOCDecision.CONTINUE  # 0.88 < 0.90

        # Case 2: 19.9 remaining (19.9%)
        # 19.9/100 < 0.2. True. Critical.
        remaining_below = Budget(financial=19.9)
        res_below = voc_engine.evaluate(trace, remaining_budget=remaining_below, total_budget=total)
        assert res_below.decision == VOCDecision.STOP  # 0.88 >= 0.81

    def test_custom_threshold_modification(self, voc_engine: VOCEngine) -> None:
        """
        Verify that if a user provides a custom threshold, the modifier applies to THAT threshold.
        """
        # Trace sim ~0.88
        trace = ReasoningTrace(steps=["0123456789", "01234567"])

        # User sets threshold to 0.99.
        # Critical budget -> 0.99 * 0.9 = 0.891.
        # Sim 0.88 < 0.891. So it should still CONTINUE.

        total = Budget(financial=10.0)
        remaining = Budget(financial=1.0)  # Critical

        res = voc_engine.evaluate(trace, threshold=0.99, remaining_budget=remaining, total_budget=total)
        assert res.decision == VOCDecision.CONTINUE
        assert "Opportunity Cost" not in res.reason
        assert "threshold 0.8910" in res.reason

        # Now set threshold to 0.95.
        # Critical -> 0.95 * 0.9 = 0.855.
        # Sim 0.88 > 0.855. STOP.
        res2 = voc_engine.evaluate(trace, threshold=0.95, remaining_budget=remaining, total_budget=total)
        assert res2.decision == VOCDecision.STOP
        assert "Opportunity Cost" in res2.reason

    def test_complex_diminishing_budget_loop(self, voc_engine: VOCEngine) -> None:
        """
        Scenario: "The Desperate Hail Mary"
        Simulate a loop where:
        1. Text similarity is constant at 0.88 (below default 0.90).
        2. Budget drains step by step.
        3. Expectation: Continue, Continue, ..., then STOP when budget hits <20%.
        """
        # Sim ~0.88
        trace = ReasoningTrace(steps=["0123456789", "01234567"])

        total = Budget(financial=100.0)

        # Step 1: 50% remaining
        res1 = voc_engine.evaluate(trace, remaining_budget=Budget(financial=50.0), total_budget=total)
        assert res1.decision == VOCDecision.CONTINUE

        # Step 2: 30% remaining
        res2 = voc_engine.evaluate(trace, remaining_budget=Budget(financial=30.0), total_budget=total)
        assert res2.decision == VOCDecision.CONTINUE

        # Step 3: 21% remaining
        res3 = voc_engine.evaluate(trace, remaining_budget=Budget(financial=21.0), total_budget=total)
        assert res3.decision == VOCDecision.CONTINUE

        # Step 4: 19% remaining -> CRITICAL -> Threshold drops to 0.81 -> STOP
        res4 = voc_engine.evaluate(trace, remaining_budget=Budget(financial=19.0), total_budget=total)
        assert res4.decision == VOCDecision.STOP
        assert "Opportunity Cost" in res4.reason
