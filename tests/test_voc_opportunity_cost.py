import pytest
from coreason_economist.models import Budget, ReasoningTrace, VOCDecision
from coreason_economist.voc import VOCEngine


class TestVOCOpportunityCost:
    """
    Tests for the Opportunity Cost logic in VOCEngine.
    """

    @pytest.fixture  # type: ignore
    def voc_engine(self) -> VOCEngine:
        return VOCEngine(default_threshold=0.90)

    @pytest.fixture  # type: ignore
    def trace(self) -> ReasoningTrace:
        return ReasoningTrace(steps=["This is the first draft.", "This is the first draft."])

    def test_evaluate_normal_budget_no_change(self, voc_engine: VOCEngine) -> None:
        """
        Test that normal budget does not affect threshold.
        """
        # ReasoningTrace(steps=["A", "B"]) # Very different, sim ~ 0
        # Similarity is low, should continue

        # trace with high similarity
        trace_high = ReasoningTrace(steps=["Hello World", "Hello World"])  # Sim 1.0

        # Normal budget (50% remaining)
        total = Budget(financial=1.0, latency_ms=1000, token_volume=1000)
        remaining = Budget(financial=0.5, latency_ms=500, token_volume=500)

        result = voc_engine.evaluate(trace_high, remaining_budget=remaining, total_budget=total)
        assert result.decision == VOCDecision.STOP
        assert "Opportunity Cost" not in result.reason
        assert "0.9000" in result.reason  # Check used threshold in message

    def test_evaluate_critical_budget_lowers_threshold(self, voc_engine: VOCEngine) -> None:
        """
        Test that critical budget lowers threshold.
        Default is 0.90. Lowered is 0.81.
        We provide texts with similarity 0.85.
        Normal -> CONTINUE (0.85 < 0.90)
        Critical -> STOP (0.85 >= 0.81)
        """
        # A: "0123456789" (10 chars)
        # B: "01234567" (8 chars)
        # Matches = 8.
        # Sim = 2*8 / (10+8) = 16/18 = 0.888...

        trace = ReasoningTrace(steps=["0123456789", "01234567"])

        # 1. Normal Budget -> Should Continue (0.88 < 0.90)
        total = Budget(financial=1.0)
        remaining = Budget(financial=0.5)

        res_normal = voc_engine.evaluate(trace, remaining_budget=remaining, total_budget=total)
        assert res_normal.decision == VOCDecision.CONTINUE
        assert "Significant change detected" in res_normal.reason

        # 2. Critical Budget -> Should Stop (0.88 >= 0.81)
        # 10% remaining
        remaining_crit = Budget(financial=0.1)

        res_crit = voc_engine.evaluate(trace, remaining_budget=remaining_crit, total_budget=total)
        assert res_crit.decision == VOCDecision.STOP
        assert "Opportunity Cost" in res_crit.reason
        assert "Threshold lowered" in res_crit.reason

    def test_is_budget_critical_financial(self, voc_engine: VOCEngine) -> None:
        total = Budget(financial=10.0)

        # 3.0 left (30%) -> Not Critical
        assert voc_engine._is_budget_critical(Budget(financial=3.0), total) is False

        # 1.0 left (10%) -> Critical
        assert voc_engine._is_budget_critical(Budget(financial=1.0), total) is True

    def test_is_budget_critical_token_volume(self, voc_engine: VOCEngine) -> None:
        total = Budget(token_volume=100)

        # 30 left (30%) -> Not Critical
        assert voc_engine._is_budget_critical(Budget(token_volume=30), total) is False

        # 10 left (10%) -> Critical
        assert voc_engine._is_budget_critical(Budget(token_volume=10), total) is True

    def test_is_budget_critical_latency(self, voc_engine: VOCEngine) -> None:
        total = Budget(latency_ms=100)

        # 30 left (30%) -> Not Critical
        assert voc_engine._is_budget_critical(Budget(latency_ms=30), total) is False

        # 10 left (10%) -> Critical
        assert voc_engine._is_budget_critical(Budget(latency_ms=10), total) is True

    def test_is_budget_critical_mixed(self, voc_engine: VOCEngine) -> None:
        """
        If one dimension is critical, it returns True.
        """
        total = Budget(financial=10.0, latency_ms=100.0)

        # Fin OK, Latency Low
        remaining = Budget(financial=5.0, latency_ms=10.0)  # 50% fin, 10% lat
        assert voc_engine._is_budget_critical(remaining, total) is True

    def test_evaluate_missing_budgets(self, voc_engine: VOCEngine) -> None:
        """
        Test behavior when budgets are not provided (should behave normally).
        """
        trace = ReasoningTrace(steps=["A", "B"])
        # Sim ~ 0
        res = voc_engine.evaluate(trace)
        assert res.decision == VOCDecision.CONTINUE
        assert "Opportunity Cost" not in res.reason
