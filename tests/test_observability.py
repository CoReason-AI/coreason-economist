import pytest
from coreason_economist.models import Budget, Decision, EconomicTrace

class TestEconomicTraceObservability:
    """Test suite for EconomicTrace observability metrics."""

    def test_compute_efficiency_metrics_actual_cost(self):
        """Test calculation using actual cost."""
        estimated = Budget(financial=0.10, latency_ms=1000.0, token_volume=1000)
        actual = Budget(financial=0.05, latency_ms=500.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated,
            actual_cost=actual,
            decision=Decision.APPROVED,
            model_used="gpt-4",
            input_tokens=500
        )

        metrics = trace.compute_efficiency_metrics()

        # Financial: 1000 tokens / $0.05 = 20,000 tokens/$
        assert metrics["tokens_per_dollar"] == 20000.0

        # Latency: 1000 tokens / 0.5 sec = 2000 tokens/sec
        assert metrics["tokens_per_second"] == 2000.0

    def test_compute_efficiency_metrics_fallback_estimated(self):
        """Test fallback to estimated cost when actual cost is None."""
        estimated = Budget(financial=0.10, latency_ms=1000.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated,
            actual_cost=None,
            decision=Decision.APPROVED,
            model_used="gpt-4",
            input_tokens=500
        )

        metrics = trace.compute_efficiency_metrics()

        # Financial: 1000 tokens / $0.10 = 10,000 tokens/$
        assert metrics["tokens_per_dollar"] == 10000.0

        # Latency: 1000 tokens / 1.0 sec = 1000 tokens/sec
        assert metrics["tokens_per_second"] == 1000.0

    def test_compute_efficiency_metrics_zero_division_safety(self):
        """Test division by zero safety."""
        estimated = Budget(financial=0.0, latency_ms=0.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated,
            decision=Decision.APPROVED,
            model_used="gpt-4",
            input_tokens=500
        )

        metrics = trace.compute_efficiency_metrics()

        assert metrics["tokens_per_dollar"] == 0.0
        assert metrics["tokens_per_second"] == 0.0

    def test_compute_efficiency_metrics_zero_tokens(self):
        """Test calculation with zero tokens."""
        estimated = Budget(financial=0.10, latency_ms=1000.0, token_volume=0)

        trace = EconomicTrace(
            estimated_cost=estimated,
            decision=Decision.APPROVED,
            model_used="gpt-4",
            input_tokens=0
        )

        metrics = trace.compute_efficiency_metrics()

        assert metrics["tokens_per_dollar"] == 0.0
        assert metrics["tokens_per_second"] == 0.0
