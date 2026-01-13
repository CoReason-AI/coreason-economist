from coreason_economist.models import Budget, Decision, EconomicTrace


class TestEconomicTraceObservability:
    """Test suite for EconomicTrace observability metrics using computed fields."""

    def test_compute_efficiency_metrics_actual_cost(self) -> None:
        """Test calculation using actual cost."""
        estimated = Budget(financial=0.10, latency_ms=1000.0, token_volume=1000)
        actual = Budget(financial=0.05, latency_ms=500.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated,
            actual_cost=actual,
            decision=Decision.APPROVED,
            model_used="gpt-4",
            input_tokens=500,
        )

        # Financial: 1000 tokens / $0.05 = 20,000 tokens/$
        assert trace.tokens_per_dollar == 20000.0

        # Latency: 1000 tokens / 0.5 sec = 2000 tokens/sec
        assert trace.tokens_per_second == 2000.0

        # Latency/Token: 500 ms / 1000 tokens = 0.5 ms/token
        assert trace.latency_per_token == 0.5

    def test_compute_efficiency_metrics_fallback_estimated(self) -> None:
        """Test fallback to estimated cost when actual cost is None."""
        estimated = Budget(financial=0.10, latency_ms=1000.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated, actual_cost=None, decision=Decision.APPROVED, model_used="gpt-4", input_tokens=500
        )

        # Financial: 1000 tokens / $0.10 = 10,000 tokens/$
        assert trace.tokens_per_dollar == 10000.0

        # Latency: 1000 tokens / 1.0 sec = 1000 tokens/sec
        assert trace.tokens_per_second == 1000.0

        # Latency/Token: 1000 ms / 1000 tokens = 1.0 ms/token
        assert trace.latency_per_token == 1.0

    def test_compute_efficiency_metrics_zero_division_safety(self) -> None:
        """Test division by zero safety."""
        estimated = Budget(financial=0.0, latency_ms=0.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated, decision=Decision.APPROVED, model_used="gpt-4", input_tokens=500
        )

        assert trace.tokens_per_dollar == 0.0
        assert trace.tokens_per_second == 0.0
        assert trace.latency_per_token == 0.0

    def test_mixed_zero_non_zero(self) -> None:
        """Test independent handling of zero denominators."""
        # Case 1: Cost is 0 (free), Latency > 0
        actual = Budget(financial=0.0, latency_ms=1000.0, token_volume=1000)
        trace_free = EconomicTrace(
            estimated_cost=actual,
            actual_cost=actual,
            decision=Decision.APPROVED,
            model_used="gpt-4-free",
            input_tokens=500,
        )
        assert trace_free.tokens_per_dollar == 0.0  # Guarded
        assert trace_free.tokens_per_second == 1000.0  # Valid
        assert trace_free.latency_per_token == 1.0  # 1000 / 1000

        # Case 2: Cost > 0, Latency is 0 (instant)
        actual_instant = Budget(financial=1.0, latency_ms=0.0, token_volume=1000)
        trace_instant = EconomicTrace(
            estimated_cost=actual_instant,
            actual_cost=actual_instant,
            decision=Decision.APPROVED,
            model_used="gpt-4-instant",
            input_tokens=500,
        )
        assert trace_instant.tokens_per_dollar == 1000.0  # Valid
        assert trace_instant.tokens_per_second == 0.0  # Guarded
        assert trace_instant.latency_per_token == 0.0  # 0 / 1000

    def test_small_values_precision(self) -> None:
        """Test calculation stability with micro-values."""
        # $0.0001 cost, 0.1ms latency (100 microseconds)
        actual = Budget(financial=0.0001, latency_ms=0.1, token_volume=100)
        trace = EconomicTrace(
            estimated_cost=actual,
            actual_cost=actual,
            decision=Decision.APPROVED,
            model_used="micro-model",
            input_tokens=10,
        )

        # 100 / 0.0001 = 1,000,000
        assert trace.tokens_per_dollar == 1_000_000.0

        # 0.1ms = 0.0001s. 100 / 0.0001 = 1,000,000
        assert trace.tokens_per_second == 1_000_000.0

        # 0.1ms / 100 = 0.001 ms/token
        assert trace.latency_per_token == 0.001

    def test_strict_actual_precedence(self) -> None:
        """Verify actual_cost fully overrides estimated_cost, even if actual has zeros."""
        estimated = Budget(financial=10.0, latency_ms=1000.0, token_volume=1000)
        # Actual was free and instant (e.g., cached), but had volume
        actual = Budget(financial=0.0, latency_ms=0.0, token_volume=1000)

        trace = EconomicTrace(
            estimated_cost=estimated,
            actual_cost=actual,
            decision=Decision.APPROVED,
            model_used="cached-model",
            input_tokens=500,
        )

        # Should use ACTUAL values (0.0), not fall back to ESTIMATED (10.0/1000.0)
        assert trace.tokens_per_dollar == 0.0
        assert trace.tokens_per_second == 0.0
        assert trace.latency_per_token == 0.0

    def test_free_transaction_handling(self) -> None:
        """Confirm that 'infinite' efficiency (free transaction) returns safe 0.0."""
        # Free transaction ($0 cost)
        actual = Budget(financial=0.0, latency_ms=1000.0, token_volume=500)
        trace = EconomicTrace(
            estimated_cost=actual,
            actual_cost=actual,
            decision=Decision.APPROVED,
            model_used="free-model",
            input_tokens=250,
        )

        # Mathematically infinite, but dashboard safe is 0.0
        assert trace.tokens_per_dollar == 0.0

    def test_compute_efficiency_metrics_zero_tokens(self) -> None:
        """Test calculation with zero tokens."""
        estimated = Budget(financial=0.10, latency_ms=1000.0, token_volume=0)

        trace = EconomicTrace(estimated_cost=estimated, decision=Decision.APPROVED, model_used="gpt-4", input_tokens=0)

        assert trace.tokens_per_dollar == 0.0
        assert trace.tokens_per_second == 0.0
        assert trace.latency_per_token == 0.0
