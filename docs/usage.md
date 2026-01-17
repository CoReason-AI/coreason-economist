# Usage & User Stories

This section describes how to use `coreason-economist` to implement cost controls and optimization in your agentic workflows.

## Story A: The "Hard Stop" (Budget Enforcement)

**Scenario:** You want to ensure a complex query does not exceed a hard financial limit (e.g., $0.05).

```python
from coreason_economist import Economist, RequestPayload, Budget, Decision

economist = Economist()

# Define the request with a strict financial budget
request = RequestPayload(
    model_name="gpt-4",
    prompt="Explain the geopolitical implications of...",
    max_budget=Budget(financial=0.05),
    estimated_output_tokens=1000,
    agent_count=1,
    rounds=1
)

# Check execution
trace = economist.check_execution(request)

if trace.decision == Decision.APPROVED:
    print(f"Approved! Estimated cost: ${trace.estimated_cost.financial:.4f}")
    # Proceed with actual execution...
elif trace.decision == Decision.REJECTED:
    print(f"Rejected: {trace.reason}")
    if trace.suggested_alternative:
        print(f"Alternative Suggestion: Use {trace.suggested_alternative.model_name}")
        # Optionally retry with the alternative
```

## Story B: The "Early Exit" (VOC Optimization)

**Scenario:** You are running a multi-round debate and want to stop if the results are converging (diminishing returns).

```python
from coreason_economist import Economist, ReasoningTrace, VOCResult, VOCDecision

economist = Economist()

# Simulate a history of reasoning steps
history = [
    "The solution is X because of A, B, C.",
    "The solution is X because of A and B."
]

trace_history = ReasoningTrace(steps=history)

# Evaluate whether to continue
voc_result: VOCResult = economist.should_continue(trace_history)

if voc_result.decision == VOCDecision.STOP:
    print(f"Stop thinking: {voc_result.reason} (Score: {voc_result.score})")
    # Terminate the loop
else:
    print("Continue reasoning...")
```

## Story C: The "Latency Veto" (Time Budgeting)

**Scenario:** A real-time application requires a response within a strict time limit (e.g., 3 seconds).

```python
from coreason_economist import Economist, RequestPayload, Budget, Decision

economist = Economist()

# Define the request with a latency budget
request = RequestPayload(
    model_name="gpt-4",
    prompt="Quick status update",
    max_budget=Budget(latency_ms=3000), # 3 seconds
    estimated_output_tokens=200
)

trace = economist.check_execution(request)

if trace.decision == Decision.REJECTED:
    print(f"Rejected due to latency constraints: {trace.reason}")
    # Fallback to a faster model or cached response
```
