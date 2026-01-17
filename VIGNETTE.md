# The Architecture and Utility of coreason_economist

### 1. The Philosophy (The Why)

In the rapidly evolving landscape of Large Language Model (LLM) agents, a new problem has emerged: **Cognitive Sprawl**. Agents, when left unchecked, can enter infinite reasoning loops, burn through expensive tokens on trivial tasks, or fail to deliver results within acceptable latency windows.

`coreason_economist` was built to solve this by acting as the **Cognitive "CFO"** for the CoReason platform. While other components decide *how* to think, this package decides *if it is worth* thinking. It introduces a strict **Return on Investment (ROI)** philosophy to agentic workflows, ensuring that the utility of generated intelligence always exceeds the cost of its production.

By implementing concepts from **Rational Metareasoning** and **Value of Computation (VOC)**, `coreason_economist` shifts the paradigm from "maximize capability at all costs" to "optimize capability within constraints." It empowers developers to build agents that are not just smart, but also economically viable and respectful of resource limits.

### 2. Under the Hood (The Dependencies & Logic)

The package relies on a lean, high-integrity stack designed for precision and observability:

*   **`pydantic`**: This is the bedrock of the system. It enforces strict data validation for `Budget`, `EconomicTrace`, and `RequestPayload` models. It handles the complex multi-currency accounting needed to track Financial (USD), Latency (ms), and Token Volume limits simultaneously.
*   **`loguru`**: Provides the granular observability required for an economic engine. Every transaction, approval, or rejection is logged with structure, allowing for the generation of `EconomicTrace` objects that feed into dashboard metrics.
*   **`difflib` (Standard Library)**: Used within the `VOCEngine`, this module powers the "stopping mechanism." By calculating the sequence similarity between reasoning steps, it detects diminishing returns without requiring heavy external dependencies.

**The Logic:**
At the center sits the **`Economist`**, an orchestrator that coordinates three specialized components:
1.  **The Pricer**: Uses heuristic rate cards to predict the cost of a request before it is sent to an LLM provider. It handles the messy reality of estimating output tokens and tool call costs.
2.  **The Budget Authority**: The gatekeeper. It evaluates the Pricer's estimates against the hard and soft limits defined in the `Budget`. If a request exceeds $0.05 or 5000ms, it is rejected *before* execution, saving real resources.
3.  **The VOC Engine**: The monitor. As an agent reasons, this engine analyzes the semantic drift between steps. If the reasoning converges (i.e., the agent is just rephrasing the same thought), it signals a "Stop" decision to prevent waste.

### 3. In Practice (The How)

`coreason_economist` is designed to be integrated directly into the control loop of your agent. Here is how it protects your resources in practice.

#### Example A: The "Pre-Flight Check" (Budget Enforcement)

Before your agent spins up a complex chain of thought, ask the Economist for permission. This ensures you never accidentally spend $10 on a $0.10 problem.

```python
from coreason_economist.economist import Economist
from coreason_economist.models import Budget, RequestPayload

# Initialize the "CFO"
economist = Economist()

# 1. Define the constraints (The Wallet)
project_budget = Budget(
    financial=0.05,       # Max $0.05
    latency_ms=5000,      # Max 5 seconds
    token_volume=8000     # Max 8k context
)

# 2. Formulate the request (The Purchase Order)
request = RequestPayload(
    model_name="gpt-4",
    prompt="Analyze the macroeconomic impact of rate cuts on tech stocks...",
    estimated_output_tokens=500,
    max_budget=project_budget
)

# 3. Ask for permission
trace = economist.check_execution(request)

if trace.decision == "APPROVED":
    print(f"Green light! Estimated cost: ${trace.estimated_cost.financial:.4f}")
    # execute_agent(request)
else:
    print(f"Rejected: {trace.reason}")
    # The trace might contain a cheaper suggestion from the Arbitrageur
    if trace.suggested_alternative:
        print(f"Try this instead: {trace.suggested_alternative.model_name}")
```

#### Example B: The "Early Exit" (Value of Computation)

When your agent is engaged in a multi-step reasoning process, use the `VOCEngine` to detect when it's time to stop.

```python
from coreason_economist.voc import VOCEngine
from coreason_economist.models import ReasoningTrace, VOCDecision

voc = VOCEngine()

# A trace of the agent's internal monologue across steps
trace = ReasoningTrace(steps=[
    "The user wants a sorting algorithm. Merge sort is stable.",
    "Merge sort is O(n log n). It is a stable sort.",
    "Merge sort complexity is O(n log n). Stability is preserved."
])

# Evaluate: Is it worth continuing to think?
result = voc.evaluate(trace)

if result.decision == VOCDecision.STOP:
    print(f"Stop thinking! {result.reason}")
    # Output: Stop thinking! Diminishing returns detected...
else:
    print("Continue reasoning...")
```
