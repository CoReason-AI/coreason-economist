# Components & Requirements

## Core Functional Requirements

### 3.1 The Budget Authority (The Controller)

**Concept:** The gatekeeper that enforces limits set by the parent application (`coreason-maco`).

*   **Multi-Currency Support:** It must manage three distinct "currencies":
    *   **Financial:** Hard dollar limits (e.g., "$0.50 max per query").
    *   **Latency:** Time budgets (e.g., "Must answer within 5 seconds").
    *   **Token Volume:** Context window limits (e.g., "Don't exceed 128k tokens").
*   **Authorization Protocol:** It provides a boolean `allow_execution(request_payload)` interface. If a request from cortex exceeds the remaining budget, it returns `False` with a specific `BudgetExhaustedError`.

### 3.2 The Pricer (The Estimator)

**Concept:** A dynamic look-up engine that predicts the cost of an action before it happens.

*   **Model Rate Cards:** Maintains an up-to-date registry of costs for supported models (Input/Output token prices for GPT-4, Claude, Llama, etc.).
*   **Heuristic Estimation:** Since we don't know the exact output length before generation, the Pricer must use heuristics (e.g., "Average summarization is 20% of input length") to forecast the final bill.
*   **Tool Pricing:** It must also estimate costs for external tool calls (e.g., API fees for searching a premium database).

### 3.3 The Arbitrageur (The Optimizer)

**Concept:** A recommendation engine that suggests cheaper alternatives.

*   **Model Routing:** If cortex requests "GPT-4" for a simple task, the Arbitrageur can intercept and suggest "GPT-4o-mini" if the difficulty score is low.
*   **Quantization Suggestions:** It should be able to recommend lower-fidelity reasoning paths if the budget is tight (e.g., "Skip the 3-round debate, do a single-shot query instead").

### 3.4 The VOC Engine (The Stopping Mechanism)

**Concept:** The "Stop Button" logic based on **Value of Computation**.

*   **Diminishing Returns Check:** During a multi-step cortex chain or a council debate, the VOC Engine analyzes the delta between steps. If Round 3's answer is 99% similar to Round 2's answer, the VOC Engine triggers a `StopIteration` signal. The marginal utility of Round 4 is effectively zero.
*   **Opportunity Cost:** It calculates whether resources are better saved for a future step in the workflow rather than burned on the current low-priority clarification.

## Integration Requirements (The Ecosystem)

*   **Pre-Flight Check (Hook for coreason-cortex):**
    *   Before cortex spins up a System 2 thread, it sends the prompt + selected model to economist. The Economist calculates the projected cost. If > Budget, it throws a **"Resource Rejection."** cortex must then catch this and try a cheaper strategy (System 1).
*   **Consensus Cap (Hook for coreason-council):**
    *   The Council asks: "Can I run 5 agents for 3 rounds?"
    *   The Economist replies: "No, you only have budget for 3 agents for 1 round." The Council must dynamically resize its topology to fit.
*   **Ledger Updates (Hook for coreason-veritas):**
    *   After every actual API call, the true cost must be logged. The Economist reconciles the "Projected" vs. "Actual" spend to improve future estimates (Self-Calibrating Pricer).

## Observability Requirements

To ensure the user knows *why* the agent stopped or chose a cheaper model, the economics must be transparent.

*   **EconomicTrace Object:** Every transaction must yield a log containing:
    *   **The Quote:** Estimated cost vs. Actual cost.
    *   **The Currency:** Token counts (In/Out) and Latency (ms).
    *   **The Decision:** Approved, Rejected, or Modified (Arbitrage).
    *   **The VOC Score:** Why the thinking stopped (e.g., "Delta < Threshold").
*   **Dashboard Metric:** The maco UI must be able to display a "Cost per Insight" metric derived from these traces.
