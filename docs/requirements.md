# Product Requirements Document: coreason-economist

**Domain:** Economic Metareasoning & Resource Optimization
**Scientific Basis:** Rational Metareasoning, Value of Computation (VOC), & Bounded Rationality
**Architectural Role:** The Cognitive "CFO" and Optimization Engine

## 1. Executive Summary

`coreason-economist` is the central banking and resource allocation authority of the CoReason platform. While `coreason-cortex` decides *how* to think, `coreason-economist` decides *if it is worth* thinking.

In an ecosystem where every "thought" (token) has a real-world cost (financial and temporal), this package prevents "Cognitive Sprawl"—the tendency of agents to enter infinite reasoning loops or use overpowered models for trivial tasks. It implements a strict **Return on Investment (ROI)** philosophy, ensuring that the utility of the generated intelligence exceeds the cost of its production.

## 2. Functional Philosophy

The agent must implement the **Resource-Rational Control Loop**:

1.  **Estimation before Execution:** No process (System 2, Council, Tool Call) runs without a pre-computed cost estimate.
2.  **Value of Computation (VOC):** The system must continuously evaluate if the *next* step of reasoning is likely to change the outcome enough to justify its cost.
3.  **Token Arbitrage:** Given a set of requirements, the system should always recommend the cheapest model that satisfies the minimum quality threshold.
4.  **Sunk Cost Awareness:** The system must be willing to terminate a "bad line of thought" immediately, regardless of how much has already been spent.

## 3. Core Functional Requirements (Component Level)

### 3.1 The Budget Authority (The Controller)

**Concept:** The gatekeeper that enforces limits set by the parent application (coreason-maco).

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

## 4. Integration Requirements (The Ecosystem)

*   **Pre-Flight Check (Hook for coreason-cortex):**
    *   Before cortex spins up a System 2 thread, it sends the prompt + selected model to economist. The Economist calculates the projected cost. If > Budget, it throws a **"Resource Rejection."** cortex must then catch this and try a cheaper strategy (System 1).
*   **Consensus Cap (Hook for coreason-council):**
    *   The Council asks: "Can I run 5 agents for 3 rounds?"
    *   The Economist replies: "No, you only have budget for 3 agents for 1 round." The Council must dynamically resize its topology to fit.
*   **Ledger Updates (Hook for coreason-veritas):**
    *   After every actual API call, the true cost must be logged. The Economist reconciles the "Projected" vs. "Actual" spend to improve future estimates (Self-Calibrating Pricer).

## 5. User Stories (Behavioral Expectations)

### Story A: The "Hard Stop" (Budget Enforcement)

**Trigger:** User sets a max cost of $0.05 per run.
**Request:** User asks a complex question requiring coreason-council (MoA).
**Calculation:** The Pricer estimates the MoA session will cost $0.12.
**Action:** The Budget Authority denies the request.
**Fallback:** It returns a recommendation to cortex: "Budget insufficient for Council. Downgrade to single-shot GPT-4o-mini?"

### Story B: The "Early Exit" (VOC Optimization)

**Trigger:** coreason-council is running a 5-round debate.
**Execution:** After Round 2, the VOC Engine compares the transcripts. It detects that the semantic drift between Round 1 and Round 2 is < 2%.
**Action:** The Economist issues a StopIteration signal: "Consensus Achieved. Further debate is waste."
**Result:** The system saves the cost of Rounds 3, 4, and 5.

### Story C: The "Latency Veto" (Time Budgeting)

**Trigger:** A real-time clinical support bot has a strict 3-second SLA.
**Request:** The agent attempts to use a Chain-of-Thought reasoning path.
**Calculation:** The Pricer estimates CoT generation will take 8 seconds based on current API latency stats.
**Action:** The Budget Authority vetoes the CoT strategy.
**Result:** It forces cortex to use a direct Answer-Only strategy to meet the SLA.

## 6. Observability Requirements

To ensure the user knows *why* the agent stopped or chose a cheaper model, the economics must be transparent.

*   **EconomicTrace Object:** Every transaction must yield a log containing:
    *   **The Quote:** Estimated cost vs. Actual cost.
    *   **The Currency:** Token counts (In/Out) and Latency (ms).
    *   **The Decision:** Approved, Rejected, or Modified (Arbitrage).
    *   **The VOC Score:** Why the thinking stopped (e.g., "Delta < Threshold").
*   **Dashboard Metric:** The maco UI must be able to display a "Cost per Insight" metric derived from these traces.
