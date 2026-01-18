Architectural Role: The Central Cognitive Execution Engine

## ---

**1. Executive Summary**

coreason-cortex is the execution engine of the CoReason platform. It acts as the "Brain" that orchestrates intelligence. Its primary mandate is to implement a **Neuro-Symbolic Architecture** that dynamically switches between **System 1** (Fast, Heuristic, Reflexive) and **System 2** (Slow, Deliberative, Logical) modes of cognition.

Unlike standard agent frameworks that default to "always-on" reasoning, coreason-cortex treats cognition as a scarce resource. It prioritizes deterministic execution (Code/FSM) first, only escalating to probabilistic reasoning (LLM) when novelty or ambiguity is detected. It aims for a "Glass Box" implementation where every thought is traceable, auditable, and interruptible.

## **2. Functional Philosophy**

The agent must implement the **Kahneman Control Loop**:

1. **Default to System 1:** Always attempt to solve the input using low-latency, low-cost heuristics or defined rules.
2. **Monitor for Surprise:** Continuously evaluate the "Confidence" or "Perplexity" of the System 1 output.
3. **Escalate to System 2:** If confidence drops below a threshold, pause execution, instantiate a deliberative reasoning chain (CoT), and override the reflex.
4. **Consolidate (Crystallize):** Successful System 2 reasoning should update the System 1 state (learning), converting high-cost reasoning into low-cost reflexes for future use.

## ---

**3. Core Functional Requirements (Component Level)**

### **3.1 The Cognitive Controller (The Orchestrator)**

**Concept:** A central entity that manages the lifecycle of a single "Thought Cycle."

* **Mode Arbitration:** It must decide *before* execution whether to route the stimulus to the Reflex Engine (S1) or the Reasoning Engine (S2). It must support "Forced Mode" (forcing S2 for high-stakes tasks) and "Auto Mode" (dynamic switching based on confidence).
* **State Management:** It maintains "Working Memory" across the lifecycle. Data generated in S1 must be accessible to S2 if an escalation occurs.
* **Output Normalization:** Whether the answer comes from a regex (S1) or a generic LLM (S2), the output format returned to the parent application must be identical.

### **3.2 System 1: The Reflex Engine**

**Concept:** A deterministic runtime for "Fast Thinking."

* **Mechanism:** Executes defined workflows, heuristics, and lookup tables (Finite State Machine).
* **Capabilities:**
  * **Pattern Matching:** Instant classification of inputs based on keywords or regex.
  * **Retrieval:** Fetching cached answers or static documents without generating new text.
* **Fallibility Signal:** It must return a **Confidence Score** or **Failure Signal**. It must *not* hallucinate; if the input doesn't fit known patterns, it must admit ignorance to trigger System 2.

### **3.3 System 2: The Reasoning Engine**

**Concept:** A probabilistic runtime for "Slow Thinking."

* **Mechanism:** Manages Large Language Model interactions, context windows, and generation parameters.
* **Capabilities:**
  * **Chain of Thought (CoT):** Supports multi-step reasoning prompts where the model "shows its work" before answering.
  * **Tool Usage:** Can pause generation to request external data (via coreason-mcp).
  * **Counterfactual Simulation:** Capable of generating multiple potential answers and evaluating them (drafting).
* **Streaming Evaluation:** Must support streaming tokens to allow for real-time interception by the **InterruptHandler**.

### **3.4 The Metacognitive Bridge (The Handoff)**

**Concept:** The logic that governs the transition from S1 to S2.

* **Trigger Conditions:** Supports defining custom triggers for escalation (e.g., Semantic Entropy, Keyword Detection, Structural Failure).
* **Context Promotion:** When escalating, the full trace of System 1's attempt (and *why* it failed) must be injected into the System 2 context window so the reasoner doesn't repeat the mistake.

### **3.5 The Crystallizer (The Learning Loop)**

**Concept:** A post-execution process that analyzes successful System 2 traces to generate new System 1 artifacts ("Slow-to-Fast" learning).

* **Trace Analysis:** After a successful System 2 resolution, it evaluates if the logic is deterministic enough to be "downgraded" to System 1.
* **Artifact Generation:** It automatically proposes a new **Regex Pattern**, **Lookup Key**, or **Cached Response** to be stored in the System 1 Reflex Engine for future requests.

### **3.6 The InterruptHandler (The Kill Switch)**

**Concept:** Asynchronous signal handling for safety and budgeting.

* **Signal Listening:** Must listen for StopIteration signals from the **Economist** (Budget Depleted) or the **User** (Cancellation).
* **Graceful Dismount:** Upon receiving a signal, it must halt the Reasoning Engine immediately but safely, logging the partial thought trace before killing the thread.

### **3.7 The EpisodicRetriever (The Bias Check)**

**Concept:** A "Pre-Computation" check before invoking System 2 to enforce consistency.

* **Retrieval:** Before instantiating the heavy System 2, the Controller queries a vector store for **"Similar Past Traces."**
* **Context Injection:** If a similar past successful trace is found, it is injected into the System 2 context window as a **Few-Shot Example** to align current reasoning with past behavior.

## ---

**4. Integration Requirements (The Ecosystem)**

This package must define abstract interfaces (hooks) for the following interactions:

* **Consensus (Hook for coreason-council):** The Reasoning Engine must support a "Consultation" mode where it halts execution to poll multiple external models (Mixture of Agents) for agreement before finalizing an S2 decision.
* **Metareasoning (Hook for coreason-economist):** Before transitioning from S1 to S2, the Controller must ask a "Budget Authority" if the compute cost is permissible. If denied, it returns a "Resource Exhausted" fallback.
* **Governance (Hook for coreason-constitution):** The Output Normalizer must support a "Veto" loop. Before returning a final answer, the payload is submitted to a constitutional check. If rejected, the Controller triggers a "Revision Loop."

## ---

**5. User Stories (Behavioral Expectations)**

### **Story A: The "Happy Path" (System 1 Success)**

Trigger: User asks "What is the study ID for the Oncology trial?"
Execution: Controller sees a regex match. Routes to Reflex Engine (S1). S1 queries a structured cache. Finds "NCT12345".
Result: Answer returned in 200ms. Cost: $0.00. System 2 is never instantiated.

### **Story B: The "Escalation & Learning" (S1 Fail -> S2 -> Crystallize)**

Trigger: User asks "Interpret the adverse event trend."
Execution: S1 fails (Capabilities Mismatch). EpisodicRetriever finds no past context. Economist approves budget. Reasoning Engine (S2) activates, analyzes data, and explains the trend.
Post-Processing: The Crystallizer analyzes the result, determines this query is common, and creates a cached "Summary Artifact" for the next 24 hours.

### **Story C: The "Intervention" (Constitutional Block)**

Trigger: System 2 generates a response suggesting a protocol deviation.
Execution: The Governance Hook scans the draft and detects a GxP violation. It sends a "Veto."
Correction: Controller re-prompts System 2 with the error. System 2 self-corrects.

## ---

**6. Observability Requirements**

To support the "Glass Box" philosophy, the coreason-cortex must treat the reasoning trace as a first-class citizen.

* **CognitiveTrace Object:** Every execution must yield a serializable object containing:
  * Path taken (S1 vs S2).
  * Triggers for switching.
  * Raw prompts/completions.
  * Latency breakdown.
  * Episodic context used.
  * Crystallization suggestions (if any).
* **Integration:** This object must be structured for consumption by coreason-veritas.

**7.  Implementation:** Please research how others have implemented. Understand the best practices. The package should make choices typically based on a catalog of **heuristics** (Regex matches), **lightweight classifiers** (BERT/embedding similarity), or **Confidence Thresholds** (perplexity checks).
