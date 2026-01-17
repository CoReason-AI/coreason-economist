# coreason-economist

**Is it worth thinking?**

The Cognitive "CFO" and Optimization Engine for the CoReason platform.

[![CI](https://github.com/CoReason-AI/coreason_economist/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_economist/actions/workflows/ci.yml)

## Overview

`coreason-economist` acts as the central banking authority for your AI agents, preventing "Cognitive Sprawl" by enforcing Return on Investment (ROI) on every thought. It manages budgets across three currencies: **Financial**, **Latency**, and **Token Volume**.

## Documentation

Full documentation is available in the `docs/` directory or at the [documentation site](https://coreason-ai.github.io/coreason_economist/).

*   [Home](docs/index.md)
*   [Components](docs/components.md)
*   [Usage](docs/usage.md)

## Getting Started

### Prerequisites

- Python 3.12+
- Poetry

### Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/CoReason-AI/coreason_economist.git
    cd coreason_economist
    ```
2.  Install dependencies:
    ```sh
    poetry install
    ```

### Usage

Run the linter:
```sh
poetry run pre-commit run --all-files
```

Run the tests:
```sh
poetry run pytest
```
