# coreason-economist

**Is it worth thinking?**

The Cognitive "CFO" and Optimization Engine for the CoReason platform.

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![Python Versions](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/coreason-economist/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/CoReason-AI/coreason_economist/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason_economist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CoReason-AI/coreason_economist/graph/badge.svg?token=placeholder)](https://codecov.io/gh/CoReason-AI/coreason_economist)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://coreason-ai.github.io/coreason_economist/)

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
