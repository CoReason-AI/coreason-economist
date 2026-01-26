from decimal import Decimal
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreason_economist.database import BudgetAccount, get_db
from coreason_economist.server import app
from fastapi.testclient import TestClient


@pytest.fixture  # type: ignore[misc]
def mock_session() -> AsyncMock:
    session = AsyncMock()

    # session.begin() is a sync method returning an async context manager.
    # We need to ensure session.begin is NOT treated as an async method returning a coroutine.
    session.begin = MagicMock()

    # The return value of begin() must be an async context manager.
    # An AsyncMock instance serves as an async context manager (has async __aenter__ and __aexit__).
    transaction_ctx = AsyncMock()
    session.begin.return_value = transaction_ctx
    transaction_ctx.__aenter__.return_value = transaction_ctx

    # Mock execute result
    session.execute.return_value = MagicMock()

    return session


@pytest.fixture  # type: ignore[misc]
def client(mock_session: AsyncMock) -> TestClient:
    async def override_get_db() -> AsyncGenerator[AsyncMock, None]:
        yield mock_session

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_authorize_budget_success(client: TestClient, mock_session: AsyncMock) -> None:
    # Setup mock return value
    mock_account = BudgetAccount(project_id="p1", balance=Decimal("10.0"))
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_account
    mock_session.execute.return_value = mock_result

    response = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 0.5})

    assert response.status_code == 200
    data = response.json()
    assert data["authorized"] is True
    assert "transaction_id" in data

    # Verify balance deduction
    assert mock_account.balance == Decimal("9.5")


def test_authorize_budget_insufficient_funds(client: TestClient, mock_session: AsyncMock) -> None:
    mock_result = MagicMock()
    mock_account = BudgetAccount(project_id="p1", balance=Decimal("0.1"))
    mock_result.scalar_one_or_none.return_value = mock_account
    mock_session.execute.return_value = mock_result

    response = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 0.5})

    assert response.status_code == 402
    assert "Insufficient funds" in response.json()["detail"]


def test_authorize_budget_auto_provision(client: TestClient, mock_session: AsyncMock) -> None:
    # Mock account not found initially
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    response = client.post("/budget/authorize", json={"project_id": "new_proj", "estimated_cost": 1.0})

    assert response.status_code == 200
    # Verify that add was called
    assert mock_session.add.called
    args, _ = mock_session.add.call_args
    new_account = args[0]
    assert new_account.project_id == "new_proj"
    assert new_account.balance == Decimal("4.0")


def test_commit_budget(client: TestClient, mock_session: AsyncMock) -> None:
    mock_account = BudgetAccount(project_id="p1", balance=Decimal("9.5"))
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_account
    mock_session.execute.return_value = mock_result

    response = client.post("/budget/commit", json={"project_id": "p1", "estimated_cost": 0.5, "actual_cost": 0.4})

    assert response.status_code == 200
    data = response.json()
    assert data["refund"] == 0.1
    assert mock_account.balance == Decimal("9.6")


def test_voc_analyze(client: TestClient) -> None:
    response = client.post("/voc/analyze", json={"task_complexity": 0.5, "current_uncertainty": 0.2})
    assert response.status_code == 200
    data = response.json()
    assert data["should_execute"] is True
    assert data["max_allowable_cost"] > 0
