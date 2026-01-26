from decimal import Decimal
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from coreason_economist.database import BudgetAccount, get_db
from coreason_economist.server import app
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError


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


def test_commit_budget_not_found(client: TestClient, mock_session: AsyncMock) -> None:
    # Mock account not found
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    response = client.post("/budget/commit", json={"project_id": "p1", "estimated_cost": 0.5, "actual_cost": 0.4})

    assert response.status_code == 404
    assert "Account not found" in response.json()["detail"]


def test_voc_analyze(client: TestClient) -> None:
    response = client.post("/voc/analyze", json={"task_complexity": 0.5, "current_uncertainty": 0.2})
    assert response.status_code == 200
    data = response.json()
    assert data["should_execute"] is True
    assert data["max_allowable_cost"] > 0


@pytest.mark.asyncio  # type: ignore[misc]
async def test_get_db() -> None:
    # Mock AsyncSessionLocal to return a mock session
    mock_session_local = MagicMock()
    mock_session_instance = AsyncMock()

    # Setup the context manager of AsyncSessionLocal
    mock_session_local.return_value.__aenter__.return_value = mock_session_instance

    # Patch the AsyncSessionLocal in coreason_economist.database
    with patch("coreason_economist.database.AsyncSessionLocal", mock_session_local):
        gen = get_db()
        session = await anext(gen)
        assert session == mock_session_instance

        # Verify closure (via generator exit)
        try:
            await anext(gen)
        except StopAsyncIteration:
            pass

        # Verify __aexit__ was called (session closed)
        assert mock_session_local.return_value.__aexit__.called


# Edge Case Tests


def test_authorize_exact_balance(client: TestClient, mock_session: AsyncMock) -> None:
    # Verify we can spend the last penny
    mock_account = BudgetAccount(project_id="p1", balance=Decimal("1.0"))
    mock_session.execute.return_value.scalar_one_or_none.return_value = mock_account

    response = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 1.0})

    assert response.status_code == 200
    assert mock_account.balance == Decimal("0.0")


def test_authorize_validation_error(client: TestClient) -> None:
    # Cost <= 0 should be rejected by Pydantic model
    response = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 0.0})
    assert response.status_code == 422

    response = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": -1.0})
    assert response.status_code == 422


def test_commit_calculations_weird_values(client: TestClient, mock_session: AsyncMock) -> None:
    # Test refund logic with unusual values
    mock_account = BudgetAccount(project_id="p1", balance=Decimal("10.0"))
    mock_session.execute.return_value.scalar_one_or_none.return_value = mock_account

    # Actual cost higher than estimated (negative refund)
    # This implies we under-reserved. Logic should subtract the difference (add negative refund).
    # Est: 1.0, Actual: 2.0 -> Refund: -1.0. Balance: 10 + (-1) = 9.
    response = client.post("/budget/commit", json={"project_id": "p1", "estimated_cost": 1.0, "actual_cost": 2.0})

    assert response.status_code == 200
    assert response.json()["refund"] == -1.0
    assert mock_account.balance == Decimal("9.0")


def test_database_error_handling(client: TestClient, mock_session: AsyncMock) -> None:
    # Simulate DB error during execution
    mock_session.execute.side_effect = SQLAlchemyError("DB Boom")

    with pytest.raises(SQLAlchemyError):
        # We expect the exception to bubble up or be handled by FastAPI's default handler (500)
        # Since we haven't added a custom exception handler in server.py, TestClient might see the raw exception
        # or a 500 depending on configuration. TestClient usually raises the exception.
        client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 1.0})


def test_complex_budget_lifecycle(client: TestClient, mock_session: AsyncMock) -> None:
    """
    Simulates a sequential workflow:
    1. Start with 10.0
    2. Reserve 5.0 (Bal -> 5.0)
    3. Commit with Actual 3.0 (Refund 2.0 -> Bal 7.0)
    4. Reserve 6.0 (Bal -> 1.0)
    5. Try Reserve 2.0 (Fail)
    """
    mock_account = BudgetAccount(project_id="p1", balance=Decimal("10.0"))
    mock_session.execute.return_value.scalar_one_or_none.return_value = mock_account

    # Step 2: Reserve 5.0
    resp1 = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 5.0})
    assert resp1.status_code == 200
    assert mock_account.balance == Decimal("5.0")

    # Step 3: Commit (Est 5, Act 3 -> Refund 2)
    resp2 = client.post("/budget/commit", json={"project_id": "p1", "estimated_cost": 5.0, "actual_cost": 3.0})
    assert resp2.status_code == 200
    assert resp2.json()["refund"] == 2.0
    assert mock_account.balance == Decimal("7.0")

    # Step 4: Reserve 6.0
    resp3 = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 6.0})
    assert resp3.status_code == 200
    assert mock_account.balance == Decimal("1.0")

    # Step 5: Try Reserve 2.0 (Fail)
    resp4 = client.post("/budget/authorize", json={"project_id": "p1", "estimated_cost": 2.0})
    assert resp4.status_code == 402
    assert "Insufficient funds" in resp4.json()["detail"]
