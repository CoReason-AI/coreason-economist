from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.database import BudgetAccount, get_db
from coreason_economist.models import Budget, RequestPayload
from coreason_economist.pricer import Pricer
from coreason_economist.server import app, get_user_context
from coreason_identity.models import UserContext
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

# --- Database Fixtures ---


@pytest.fixture  # type: ignore[misc]
def mock_session() -> AsyncMock:
    # Session is an AsyncMock
    session = AsyncMock(spec=AsyncSession)

    # Configure session.begin() to be an async context manager
    # It returns an object that has __aenter__ and __aexit__
    session.begin.return_value.__aenter__.return_value = session
    session.begin.return_value.__aexit__.return_value = None

    # Configure execute to return a result object with scalar_one_or_none
    mock_result = MagicMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)  # Default

    # session.execute is async, so it returns a coroutine that returns result
    session.execute.return_value = mock_result

    return session


@pytest.fixture  # type: ignore[misc]
def client(mock_session: AsyncMock) -> TestClient:
    app.dependency_overrides[get_db] = lambda: mock_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# --- Unit Tests for Arbitrageur ---


def test_arbitrage_premium_user_no_downgrade() -> None:
    pricer = Pricer()  # Uses default rates (gpt-4o is expensive)
    arb = Arbitrageur(pricer=pricer)

    # Request: Easy task, expensive model
    request = RequestPayload(
        model_name="gpt-4o",
        prompt="hello",
        estimated_output_tokens=10,
        difficulty_score=0.2,  # Very easy
    )

    # Premium User
    user = UserContext(user_id="u1", email="u1@example.com", groups=["Premium"])

    # Should NOT recommend alternative (downgrade disabled)
    recommendation = arb.recommend_alternative(request, user_context=user)
    assert recommendation is None


def test_arbitrage_free_user_aggressive_downgrade() -> None:
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer, threshold=0.5)

    # Request: Medium task (0.7), expensive model
    # Normally (0.7 >= 0.5) this would NOT downgrade.
    request = RequestPayload(model_name="gpt-4o", prompt="hello", estimated_output_tokens=10, difficulty_score=0.7)

    # Free User -> Threshold becomes 0.8
    user = UserContext(user_id="u2", email="u2@example.com", groups=["Free"])

    # Should recommend alternative (downgrade triggered because 0.7 < 0.8)
    recommendation = arb.recommend_alternative(request, user_context=user)
    assert recommendation is not None
    assert recommendation.model_name != "gpt-4o"
    # Should suggest something cheaper like gpt-4o-mini or llama
    assert recommendation.model_name in ["gpt-4o-mini", "llama-3.1-70b"]


def test_arbitrage_budget_exceeded_premium_user() -> None:
    # Even Premium users get downgraded if they hit HARD budget limits
    pricer = Pricer()
    arb = Arbitrageur(pricer=pricer)

    # Low budget (enough for cheap models, not for gpt-4o)
    budget = Budget(financial=0.0001)
    request = RequestPayload(
        model_name="gpt-4o", prompt="hello" * 100, estimated_output_tokens=100, max_budget=budget, difficulty_score=0.9
    )

    user = UserContext(user_id="u1", email="u1@example.com", groups=["Premium"])

    recommendation = arb.recommend_alternative(request, user_context=user)
    # Should downgrade or reduce topology because budget is exceeded
    assert recommendation is not None


# --- Integration Tests for Server (with Mocks) ---


@pytest.mark.asyncio  # type: ignore[misc]
async def test_authorize_new_project_sets_owner(mock_session: AsyncMock) -> None:
    # Setup
    # scalar_one_or_none returns None (default in fixture)

    # Context
    user = UserContext(user_id="owner_1", email="owner1@example.com", groups=[])
    app.dependency_overrides[get_user_context] = lambda: user
    app.dependency_overrides[get_db] = lambda: mock_session

    client = TestClient(app)

    response = client.post("/budget/authorize", json={"project_id": "proj_1", "estimated_cost": 0.01})

    assert response.status_code == 200
    assert response.json()["authorized"] is True

    # Verify DB add was called with correct owner_id
    # session.add is synchronous
    args, _ = mock_session.add.call_args
    account = args[0]
    assert isinstance(account, BudgetAccount)
    assert account.project_id == "proj_1"
    assert account.owner_id == "owner_1"


@pytest.mark.asyncio  # type: ignore[misc]
async def test_authorize_existing_project_owner_mismatch(mock_session: AsyncMock) -> None:
    # Setup
    existing_account = BudgetAccount(project_id="proj_1", balance=Decimal("10.0"), owner_id="owner_1")
    # Set return value for this test
    mock_session.execute.return_value.scalar_one_or_none.return_value = existing_account

    # Context: Different user
    user = UserContext(user_id="intruder", email="intruder@example.com", groups=[])
    app.dependency_overrides[get_user_context] = lambda: user
    app.dependency_overrides[get_db] = lambda: mock_session

    client = TestClient(app)

    response = client.post("/budget/authorize", json={"project_id": "proj_1", "estimated_cost": 0.01})

    assert response.status_code == 403
    assert "not own" in response.json()["detail"]


@pytest.mark.asyncio  # type: ignore[misc]
async def test_authorize_existing_project_admin_override(mock_session: AsyncMock) -> None:
    # Setup
    existing_account = BudgetAccount(project_id="proj_1", balance=Decimal("10.0"), owner_id="owner_1")
    mock_session.execute.return_value.scalar_one_or_none.return_value = existing_account

    # Context: Admin user (different ID but has Admin group)
    user = UserContext(user_id="admin_user", email="admin@example.com", groups=["Admin"])
    app.dependency_overrides[get_user_context] = lambda: user
    app.dependency_overrides[get_db] = lambda: mock_session

    client = TestClient(app)

    response = client.post("/budget/authorize", json={"project_id": "proj_1", "estimated_cost": 0.01})

    assert response.status_code == 200
    assert response.json()["authorized"] is True


def test_get_user_context_raises_401_if_not_overridden(mock_session: AsyncMock) -> None:
    # Verify default dependency raises 401
    app.dependency_overrides.clear()
    # Ensure get_db is still overridden to avoid DB connection issues
    app.dependency_overrides[get_db] = lambda: mock_session
    with TestClient(app) as client:
        response = client.post("/budget/authorize", json={"project_id": "p", "estimated_cost": 1})
        assert response.status_code == 401


@pytest.mark.asyncio  # type: ignore[misc]
async def test_commit_budget_owner_mismatch(mock_session: AsyncMock) -> None:
    # Setup
    existing_account = BudgetAccount(project_id="proj_1", balance=Decimal("10.0"), owner_id="owner_1")
    mock_session.execute.return_value.scalar_one_or_none.return_value = existing_account

    user = UserContext(user_id="intruder", email="intruder@example.com", groups=[])
    app.dependency_overrides[get_user_context] = lambda: user
    app.dependency_overrides[get_db] = lambda: mock_session

    client = TestClient(app)
    response = client.post("/budget/commit", json={"project_id": "proj_1", "estimated_cost": 1.0, "actual_cost": 0.5})
    assert response.status_code == 403
