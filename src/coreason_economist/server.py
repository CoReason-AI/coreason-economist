import uuid
from decimal import Decimal
from typing import Annotated, Any, Dict

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coreason_economist.database import BudgetAccount, get_db, settings
from coreason_economist.models import (
    AuthorizeRequest,
    AuthorizeResponse,
    CommitRequest,
    VocAnalyzeRequest,
    VocAnalyzeResponse,
)
from coreason_economist.voc import VOCEngine
from coreason_identity.models import UserContext

app = FastAPI(title="Cost Control Microservice")

# Annotated dependency for Ruff B008
SessionDep = Annotated[AsyncSession, Depends(get_db)]


async def get_user_context() -> UserContext:
    """
    Dependency to get the current authenticated user context.
    This should be overridden by the application authentication layer.
    """
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


UserContextDep = Annotated[UserContext, Depends(get_user_context)]


@app.post("/budget/authorize", response_model=AuthorizeResponse)  # type: ignore[misc]
async def authorize_budget(
    request: AuthorizeRequest, session: SessionDep, user_context: UserContextDep
) -> AuthorizeResponse:
    async with session.begin():
        # Row Locking
        stmt = select(BudgetAccount).where(BudgetAccount.project_id == request.project_id).with_for_update()
        result = await session.execute(stmt)
        account = result.scalar_one_or_none()

        if not account:
            # Auto-provision
            initial = settings.INITIAL_BUDGET_TIER
            account = BudgetAccount(
                project_id=request.project_id, balance=Decimal(initial), owner_id=user_context.user_id
            )
            session.add(account)
            await session.flush()  # Ensure it's there for logic
        else:
            # Ownership Check
            is_admin = "Admin" in (user_context.groups or [])
            if account.owner_id != user_context.user_id and not is_admin:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN, detail="You do not own this project budget."
                )

        cost = Decimal(str(request.estimated_cost))
        if account.balance < cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient funds. Balance: {account.balance}, Required: {cost}",
            )

        account.balance -= cost

    return AuthorizeResponse(authorized=True, transaction_id=f"tx_{uuid.uuid4().hex}")


@app.post("/budget/commit")  # type: ignore[misc]
async def commit_budget(
    request: CommitRequest, session: SessionDep, user_context: UserContextDep
) -> Dict[str, Any]:
    async with session.begin():
        stmt = select(BudgetAccount).where(BudgetAccount.project_id == request.project_id).with_for_update()
        result = await session.execute(stmt)
        account = result.scalar_one_or_none()

        if not account:
            raise HTTPException(status_code=404, detail="Account not found")

        # Ownership Check
        is_admin = "Admin" in (user_context.groups or [])
        if account.owner_id != user_context.user_id and not is_admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not own this project budget.")

        estimated = Decimal(str(request.estimated_cost))
        actual = Decimal(str(request.actual_cost))
        refund = estimated - actual

        account.balance += refund

    return {"status": "committed", "refund": float(refund)}


@app.post("/voc/analyze", response_model=VocAnalyzeResponse)  # type: ignore[misc]
async def analyze_voc(request: VocAnalyzeRequest) -> VocAnalyzeResponse:
    # Instantiate logic components
    voc_engine = VOCEngine()

    # Run the calculation
    should_execute, max_allowable_cost = voc_engine.assess_viability(
        task_complexity=request.task_complexity, current_uncertainty=request.current_uncertainty
    )

    return VocAnalyzeResponse(should_execute=should_execute, max_allowable_cost=max_allowable_cost)
