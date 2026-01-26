import uuid
from decimal import Decimal
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from coreason_economist.database import get_db, BudgetAccount, settings
from coreason_economist.models import (
    AuthorizeRequest, AuthorizeResponse,
    CommitRequest,
    VocAnalyzeRequest, VocAnalyzeResponse
)
from coreason_economist.pricer import Pricer
from coreason_economist.arbitrageur import Arbitrageur
from coreason_economist.voc import VOCEngine

app = FastAPI(title="Cost Control Microservice")

@app.post("/budget/authorize", response_model=AuthorizeResponse)
async def authorize_budget(
    request: AuthorizeRequest,
    session: AsyncSession = Depends(get_db)
):
    async with session.begin():
        # Row Locking
        stmt = select(BudgetAccount).where(BudgetAccount.project_id == request.project_id).with_for_update()
        result = await session.execute(stmt)
        account = result.scalar_one_or_none()

        if not account:
            # Auto-provision
            initial = settings.INITIAL_BUDGET_TIER
            account = BudgetAccount(project_id=request.project_id, balance=Decimal(initial))
            session.add(account)
            await session.flush() # Ensure it's there for logic

        cost = Decimal(str(request.estimated_cost))
        if account.balance < cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=f"Insufficient funds. Balance: {account.balance}, Required: {cost}"
            )

        account.balance -= cost

    return AuthorizeResponse(
        authorized=True,
        transaction_id=f"tx_{uuid.uuid4().hex}"
    )

@app.post("/budget/commit")
async def commit_budget(
    request: CommitRequest,
    session: AsyncSession = Depends(get_db)
):
    async with session.begin():
        stmt = select(BudgetAccount).where(BudgetAccount.project_id == request.project_id).with_for_update()
        result = await session.execute(stmt)
        account = result.scalar_one_or_none()

        if not account:
            raise HTTPException(status_code=404, detail="Account not found")

        estimated = Decimal(str(request.estimated_cost))
        actual = Decimal(str(request.actual_cost))
        refund = estimated - actual

        account.balance += refund

    return {"status": "committed", "refund": float(refund)}

@app.post("/voc/analyze", response_model=VocAnalyzeResponse)
async def analyze_voc(request: VocAnalyzeRequest):
    # Instantiate logic components
    voc_engine = VOCEngine()

    # Run the calculation
    should_execute, max_allowable_cost = voc_engine.assess_viability(
        task_complexity=request.task_complexity,
        current_uncertainty=request.current_uncertainty
    )

    return VocAnalyzeResponse(should_execute=should_execute, max_allowable_cost=max_allowable_cost)
