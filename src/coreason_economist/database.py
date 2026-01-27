from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator

from pydantic_settings import BaseSettings
from sqlalchemy import DateTime, Numeric, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Settings(BaseSettings):  # type: ignore[misc]
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/dbname"  # Default or override via env
    INITIAL_BUDGET_TIER: float = 5.0


settings = Settings()

# Setup Engine
engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):  # type: ignore[misc]
    pass


class BudgetAccount(Base):
    __tablename__ = "budget_accounts"

    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    owner_id: Mapped[str] = mapped_column(String, nullable=True)
    balance: Mapped[Decimal] = mapped_column(Numeric(10, 4))
    currency: Mapped[str] = mapped_column(String, default="USD")
    last_updated: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
