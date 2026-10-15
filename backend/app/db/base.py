from datetime import datetime, timezone
from sqlalchemy.orm import declarative_base, declared_attr
from sqlalchemy import Column, Integer, DateTime

DeclarativeBase = declarative_base()


class Base(DeclarativeBase):
    __abstract__ = True

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
