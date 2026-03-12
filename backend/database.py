"""
Database setup — SQLite (dev) with a clean path to Postgres (production).

To switch to Postgres, change DATABASE_URL to:
    postgresql://user:password@host:5432/creatorDynamix

Everything else stays the same — SQLAlchemy abstracts the driver.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

# SQLite file lives in backend/data/ — excluded from git via .gitignore
_DATA_DIR = Path(__file__).parent / "data"
_DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{_DATA_DIR / 'predictions.db'}",
)

# check_same_thread=False is SQLite-specific — safe for FastAPI's async handlers
# when using synchronous sessions (which we do here)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# Base class for all ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Dependency for FastAPI route injection
# ---------------------------------------------------------------------------

def get_db():
    """
    Yield a database session for the duration of a request, then close it.
    Usage in routes:
        def my_route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
