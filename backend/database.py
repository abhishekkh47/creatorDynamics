"""
Database setup.

Reads DATABASE_URL from .env (via python-dotenv).
SQLite for local testing, Postgres for production — swap is one line in .env.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Load .env from the backend directory
load_dotenv(Path(__file__).parent / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/predictions.db")

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
