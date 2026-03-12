"""
Alembic environment configuration.

Key points:
- DATABASE_URL is loaded from backend/.env (never hardcoded here)
- target_metadata points to our ORM Base so Alembic can autogenerate migrations
- We create the engine directly (not via configparser) to safely handle
  special characters in passwords (e.g. @ encoded as %40)
"""

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

from alembic import context

# Load .env so DATABASE_URL is available
load_dotenv(Path(__file__).parent.parent / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Add it to backend/.env")
    sys.exit(1)

# Alembic Config object — gives access to alembic.ini values
config = context.config

# Set up Python logging from alembic.ini [loggers] section
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import our Base and all models so Alembic can see the full schema.
# Every model file that defines tables must be imported here.
from database import Base  # noqa: E402
import db_models  # noqa: E402, F401  — registers Account, Post, FeatureStore, Prediction

target_metadata = Base.metadata


# ---------------------------------------------------------------------------
# Offline mode  — generate a .sql script without connecting to the DB
# ---------------------------------------------------------------------------

def run_migrations_offline() -> None:
    """
    Render migrations as SQL to stdout instead of running them.
    Useful for review before applying.

    Usage:  alembic upgrade head --sql > migration.sql
    """
    context.configure(
        url=DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ---------------------------------------------------------------------------
# Online mode  — apply migrations directly to the live database
# ---------------------------------------------------------------------------

def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    # Create engine directly from the env var — bypasses configparser so
    # special characters in passwords (%40 etc.) are handled correctly.
    connectable = create_engine(DATABASE_URL, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,           # detect column type changes
            compare_server_default=True, # detect default value changes
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
