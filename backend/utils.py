"""
Shared primitive utilities.

No imports from application modules — keeps this importable by anyone
(db_models, routers, serializers) without circular dependency risk.
"""

from datetime import datetime, timezone
from typing import Optional


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fmt(dt: Optional[datetime]) -> Optional[str]:
    """ISO-8601 string for a datetime, or None if the datetime is None."""
    return dt.isoformat() if dt else None
