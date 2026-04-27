"""Serialization helpers for API responses."""

from __future__ import annotations

from dataclasses import is_dataclass
from datetime import datetime


def jsonable(value):
    """Convert simple dataclass/datetime containers into JSON-compatible values."""
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in value.__dict__.items()}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value
