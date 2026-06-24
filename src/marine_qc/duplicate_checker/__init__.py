"""Marine Duplicate Checker package."""

from __future__ import annotations

from .duplicates import duplicate_check, flag_duplicates, get_duplicates, remove_duplicates


__all__ = [
    "duplicate_check",
    "flag_duplicates",
    "get_duplicates",
    "remove_duplicates",
]
