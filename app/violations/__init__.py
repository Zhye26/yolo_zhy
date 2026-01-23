"""Violations module for deduplication and event management."""
from .dedup_fsm import ViolationDeduper, ViolationStateEntry

__all__ = [
    "ViolationDeduper",
    "ViolationStateEntry",
]
