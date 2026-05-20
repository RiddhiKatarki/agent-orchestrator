"""Blackboard — thread-safe typed key-value store shared across all agents."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar

T = TypeVar("T")


@dataclass
class BlackboardEvent:
    key: str
    value: Any
    agent: str
    timestamp: float = field(default_factory=time.time)


class Blackboard:
    """
    The shared memory space all agents read from and write to.

    Agents declare which keys they `reads` and `writes`. This lets the executor
    automatically detect dependencies and decide what can run in parallel.
    """

    def __init__(self, initial: dict[str, Any] | None = None) -> None:
        self._store: dict[str, Any] = {}
        self._history: list[BlackboardEvent] = []
        self._lock = threading.Lock()

        if initial:
            for k, v in initial.items():
                self.set(k, v, agent="init")

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any, agent: str = "system") -> None:
        with self._lock:
            self._store[key] = value
            self._history.append(
                BlackboardEvent(key=key, value=value, agent=agent, timestamp=time.time())
            )

    def get(self, key: str, type_: Type[T] | None = None, default: Any = None) -> T:
        with self._lock:
            value = self._store.get(key, default)
        if type_ is not None and value is not None and not isinstance(value, type_):
            raise TypeError(
                f"Blackboard key '{key}' expected {type_.__name__}, "
                f"got {type(value).__name__}"
            )
        return value  # type: ignore[return-value]

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._store

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the current store."""
        with self._lock:
            return dict(self._store)

    def history(self) -> list[BlackboardEvent]:
        with self._lock:
            return list(self._history)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def __repr__(self) -> str:
        with self._lock:
            return f"Blackboard({list(self._store.keys())})"
