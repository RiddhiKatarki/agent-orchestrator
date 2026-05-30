"""Checkpoint — persist and restore blackboard state for resumable pipelines."""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.core.blackboard import Blackboard


class CheckpointStore(ABC):
    """
    Abstract base for checkpoint backends.

    A checkpoint records:
      - run_id        : stable identifier for a pipeline run
      - node_index    : the index of the *next* node to execute (i.e. how far we got)
      - board snapshot: full blackboard state at that point
      - timestamp     : when the checkpoint was written
    """

    @abstractmethod
    def save(self, run_id: str, node_index: int, board: "Blackboard") -> None:
        """Persist the current board state and node position."""

    @abstractmethod
    def load(self, run_id: str) -> "tuple[int, dict[str, Any]] | None":
        """
        Return (node_index, board_snapshot) for the given run_id, or None if
        no checkpoint exists.
        """

    @abstractmethod
    def delete(self, run_id: str) -> None:
        """Remove the checkpoint (call after a successful completed run)."""

    @abstractmethod
    def list_runs(self) -> list[dict[str, Any]]:
        """Return metadata for all stored checkpoints."""


class FileCheckpointStore(CheckpointStore):
    """
    Stores each checkpoint as a JSON file under `directory/<run_id>.json`.

    The file is written atomically (write to a temp file, then rename) so a
    crash mid-write cannot corrupt an existing checkpoint.

    Usage
    -----
    store = FileCheckpointStore(".checkpoints")

    board = pipeline.run(
        initial={"input": "..."},
        run_id="my-run-001",
        checkpoint_store=store,
    )

    # Later, to resume after a crash:
    board = pipeline.run(
        run_id="my-run-001",
        checkpoint_store=store,
        resume=True,
    )
    """

    def __init__(self, directory: str = ".checkpoints") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        safe = run_id.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe}.json"

    def save(self, run_id: str, node_index: int, board: "Blackboard") -> None:
        data = {
            "run_id": run_id,
            "node_index": node_index,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "board": _serialise(board.snapshot()),
        }
        path = self._path(run_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        os.replace(tmp, path)   # atomic on POSIX

    def load(self, run_id: str) -> "tuple[int, dict[str, Any]] | None":
        path = self._path(run_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data["node_index"], data["board"]

    def delete(self, run_id: str) -> None:
        path = self._path(run_id)
        if path.exists():
            path.unlink()

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for p in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                runs.append({
                    "run_id": data["run_id"],
                    "node_index": data["node_index"],
                    "timestamp": data["timestamp"],
                    "file": str(p),
                })
            except Exception:
                pass
        return runs


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """
    Recursively make an object JSON-serialisable.

    Pydantic models → dict, sets → list, everything else → str fallback.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, set):
        return [_serialise(i) for i in sorted(obj, key=str)]
    # Pydantic BaseModel
    if hasattr(obj, "model_dump"):
        return _serialise(obj.model_dump())
    # dataclass
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        return _serialise(dataclasses.asdict(obj))
    return str(obj)
