"""Pipeline node types — the building blocks of a workflow DAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from orchestrator.core.agent import Agent
    from orchestrator.core.blackboard import Blackboard


# ------------------------------------------------------------------
# Node types
# ------------------------------------------------------------------

@dataclass
class SequentialNode:
    """Run a single agent, then continue."""
    agent: "Agent"


@dataclass
class ParallelNode:
    """Run multiple agents concurrently, then continue when all finish."""
    agents: list["Agent"]


@dataclass
class LoopNode:
    """
    Repeatedly run a sequence of agents until `until(board)` returns True
    or `max_iterations` is hit.

    On each iteration the executor writes `_loop_iteration` (int, 0-based)
    to the board so agents inside the loop can inspect which round they're in.

    If `max_reached` is "warn" (default), execution continues after the loop.
    If "raise", a RuntimeError is raised.
    """
    agents: list["Agent"]
    until: Callable[["Blackboard"], bool]
    max_iterations: int = 3
    max_reached: str = "warn"  # "warn" | "raise"


@dataclass
class BranchNode:
    """
    Conditionally route to one of two agents (or sub-pipelines) based on
    the current blackboard state.
    """
    condition: Callable[["Blackboard"], bool]
    if_true: "Agent | list[Node]"
    if_false: "Agent | list[Node]"


@dataclass
class FanOutNode:
    """
    Clone the blackboard N times with different payloads, run one agent
    per clone in parallel, then merge all results back with a FanIn agent.

    `inputs`  — list of dicts; each dict is set on a board clone before the
                worker agent runs. Typically sets a single differentiating key.
    `output_key` — the board key each worker writes; collected into a list
                   and written as `gather_key` for the FanIn agent.
    `gather_key`  — key written to the main board containing list of results.
    """
    worker: "Agent"
    inputs: list[dict]
    output_key: str
    gather_key: str
    fanin: "Agent | None" = None


# Union type for type hints elsewhere
Node = SequentialNode | ParallelNode | LoopNode | BranchNode | FanOutNode
