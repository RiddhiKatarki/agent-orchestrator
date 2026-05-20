"""Pipeline — fluent DSL for composing agent workflows."""

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

import anyio

from orchestrator.core.blackboard import Blackboard
from orchestrator.core.executor import Executor
from orchestrator.patterns.nodes import (
    BranchNode,
    FanOutNode,
    LoopNode,
    Node,
    ParallelNode,
    SequentialNode,
)

if TYPE_CHECKING:
    from orchestrator.core.agent import Agent
    from orchestrator.core.tracer import Tracer


class Pipeline:
    """
    Fluent builder for agent workflows.

    Usage
    -----
    pipeline = (
        Pipeline(name="My Workflow")
        .then(agent_a)
        .parallel(agent_b, agent_c)
        .loop(
            agents=[agent_d, agent_e],
            until=lambda b: b.get("done") is True,
            max_iterations=3,
        )
        .branch(
            condition=lambda b: b.get("needs_review") is True,
            if_true=review_agent,
            if_false=skip_agent,
        )
    )

    board = pipeline.run({"input": "my idea"})
    """

    def __init__(self, name: str = "Pipeline") -> None:
        self.name = name
        self._nodes: list[Node] = []

    # ------------------------------------------------------------------
    # Fluent builders
    # ------------------------------------------------------------------

    def then(self, agent: "Agent") -> "Pipeline":
        """Run a single agent sequentially."""
        self._nodes.append(SequentialNode(agent=agent))
        return self

    def parallel(self, *agents: "Agent") -> "Pipeline":
        """Run multiple agents concurrently, continue when all finish."""
        self._nodes.append(ParallelNode(agents=list(agents)))
        return self

    def loop(
        self,
        agents: "list[Agent] | Agent",
        *,
        until: Callable[[Blackboard], bool],
        max_iterations: int = 3,
        max_reached: str = "warn",
    ) -> "Pipeline":
        """
        Repeat a sequence of agents until `until(board)` is True.

        Parameters
        ----------
        agents         : Single agent or list of agents to run each iteration.
        until          : Callable that receives the board; return True to stop.
        max_iterations : Hard cap on iterations.
        max_reached    : "warn" (default) to continue, "raise" to error out.
        """
        if not isinstance(agents, list):
            agents = [agents]
        self._nodes.append(
            LoopNode(
                agents=agents,
                until=until,
                max_iterations=max_iterations,
                max_reached=max_reached,
            )
        )
        return self

    def branch(
        self,
        condition: Callable[[Blackboard], bool],
        *,
        if_true: "Agent | list[Node]",
        if_false: "Agent | list[Node]",
    ) -> "Pipeline":
        """Route execution based on a board condition."""
        self._nodes.append(
            BranchNode(condition=condition, if_true=if_true, if_false=if_false)
        )
        return self

    def fanout(
        self,
        worker: "Agent",
        *,
        inputs: list[dict],
        output_key: str,
        gather_key: str,
        fanin: "Agent | None" = None,
    ) -> "Pipeline":
        """
        Fan a worker agent out across multiple inputs in parallel, then
        optionally run a fanin agent to aggregate results.
        """
        self._nodes.append(
            FanOutNode(
                worker=worker,
                inputs=inputs,
                output_key=output_key,
                gather_key=gather_key,
                fanin=fanin,
            )
        )
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        initial: "str | dict[str, Any] | None" = None,
        *,
        tracer: "Tracer | None" = None,
    ) -> Blackboard:
        """
        Execute the pipeline synchronously.

        Parameters
        ----------
        initial : Seed the blackboard before any agent runs.
                  - str  → written as board["input"]
                  - dict → each key/value written to the board
        tracer  : Optional Tracer to record token usage and timing.

        Returns the final Blackboard state.
        """
        board = Blackboard()

        if isinstance(initial, str):
            board.set("input", initial, agent="user")
        elif isinstance(initial, dict):
            for k, v in initial.items():
                board.set(k, v, agent="user")

        executor = Executor(tracer=tracer)
        anyio.run(executor.run_pipeline, self._nodes, board)
        return board

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, nodes={len(self._nodes)})"
