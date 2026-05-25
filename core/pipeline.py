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
    from orchestrator.core.checkpoint import CheckpointStore
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
    # Inspection
    # ------------------------------------------------------------------

    def preview(self) -> "Pipeline":
        """
        Print the pipeline DAG to the terminal without running it.
        Returns self so it can be chained: pipeline.preview().run(...)
        """
        from orchestrator.rendering.dag import render
        render(self, tracer=None)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        initial: "str | dict[str, Any] | None" = None,
        *,
        tracer: "Tracer | None" = None,
        show_dag: bool = False,
        run_id: str | None = None,
        checkpoint_store: "CheckpointStore | None" = None,
        resume: bool = False,
    ) -> Blackboard:
        """
        Execute the pipeline synchronously.

        Parameters
        ----------
        initial           : Seed the blackboard before any agent runs.
                            - str  → written as board["input"]
                            - dict → each key/value written to the board
        tracer            : Optional Tracer to record token usage and timing.
        show_dag          : If True, render the pipeline DAG before execution.
        run_id            : Stable identifier for this run. Required when using
                            checkpoint_store.
        checkpoint_store  : A CheckpointStore instance. When set, the blackboard
                            is persisted to disk after every node completes.
        resume            : If True and a checkpoint exists for run_id, restore
                            the board from that checkpoint and skip already-
                            completed nodes. Requires checkpoint_store + run_id.

        Returns the final Blackboard state.
        """
        from rich.console import Console
        _console = Console()

        if show_dag:
            self.preview()

        start_index = 0
        board = Blackboard()

        # --- Seed initial values ---
        if isinstance(initial, str):
            board.set("input", initial, agent="user")
        elif isinstance(initial, dict):
            for k, v in initial.items():
                board.set(k, v, agent="user")

        # --- Resume from checkpoint ---
        if resume and checkpoint_store is not None and run_id is not None:
            saved = checkpoint_store.load(run_id)
            if saved is not None:
                start_index, snapshot = saved
                for k, v in snapshot.items():
                    board.set(k, v, agent="checkpoint")
                _console.print(
                    f"\n[bold green]⟳ Resuming run '{run_id}' "
                    f"from node {start_index}/{len(self._nodes)}[/bold green]"
                )
            else:
                _console.print(
                    f"\n[yellow]No checkpoint found for run_id='{run_id}' — "
                    f"starting from scratch[/yellow]"
                )

        # --- Checkpoint callback ---
        def _on_node_complete(next_index: int, board: Blackboard) -> None:
            if checkpoint_store is not None and run_id is not None:
                checkpoint_store.save(run_id, next_index, board)

        executor = Executor(
            tracer=tracer,
            on_node_complete=_on_node_complete if checkpoint_store else None,
            start_index=start_index,
        )
        anyio.run(executor.run_pipeline, self._nodes, board)

        # --- Delete checkpoint on clean completion ---
        if checkpoint_store is not None and run_id is not None:
            checkpoint_store.delete(run_id)
            _console.print(
                f"[dim]✓ Checkpoint '{run_id}' cleared after successful run[/dim]"
            )

        return board

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, nodes={len(self._nodes)})"
