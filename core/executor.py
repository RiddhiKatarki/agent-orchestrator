"""Executor — async engine that walks the pipeline DAG and runs agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

import anyio
from rich.console import Console

from orchestrator.patterns.nodes import (
    BranchNode,
    FanOutNode,
    LoopNode,
    Node,
    ParallelNode,
    SequentialNode,
)

if TYPE_CHECKING:
    from orchestrator.core.blackboard import Blackboard
    from orchestrator.core.tracer import Tracer
    from orchestrator.core.agent import Agent

console = Console()


async def _run_with_timeout(agent: "Agent", board: "Blackboard", tracer: "Tracer | None") -> None:
    """
    Run an agent with optional timeout and error-fallback enforcement.

    Timeout behaviour (agent.timeout_action):
      "skip"    — log a warning and continue
      "raise"   — raise TimeoutError, aborting the pipeline
      <Agent>   — run the fallback agent instead

    Error fallback (agent.error_fallback):
      None      — exception propagates normally
      <Agent>   — run this agent instead when any exception is raised
    """
    timeout = agent.timeout_seconds

    try:
        if timeout is None:
            await anyio.to_thread.run_sync(lambda: agent.run(board, tracer))
            return

        with anyio.move_on_after(timeout) as scope:
            await anyio.to_thread.run_sync(
                lambda: agent.run(board, tracer), cancellable=True
            )

        if scope.cancelled_caught:
            action = agent.timeout_action
            if action == "skip":
                console.print(
                    f"\n[bold yellow]⏱ {agent.name} timed out after {timeout}s — skipping[/bold yellow]"
                )
            elif action == "raise":
                raise TimeoutError(
                    f"{agent.name} exceeded timeout_seconds={timeout}"
                )
            else:
                # Assume it's a fallback Agent instance
                console.print(
                    f"\n[bold yellow]⏱ {agent.name} timed out after {timeout}s — "
                    f"running fallback: {action.name}[/bold yellow]"
                )
                await anyio.to_thread.run_sync(lambda: action.run(board, tracer))

    except TimeoutError:
        raise  # timeout_action="raise" is intentional — don't intercept with error_fallback
    except Exception as exc:
        fallback = agent.error_fallback
        if fallback is None:
            raise
        console.print(
            f"\n[bold red]✖ {agent.name} failed: {exc}[/bold red]"
            f"\n[yellow]  → running error fallback: {fallback.name}[/yellow]"
        )
        await anyio.to_thread.run_sync(lambda: fallback.run(board, tracer), cancellable=True)


class Executor:
    """
    Walks a list of pipeline nodes and runs each one.

    - SequentialNode  → runs one agent, blocks until done
    - ParallelNode    → runs all agents concurrently via anyio task group
    - LoopNode        → repeatedly runs a sequence until condition or max hit
    - BranchNode      → evaluates condition, routes to one of two paths
    - FanOutNode      → fans a worker out across N inputs, gathers results

    Optional callback
    -----------------
    on_node_complete(node_index, board) is called after every top-level node
    finishes. The pipeline uses this to write checkpoints.
    """

    def __init__(
        self,
        tracer: "Tracer | None" = None,
        on_node_complete: "callable | None" = None,
        start_index: int = 0,
    ) -> None:
        self.tracer = tracer
        self.on_node_complete = on_node_complete
        self.start_index = start_index

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run_pipeline(self, nodes: list[Node], board: "Blackboard") -> None:
        for absolute_index, node in enumerate(nodes):
            if absolute_index < self.start_index:
                continue
            await self._run_node(node, board)
            if self.on_node_complete is not None:
                self.on_node_complete(absolute_index + 1, board)

    # ------------------------------------------------------------------
    # Node dispatch
    # ------------------------------------------------------------------

    async def _run_node(self, node: Node, board: "Blackboard") -> None:
        if isinstance(node, SequentialNode):
            await self._run_sequential(node, board)
        elif isinstance(node, ParallelNode):
            await self._run_parallel(node, board)
        elif isinstance(node, LoopNode):
            await self._run_loop(node, board)
        elif isinstance(node, BranchNode):
            await self._run_branch(node, board)
        elif isinstance(node, FanOutNode):
            await self._run_fanout(node, board)
        else:
            raise TypeError(f"Unknown node type: {type(node)}")

    # ------------------------------------------------------------------
    # Sequential
    # ------------------------------------------------------------------

    async def _run_sequential(self, node: SequentialNode, board: "Blackboard") -> None:
        await _run_with_timeout(node.agent, board, self.tracer)

    # ------------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------------

    async def _run_parallel(self, node: ParallelNode, board: "Blackboard") -> None:
        console.print(
            f"\n[bold blue]⚡ Running {len(node.agents)} agents in parallel: "
            f"{', '.join(a.name for a in node.agents)}[/bold blue]"
        )

        async with anyio.create_task_group() as tg:
            for agent in node.agents:
                tg.start_soon(_run_with_timeout, agent, board, self.tracer)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    async def _run_loop(self, node: LoopNode, board: "Blackboard") -> None:
        for iteration in range(node.max_iterations):
            board.set("_loop_iteration", iteration, agent="executor")
            console.print(
                f"\n[bold yellow]↻ Loop iteration {iteration + 1}/{node.max_iterations}[/bold yellow]"
            )

            for agent in node.agents:
                await _run_with_timeout(agent, board, self.tracer)

            if node.until(board):
                console.print("[green]✓ Loop condition satisfied[/green]")
                return

        # Max iterations reached
        msg = (
            f"Loop hit max_iterations={node.max_iterations} without satisfying condition"
        )
        if node.max_reached == "raise":
            raise RuntimeError(msg)
        console.print(f"[yellow]⚠ {msg} — continuing with best-effort result[/yellow]")

    # ------------------------------------------------------------------
    # Branch
    # ------------------------------------------------------------------

    async def _run_branch(self, node: BranchNode, board: "Blackboard") -> None:
        result = node.condition(board)
        target = node.if_true if result else node.if_false
        console.print(
            f"\n[bold blue]⑂ Branch: condition={result} → "
            f"{'if_true' if result else 'if_false'}[/bold blue]"
        )

        if isinstance(target, list):
            await self.run_pipeline(target, board)
        else:
            await _run_with_timeout(target, board, self.tracer)

    # ------------------------------------------------------------------
    # FanOut / FanIn
    # ------------------------------------------------------------------

    async def _run_fanout(self, node: FanOutNode, board: "Blackboard") -> None:
        console.print(
            f"\n[bold blue]⑁ FanOut: {node.worker.name} × {len(node.inputs)} inputs[/bold blue]"
        )

        results: list = [None] * len(node.inputs)

        async def run_worker(idx: int, extra: dict) -> None:
            # Each worker gets its own blackboard clone so writes don't collide
            from orchestrator.core.blackboard import Blackboard as BB
            clone = BB(initial=board.snapshot())
            for k, v in extra.items():
                clone.set(k, v, agent="fanout")
            await _run_with_timeout(node.worker, clone, self.tracer)
            results[idx] = clone.get(node.output_key)

        async with anyio.create_task_group() as tg:
            for i, extra in enumerate(node.inputs):
                tg.start_soon(run_worker, i, extra)

        # Write gathered results to main board
        board.set(node.gather_key, results, agent="fanout")

        # Optionally run a FanIn agent on the main board
        if node.fanin:
            console.print(f"[bold blue]⑁ FanIn: {node.fanin.name}[/bold blue]")
            await _run_with_timeout(node.fanin, board, self.tracer)
