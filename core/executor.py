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


class Executor:
    """
    Walks a list of pipeline nodes and runs each one.

    - SequentialNode  → runs one agent, blocks until done
    - ParallelNode    → runs all agents concurrently via anyio task group
    - LoopNode        → repeatedly runs a sequence until condition or max hit
    - BranchNode      → evaluates condition, routes to one of two paths
    - FanOutNode      → fans a worker out across N inputs, gathers results
    """

    def __init__(self, tracer: "Tracer | None" = None) -> None:
        self.tracer = tracer

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run_pipeline(self, nodes: list[Node], board: "Blackboard") -> None:
        for node in nodes:
            await self._run_node(node, board)

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
        tracer = self.tracer
        agent = node.agent
        await anyio.to_thread.run_sync(lambda: agent.run(board, tracer))

    # ------------------------------------------------------------------
    # Parallel
    # ------------------------------------------------------------------

    async def _run_parallel(self, node: ParallelNode, board: "Blackboard") -> None:
        console.print(
            f"\n[bold blue]⚡ Running {len(node.agents)} agents in parallel: "
            f"{', '.join(a.name for a in node.agents)}[/bold blue]"
        )
        tracer = self.tracer

        async with anyio.create_task_group() as tg:
            for agent in node.agents:
                tg.start_soon(
                    anyio.to_thread.run_sync,
                    lambda a=agent: a.run(board, tracer),
                )

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
                await anyio.to_thread.run_sync(
                    lambda a=agent: a.run(board, self.tracer)
                )

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
            await anyio.to_thread.run_sync(lambda: target.run(board, self.tracer))

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
            await anyio.to_thread.run_sync(
                lambda: node.worker.run(clone, self.tracer)
            )
            results[idx] = clone.get(node.output_key)

        async with anyio.create_task_group() as tg:
            for i, extra in enumerate(node.inputs):
                tg.start_soon(run_worker, i, extra)

        # Write gathered results to main board
        board.set(node.gather_key, results, agent="fanout")

        # Optionally run a FanIn agent on the main board
        if node.fanin:
            console.print(f"[bold blue]⑁ FanIn: {node.fanin.name}[/bold blue]")
            await anyio.to_thread.run_sync(
                lambda: node.fanin.run(board, self.tracer)
            )
