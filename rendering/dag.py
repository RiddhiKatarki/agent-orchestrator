"""DAG renderer — prints a Rich tree showing the pipeline structure and run results."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from orchestrator.patterns.nodes import (
    BranchNode,
    FanOutNode,
    LoopNode,
    Node,
    ParallelNode,
    SequentialNode,
)

if TYPE_CHECKING:
    from orchestrator.core.pipeline import Pipeline
    from orchestrator.core.tracer import Tracer, AgentSpan

console = Console()


def _span_suffix(agent_name: str, tracer: "Tracer | None") -> str:
    """Return a timing/status string for a completed agent span."""
    if tracer is None:
        return ""
    spans = {s.agent_name: s for s in tracer.spans}
    span = spans.get(agent_name)
    if span is None:
        return ""
    icon = "✅" if span.success else "❌"
    return f"  {icon} {span.elapsed:.1f}s  ${span.cost_usd:.4f}"


def _add_node(tree: Tree, node: Node, tracer: "Tracer | None") -> None:
    if isinstance(node, SequentialNode):
        label = Text(node.agent.name, style="bold")
        label.append(_span_suffix(node.agent.name, tracer), style="dim")
        tree.add(label)

    elif isinstance(node, ParallelNode):
        branch = tree.add(Text("┌─ parallel ─┐", style="bold blue"))
        for agent in node.agents:
            label = Text(f"  {agent.name}", style="bold")
            label.append(_span_suffix(agent.name, tracer), style="dim")
            branch.add(label)
        branch.add(Text("└────────────┘", style="bold blue"))

    elif isinstance(node, LoopNode):
        branch = tree.add(
            Text(f"loop (max {node.max_iterations})", style="bold yellow")
        )
        for agent in node.agents:
            label = Text(f"  {agent.name}", style="bold")
            # Loops may have multiple spans — show them all
            if tracer:
                agent_spans = [s for s in tracer.spans if s.agent_name == agent.name]
                for i, span in enumerate(agent_spans):
                    icon = "✅" if span.success else "❌"
                    label.append(
                        f"  iter {i + 1}: {icon} {span.elapsed:.1f}s", style="dim"
                    )
            branch.add(label)

    elif isinstance(node, BranchNode):
        branch = tree.add(Text("⑂ branch", style="bold magenta"))
        branch.add(Text(f"  if_true  → {_agent_name(node.if_true)}", style="green"))
        branch.add(Text(f"  if_false → {_agent_name(node.if_false)}", style="red"))

    elif isinstance(node, FanOutNode):
        branch = tree.add(
            Text(
                f"⑁ fanout × {len(node.inputs)} → {node.worker.name}",
                style="bold cyan",
            )
        )
        if node.fanin:
            branch.add(Text(f"  fanin: {node.fanin.name}", style="cyan"))


def _agent_name(target) -> str:
    if isinstance(target, list):
        return f"[{len(target)} nodes]"
    return target.name


def render(pipeline: "Pipeline", tracer: "Tracer | None" = None) -> None:
    """Print a tree representation of the pipeline, optionally decorated with tracer data."""
    title = Text(f"Pipeline: {pipeline.name}", style="bold magenta")
    tree = Tree(title)

    for node in pipeline._nodes:
        _add_node(tree, node, tracer)

    console.print()
    console.print(tree)
    console.print()
