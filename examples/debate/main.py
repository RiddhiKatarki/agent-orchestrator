#!/usr/bin/env python3
"""
Multi-agent Debate — two debaters argue a proposition across multiple rounds,
then a judge delivers a verdict.

Usage:
    python -m orchestrator.examples.debate.main "AI will eliminate more jobs than it creates"
    python -m orchestrator.examples.debate.main   # uses default topic
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from orchestrator import Pipeline, Tracer
from orchestrator.rendering import dag
from orchestrator.examples.debate.agents import (
    JudgeAgent,
    OpponentAgent,
    OpponentRebuttalAgent,
    ProponentAgent,
    ProponentRebuttalAgent,
)
from orchestrator.examples.debate.models import Verdict

console = Console()

DEFAULT_TOPIC = "AI will eliminate more jobs than it creates over the next decade"
DEBATE_ROUNDS = 2  # rebuttal rounds each side gets


def build_pipeline() -> Pipeline:
    return (
        Pipeline(name="Debate")
        # Opening statements — both sides argue in parallel
        .parallel(ProponentAgent(), OpponentAgent())
        # Rebuttal rounds — sequential within each round, parallel between sides
        .loop(
            agents=[ProponentRebuttalAgent(), OpponentRebuttalAgent()],
            until=lambda b: (b.get("_loop_iteration") or 0) >= DEBATE_ROUNDS - 1,
            max_iterations=DEBATE_ROUNDS,
        )
        # Judge delivers verdict
        .then(JudgeAgent())
    )


def render_verdict(verdict: Verdict, topic: str) -> None:
    winner_label = {
        "for": "[green]FOR[/green]",
        "against": "[red]AGAINST[/red]",
        "draw": "[yellow]DRAW[/yellow]",
    }.get(verdict.winner, verdict.winner)

    console.print()
    console.print(Rule("[bold magenta]⚖  VERDICT[/bold magenta]"))
    console.print(f"\n[bold]Proposition:[/bold] {topic}")
    console.print(f"[bold]Winner:[/bold] {winner_label}")
    console.print(
        f"[bold]Scores:[/bold] "
        f"FOR {verdict.scores.get('for', '?')}/10  vs  "
        f"AGAINST {verdict.scores.get('against', '?')}/10"
    )
    console.print(f"\n[bold]Reasoning:[/bold]\n{verdict.reasoning}")
    console.print(
        Panel(verdict.strongest_argument, title="Strongest argument", border_style="green")
    )
    console.print(
        Panel(verdict.weakest_argument, title="Weakest argument", border_style="red")
    )
    console.print(f"\n[dim]{verdict.summary}[/dim]")


def main() -> None:
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_TOPIC

    console.print("\n[bold magenta]╔══════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║       DEBATE CHAMBER         ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════╝[/bold magenta]")
    console.print(f'\n[bold]Proposition:[/bold] "{topic}"\n')

    pipeline = build_pipeline()
    tracer = Tracer()

    board = pipeline.run({"topic": topic}, tracer=tracer)

    # DAG + cost summary
    dag.render(pipeline, tracer)
    tracer.render()

    # Verdict
    verdict: Verdict | None = board.get("verdict")
    if verdict:
        render_verdict(verdict, topic)


if __name__ == "__main__":
    main()
