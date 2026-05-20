#!/usr/bin/env python3
"""
AI Startup — refactored to run on the orchestrator framework.

Usage:
    python -m orchestrator.examples.ai_startup.main "Your startup idea here"
    python -m orchestrator.examples.ai_startup.main   # uses default idea
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console

from orchestrator import Pipeline, Tracer
from orchestrator.rendering import dag
from orchestrator.examples.ai_startup.agents import (
    ArchitectAgent,
    DesignerAgent,
    EngineerAgent,
    PMAgent,
    QAAgent,
)
from orchestrator.examples.ai_startup.models import CodeArtifact, PRD

console = Console()

DEFAULT_IDEA = (
    "A CLI tool that watches a directory and automatically commits changes to Git "
    "with AI-generated commit messages based on the diff."
)


def build_pipeline() -> Pipeline:
    return (
        Pipeline(name="AI Startup")
        .then(PMAgent())
        .parallel(ArchitectAgent(), DesignerAgent())
        .loop(
            agents=[EngineerAgent(), QAAgent()],
            until=lambda b: b.get("qa_passed") is True,
            max_iterations=3,
        )
    )


def save_output(board) -> None:
    prd: PRD | None = board.get("prd")
    code: CodeArtifact | None = board.get("code")

    if not code:
        return

    title = prd.title.replace(" ", "_").lower() if prd else "output"
    out_dir = Path("output") / title
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in code.files.items():
        file_path = out_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    summary = {
        "idea": board.get("input"),
        "prd": prd.model_dump() if prd else None,
        "architecture": board.get("architecture").model_dump() if board.get("architecture") else None,
        "setup": code.setup_instructions,
        "dependencies": code.dependencies,
        "qa_passed": board.get("qa_passed"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print(f"\n[bold green]✓ Output saved to: {out_dir}/[/bold green]")


def main() -> None:
    idea = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_IDEA

    console.print("\n[bold magenta]╔══════════════════════════════╗[/bold magenta]")
    console.print("[bold magenta]║      AI STARTUP TEAM         ║[/bold magenta]")
    console.print("[bold magenta]╚══════════════════════════════╝[/bold magenta]")
    console.print(f"\n[bold]Idea:[/bold] {idea}\n")

    pipeline = build_pipeline()
    tracer = Tracer()

    board = pipeline.run(idea, tracer=tracer)

    # Print DAG with timing
    dag.render(pipeline, tracer)

    # Print cost summary
    tracer.render()

    # Deliver final output
    qa_report = board.get("qa_report")
    if qa_report:
        status = "✅ PASSED" if qa_report.passed else "⚠️  BEST EFFORT"
        console.print(f"\n[bold]QA Status:[/bold] {status}")
        if qa_report.suggestions:
            console.print("[bold]Suggestions for v2:[/bold]")
            for s in qa_report.suggestions:
                console.print(f"  • {s}")

    save_output(board)


if __name__ == "__main__":
    main()
