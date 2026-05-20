"""AI Startup agents — refactored to use the orchestrator framework."""

from __future__ import annotations

import json
import os

from orchestrator.core.agent import Agent, BACKEND_ANTHROPIC, BACKEND_OPENAI_COMPAT
from orchestrator.core.blackboard import Blackboard
from orchestrator.examples.ai_startup.models import (
    PRD,
    Architecture,
    CodeArtifact,
    DesignSpec,
    QAReport,
)

# ---------------------------------------------------------------------------
# Backend config via environment variables — no code changes needed to switch.
#
#   Anthropic (default):
#     export ANTHROPIC_API_KEY="sk-ant-..."
#
#   Redpill / any OpenAI-compatible provider:
#     export AGENT_BACKEND="openai_compatible"
#     export OPENAI_COMPAT_BASE_URL="https://api.redpill.ai/v1"
#     export OPENAI_COMPAT_API_KEY="sk-rp-..."
#     export AGENT_MODEL="minimax/minimax-m2.5"
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("AGENT_BACKEND", BACKEND_ANTHROPIC)
_MODEL   = os.environ.get("AGENT_MODEL", "claude-opus-4-6")
_THINKING = _BACKEND == BACKEND_ANTHROPIC  # thinking only supported on Anthropic


# ──────────────────────────────────────────────
# PM
# ──────────────────────────────────────────────

class PMAgent(Agent):
    name = "PM"
    role = "Product Manager — turns the idea into a PRD"
    system_prompt = """You are a world-class Product Manager at a fast-moving startup.
Your job is to take a raw idea and produce a tight, actionable Product Requirements Document (PRD).
Be concise and opinionated. Avoid fluff. Focus on what users actually need.

Respond ONLY with valid JSON matching this exact structure:
{
  "title": "short product name",
  "problem_statement": "one paragraph describing the core problem",
  "target_users": "who this is for",
  "user_stories": ["As a X, I want Y, so that Z", ...],
  "acceptance_criteria": ["Given X, when Y, then Z", ...],
  "out_of_scope": ["explicitly excluded feature", ...]
}"""
    backend = _BACKEND
    model = _MODEL
    use_thinking = _THINKING
    output_schema = PRD
    reads = []
    writes = ["prd"]

    def build_prompt(self, board: Blackboard) -> str:
        idea = board.get("input", str) or ""
        return f"Write a PRD for this idea:\n\n{idea}"


# ──────────────────────────────────────────────
# Architect
# ──────────────────────────────────────────────

class ArchitectAgent(Agent):
    name = "Architect"
    role = "Software Architect — designs the technical architecture"
    system_prompt = """You are a senior software architect. Given a PRD, you design a lean,
practical technical architecture for a startup MVP.
Prefer simple, proven technology over cutting-edge complexity.
Make concrete decisions — don't hedge.

Respond ONLY with valid JSON matching this exact structure:
{
  "tech_stack": {"frontend": "...", "backend": "...", "database": "...", "hosting": "..."},
  "components": ["Component: description", ...],
  "api_contracts": ["POST /endpoint — description", ...],
  "data_models": ["ModelName: field1, field2, ...", ...],
  "design_decisions": ["Decision and rationale", ...]
}"""
    backend = _BACKEND
    model = _MODEL
    use_thinking = _THINKING
    output_schema = Architecture
    reads = ["prd"]
    writes = ["architecture"]

    def build_prompt(self, board: Blackboard) -> str:
        prd: PRD = board.get("prd", PRD)
        return (
            f"Design the technical architecture for this PRD:\n\n"
            f"Title: {prd.title}\n"
            f"Problem: {prd.problem_statement}\n"
            f"Users: {prd.target_users}\n"
            f"Stories:\n" + "\n".join(f"- {s}" for s in prd.user_stories) + "\n"
            f"Acceptance criteria:\n" + "\n".join(f"- {c}" for c in prd.acceptance_criteria)
        )


# ──────────────────────────────────────────────
# Designer
# ──────────────────────────────────────────────

class DesignerAgent(Agent):
    name = "Designer"
    role = "UX/UI Designer — creates interface specs and wireframes"
    system_prompt = """You are a senior UX/UI designer. Given a PRD, you create clean, user-centric
interface specifications for a startup MVP.
Use ASCII art for wireframes. Be specific about components, interactions, and layout.

Respond ONLY with valid JSON matching this exact structure:
{
  "wireframes": [
    "Screen Name:\\n+--------+\\n| ...    |\\n+--------+",
    ...
  ],
  "color_palette": ["#hexcode — purpose", ...],
  "component_specs": ["ComponentName: description, props, behavior", ...],
  "ux_notes": ["UX principle or interaction note", ...]
}"""
    backend = _BACKEND
    model = _MODEL
    use_thinking = _THINKING
    output_schema = DesignSpec
    reads = ["prd"]
    writes = ["design"]

    def build_prompt(self, board: Blackboard) -> str:
        prd: PRD = board.get("prd", PRD)
        return (
            f"Create UX/UI specs and wireframes for this product:\n\n"
            f"Title: {prd.title}\n"
            f"Problem: {prd.problem_statement}\n"
            f"Users: {prd.target_users}\n"
            f"User Stories:\n" + "\n".join(f"- {s}" for s in prd.user_stories)
        )


# ──────────────────────────────────────────────
# Engineer
# ──────────────────────────────────────────────

class EngineerAgent(Agent):
    name = "Engineer"
    role = "Full-stack Engineer — writes the MVP code"
    system_prompt = """You are a senior full-stack engineer. Given a PRD, architecture, and design specs,
you write working, clean code for a startup MVP.
Write complete, runnable files. Don't leave placeholders like "# TODO" or "your logic here".
Use the exact tech stack from the architecture.

Respond ONLY with valid JSON matching this exact structure:
{
  "files": {
    "filename.ext": "full file content here",
    "another_file.ext": "full file content here"
  },
  "setup_instructions": "step by step instructions to run the project",
  "dependencies": ["package==version", ...]
}"""
    backend = _BACKEND
    model = _MODEL
    use_thinking = _THINKING
    max_tokens = 32768
    output_schema = CodeArtifact
    reads = ["prd", "architecture", "design", "bugs"]
    writes = ["code"]

    def build_prompt(self, board: Blackboard) -> str:
        prd: PRD = board.get("prd", PRD)
        arch: Architecture = board.get("architecture", Architecture)
        design: DesignSpec = board.get("design", DesignSpec)
        bugs: list[str] | None = board.get("bugs")
        iteration: int = board.get("_loop_iteration", int) or 0

        label = f"(iteration {iteration + 1})" if iteration > 0 else ""
        prompt = (
            f"Write the complete MVP implementation {label}\n\n"
            f"PRD:\n"
            f"  Title: {prd.title}\n"
            f"  Problem: {prd.problem_statement}\n"
            f"  Users: {prd.target_users}\n"
            f"  Acceptance criteria:\n"
            + "\n".join(f"    - {c}" for c in prd.acceptance_criteria)
            + f"\n\nArchitecture:\n"
            f"  Tech stack: {json.dumps(arch.tech_stack, indent=2)}\n"
            f"  Components:\n" + "\n".join(f"  - {c}" for c in arch.components)
            + f"\n  API contracts:\n" + "\n".join(f"  - {a}" for a in arch.api_contracts)
            + f"\n  Data models:\n" + "\n".join(f"  - {d}" for d in arch.data_models)
            + f"\n\nDesign:\n"
            f"  Components:\n" + "\n".join(f"  - {c}" for c in design.component_specs)
            + f"\n  UX notes:\n" + "\n".join(f"  - {n}" for n in design.ux_notes)
        )

        if bugs:
            prompt += (
                f"\n\nQA found these bugs — fix them:\n"
                + "\n".join(f"  - {b}" for b in bugs)
            )

        return prompt


# ──────────────────────────────────────────────
# QA
# ──────────────────────────────────────────────

class QAAgent(Agent):
    name = "QA"
    role = "QA Engineer — reviews code against the PRD"
    system_prompt = """You are a senior QA engineer. You review code against requirements and
identify real bugs, missing features, and quality issues.
Be specific — reference file names and line numbers where possible.
Don't invent problems; only flag genuine issues.

Respond ONLY with valid JSON matching this exact structure:
{
  "passed": true or false,
  "test_cases": ["Test case description — PASS or FAIL", ...],
  "bugs_found": ["Bug description with file/line reference", ...],
  "suggestions": ["Optional improvement", ...]
}

Set "passed" to true only if all acceptance criteria are met and no critical bugs found."""
    backend = _BACKEND
    model = _MODEL
    use_thinking = _THINKING
    output_schema = QAReport
    reads = ["prd", "code"]
    writes = ["qa_report"]

    def build_prompt(self, board: Blackboard) -> str:
        prd: PRD = board.get("prd", PRD)
        code: CodeArtifact = board.get("code", CodeArtifact)

        files_block = "\n".join(
            f"\n--- {fname} ---\n{content}"
            for fname, content in code.files.items()
        )
        return (
            f"Review this code against the acceptance criteria.\n\n"
            f"Acceptance criteria:\n"
            + "\n".join(f"- {c}" for c in prd.acceptance_criteria)
            + f"\n\nCode:\n{files_block}\n\nSetup:\n{code.setup_instructions}"
        )

    def write_outputs(self, result: QAReport, board: Blackboard) -> None:
        board.set("qa_report", result, agent=self.name)
        # Write qa_passed separately so the loop condition can read it cheaply
        board.set("qa_passed", result.passed, agent=self.name)
        # Forward bugs to the board so EngineerAgent can pick them up next round
        if not result.passed:
            board.set("bugs", result.bugs_found, agent=self.name)

        from rich.console import Console
        c = Console()
        if result.passed:
            c.print("\n[green]✓ QA PASSED[/green]")
        else:
            c.print(f"\n[red]✗ QA FAILED — {len(result.bugs_found)} bug(s)[/red]")
            for bug in result.bugs_found:
                c.print(f"  • {bug}")
