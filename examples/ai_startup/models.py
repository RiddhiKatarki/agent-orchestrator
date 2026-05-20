"""Pydantic output schemas for the AI Startup pipeline."""

from __future__ import annotations

from pydantic import BaseModel


class PRD(BaseModel):
    title: str
    problem_statement: str
    target_users: str
    user_stories: list[str]
    acceptance_criteria: list[str]
    out_of_scope: list[str] = []


class Architecture(BaseModel):
    tech_stack: dict[str, str]
    components: list[str]
    api_contracts: list[str] = []
    data_models: list[str] = []
    design_decisions: list[str] = []


class DesignSpec(BaseModel):
    wireframes: list[str] = []
    color_palette: list[str] = []
    component_specs: list[str] = []
    ux_notes: list[str] = []


class CodeArtifact(BaseModel):
    files: dict[str, str]
    setup_instructions: str
    dependencies: list[str] = []


class QAReport(BaseModel):
    passed: bool
    test_cases: list[str] = []
    bugs_found: list[str] = []
    suggestions: list[str] = []
