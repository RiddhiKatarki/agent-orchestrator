"""Agent registry — loads YAML agent definitions and builds Agent subclasses dynamically."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import yaml

# orchestrator/ root — makes `ui` importable
_orch_root = Path(__file__).parent.parent
# project/ root — makes `orchestrator` package importable
_proj_root = _orch_root.parent
for _p in (_orch_root, _proj_root):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from orchestrator.core.agent import Agent, DEFAULT_MODEL, BACKEND_ANTHROPIC

AGENTS_DIR = Path(__file__).parent / "agents"


def load_registry() -> list[dict]:
    """Return all agent specs from *.yaml files in the agents/ directory."""
    specs: list[dict] = []
    for path in sorted(AGENTS_DIR.glob("*.yaml")):
        with open(path) as f:
            spec = yaml.safe_load(f)
        spec["_source"] = path.name
        specs.append(spec)
    return specs


def build_agent(spec: dict) -> Agent:
    """
    Dynamically create an Agent subclass from a spec dict and return an instance.

    The spec dict mirrors the YAML format:
      name, role, system_prompt, model, backend, base_url, api_key_env,
      reads, writes, max_tokens, use_thinking, timeout_seconds,
      timeout_action, max_validation_retries
    """
    attrs: dict[str, Any] = {
        "name": spec["name"],
        "role": spec.get("role", ""),
        "system_prompt": spec.get("system_prompt", ""),
        "model": spec.get("model", DEFAULT_MODEL),
        "backend": spec.get("backend", BACKEND_ANTHROPIC),
        "base_url": spec.get("base_url", ""),
        "api_key_env": spec.get("api_key_env", "OPENAI_COMPAT_API_KEY"),
        "reads": list(spec.get("reads") or []),
        "writes": list(spec.get("writes") or []),
        "max_tokens": spec.get("max_tokens", 4096),
        "use_thinking": spec.get("use_thinking", False),
        "output_schema": None,
        "timeout_seconds": spec.get("timeout_seconds", None),
        "timeout_action": spec.get("timeout_action", "skip"),
        "error_fallback": None,
        "max_validation_retries": spec.get("max_validation_retries", 2),
    }
    cls = type(spec["name"], (Agent,), attrs)
    return cls()


def save_agent(spec: dict) -> Path:
    """Persist a spec dict as a YAML file in the agents/ directory."""
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = spec["name"].lower().replace(" ", "_")
    path = AGENTS_DIR / f"{safe_name}.yaml"
    with open(path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
    return path


def render_dag_text(pipeline) -> str:
    """
    Render the pipeline DAG as plain text (no ANSI codes) and return the string.
    Uses the same tree-building logic as dag.render() but with a captured console.
    """
    from rich.console import Console
    from rich.tree import Tree
    from rich.text import Text
    from orchestrator.rendering.dag import _add_node

    buf = io.StringIO()
    cap = Console(file=buf, no_color=True, highlight=False, width=72)

    title = Text(f"Pipeline: {pipeline.name}")
    tree = Tree(title)
    for node in pipeline._nodes:
        _add_node(tree, node, tracer=None)

    cap.print()
    cap.print(tree)
    cap.print()
    return buf.getvalue()
