"""
Orchestrator UI — Phase 1
Run with:  streamlit run ui/app.py  (from the orchestrator/ root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# orchestrator/ root — makes `ui` importable
_orch_root = Path(__file__).parent.parent
# project/ root — makes `orchestrator` package importable
_proj_root = _orch_root.parent
for _p in (_orch_root, _proj_root):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from orchestrator.core.pipeline import Pipeline
from orchestrator.core.tracer import Tracer
from ui.registry import build_agent, load_registry, render_dag_text, save_agent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agent Orchestrator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------
if "steps" not in st.session_state:
    st.session_state.steps: list[dict] = []          # ordered pipeline steps
if "seed_pairs" not in st.session_state:
    st.session_state.seed_pairs = [{"key": "input", "value": ""}]
if "result_board" not in st.session_state:
    st.session_state.result_board: dict | None = None
if "result_tracer" not in st.session_state:
    st.session_state.result_tracer: list[dict] | None = None
if "run_error" not in st.session_state:
    st.session_state.run_error: str | None = None
if "dag_text" not in st.session_state:
    st.session_state.dag_text: str | None = None
if "show_create" not in st.session_state:
    st.session_state.show_create = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move(lst: list, i: int, direction: int) -> None:
    j = i + direction
    lst[i], lst[j] = lst[j], lst[i]


def _build_pipeline(name: str = "UI Pipeline") -> Pipeline:
    pipeline = Pipeline(name=name)
    for step in st.session_state.steps:
        agent = build_agent(step)
        pipeline.then(agent)
    return pipeline


def _seed_dict() -> dict:
    return {
        p["key"]: p["value"]
        for p in st.session_state.seed_pairs
        if p["key"].strip()
    }


# ---------------------------------------------------------------------------
# Sidebar — Agent Library
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Agent Library")
    st.caption("Click **Add** to append an agent to your pipeline.")

    search = st.text_input("Search agents", placeholder="Filter by name or role…")
    registry = load_registry()
    filtered = (
        [s for s in registry
         if search.lower() in s["name"].lower()
         or search.lower() in s.get("role", "").lower()]
        if search else registry
    )

    for spec in filtered:
        with st.expander(f"**{spec['name']}**"):
            st.caption(spec.get("role", ""))
            col_m, col_b = st.columns(2)
            col_m.markdown(f"`{spec.get('model', '—')}`")
            col_b.markdown(f"`{spec.get('backend', 'anthropic')}`")
            if spec.get("reads"):
                st.markdown(f"**Reads:** {', '.join(spec['reads'])}")
            if spec.get("writes"):
                st.markdown(f"**Writes:** {', '.join(spec['writes'])}")
            if st.button("Add to Pipeline", key=f"add_{spec['name']}", use_container_width=True):
                st.session_state.steps.append(dict(spec))
                st.rerun()

    st.divider()

    # ---- Create New Agent ----
    if st.button("+ Create New Agent", use_container_width=True):
        st.session_state.show_create = not st.session_state.show_create

    if st.session_state.show_create:
        st.subheader("New Agent")
        with st.form("create_agent_form", clear_on_submit=True):
            c_name = st.text_input("Name*", placeholder="MyAgent")
            c_role = st.text_input("Role", placeholder="One-line description")
            c_model = st.text_input("Model", value="claude-opus-4-6")
            c_backend = st.selectbox("Backend", ["anthropic", "openai_compatible"])
            c_base_url = st.text_input("Base URL", placeholder="(openai_compatible only)")
            c_system = st.text_area("System Prompt*", height=120)
            c_reads = st.text_input("Reads (comma-separated keys)")
            c_writes = st.text_input("Writes (comma-separated keys)")
            c_max_tokens = st.number_input("Max Tokens", value=2048, min_value=256, step=256)
            submitted = st.form_submit_button("Save Agent", use_container_width=True)

        if submitted:
            if not c_name or not c_system:
                st.error("Name and System Prompt are required.")
            else:
                new_spec = {
                    "name": c_name.strip(),
                    "role": c_role.strip(),
                    "model": c_model.strip(),
                    "backend": c_backend,
                    "base_url": c_base_url.strip(),
                    "system_prompt": c_system.strip(),
                    "reads": [k.strip() for k in c_reads.split(",") if k.strip()],
                    "writes": [k.strip() for k in c_writes.split(",") if k.strip()],
                    "max_tokens": int(c_max_tokens),
                    "use_thinking": False,
                }
                save_agent(new_spec)
                st.success(f"Saved `{c_name}` to the registry. Reload the sidebar to see it.")
                st.session_state.show_create = False
                st.rerun()

    st.divider()
    st.caption("Set `ANTHROPIC_API_KEY` (Anthropic) or `OPENAI_COMPAT_BASE_URL` + `OPENAI_COMPAT_API_KEY` (other backends) as environment variables before running.")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.title("Agent Orchestrator")
st.caption("Build sequential multi-agent pipelines without writing code.")

# ---------------------------------------------------------------------------
# Two-column layout: Pipeline Builder | Run Controls
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2], gap="large")

# ---- Left: Pipeline Builder ----
with col_left:
    st.subheader("Pipeline")

    if not st.session_state.steps:
        st.info("No agents added yet. Use the sidebar to add agents to your pipeline.")
    else:
        for i, step in enumerate(st.session_state.steps):
            with st.container(border=True):
                hdr, btn_up, btn_dn, btn_rm = st.columns([5, 1, 1, 1])
                hdr.markdown(f"**{i + 1}. {step['name']}**  \n{step.get('role', '')}")

                if btn_up.button("↑", key=f"up_{i}", disabled=(i == 0)):
                    _move(st.session_state.steps, i, -1)
                    st.rerun()

                if btn_dn.button("↓", key=f"dn_{i}",
                                 disabled=(i == len(st.session_state.steps) - 1)):
                    _move(st.session_state.steps, i, 1)
                    st.rerun()

                if btn_rm.button("✕", key=f"rm_{i}"):
                    st.session_state.steps.pop(i)
                    st.rerun()

                detail_l, detail_r = st.columns(2)
                detail_l.caption(
                    f"Model: `{step.get('model', '—')}`  |  "
                    f"Backend: `{step.get('backend', 'anthropic')}`"
                )
                reads = step.get("reads") or []
                writes = step.get("writes") or []
                detail_r.caption(
                    f"Reads: `{', '.join(reads) or '—'}`  |  "
                    f"Writes: `{', '.join(writes) or '—'}`"
                )

        if st.button("Clear Pipeline", use_container_width=True):
            st.session_state.steps = []
            st.session_state.result_board = None
            st.session_state.result_tracer = None
            st.session_state.run_error = None
            st.session_state.dag_text = None
            st.rerun()

# ---- Right: Seed Input + Controls ----
with col_right:
    st.subheader("Seed Input")
    st.caption("Key-value pairs written to the blackboard before the pipeline runs.")

    for j, pair in enumerate(st.session_state.seed_pairs):
        r1, r2, r3 = st.columns([2, 4, 1])
        st.session_state.seed_pairs[j]["key"] = r1.text_input(
            "Key", value=pair["key"], key=f"sk_{j}", label_visibility="collapsed",
            placeholder="key"
        )
        st.session_state.seed_pairs[j]["value"] = r2.text_input(
            "Value", value=pair["value"], key=f"sv_{j}", label_visibility="collapsed",
            placeholder="value"
        )
        if r3.button("−", key=f"srm_{j}", disabled=(len(st.session_state.seed_pairs) == 1)):
            st.session_state.seed_pairs.pop(j)
            st.rerun()

    if st.button("+ Add key", use_container_width=True):
        st.session_state.seed_pairs.append({"key": "", "value": ""})
        st.rerun()

    st.divider()

    pipeline_name = st.text_input("Pipeline name", value="My Pipeline")

    dag_btn, run_btn = st.columns(2)

    if dag_btn.button("Preview DAG", use_container_width=True,
                      disabled=not st.session_state.steps):
        pipeline = _build_pipeline(pipeline_name)
        st.session_state.dag_text = render_dag_text(pipeline)

    run_clicked = run_btn.button(
        "Run Pipeline", type="primary", use_container_width=True,
        disabled=not st.session_state.steps,
    )

# ---------------------------------------------------------------------------
# DAG preview (shown inline below controls when requested)
# ---------------------------------------------------------------------------
if st.session_state.dag_text:
    with col_right:
        st.code(st.session_state.dag_text, language=None)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if run_clicked:
    st.session_state.result_board = None
    st.session_state.result_tracer = None
    st.session_state.run_error = None
    st.session_state.dag_text = None

    pipeline = _build_pipeline(pipeline_name)
    seed = _seed_dict()
    tracer = Tracer()

    with st.spinner("Running pipeline… (output streaming to terminal)"):
        try:
            board = pipeline.run(initial=seed or None, tracer=tracer)
            st.session_state.result_board = board.snapshot()
            st.session_state.result_tracer = tracer.export()
        except Exception as exc:
            st.session_state.run_error = str(exc)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.run_error:
    st.error(f"**Pipeline failed:** {st.session_state.run_error}")

if st.session_state.result_board is not None:
    st.divider()
    st.subheader("Results")

    tab_board, tab_tracer, tab_raw = st.tabs(["Blackboard", "Tracer", "Raw JSON"])

    with tab_board:
        board = st.session_state.result_board
        if not board:
            st.info("Blackboard is empty.")
        else:
            for key, value in board.items():
                with st.expander(f"`{key}`", expanded=True):
                    if isinstance(value, (dict, list)):
                        st.json(value)
                    else:
                        st.text(str(value))

    with tab_tracer:
        spans = st.session_state.result_tracer
        if not spans:
            st.info("No tracer data.")
        else:
            import pandas as pd

            df = pd.DataFrame(spans)[
                ["agent", "model", "success", "input_tokens",
                 "output_tokens", "cost_usd", "elapsed_s", "retries", "error"]
            ]
            df["cost_usd"] = df["cost_usd"].map("${:.4f}".format)
            df["elapsed_s"] = df["elapsed_s"].map("{:.1f}s".format)
            st.dataframe(df, use_container_width=True, hide_index=True)

            total_cost = sum(s["cost_usd"] for s in spans
                             if isinstance(s["cost_usd"], float))
            total_tok = sum(s["input_tokens"] + s["output_tokens"] for s in spans)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total tokens", f"{total_tok:,}")
            m2.metric("Total cost", f"${total_cost:.4f}")
            m3.metric("Agents run", len(spans))

    with tab_raw:
        st.json(st.session_state.result_board)
