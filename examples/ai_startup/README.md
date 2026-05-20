# Example: AI Startup Team

Takes a one-line product idea and builds a complete full-stack MVP using five specialist agents.

## Pipeline

```
Input: "A budgeting app for freelancers"
       │
       ▼
  ┌─────────┐
  │   PM    │  Writes a PRD (title, problem, user stories, acceptance criteria)
  └─────────┘
       │
       ▼ (parallel)
  ┌──────────┐  ┌──────────┐
  │Architect │  │ Designer │  Run simultaneously — no shared writes
  └──────────┘  └──────────┘
  Tech stack,    Wireframes,
  API contracts, color palette,
  data models    component specs
       │
       └──────────── merged on blackboard ──────────────┐
                                                         ▼
                                                  ┌──────────┐
                                                  │ Engineer │  Reads PRD + arch + design
                                                  └──────────┘  Writes complete code files
                                                         │
                                                         ▼
                                                  ┌──────────┐
                                                  │   QA     │  Reviews code vs acceptance criteria
                                                  └──────────┘  Loops back to Engineer on failure
                                                               (max 3 iterations)
```

## Agents

| Agent | Reads | Writes |
|---|---|---|
| PM | `input` | `prd` |
| Architect | `prd` | `architecture` |
| Designer | `prd` | `design` |
| Engineer | `prd`, `architecture`, `design`, `bugs` | `code` |
| QA | `prd`, `code` | `qa_report`, `qa_passed`, `bugs` |

## Usage

```bash
# Default idea
python -m orchestrator.examples.ai_startup.main

# Custom idea
python -m orchestrator.examples.ai_startup.main "A task manager for remote teams"
```

## Output

Generated files are saved to `output/<product_name>/`:

```
output/freelanceflow/
├── backend/
│   ├── src/
│   │   ├── server.js
│   │   └── routes/
│   └── prisma/schema.prisma
├── frontend/
│   └── src/
│       ├── App.jsx
│       └── components/
└── summary.json
```

## Backend configuration

```bash
# Anthropic (default)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI-compatible (Redpill, Groq, etc.)
export AGENT_BACKEND="openai_compatible"
export OPENAI_COMPAT_BASE_URL="https://api.redpill.ai/v1"
export OPENAI_COMPAT_API_KEY="sk-rp-..."
export AGENT_MODEL="minimax/minimax-m2.5"
```
