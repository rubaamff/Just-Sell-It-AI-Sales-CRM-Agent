# saudi-companies-directory — installations & modifications

This file tracks every package install and system change made on
**2026-06-23** when the project was refactored to a LangGraph-based
agentic system with guardrails.

## Packages installed (essential — keep until project decommission)

All four are pinned in `requirements.txt` and are required for the agents
in `agents/*.py` and `utils/scorer.py` to import.

| Package              | Version | Installed at                                                                 | Size  |
|----------------------|---------|------------------------------------------------------------------------------|-------|
| `langgraph`          | 0.2.76  | `/Users/abdulrhman/miniconda3/envs/SDA/lib/python3.11/site-packages/langgraph` | 1.8 MB |
| `langgraph-checkpoint` | 2.1.2 | same site-packages dir                                                       | 24 KB (dist-info; package files merged into langgraph tree) |
| `langgraph-sdk`      | 0.1.74  | `…/site-packages/langgraph_sdk`                                              | 588 KB |
| `ddgs`               | 9.14.4  | `…/site-packages/ddgs`                                                       | 636 KB |

Total on-disk: **~3 MB**.

Also touched:
- `requirements.txt` — added `langgraph>=0.2.50,<0.3` and `pydantic>=2.5,<3`.
- Uninstalled `langgraph-prebuilt 1.1.0` (orphaned after downgrading
  langgraph from 1.2.0 to 0.2.76 to match `langchain==0.3.*`).

No system files outside the Python environment were modified.

## macOS OpenMP workaround (2026-06-23)

`app.py` and `utils/rag.py` set `KMP_DUPLICATE_LIB_OK=TRUE` at the top
(before any other import). This is a workaround — not a fix — for a
known macOS issue: `faiss-cpu` and numpy each link their own
`libomp.dylib`, and on certain code paths (proposal agent's second
RAG cycle is the reliable trigger) the process aborts with:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized.
```

The env var is the workaround OpenMP itself recommends in the error
text. It is safe in this app because we don't run parallel OpenMP
workloads — FAISS lookups, OpenAI calls, and DuckDuckGo searches all
happen sequentially.

The proper fix is environment-level (ensure only one libomp lives in
the conda env). If you want to remove the workaround later, audit:
`conda list | grep -iE "faiss|numpy|mkl|openmp|libomp"` and reconcile
the openmp providers.

## Instructions for the next agent

Read this once the **project itself is being decommissioned or migrated
away from LangGraph**. Only then is it safe to remove the packages above.

The project is currently in active development — the agents under
`agents/` and the scorer under `utils/scorer.py` import LangGraph at
module load. Removing it will break `app.py`. Do NOT clean up while:

1. Any tab in `app.py` still references the agents (grep `from agents.`).
2. `requirements.txt` still pins `langgraph` / `pydantic`.
3. There are no replacement implementations for the LangGraph workflows.

When (and only when) the above no longer hold, run:

```sh
pip uninstall -y langgraph langgraph-checkpoint langgraph-sdk ddgs
# pydantic is a transitive dep of many libs; do not blindly uninstall.
```

Then remove the matching lines from `requirements.txt` and delete this
file.
