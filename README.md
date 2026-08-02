# Bonton Autonomous Capacity Orchestration & Order Processing Solution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-blue.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)

This repository contains a working MVP for an agentic production-planning assistant for Bonton Technomake. The current implementation combines a deterministic capacity engine with a Gemini-powered LangGraph ReAct agent so that planning questions can be answered with both calculation and explanation.

## What the current implementation does

The project currently focuses on three core capabilities:

- Deterministic shop-capacity analysis across four manufacturing steps:
  - Press Shop (S1)
  - Fabrication (S2)
  - Powder Coating (S3)
  - Assembly (S4)
- Makespan comparison between monolithic batching and transfer batching.
- Intervention analysis for shortfall relief using rough cost estimates for overtime, subcontracting, and yield improvement.

## Main components

### 1. Deterministic capacity engine

The logic in model.py defines a CapacityEngine with:

- shop configuration data for the four shops
- unit-flow calculations for Press, Fabrication, and Assembly
- batch-process calculations for Powder Coating with yield and rework assumptions
- effective rate, required hours, shortfall, and constraint detection

### 2. LangChain tools

The file exposes three tools for the agent:

- tool_compute_plant_capacity(order_qty)
- tool_calculate_makespans(order_qty, transfer_batch_size=100)
- tool_evaluate_interventions(order_qty, shortfall_hours)

These tools provide the numerical grounding for the agent.

### 3. Gemini ReAct agent

The build_bonton_agent() function builds a LangGraph ReAct agent using:

- ChatGoogleGenerativeAI with the gemini-3.1-flash-lite model
- the three capacity tools above
- a system prompt that instructs the agent to reason from tool output

## Repository files

- model.py: the main implementation containing the capacity engine, tools, and agent setup
- setup.py: installs the Python dependencies required by the current MVP
- README.md: project documentation

## Prerequisites

- Python 3.10+
- A Google Gemini API key

## Setup

On Windows, the workflow is:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python setup.py
```

After installing dependencies, update the placeholder API key in model.py:

```python
google_api_key="YOUR_GEMINI_API_KEY"
```

## Run the demo

```powershell
python model.py
```

The script runs a sample request for 1,000 units and prints the agent's analysis of capacity, bottlenecks, makespan improvement, and intervention recommendations.

## Notes

- This is an MVP implementation, not a full production planning platform.
- The current version relies on a hardcoded Gemini API key placeholder and is intended as a starting point for further expansion.
- All core calculations are deterministic and intended to be explainable and auditable.
