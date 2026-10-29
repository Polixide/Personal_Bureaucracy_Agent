# Personal Bureaucracy Digital Twin: Stateful Agentic Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Framework: LangGraph](https://img.shields.io/badge/Framework-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Course: CS58012](https://img.shields.io/badge/Course-CS58012%20Agentic%20AI-blue.svg)](https://www.sabanciuniv.edu/)
[![Status: Milestone 1](https://img.shields.io/badge/Status-In%20Development-green.svg)](/)

---

## 📌 Introduction

This repository contains the official implementation of the **Personal Bureaucracy Digital Twin**, a stateful hybrid agentic assistant designed to guide international students through complex, multi-week administrative workflows (such as obtaining a Turkish residence permit / *İkamet*).

Bureaucratic procedures are inherently dependency-driven and time-sensitive. Standard conversational LLMs or static checklists fail because they lack persistent long-term memory, make errors in date logic, and cannot recover safely from runtime failures.

> [!IMPORTANT]
> **Core Architectural Rule:** Use LLMs strictly for semantic interpretation and flexible planning, but rely on **deterministic Python code** for exact rules, date arithmetic, and schema validations.

---

## 🚀 Key Features

- **LangGraph Orchestrator-Workers Topology:** A Central Supervisor dynamically evaluates the user state and delegates execution to specialized workers (*Document Analyzer*, *Requirement Engine*, *Action Planner*).
- **Zero-Token & Deterministic Guardrails:** Pre-checks (**Gate 0**) reject malformed inputs before spending LLM tokens, while post-checks (**Gate 1**) enforce strict prerequisite and date consistency.
- **Persistent Digital Twin State:** SQLite-backed checkpointing maintains state continuity across sessions, tracking uploaded documents, validity dates, active blockers, and completed steps.
- **Grounded & Bounded Reflexion Loop:** When Gate 1 detects an invalid plan, a structured error feedback loop triggers bounded replanning, strictly capped by a **Hard Circuit Breaker** ($t \le 3$).
- **Human-in-the-Loop Approval:** Consequential or official submissions require explicit user confirmation before execution.

> [!NOTE]
> This prototype is specifically targeted at the Turkish residence permit (*İkamet*) workflow, but the orchestration engine is domain-agnostic and can be adapted to other procedures by swapping the rule base.

---

## 🏗️ System Architecture

The workflow combines semantic LLM planning with deterministic safety gates:

```text
User Goal / New Event 
       │
       ▼
[Gate 0: Input Pre-Checks] ──(Rejects Malformed Files)
       │
       ▼
{ Digital Twin / Shared State } ──(SQLite Persistent Checkpointer)
       │
       ▼
[Supervisor / Orchestrator] ──(State-Aware Dynamic Routing)
       ├── 📄 Document Analyzer (Parser + LLM + Pydantic)
       ├── ⚙️ Requirement Engine (Deterministic Rules)
       └── 🗓️ Action Planner (LLM + Scheduling Tools)
       │
       ▼
[Gate 1: Deterministic Validator] ──(Checks Dates, Schemas & Prereqs)
       │
       ├── [PASS] ──► Human Approval ──► State Update ──► Final Output
       └── [FAIL] ──► Reflection / Replan Loop (Max 3 retries)
```

> [!WARNING]
> **Safety Guardrail:** The system will never autonomously submit official forms, make payments, or bypass authentication on government portals. Human approval is mandatory.

---

## 📂 Project Structure

The codebase is organized into clean, modular workstreams designed for parallel collaboration across our team of 4 data science engineers:

```text
personal-bureaucracy-digital-twin/
├── config.py                 # Global configurations (paths, DB settings, circuit breaker thresholds)
│
├── data/
│   ├── rules_ikamet.json     # Versioned, authoritative administrative rule base
│   └── synthetic_samples/    # Test document samples and mock student profiles
│
├── src/
│   ├── core/
│   │   ├── state.py          # AgentState definition, Pydantic schemas, and state reducers
│   │   └── graph.py          # LangGraph graph definition, nodes, and conditional edges
│   │
│   ├── agents/
│   │   ├── supervisor.py     # Central Orchestrator node
│   │   └── planner.py        # Action & Deadline Planner node
│   │
│   ├── workers/
│   │   ├── document_analyzer.py # Document parsing & Pydantic structured output extraction
│   │   └── rules_engine.py      # Deterministic requirement and date logic checker
│   │
│   ├── guardrails/
│   │   ├── gate_0.py         # Zero-token input and schema validator
│   │   └── gate_1.py         # Deterministic post-execution validator
│   │
│   ├── memory/
│   │   ├── checkpointer.py   # SQLite persistence layer for Digital Twin state
│   │   └── reflection.py     # Bounded reflection mechanism and episodic memory
│   │
│   └── utils/
│       └── helpers.py        # Date formatting, logging, and helper utilities
│
├── tests/                    # Synthetic scenario runner & automated unit tests
│   ├── test_rules.py
│   ├── test_gateways.py
│   └── test_scenarios.py
│
├── scripts/                  # Execution scripts
│   └── run_demo.py           # End-to-end CLI scenario runner
│
├── requirements.txt
├── LICENSE                   # MIT License
└── README.md
```

> [!TIP]
> **Suggested Team Task Division (4 Members):**
> - **Member 1 (Graph & Orchestration):** `src/core/` and `src/agents/` (LangGraph setup, Supervisor logic, state reducers).
> - **Member 2 (Document Analysis & Extraction):** `src/workers/document_analyzer.py` (PDF parsing, OCR integration, Pydantic schemas).
> - **Member 3 (Rules Engine & Knowledge Base):** `src/workers/rules_engine.py` and `data/` (Encoding deterministic dependency rules, date logic).
> - **Member 4 (Guardrails, Persistence & Testing):** `src/guardrails/`, `src/memory/`, and `tests/` (Gates 0/1, SQLite checkpointer, circuit breaker test harness).

---

## 🛠️ Installation & Quickstart

### Prerequisites
- Python 3.11+
- SQLite3

### Setup

```bash
# Clone repository
git clone [https://github.com/your-username/personal-bureaucracy-digital-twin.git](https://github.com/your-username/personal-bureaucracy-digital-twin.git)
cd personal-bureaucracy-digital-twin

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running a Test Scenario

To test the system against a sample event (e.g., uploading a tuition payment receipt):

```bash
python -m scripts.run_demo --scenario tuition_receipt
```

---

## 👥 Authors

- **Neda Mohamed** — Sabancı University
- **Daniele Catalano** ([@Polixide](https://github.com/Polixide)) — Politecnico di Torino / Sabancı University
- **Andrea Brambilla** — Politecnico di Torino / Sabancı University
- **Marco Belforti** — Politecnico di Torino / Sabancı University

*CS58012: Agentic AI Systems Architecture and Deployment — Sabancı University 2026*
