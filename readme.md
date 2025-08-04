# Astronomicon

> Retrieval-Augmented Generation over Warhammer 40,000 lore.  
> I got tired of my friends constantly asking me questions about 40K, so I built this to answer them fast—with sources and provenance.

![UI](./ui.png)

---

## Table of Contents

1. [Motivation](#motivation)  
2. [High-Level Design](#high-level-design)  
3. [Architecture & Components](#architecture--components)  
4. [Installation](#installation)  
5. [Configuration](#configuration)  
6. [Usage](#usage)  
7. [Extending / Adding New Lore](#extending--adding-new-lore)  
8. [Answer Generation & Citation Policy](#answer-generation--citation-policy)  
9. [Evaluation & Metrics](#evaluation--metrics)  
10. [Development Workflow](#development-workflow)  
11. [Deployment (Free / No-Cost Mode)](#deployment-free--no-cost-mode)  
12. [Troubleshooting](#troubleshooting)  
13. [Future Work](#future-work)  
14. [Contributing](#contributing)  
15. [License](#license)

---

## Motivation

Warhammer 40K lore is vast, fragmented, and constantly referenced in casual conversation. The goal of **Astronomicon** is simple: stop repeating yourself.  
Ask a question. Get a concise, sourced answer immediately. No manual searching through codices, no fuzzy memory. Reliable retrieval with provenance, wrapped in a minimal, practical service.

---

## High-Level Design

Astronomicon is a **Retrieval-Augmented Generation (RAG)** service that layers:
- **Document retrieval** over a local lore corpus,
- Optional **semantic augmentation** via embeddings and an LLM synthesizer,
- **Answer assembly** with explicit source attribution.

It falls back gracefully to a lightweight, zero-key mode. Upgrade to richer embedding+LLM modes if desired.

---

## Architecture & Components

### Core paths

- **Data ingestion / corpus**: Local text files (Warhammer 40K compendium) are split into queryable chunks.
- **Retriever**: Finds relevant passages given a user question (vector or keyword based depending on mode).
- **Synthesizer (optional)**: Combines retrieved context into a coherent answer with optional LLM assistance.
- **Provenance layer**: Tracks and surfaces which source chunks contributed to each answer.
- **API layer**: FastAPI exposes endpoints for querying, health, and UI.

### Key modules

- `rag/` (or equivalent): Initialization, vector store management, retrieval logic, embedding handling.
- `endpoints/`: HTTP router exposing:
  - `/query/` — question input, returns answer + sources.
  - `/healthz` — liveness/version.
  - `/ui` — lightweight HTML frontend.
- `main.py`: Application bootstrap, lifespan warming of RAG backend.
- Embedding backend (optional): plugs into OpenAI or local embedding provider.
- Chat/answer synthesis backend (optional): uses an LLM to fuse retrieved context into fluent answers when enabled.

---

## Installation

Assumes Python 3.11+.

```bash
# clone repository
git clone <your-repo-url>
cd Astronomicon

# create venv and activate
python -m venv .venv
source .venv/bin/activate      # or `.venv\Scripts\activate` on Windows

# install dependencies
pip install -r requirements.txt

## OpenAI-enhanced Mode

If you want the OpenAI-enhanced mode:
```bash
export OPENAI_API_KEY=your_key_here
