Finance RAG & Agent MVP

Overview
- Stage 1: Runnable RAG MVP over sample financial documents with a simple evaluator to compare retrieval‑grounded answers vs. unsupported ones.
- Stage 2: Scaffolding for LoRA/PEFT fine‑tuning on finance data (code stub + instructions).
- Stage 3: Agent scaffolding for ingestion, compliance, and report summarization.
- Stage 4: Monitoring + API licensing stubs (usage tracking, keys).

Quick Start (Stage 1)
- Requires: Python 3.9+ (no external packages needed).
- Run demo: `python3 scripts/demo_stage1.py`
 - Optional (Ollama-backed): `python3 scripts/demo_stage1_ollama.py`

What the demo does
- Loads and cleans sample financial domain documents.
- Builds a basic TF‑IDF vector store and retrieves top‑k chunks.
- Uses a local, deterministic “LLM” that composes answers from retrieved text with citations, minimizing hallucinations.
- Evaluates citation coverage and flags unsupported sentences.

Project Structure
- `data/raw/` sample regulatory filing, investment report, market commentary.
- `data/clean/` normalized copies produced by the loader.
- `src/rag/` chunking, vector store, and RAG pipeline.
- `src/llm/` local answer composer (LLM stub) + interface.
- `src/eval/` lightweight hallucination checks and metrics.
- `src/agents/` agent stubs for ingestion, compliance, summarization (Stage 3).
- `src/train/` LoRA/PEFT training stub and instructions (Stage 2).
- `src/server/` API usage/licensing/monitoring stubs (Stage 4).
- `scripts/` runnable demos and utilities.
 - `docs/END_TO_END.md` end‑to‑end documentation with diagrams.

Notes
- The Stage 1 MVP is dependency‑free and runs offline.
- For Stages 2–4, install the listed dependencies and connect your GPU/LLM APIs when ready.

Using Ollama (Local LLM)
- Install Ollama and pull a model, e.g.: `ollama pull llama3` (or `mistral`)
- Ensure the daemon is running (default: `http://localhost:11434`).
- Set model via env: `export OLLAMA_MODEL=llama3` (default: `llama3`).
- Run: `python3 scripts/demo_stage1_ollama.py`
- The prompt instructs the model to answer only from provided sources and cite them as [n].

Stage 2: Fine‑Tuning (Stub)
- File: `src/train/lora_finetune_stub.py`
- Usage: `python3 src/train/lora_finetune_stub.py --model mistral-7b --data path/to/jsonl`
- Replace with real training using HuggingFace Transformers + PEFT when your environment is ready (GPU, packages).

Stage 3: Agents (Stubs)
- Ingestion: `src/agents/ingestion_agent.py`
- Compliance: `src/agents/compliance_agent.py`
- Report: `src/agents/report_agent.py`
- Integrate these with the RAG pipeline for automated workflows.

Stage 4: API + Monitoring (Stubs)
- API server (dev): `python3 src/server/api_stub.py` then POST `http://localhost:8080/ask` with `{ "query": "..." }`
- Monitoring: `src/monitoring/metrics_stub.py` for simple counters/timings; replace with Prometheus/OpenTelemetry later.

Dashboard (Port 4000)
- Start server: `python3 src/server/rag_server.py --port 4000`
- Open: `http://localhost:4000/`
- Type a query and press Enter or click Ask. The UI shows the grounded answer, citations, evaluation metrics, and top contexts.
- Prefer a different port? Pass `--port 0` (or set `$PORT`) and the server will bind to an available port and print the chosen value. You can also set `--host` if you need to limit binding to `127.0.0.1`.
