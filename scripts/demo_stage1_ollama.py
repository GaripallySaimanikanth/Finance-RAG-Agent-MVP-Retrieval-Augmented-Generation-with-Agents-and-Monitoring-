#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data.loader import load_and_clean
from rag.vector_store import DocumentStore
from rag.pipeline import RagPipeline
from llm.ollama_client import OllamaAnswerer
from eval.metrics import evaluate_answer


def main():
    raw_dir = ROOT / "data" / "raw"
    clean_dir = ROOT / "data" / "clean"
    os.makedirs(clean_dir, exist_ok=True)

    print("[Stage 1 + Ollama] Loading and cleaning documents…")
    docs = load_and_clean(raw_dir, clean_dir)
    print(f"Loaded {len(docs)} documents; creating chunks and index…")

    store = DocumentStore(chunk_size=600, chunk_overlap=80)
    store.fit([d["content"] for d in docs], meta=[{"source": d["path"]} for d in docs])

    model = os.environ.get("OLLAMA_MODEL", "llama3")
    print(f"Using Ollama model: {model}")
    rag = RagPipeline(store=store, answerer=OllamaAnswerer(model=model))

    queries = [
        "What were the key risk factors disclosed?",
        "Summarize the investment strategy described.",
        "What market outlook was discussed regarding inflation?",
    ]

    for q in queries:
        print("\n=== Query ===\n" + q)
        try:
            result = rag.answer(q, top_k=4)
        except Exception as e:
            print("Error calling Ollama:", e)
            print("Is Ollama running and the model pulled? See README for setup.")
            return

        print("\n--- Answer ---\n" + result["answer"]) 
        print("\n--- Citations (parsed [n]) ---")
        if result["citations"]:
            for c in result["citations"]:
                print(f"- {c['source']} #L{c['line']}")
        else:
            print("(no explicit [n] citations detected in model output)")

        metrics = evaluate_answer(result["answer"], [c["text"] for c in result["contexts"]])
        print("\n--- Evaluation ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.3f}")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()

