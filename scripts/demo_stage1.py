#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from data.loader import load_and_clean
from rag.vector_store import DocumentStore
from rag.pipeline import RagPipeline
from llm.local import LocalAnswerComposer
from eval.metrics import evaluate_answer


def main():
    raw_dir = ROOT / "data" / "raw"
    clean_dir = ROOT / "data" / "clean"
    os.makedirs(clean_dir, exist_ok=True)

    print("[Stage 1] Loading and cleaning documents…")
    docs = load_and_clean(raw_dir, clean_dir)
    print(f"Loaded {len(docs)} documents; creating chunks and index…")

    store = DocumentStore(chunk_size=600, chunk_overlap=80)
    store.fit([d["content"] for d in docs], meta=[{"source": d["path"]} for d in docs])

    rag = RagPipeline(store=store, answerer=LocalAnswerComposer())

    queries = [
        "What were the key risk factors disclosed?",
        "Summarize the investment strategy described.",
        "What market outlook was discussed regarding inflation?",
    ]

    for q in queries:
        print("\n=== Query ===\n" + q)
        result = rag.answer(q, top_k=4)
        print("\n--- Answer ---\n" + result["answer"]) 
        print("\n--- Citations ---")
        for c in result["citations"]:
            print(f"- {c['source']} #L{c['line']}")

        metrics = evaluate_answer(result["answer"], [c["text"] for c in result["contexts"]])
        print("\n--- Evaluation ---")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.3f}")
            else:
                print(f"{k}: {v}")


if __name__ == "__main__":
    main()

