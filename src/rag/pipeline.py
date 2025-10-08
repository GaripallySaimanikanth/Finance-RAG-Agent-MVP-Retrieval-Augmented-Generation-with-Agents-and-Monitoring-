from typing import Dict, List
from .vector_store import DocumentStore
from llm.local import LocalAnswerComposer


class RagPipeline:
    def __init__(self, store: DocumentStore, answerer: LocalAnswerComposer):
        self.store = store
        self.answerer = answerer

    def answer(self, query: str, top_k: int = 4) -> Dict:
        hits = self.store.query(query, top_k=top_k)
        contexts = []
        for score, doc in hits:
            contexts.append({
                "score": score,
                "text": doc["text"],
                "source": doc["meta"].get("source", "unknown")
            })
        composed = self.answerer.compose(query, contexts)
        return {
            "query": query,
            "answer": composed["answer"],
            "citations": composed["citations"],
            "contexts": contexts,
        }

