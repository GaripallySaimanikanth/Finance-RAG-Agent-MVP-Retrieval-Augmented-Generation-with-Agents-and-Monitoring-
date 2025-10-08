from typing import Dict, List
from utils.text import split_sentences


class LocalAnswerComposer:
    """
    A deterministic, citation-focused answer composer that selects
    sentences from retrieved context and lightly summarizes to reduce
    hallucinations. It prefers quoting directly, with source markers.
    """

    def compose(self, query: str, contexts: List[Dict]) -> Dict:
        # Rank sentences by overlap with query terms, pick top few.
        q_terms = set(w.lower() for w in query.split() if len(w) > 2)
        scored = []
        for ctx in contexts:
            for i, sent in enumerate(split_sentences(ctx["text"])):
                terms = set(w.lower() for w in sent.split() if len(w) > 2)
                score = len(q_terms & terms) + 1e-6 * (len(terms))
                if score > 0:
                    scored.append((score, sent.strip(), ctx["source"], i + 1))

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = scored[:4] if scored else []

        if not picked and contexts:
            # Fallback: take the first sentence of the top context
            top_ctx = contexts[0]
            first = split_sentences(top_ctx["text"])[:1]
            picked = [(0.0, first[0], top_ctx["source"], 1)] if first else []

        lines = []
        citations = []
        for _, s, src, line_no in picked:
            lines.append(s)
            citations.append({"source": src, "line": line_no})

        if lines:
            answer = " ".join(lines)
        else:
            answer = "No directly supported answer was found in the provided documents."

        # Light prompt-like header
        answer = (
            "Answer (grounded in retrieved sources):\n" + answer
        )

        return {"answer": answer, "citations": citations}

