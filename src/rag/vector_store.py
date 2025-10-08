import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from .chunker import make_chunks


def _tokenize(text: str) -> List[str]:
    # Lowercase alphanumerics; minimal tokenizer
    out = []
    w = []
    for ch in text.lower():
        if ch.isalnum():
            w.append(ch)
        else:
            if w:
                out.append("".join(w))
                w = []
    if w:
        out.append("".join(w))
    return out


class DocumentStore:
    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs: List[Dict] = []
        self.vocab_df = defaultdict(int)
        self.num_docs = 0
        self.vectors: List[Dict[str, float]] = []  # sparse tf-idf

    def _tf(self, tokens: List[str]) -> Dict[str, float]:
        c = Counter(tokens)
        total = sum(c.values()) or 1
        return {t: v / total for t, v in c.items()}

    def _idf(self, term: str) -> float:
        # smoothed idf
        df = self.vocab_df.get(term, 0) + 1
        return math.log((self.num_docs + 1) / df) + 1.0

    def _tfidf(self, tokens: List[str]) -> Dict[str, float]:
        tf = self._tf(tokens)
        return {t: tf_v * self._idf(t) for t, tf_v in tf.items()}

    def fit(self, documents: List[str], meta: List[Dict] = None):
        self.docs = []
        self.vectors = []
        self.vocab_df = defaultdict(int)
        self.num_docs = 0

        meta = meta or [{} for _ in documents]
        for i, (doc, m) in enumerate(zip(documents, meta)):
            chunks = make_chunks(doc, self.chunk_size, self.chunk_overlap)
            for ch in chunks:
                tokens = _tokenize(ch["text"]) 
                # update df
                for t in set(tokens):
                    self.vocab_df[t] += 1
                self.num_docs += 1
                self.docs.append({"text": ch["text"], "meta": m})
                self.vectors.append({"__tokens__": tokens})  # placeholder for now

        # finalize tf-idf using df
        for v in self.vectors:
            tokens = v.pop("__tokens__")
            tfidf = self._tfidf(tokens)
            v.update(tfidf)

    def _sim(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        # cosine similarity for sparse dicts
        if not v1 or not v2:
            return 0.0
        dot = 0.0
        for k, a in v1.items():
            b = v2.get(k)
            if b is not None:
                dot += a * b
        n1 = math.sqrt(sum(a * a for a in v1.values())) or 1e-12
        n2 = math.sqrt(sum(b * b for b in v2.values())) or 1e-12
        return dot / (n1 * n2)

    def query(self, text: str, top_k: int = 4) -> List[Tuple[float, Dict]]:
        q_vec = self._tfidf(_tokenize(text))
        sims = [(self._sim(q_vec, v), d) for v, d in zip(self.vectors, self.docs)]
        sims.sort(key=lambda x: x[0], reverse=True)
        return sims[:top_k]

