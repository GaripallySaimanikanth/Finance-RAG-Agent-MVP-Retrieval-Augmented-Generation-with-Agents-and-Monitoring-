import re
from typing import Dict, List


def _tokenize_simple(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def support_coverage(answer: str, sources: List[str]) -> float:
    """Fraction of answer tokens present in any source string."""
    a_toks = set(_tokenize_simple(answer))
    if not a_toks:
        return 0.0
    s_toks = set()
    for s in sources:
        s_toks.update(_tokenize_simple(s))
    return len(a_toks & s_toks) / max(1, len(a_toks))


def unsupported_sentences(answer: str, sources: List[str]) -> List[str]:
    """Return sentences with low lexical overlap to sources (likely hallucinations)."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    bad = []
    s_toks = set()
    for s in sources:
        s_toks.update(_tokenize_simple(s))
    for sent in sentences:
        a_toks = set(_tokenize_simple(sent))
        if not a_toks:
            continue
        overlap = len(a_toks & s_toks) / max(1, len(a_toks))
        if overlap < 0.35:  # heuristic threshold
            bad.append(sent)
    return bad


def evaluate_answer(answer: str, sources: List[str]) -> Dict:
    cov = support_coverage(answer, sources)
    bad = unsupported_sentences(answer, sources)
    return {
        "support_coverage": cov,
        "unsupported_sentence_count": len(bad),
        "unsupported_sentences": bad,
    }

