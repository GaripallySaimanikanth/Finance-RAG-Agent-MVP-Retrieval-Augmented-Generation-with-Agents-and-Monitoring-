from typing import List, Dict


def make_chunks(text: str, size: int = 600, overlap: int = 80) -> List[Dict]:
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + size)
        chunk_tokens = tokens[start:end]
        chunk = " ".join(chunk_tokens).strip()
        if chunk:
            chunks.append({"text": chunk, "start_token": start, "end_token": end})
        if end == len(tokens):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

