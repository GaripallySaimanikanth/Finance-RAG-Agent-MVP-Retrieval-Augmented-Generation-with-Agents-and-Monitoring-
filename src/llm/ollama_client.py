import json
import os
import re
from typing import Dict, List, Optional
from urllib import request, error


class OllamaClient:
    def __init__(self, host: str = None):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def generate(self, model: str, prompt: str, options: Optional[Dict] = None, timeout: int = 120) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if options:
            payload["options"] = options
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                obj = json.loads(body.decode("utf-8"))
                return obj.get("response", "")
        except error.URLError as e:
            raise RuntimeError(f"Failed to reach Ollama at {url}: {e}")


class OllamaAnswerer:
    def __init__(self, model: str = None, temperature: float = 0.1):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3")
        self.client = OllamaClient()
        self.temperature = temperature

    def _build_prompt(self, query: str, contexts: List[Dict]) -> str:
        lines = [
            "You are a financial RAG assistant. Answer ONLY using the provided sources.",
            "If the answer is not contained in the sources, say: 'I don't know based on the provided sources.'",
            "Cite sources with bracketed numbers like [1], [2] corresponding to the source list.",
            "Be concise and factual. No speculation.",
            "\nQuestion:",
            query,
            "\nSources:",
        ]
        for i, c in enumerate(contexts, 1):
            # Truncate overly long contexts
            txt = c["text"]
            if len(txt) > 1200:
                txt = txt[:1200] + "…"
            lines.append(f"[{i}] ({c['source']})\n{txt}")
        lines.append("\nAnswer (with citations):")
        return "\n".join(lines)

    def _extract_citations(self, answer: str, contexts: List[Dict]) -> List[Dict]:
        cited = []
        for m in re.finditer(r"\[(\d+)\]", answer):
            idx = int(m.group(1))
            if 1 <= idx <= len(contexts):
                cited.append({"source": contexts[idx - 1]["source"], "line": 1})
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for c in cited:
            key = (c["source"], c["line"]) 
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        return uniq

    def compose(self, query: str, contexts: List[Dict]) -> Dict:
        prompt = self._build_prompt(query, contexts)
        options = {"temperature": self.temperature}
        text = self.client.generate(self.model, prompt, options=options)
        citations = self._extract_citations(text, contexts)
        return {"answer": text, "citations": citations}

