from typing import Dict, List


class ReportAgent:
    """Stage 3: Summarizes retrieved contexts into a simple structured report."""

    def build_report(self, query: str, contexts: List[Dict], answer: str) -> str:
        lines = [
            f"Query: {query}",
            "\nTop Context Snippets:",
        ]
        for i, c in enumerate(contexts[:3], 1):
            lines.append(f"[{i}] ({c['source']}) {c['text'][:240]}…")
        lines.append("\nAnswer:\n" + answer)
        return "\n".join(lines)

