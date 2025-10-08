import re
from typing import List, Dict


class ComplianceAgent:
    """Stage 3: Simple compliance checks (keywords, disclosure presence)."""

    def __init__(self, required_disclaimers: List[str] = None):
        self.required = required_disclaimers or [
            "not investment advice",
            "past performance is not indicative",
        ]

    def check(self, answer: str) -> Dict:
        lower = answer.lower()
        missing = [d for d in self.required if d not in lower]
        has_tickers = bool(re.search(r"\b[A-Z]{2,5}\b", answer))
        return {
            "missing_disclaimers": missing,
            "contains_tickers": has_tickers,
        }

