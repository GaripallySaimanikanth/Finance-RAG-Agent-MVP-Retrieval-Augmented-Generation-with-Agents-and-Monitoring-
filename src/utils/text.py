import re
from typing import List


def normalize_whitespace(s: str) -> str:
    s = s.replace("\u00A0", " ")
    s = re.sub(r"[\t\r\f]+", " ", s)
    s = re.sub(r" +", " ", s)
    s = re.sub(r" *\n+ *", "\n", s)
    return s.strip()


def split_sentences(text: str) -> List[str]:
    # Simple sentence splitter: split on ".", "?", "!" while preserving basic abbreviations.
    # Not perfect, but good enough for the MVP.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    return [p.strip() for p in parts if p.strip()]

