from pathlib import Path
from typing import List, Dict
from utils.text import normalize_whitespace


def load_and_clean(raw_dir: Path, clean_dir: Path) -> List[Dict]:
    docs = []
    for path in sorted(raw_dir.glob("**/*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        clean = normalize_whitespace(text)
        out = clean_dir / path.name
        out.write_text(clean, encoding="utf-8")
        docs.append({"path": str(out), "content": clean})
    return docs

