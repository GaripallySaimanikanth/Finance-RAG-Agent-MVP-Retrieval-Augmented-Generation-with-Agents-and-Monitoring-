from pathlib import Path
from typing import List
from data.loader import load_and_clean


class IngestionAgent:
    """Stage 3: Pulls documents from a folder or source and refreshes the index."""

    def __init__(self, raw_dir: Path, clean_dir: Path):
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir

    def run(self) -> List[dict]:
        docs = load_and_clean(self.raw_dir, self.clean_dir)
        return docs

