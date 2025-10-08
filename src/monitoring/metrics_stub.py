from time import time
from typing import Dict


class MetricsStub:
    """Stage 4: Minimal in-process metrics collector."""

    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.timings: Dict[str, float] = {}

    def inc(self, key: str, n: int = 1):
        self.counters[key] = self.counters.get(key, 0) + n

    def time(self, key: str):
        start = time()
        def stop():
            self.timings[key] = self.timings.get(key, 0.0) + (time() - start)
        return stop

    def snapshot(self) -> Dict:
        return {"counters": dict(self.counters), "timings": dict(self.timings)}

