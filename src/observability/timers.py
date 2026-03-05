# src/observability/timers.py
from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


class Timers:
    def __init__(self) -> None:
        self.ms: Dict[str, float] = {}

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.ms[name] = (t1 - t0) * 1000.0

    def set_ms(self, name: str, value_ms: float) -> None:
        self.ms[name] = float(value_ms)