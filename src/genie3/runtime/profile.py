"""Lightweight runtime profiling utilities for stage timing."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class RuntimeProfile:
    """Collect simple elapsed-time measurements for a run."""

    timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure elapsed wall-clock time for a named stage."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (
                time.perf_counter() - start_time
            )

    def to_dict(self) -> dict[str, float]:
        """Return timing data rounded for stable metadata/log output."""
        return {name: round(seconds, 6) for name, seconds in self.timings.items()}
