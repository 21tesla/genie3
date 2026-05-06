"""Lightweight terminal progress/status reporting for Genie3 CLI runs."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from typing import Iterator


class ProgressReporter:
    """Minimal in-place status reporter for the primary terminal process."""

    def __init__(self) -> None:
        self._status: str | None = None
        self._last_rendered_width = 0
        self._stream = self._resolve_stream()
        self._enabled = self._should_enable()

    def _is_primary_process(self) -> bool:
        local_rank = os.environ.get("LOCAL_RANK")
        rank = os.environ.get("RANK")
        if local_rank is not None:
            return local_rank == "0"
        if rank is not None:
            return rank == "0"
        return True

    def _should_enable(self) -> bool:
        if not self._is_primary_process():
            return False
        return self._stream is not None

    def _resolve_stream(self):
        if not self._is_primary_process():
            return None
        try:
            return open("/dev/tty", "w")
        except OSError:
            if sys.stdout.isatty():
                return sys.stdout
            return None

    def set_status(self, message: str) -> None:
        if not self._enabled:
            return
        rendered = f"   Status: {message}"
        padding = max(self._last_rendered_width - len(rendered), 0)
        assert self._stream is not None
        self._stream.write("\r" + rendered + (" " * padding))
        self._stream.flush()
        self._status = message
        self._last_rendered_width = len(rendered)

    def clear_status(self) -> None:
        if not self._enabled or self._status is None:
            return
        assert self._stream is not None
        self._stream.write("\r" + (" " * self._last_rendered_width) + "\r")
        self._stream.flush()
        self._status = None
        self._last_rendered_width = 0

    def start_stage(self, name: str, total: int | None = None) -> None:
        del name, total

    def advance(self, amount: int = 1, message: str | None = None) -> None:
        del amount, message

    def finish_stage(self, name: str) -> None:
        del name

    @contextmanager
    def stage(self, name: str, total: int | None = None) -> Iterator[None]:
        self.start_stage(name, total=total)
        try:
            yield
        finally:
            self.finish_stage(name)

    def close(self) -> None:
        """Release progress resources."""
        self.clear_status()
        if self._stream not in {None, sys.stdout, sys.stderr}:
            self._stream.close()
        self._stream = None


_ACTIVE_REPORTER: ProgressReporter | None = None


def set_active_reporter(reporter: ProgressReporter | None) -> None:
    global _ACTIVE_REPORTER
    _ACTIVE_REPORTER = reporter


def get_active_reporter() -> ProgressReporter:
    global _ACTIVE_REPORTER
    if _ACTIVE_REPORTER is None:
        _ACTIVE_REPORTER = ProgressReporter()
    return _ACTIVE_REPORTER
