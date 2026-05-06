"""Logging utilities for detailed Genie3 run logs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


LOGGER_NAME = "genie3"


def _is_primary_process() -> bool:
    """Return True for the primary CLI process/rank that should log to stdout."""
    local_rank = os.environ.get("LOCAL_RANK")
    rank = os.environ.get("RANK")
    if local_rank is not None:
        return local_rank == "0"
    if rank is not None:
        return rank == "0"
    return True


def configure_run_logger(
    *,
    log_file: Path,
    verbose: bool,
) -> logging.Logger:
    """Configure and return the per-run Genie3 logger."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)

    if _is_primary_process():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO if not verbose else logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Flush, close, and detach all handlers from `logger`."""
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
