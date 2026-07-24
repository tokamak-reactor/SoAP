"""Simple logging for SOLPS-ITER package."""

from __future__ import annotations

import sys

LOG_LEVELS = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "DEBUG"}


def log_msg(level: int, current_level: int = 2, msg: str = "", *args) -> None:
    """Log a message if level <= current_level.

    Levels: 1=ERROR, 2=WARN, 3=INFO, 4=DEBUG
    """
    if level <= current_level:
        tag = LOG_LEVELS.get(level, "INFO")
        if args:
            msg = msg % args
        print(f"[{tag}] {msg}", file=sys.stderr)
