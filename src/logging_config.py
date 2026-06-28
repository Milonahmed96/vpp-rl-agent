"""Centralised logging configuration for the VPP RL agent.

Every module obtains its logger via :func:`get_logger` so that the whole
project shares a single, consistently-formatted logging setup. ``print`` is
never used anywhere in the codebase.
"""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once for the whole process.

    Args:
        level: Optional logging level name (e.g. ``"INFO"``). When ``None`` the
            ``LOG_LEVEL`` environment variable is consulted, defaulting to
            ``"INFO"``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(level=resolved, format=_LOG_FORMAT)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    configure_logging()
    return logging.getLogger(name)
