"""
Logging configuration for the RAG API.

Uses Python's standard logging with structured output. We avoid third-party
logging frameworks (loguru, structlog) deliberately to keep dependencies minimal
and make the system portable. The format is structured so logs can be piped to
any aggregator (CloudWatch, Datadog, ELK) without changes.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: Optional[str] = "INFO") -> None:
    """
    Configure root logger with a consistent format across all modules.

    Format includes timestamp, level, module name, and message — sufficient
    for production log aggregation. Call once at application startup.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    else:
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Use __name__ as the name in each module."""
    return logging.getLogger(name)
