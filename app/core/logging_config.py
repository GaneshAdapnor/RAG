"""Compatibility wrapper for logging configuration helpers."""

import logging

from app.utils.logging import configure_logging


def setup_logging(level: str | None = "INFO") -> None:
    configure_logging(level or "INFO")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
