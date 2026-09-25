"""Structured logging setup with Loguru."""

import logging
import sys
from contextvars import ContextVar
from pathlib import Path

from loguru import logger

correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

LOG_FORMAT_JSON = (
    '{"time": "{time}", "level": "{level}", "module": "{name}", '
    '"function": "{function}", "line": {line}, "message": {message}, '
    '"correlation_id": "' + "{extra[correlation_id]}" + '"}'
)


def setup_logging(
    level: str = "INFO",
    json: bool = False,
    log_file: str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure Loguru logger with structured output."""

    # Remove default handler
    logger.remove()

    # Determine format
    fmt = format_string or LOG_FORMAT

    # Console output
    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=not json,
        backtrace=True,
        diagnose=False,
    )

    # File output
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if json:
            logger.add(
                log_path,
                format="{message}",
                level=level,
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                serialize=True,
                backtrace=True,
                diagnose=False,
            )
        else:
            logger.add(
                log_path,
                format=fmt,
                level=level,
                rotation="100 MB",
                retention="30 days",
                compression="gz",
                backtrace=True,
                diagnose=False,
            )

    # Intercept standard logging
    class InterceptHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            level_name = record.levelname
            if record.name:
                name = record.name
            else:
                name = "root"
            try:
                logger.log(
                    level_name,
                    record.getMessage(),
                    name=name,
                    function=record.funcName or "?",
                    line=record.lineno or 0,
                )
            except Exception:
                pass

    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(InterceptHandler())

    # Quiet down noisy libraries
    for noisy in ["httpx", "httpcore", "charset_normalizer", "favicon"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def add_correlation_id(correlation_id: str) -> None:
    """Attach a correlation ID to the current context."""
    correlation_id_var.set(correlation_id)


def get_correlation_id() -> str:
    """Get the current correlation ID from context."""
    return correlation_id_var.get()


class LogMixin:
    """Mixin class that provides a logger property."""

    @property
    def log(self):
        return logger.bind(correlation_id=get_correlation_id())


def configure_from_settings(settings) -> None:
    """Configure logging from Settings object."""
    setup_logging(
        level=settings.log_level,
        json=settings.log_json,
        log_file=settings.log_file,
    )
