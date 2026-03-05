"""
Centralized logging configuration.

Created by: Alex Myasnikov
Date: February 10, 2026
"""

import logging
import sys
import json
from logging.handlers import QueueHandler
from typing import Optional
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """Format log records as JSON with support for custom fields."""

    # Standard LogRecord attributes to exclude from custom fields
    RESERVED_ATTRS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName', 'levelname',
        'levelno', 'lineno', 'module', 'msecs', 'message', 'pathname', 'process',
        'processName', 'relativeCreated', 'thread', 'threadName', 'exc_info',
        'exc_text', 'stack_info', 'taskName'
    }

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name
        }

        # Add custom fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in self.RESERVED_ATTRS:
                log_data[key] = value

        # Add message only if no custom data provided
        if not any(k not in {'timestamp', 'level', 'logger'} for k in log_data.keys()):
            log_data['message'] = record.getMessage()

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def configure_logging(log_file: Optional[str] = None) -> None:
    """
    Configure global logging settings. Always outputs JSON format.

    Args:
        log_file: If provided, log to file only (no stdout)
                  If None, log to stdout
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Always use JSON formatter
    formatter = JsonFormatter()

    if log_file:
        # File handler only
        handler = logging.FileHandler(log_file, mode='a')
    else:
        # Stdout handler
        handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def configure_logging_queue(queue) -> None:
    """Configure worker-process logging to send records to a shared queue."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(queue))


def get_logger(name: str) -> logging.Logger:
    """
    Get configured logger for given module name.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured globally
    if not logging.getLogger().handlers:
        configure_logging()

    return logger


def log_structured(logger: logging.Logger, level: str, **fields) -> None:
    """
    Log structured data as custom JSON fields.

    Args:
        logger: Logger instance
        level: Log level ('info', 'warning', 'error', 'debug')
        **fields: Arbitrary key-value pairs to log as JSON fields

    Example:
        log_structured(
            logger, 'info',
            event='relator_processed',
            relator_name='R1',
            original_length=100,
            new_length=85,
            reduction=15
        )
    """
    log_method = getattr(logger, level.lower())
    # Empty message, all data in extra
    log_method('', extra=fields)
