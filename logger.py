"""
Logging configuration module for the multi-agent platform.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from .config import config


class Logger:
    """Centralized logging manager."""

    _loggers = {}

    @staticmethod
    def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get or create a logger with the specified name.

        Args:
            name: Logger name (typically __name__)
            log_file: Optional specific log file path

        Returns:
            Configured logger instance
        """
        if name in Logger._loggers:
            return Logger._loggers[name]

        logger = logging.getLogger(name)

        # Prevent duplicate handlers
        if logger.handlers:
            return logger

        # Get log level from config
        log_level = config.get('logging.level', 'INFO')
        logger.setLevel(getattr(logging, log_level))

        # Create formatters
        log_format = config.get(
            'logging.format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        formatter = logging.Formatter(log_format)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file is None:
            log_file = config.get('logging.file', './logs/app.log')

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = config.get('logging.max_bytes', 10485760)  # 10MB
        backup_count = config.get('logging.backup_count', 5)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        Logger._loggers[name] = logger
        return logger


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger."""
    return Logger.get_logger(name)
