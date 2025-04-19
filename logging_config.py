#!/usr/bin/env python3
import os
import sys
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

# Global dict to track loggers we've created
_loggers: Dict[str, logging.Logger] = {}

def configure_logging(
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Configure the logging system for the application.
    Creates the log directory if it doesn't exist.
    
    Args:
        log_dir: Directory to store log files
        console_level: Logging level for console output
        file_level: Logging level for file output
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Define log file paths
    main_log = log_path / "ai_core_run.log"
    error_log = log_path / "errors.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs
    
    # Remove any existing handlers to prevent duplicates on reconfiguration
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    default_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
    detailed_format = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(funcName)s:%(lineno)d | %(message)s"
    
    default_formatter = logging.Formatter(default_format)
    detailed_formatter = logging.Formatter(detailed_format)
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(default_formatter)
    root_logger.addHandler(console)
    
    # File handler for all logs
    file_handler = RotatingFileHandler(
        main_log, 
        maxBytes=max_bytes, 
        backupCount=backup_count
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # File handler for errors only
    error_handler = RotatingFileHandler(
        error_log, 
        maxBytes=max_bytes, 
        backupCount=backup_count
    )
    error_handler.setLevel(logging.WARNING)  # WARNING and above
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Log initialization
    root_logger.info(f"Logging initialized. Main log: {main_log}, Error log: {error_log}")


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    Ensures we don't create duplicate loggers.
    
    Args:
        name: Logger name, typically __name__ from the module
        
    Returns:
        A configured logger instance
    """
    global _loggers
    
    # Return existing logger if we've created it before
    if name in _loggers:
        return _loggers[name]
        
    # Create a new logger
    logger = logging.getLogger(name)
    
    # Store for future use
    _loggers[name] = logger
    
    return logger


if __name__ == "__main__":
    """Simple test when run directly"""
    # Configure logging
    configure_logging()
    
    # Get a test logger
    logger = get_logger("LoggingTest")
    
    # Test all log levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    try:
        # Generate an exception
        x = 1 / 0
    except Exception as e:
        logger.exception(f"Caught exception: {e}")
    
    print("Logging test complete. Check logs directory for output files.")
