import logging
from enum import StrEnum

# Define log formats
LOG_FORMAT_DEBUG = "[%(levelname)s]:\t%(message)s:%(pathname)s:%(funcName)s:%(lineno)d"
LOG_FORMAT_DEFAULT = "%(asctime)s - %(levelname)s - %(message)s"

# Define log levels as an enum for reference
class LogLevels(StrEnum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"

def configure_logging(log_level: str = "ERROR", log_file: str = None):
    """Configure logging with a specified log level and optional file output."""
    log_level = log_level.upper()
    if log_level not in logging._nameToLevel:
        log_level = "ERROR"
    log_format = LOG_FORMAT_DEBUG if log_level == "DEBUG" else LOG_FORMAT_DEFAULT
    handlers = [logging.StreamHandler()]  # Console output
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    # Clear existing handlers to avoid duplicates
    logging.getLogger().handlers = []
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)