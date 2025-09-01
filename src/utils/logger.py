import logging
import sys
from src.config import settings

# Set this to control where your logs go
LOG_DIR = settings.LOG_DIR

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file paths
ALL_LOG_FILE = LOG_DIR / "all.log"
ERROR_LOG_FILE = LOG_DIR / "error.log"

# Logger setup
logger = logging.getLogger("app_logger")
logger.setLevel(logging.DEBUG)  # Log everything; handlers filter output

# Formatter
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s in %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler (for stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# File handler - all logs
file_handler_all = logging.FileHandler(ALL_LOG_FILE, mode='a', encoding='utf-8')
file_handler_all.setLevel(logging.DEBUG)
file_handler_all.setFormatter(formatter)

# File handler - only errors
file_handler_error = logging.FileHandler(ERROR_LOG_FILE, mode='a', encoding='utf-8')
file_handler_error.setLevel(logging.ERROR)
file_handler_error.setFormatter(formatter)

# Prevent adding multiple handlers in reload scenarios
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler_all)
    logger.addHandler(file_handler_error)

# Optional: silence overly verbose third-party loggers
for noisy_logger in ("uvicorn.access", "uvicorn.error"):
    logging.getLogger(noisy_logger).propagate = True