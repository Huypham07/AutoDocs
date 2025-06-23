import logging
import os
from pathlib import Path

def setup_logging(format: str = None):
    # log directory
    base_dir = Path(__file__).parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    default_log_file = log_dir / "application.log"

    # Get and validate log level
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_file_path = Path(os.environ.get(
        "LOG_FILE_PATH", str(default_log_file)))

    log_dir_resolved = log_dir.resolve()
    resolved_path = log_file_path.resolve()
    if not str(resolved_path).startswith(str(log_dir_resolved) + os.sep):
        raise ValueError(
            f"LOG_FILE_PATH '{log_file_path}' is outside the trusted log directory '{log_dir_resolved}'"
        )
    # Ensure parent dirs exist for the log file
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging handlers
    default_format = "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    log_format = format or default_format
    
    logging.basicConfig(
        level=log_level,
        format = log_format,
        handlers=[
            logging.FileHandler(resolved_path),
            logging.StreamHandler()
        ],
        force=True
    )

    # Initial debug message to confirm configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level_str}, File: {resolved_path}")

