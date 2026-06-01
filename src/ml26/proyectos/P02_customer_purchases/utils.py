import logging
from datetime import datetime
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
LOGS_DIR = CURRENT_FILE.parent / "logs"


def setup_logger(name: str, log_dir: Path = LOGS_DIR) -> logging.Logger:
    """
    Creates a logger that writes both to console and file.
    """

    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"

    logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # Prevent duplicated handlers
    if logger.handlers:
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Logger initialized at: {log_file}")

    return logger