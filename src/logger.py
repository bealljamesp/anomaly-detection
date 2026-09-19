from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "anomaly_detection") -> logging.Logger:
    """Configure and return a production-grade structured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console Handler
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        c_handler.setFormatter(c_format)
        logger.addHandler(c_handler)

        # File Handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        f_handler = logging.FileHandler(log_dir / "execution.log")
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        )
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger
