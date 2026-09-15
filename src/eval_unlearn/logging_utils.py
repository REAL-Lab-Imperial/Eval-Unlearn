import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger with standard formatting.

    Args:
        name: The name of the logger (typically __name__).
        level: The logging level (default: logging.INFO).

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if get_logger is called repeatedly
    if not logger.handlers:
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Prevent propagation to root logger to avoid duplicate logs if root is configured
        logger.propagate = False

    return logger
