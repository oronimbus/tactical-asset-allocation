"""Configure package level logger."""
import logging


def setup_logger():
    """Basic configuration of logger for consistent format."""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
        level=logging.INFO,
    )
