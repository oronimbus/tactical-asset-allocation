"""Configure package level logger."""

import logging


def setup_logger():
    """Set up basic configuration of logger for consistent format."""
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
        level=logging.INFO,
    )
