import logging
from os import PathLike
from pathlib import Path

LOGGER_NAME = __name__

logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # prevent duplicate logs via root

if not logger.handlers:  # avoid re-adding on reload
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

PROJECT_DIR = Path(__file__).parents[2]
FIGURES_DIR = PROJECT_DIR / "figures"
DATA_DIR = PROJECT_DIR / "data"
