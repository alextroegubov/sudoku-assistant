import numpy as np
import logging
from pathlib import Path
import cv2 as cv
import datetime


def cls_output_to_grid(confs, labels, all_digits, threshold=0.8):
    """Convert classifier output to numpy sudoku grid"""
    sudoku_grid = np.zeros(81, dtype=int)
    label_idx = 0
    for i, digit in enumerate(all_digits):
        if digit is not None:
            if confs[label_idx] > threshold:
                sudoku_grid[i] = labels[label_idx] + 1
            label_idx += 1

    return sudoku_grid.reshape((9, 9))


def get_image_hash(image: np.ndarray) -> int:
    """Get custom hash of image for logging files"""
    data_hash = hash(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    image_hash = hash(np.sum(image.astype(np.uint64)))

    return hash((data_hash, image_hash))


# Logging setup
LOG_DIR = Path.cwd() / "logs"
LOG_FILE = LOG_DIR / "sudoku-assistant.log"
IMG_LOG_DIR = LOG_DIR / "images"

if IMG_LOG_DIR.exists():
    for file in IMG_LOG_DIR.glob('*'):
        file.unlink()
    IMG_LOG_DIR.rmdir()

IMG_LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s: %(message)s"


def setup_logging():
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding='utf-8'),
            # logging.StreamHandler(),
        ],
    )


def get_logger(module_name: str):
    """Returns a logger for a given module name."""
    return logging.getLogger(module_name)


def save_debug_image(image: np.ndarray, filename: Path, logging_level: int, description: str = ""):
    """Save an image for debugging and log the action."""
    logger = get_logger(__name__)  # Get logger for this module

    if not logger.isEnabledFor(logging_level):
        return

    if image is None:
        logger.warning(f"Attempted to save a None image: {filename.name}")
        return

    save_path = IMG_LOG_DIR / filename
    success = cv.imwrite(str(save_path), image)

    if success:
        logger.info(f"Saved image: {save_path} - {description}")
    else:
        logger.error(f"Failed to save image: {filename.name}")


setup_logging()
