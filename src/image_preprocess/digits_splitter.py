"""Module for splitting sudoku field into digits"""

from abc import ABC, abstractmethod
import datetime
from pathlib import Path

import numpy as np
import cv2 as cv

from src.utils import get_logger, save_debug_image


logger = get_logger(__name__)


class DigitsSplitter(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, with_not_nones: bool = False):
        raise NotImplementedError


class SimpleSplitter(DigitsSplitter):

    MARGIN = 0.05
    MIN_CNTR_AREA = 10

    def __init__(self):
        super().__init__()
        self.filestem: str = ""

    def split_into_digits(self, image: np.ndarray) -> list[np.ndarray | None]:
        height, width = image.shape
        # approximate cell size
        x_step = width / 9
        y_step = height / 9

        # some margins to remove rest of grid
        m_y = self.MARGIN * y_step
        m_x = self.MARGIN * x_step

        lst_digits: list[np.ndarray | None] = []

        for i_y, y in enumerate(np.arange(0, height, y_step)):
            for i_x, x in enumerate(np.arange(0, width, x_step)):

                y1 = int(y + m_y)
                y2 = int(y + y_step - m_y)

                x1 = int(x + m_x)
                x2 = int(x + x_step - m_x)

                roi = image[y1:y2, x1:x2].copy()

                cntrs, _ = cv.findContours(roi, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                cntrs_area = np.array([cv.contourArea(cntr) for cntr in cntrs])

                if len(cntrs) == 0 or len(cntrs) > 4 or cntrs_area.max() < self.MIN_CNTR_AREA:
                    digit = None
                else:
                    digit = roi

                save_debug_image(
                    roi, Path(self.filestem + f"_{i_y}_{i_x}_{digit is not None}").with_suffix(".jpg")
                )

                lst_digits.append(digit)

        return lst_digits

    def __call__(self, image: np.ndarray, with_not_nones: bool = False):

        file_id = hash(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        logger.info("Use %s as splitter. Recognition file id %s", self.__class__.__name__, file_id)

        self.filestem = f"{self.__class__.__name__}_{file_id}"
        save_debug_image(image, Path(self.filestem).with_suffix(".jpg"), "Image before splitting")

        all_digits = self.split_into_digits(image)
        if with_not_nones:
            not_none_digits = [d for d in all_digits if d is not None]
            return all_digits, not_none_digits
        else:
            return all_digits
