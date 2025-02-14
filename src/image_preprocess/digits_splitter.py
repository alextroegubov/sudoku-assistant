"""Module for splitting sudoku field into digits"""

from abc import ABC, abstractmethod
from pathlib import Path
import logging

import numpy as np
import cv2 as cv

from src.utils import get_logger, save_debug_image, get_image_hash


logger = get_logger(__name__)


class DigitsSplitter(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray, with_not_nones: bool = False):
        raise NotImplementedError


class SimpleSplitter(DigitsSplitter):

    # relative to cell width
    MARGIN = 0.05
    # relative to cell area
    MIN_AREA_COEFF = 1 / 36
    # relative to cell height
    MIN_ARC_COEFF = 1.0
    WHITE = 255

    def __init__(self):
        super().__init__()
        self.filestem: str = ""

    def split_into_digits(self, image: np.ndarray) -> list[np.ndarray | None]:
        height, width = image.shape
        # approximate cell size
        x_step = width / 9
        y_step = height / 9

        lst_digits = []

        for i_y, y in enumerate(np.arange(0, height, y_step)):
            for i_x, x in enumerate(np.arange(0, width, x_step)):

                proc_roi, digit = self._extract_digit(image, x, y)

                save_debug_image(
                    proc_roi,
                    Path(self.filestem + f"_{i_y}_{i_x}_{digit is not None}").with_suffix(".jpg"),
                    logging.DEBUG,
                )
                lst_digits.append(digit)

        save_debug_image(
            image, Path(self.filestem + f"_roi_correction").with_suffix(".jpg"), logging.DEBUG
        )

        return lst_digits

    def _extract_digit(
        self, image: np.ndarray, x: float, y: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extracts a single digit from a given cell region."""
        # cell size
        y_step, x_step = image.shape[0] / 9, image.shape[1] / 9
        # margins
        m_y, m_x = self.MARGIN * y_step, self.MARGIN * x_step
        # roi rectangle
        x1, y1 = int(x + m_x), int(y + m_y)
        x2, y2 = int(x + x_step - m_x), int(y + y_step - m_y)
        # binary image here after extractor
        roi = image[y1:y2, x1:x2].copy()
        roi = self._remove_noise(roi)

        if logger.isEnabledFor(logging.DEBUG):
            image[y1:y2, x1:x2] = roi

        cntrs, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        digit = None if len(cntrs) == 0 or len(cntrs) > 4 else roi

        return roi, digit

    def _remove_noise(self, roi: np.ndarray) -> np.ndarray:
        """Removes small noise elements from the image using contours filtering."""
        min_area = self.MIN_AREA_COEFF * np.prod(roi.shape)
        min_arc = self.MIN_ARC_COEFF * roi.shape[0]

        # take external contours, find perimetr and area
        cntrs, _ = cv.findContours(roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # mask to remove too small contours as noise:
        noise_mask = np.zeros_like(roi)

        noise_cntrs = [
            cntr
            for cntr in cntrs
            if cv.contourArea(cntr) < min_area and cv.arcLength(cntr, closed=True) < min_arc
        ]
        cv.drawContours(noise_mask, noise_cntrs, -1, color=self.WHITE, thickness=cv.FILLED)

        roi = cv.bitwise_and(roi, cv.bitwise_not(noise_mask))
        roi = cv.morphologyEx(roi, cv.MORPH_CLOSE, kernel=np.ones((3, 3)))

        return roi

    def __call__(self, image: np.ndarray, with_not_nones: bool = False):

        file_id = get_image_hash(image)
        logger.info("Use %s as splitter. Recognition file id %s", self.__class__.__name__, file_id)

        self.filestem = f"{self.__class__.__name__}_{file_id}"
        save_debug_image(
            image, Path(self.filestem).with_suffix(".jpg"), logging.INFO, "Image before splitting"
        )

        all_digits = self.split_into_digits(image)
        if with_not_nones:
            not_none_digits = [d for d in all_digits if d is not None]
            return all_digits, not_none_digits
        else:
            return all_digits
