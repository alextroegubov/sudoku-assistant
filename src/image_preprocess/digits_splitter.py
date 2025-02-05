"""Module for splitting sudoku field into digits"""

import numpy as np
import cv2 as cv


class DigitsSplitter:

    MARGIN = 0.05
    MIN_CNTR_AREA = 10

    def __init__(self):
        self.image = np.zeros((10, 10))

    def split_into_digits(self):
        height, width = self.image.shape
        # approximate cell size
        x_step = width / 9
        y_step = height / 9

        # some margins to remove rest of grid
        m_y = self.MARGIN * y_step
        m_x = self.MARGIN * x_step

        lst_digits = []

        for i_y, y in enumerate(np.arange(0, height, y_step)):
            for i_x, x in enumerate(np.arange(0, width, x_step)):

                y1 = int(y + m_y)
                y2 = int(y + y_step - m_y)

                x1 = int(x + m_x)
                x2 = int(x + x_step - m_x)

                roi = self.image[y1:y2, x1:x2]
                roi_area = np.prod(roi.shape)

                cntrs, _ = cv.findContours(roi, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
                cntrs_area = np.array([cv.contourArea(cntr) for cntr in cntrs])

                if (
                    len(cntrs) == 0
                    or len(cntrs) > 4
                    or cntrs_area.max() < self.MIN_CNTR_AREA
                ):
                    lst_digits.append(None)
                else:
                    lst_digits.append(roi.copy())

        return lst_digits

    def __call__(self, image: np.ndarray, with_not_nones: bool = False):
        self.image = image

        all_digits = self.split_into_digits()
        if with_not_nones:
            not_none_digits = [d for d in all_digits if d is not None]
            return all_digits, not_none_digits
        else:
            return all_digits
