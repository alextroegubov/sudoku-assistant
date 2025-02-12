"""Module for extracting sudoku field from image"""

import numpy as np
import cv2 as cv
from pathlib import Path
import logging

from src.exceptions import NoContoursError, NoRectContour
from src.utils import get_logger, save_debug_image, get_image_hash


logger = get_logger(__name__)


class FieldExtractor:
    """Class for extracting field from the image"""

    FIELD_ARC_EPSILON = 0.025
    ADA_THRESH_BLOCK_SIZE = 25
    ADA_THRESH_C = 10

    def __init__(self):
        self.original_image = np.zeros((10, 10), dtype=np.uint8)
        self.image = np.zeros((10, 10), dtype=np.uint8)
        self.filestem: str = ""

    def dist(self, pt1, pt2):
        """L2 distance"""
        return np.sqrt(np.sum(pt1 - pt2) ** 2)

    def remove_grid(self):
        """Remove grid using Hough Lines detection algorightm"""
        height, width = self.image.shape
        # draw white border to make outer square consistent
        cv.rectangle(self.image, pt1=(0, 0), pt2=(width, height), color=255, thickness=5)

        min_line_length = int(height / 9 * 0.80)
        max_line_gap = 5

        lines = cv.HoughLinesP(
            self.image, 1, np.pi / 180, 200, minLineLength=min_line_length, maxLineGap=max_line_gap
        )

        if lines is None:
            raise ValueError("Found no grid lines")

        # Create a mask for the lines
        mask = np.zeros_like(self.image)
        cell_height = height / 9
        cell_width = width / 9

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # angle from 0 to 90 in degrees
            angle = np.arctan2(abs(y1 - y2), abs(x1 - x2)) / np.pi * 180

            is_vert = angle > 80
            is_hor = angle < 10

            # [cell_n, cell_n  + 1]
            cell_n = x1 // int(cell_width)
            is_mispos_vert = (
                min(abs(x1 - cell_n * cell_width), abs(x1 - (cell_n + 1) * cell_width))
                > 0.2 * cell_width
            )

            cell_n = y1 // int(cell_height)
            is_mispos_hor = (
                min(abs(y1 - cell_n * cell_height), abs(y1 - (cell_n + 1) * cell_height))
                > 0.2 * cell_width
            )

            if (is_hor and not is_mispos_hor) or (is_vert and not is_mispos_vert):
                cv.line(mask, (x1, y1), (x2, y2), 255, 4)

        # make grid mask solid
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3)))
        save_debug_image(
            mask, Path(self.filestem + "_grid_mask").with_suffix(".jpg"), logging.DEBUG
        )
        # apply mask to remove grid
        self.image = cv.bitwise_and(self.image, cv.bitwise_not(mask))

    def image_preprocess(self):
        """Grayscale + blur + adaptive thresholding"""
        # grayscale
        self.image = cv.cvtColor(self.image, code=cv.COLOR_BGR2GRAY)
        # slight blur to remove noise
        self.image = cv.GaussianBlur(self.image, (5, 5), sigmaX=1)
        # apply adaptive threshold to binarize image
        self.image = cv.adaptiveThreshold(
            self.image,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            self.ADA_THRESH_BLOCK_SIZE,
            self.ADA_THRESH_C,
        )

    def crop_sudoku_field(self):
        """Find sudoku field on bin image, remove perspective and crop"""

        # find the field contour
        cntrs, _ = cv.findContours(self.image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        rect_corners = self.select_best_countour(cntrs)
        self.remove_perspective(rect_corners)
        # make digits and grid more solid
        self.image = cv.morphologyEx(self.image, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    def select_best_countour(self, cntrs: np.ndarray):
        """Select the largest contour which can be approximated with 4 points"""

        sorted_cntrs = sorted(cntrs, key=cv.contourArea, reverse=True)
        if not sorted_cntrs:
            raise NoContoursError()

        for cntr in sorted_cntrs:
            epsilon = self.FIELD_ARC_EPSILON * cv.arcLength(cntr, closed=True)
            corner_points = cv.approxPolyDP(cntr, epsilon, closed=True)

            if len(corner_points) == 4:
                return corner_points

        raise NoRectContour()

    def remove_perspective(self, corners: np.ndarray):
        """Remove perspective with homography matrix"""
        tl, tr, br, bl = self.reorder_cntr_corners(corners.squeeze())

        width = max(self.dist(tl, tr), self.dist(bl, br))
        height = max(self.dist(tl, bl), self.dist(tr, br))

        width, height = int(width), int(height)

        corners_old = np.array([tl, tr, br, bl], dtype=np.float32)
        corners_new = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        matrix = cv.getPerspectiveTransform(corners_old, corners_new)

        self.image = cv.warpPerspective(self.image, matrix, (width, height))

    def reorder_cntr_corners(self, points: np.ndarray):
        """Reorder contour points as tl, tr, br, bl"""
        if not points.shape == (4, 2):
            raise ValueError("points should have (4, 2) shape")
        # center
        x_c = points[:, 0].mean()
        y_c = points[:, 1].mean()

        tl = tr = br = bl = 0

        for p in points:
            x, y = p
            if x < x_c and y < y_c:
                tl = p
            elif x > x_c and y < y_c:
                tr = p
            elif x > x_c and y > y_c:
                br = p
            elif x < x_c and y > y_c:
                bl = p

        return (tl, tr, br, bl)

    def image_postprocess(self):
        self.image = cv.morphologyEx(self.image, cv.MORPH_CLOSE, kernel=np.ones((3, 3)))
        self.image = cv.erode(self.image, np.ones((3, 3)))
        _, self.image = cv.threshold(self.image, 1, 255, cv.THRESH_BINARY)

    def get_sudoku_field(self):
        save_debug_image(
            self.image, Path(self.filestem + "_orig").with_suffix(".jpg"), logging.INFO
        )
        self.image_preprocess()
        save_debug_image(
            self.image, Path(self.filestem + "_preproc").with_suffix(".jpg"), logging.DEBUG
        )
        self.crop_sudoku_field()
        save_debug_image(
            self.image, Path(self.filestem + "_field").with_suffix(".jpg"), logging.DEBUG
        )
        self.remove_grid()
        save_debug_image(
            self.image, Path(self.filestem + "_clear_field").with_suffix(".jpg"), logging.DEBUG
        )
        self.image_postprocess()
        save_debug_image(
            self.image, Path(self.filestem + "_postproc").with_suffix(".jpg"), logging.INFO
        )

    def __call__(self, image: np.ndarray):

        file_id = get_image_hash(image)
        logger.info("Apply field extractor. File id %s", file_id)

        self.filestem = f"{self.__class__.__name__}_{file_id}"
        self.image = image.copy()
        self.get_sudoku_field()
        return self.image
