"""Module for extracting sudoku field from image"""

import numpy as np
import cv2 as cv
from pathlib import Path
import logging

from src.exceptions import InvalidImageError, FieldExtractionError, OpenCvError

from src.utils import get_logger, save_debug_image, get_image_hash


logger = get_logger(__name__)


class FieldExtractor:
    """Class for extracting field from the image"""

    FIELD_ARC_EPSILON = 0.025
    # preprocess
    ADA_THRESH_BLOCK_SIZE = 25
    ADA_THRESH_C = 10
    GAUSS_KERNEL = (5, 5)

    # max aspect ration w/h
    MAX_ASPECT_RATIO_TOL = 0.25
    # min size 40px * 9 digits = 360 px
    MIN_FIELD_SIDE_SIZE = 360
    # minimum number of detected lines
    MIN_LINES_DETECTED = 15
    # minimum number of contours in grid
    MIN_CNTRS_IN_GRID = 40

    # lines extraction
    VERTICAL_ANGLE_THRESHOLD = 80
    HORIZONTAL_ANGLE_THRESHOLD = 10
    MISALIGNMENT_THRESHOLD = 0.2

    WHITE = 255
    BLACK = 0

    @staticmethod
    def dist(pt1, pt2):
        """L2 distance"""
        return np.sqrt(np.sum(pt1 - pt2) ** 2)

    @staticmethod
    def _calculate_angle(x1: int, y1: int, x2: int, y2: int) -> float:
        """Computes the angle between two points line and horizon in degrees [from 0 to pi/2]"""
        return np.arctan2(abs(y1 - y2), abs(x1 - x2)) * (180 / np.pi)

    def __init__(self):
        self.image: np.ndarray = np.empty((0, 0), dtype=np.uint8)
        self.filestem: str = ""

    def _detect_hough_lines(self, height: int):
        """Detects grid lines using Hough Transform."""
        min_line_length = int(height / 9 * 0.80)
        max_line_gap = 5

        lines = cv.HoughLinesP(
            self.image, 2, np.pi / 180, 200, minLineLength=min_line_length, maxLineGap=max_line_gap
        )

        return lines

    def _is_mispaligned_line(self, coord: int, cell_size: float) -> bool:

        grid_lines_coords = np.arange(0, 10) * cell_size
        min_dist_to_grid_line = np.min(np.abs(coord - grid_lines_coords))

        return min_dist_to_grid_line > self.MISALIGNMENT_THRESHOLD * cell_size

    def _create_grid_lines_mask(self, lines):
        """Create grid mask based on lines"""
        height, width = self.image.shape

        # Create a mask for the lines
        mask = np.zeros_like(self.image, dtype=np.uint8)

        cell_height = height / 9
        cell_width = width / 9

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # angle from 0 to 90 in degrees
            angle = self._calculate_angle(x1, y1, x2, y2)
            is_vert = angle > self.VERTICAL_ANGLE_THRESHOLD
            is_misaligned_vert = self._is_mispaligned_line(x1, cell_width)

            is_hor = angle < self.HORIZONTAL_ANGLE_THRESHOLD
            is_misaligned_hor = self._is_mispaligned_line(y1, cell_height)

            if (is_hor and not is_misaligned_hor) or (is_vert and not is_misaligned_vert):
                cv.line(mask, (x1, y1), (x2, y2), self.WHITE, 4)

        # make grid mask solid
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3)))
        save_debug_image(
            mask, Path(self.filestem + "_grid_mask").with_suffix(".jpg"), logging.DEBUG
        )

        return mask

    def remove_grid(self):
        """Remove grid using Hough Lines detection algorightm"""
        height, width = self.image.shape

        # Draw white border to make outer square consistent
        cv.rectangle(self.image, (0, 0), (width, height), color=self.WHITE, thickness=5)
        # Detect lines
        lines = self._detect_hough_lines(height)

        logging.info("%s", len(lines))

        if lines is None or len(lines) < self.MIN_LINES_DETECTED:
            raise FieldExtractionError(f"Detected less than {self.MIN_LINES_DETECTED} grid lines")

        # apply grid mask to remove grid
        grid_mask = self._create_grid_lines_mask(lines)
        cntrs, _ = cv.findContours(self.image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        if len(cntrs) < self.MIN_CNTRS_IN_GRID:
            raise InvalidImageError(
                f"Detected {len(cntrs)} contours out of {self.MIN_CNTRS_IN_GRID} in grid"
            )

        self.image = cv.bitwise_and(self.image, cv.bitwise_not(grid_mask))

    def image_preprocess(self):
        """Grayscale + blur + adaptive thresholding"""
        self.image = cv.cvtColor(self.image, code=cv.COLOR_BGR2GRAY)
        self.image = cv.GaussianBlur(self.image, self.GAUSS_KERNEL, sigmaX=1)
        self.image = cv.adaptiveThreshold(
            self.image,
            self.WHITE,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY_INV,
            self.ADA_THRESH_BLOCK_SIZE,
            self.ADA_THRESH_C,
        )

    def crop_sudoku_field(self):
        """Find sudoku field on binary image, remove perspective and crop"""
        cntrs, _ = cv.findContours(self.image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        rect_corners = self.select_best_countour(cntrs)
        self.remove_perspective(rect_corners)
        # make grid and digits more solid
        self.image = cv.morphologyEx(self.image, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        self.image = cv.threshold(self.image, 1, 255, cv.THRESH_BINARY)[1]
        # check aspect ratio and and size of the field
        h, w = self.image.shape

        if abs(w / h - 1) > self.MAX_ASPECT_RATIO_TOL:
            save_debug_image(self.image, Path(self.filestem + "invalid" + ".jpg"), logging.INFO)
            raise FieldExtractionError(f"Invalid aspect ratio for sudoku field: {w / h:.3}")
        elif min(h, w) < self.MIN_FIELD_SIDE_SIZE:
            save_debug_image(self.image, Path(self.filestem + "invalid" + ".jpg"), logging.INFO)
            raise FieldExtractionError(f"Invalid sudoku field size: ({w}, {h})")

    def select_best_countour(self, cntrs: np.ndarray):
        """Select the largest contour which can be approximated with 4 points"""
        if not cntrs:
            raise FieldExtractionError("No best contour for sudoku grid found")

        for cntr in sorted(cntrs, key=cv.contourArea, reverse=True):
            epsilon = self.FIELD_ARC_EPSILON * cv.arcLength(cntr, closed=True)
            corner_points = cv.approxPolyDP(cntr, epsilon, closed=True)

            if len(corner_points) == 4:
                return corner_points

        raise FieldExtractionError("No rectangular contour for sudoku grid found")

    def remove_perspective(self, corners: np.ndarray):
        """Remove perspective with homography matrix"""
        tl, tr, br, bl = self.reorder_cntr_corners(corners.squeeze())

        width = int(max(self.dist(tl, tr), self.dist(bl, br)))
        height = int(max(self.dist(tl, bl), self.dist(tr, br)))

        corners_old = np.array([tl, tr, br, bl], dtype=np.float32)
        corners_new = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

        matrix = cv.getPerspectiveTransform(corners_old, corners_new)

        self.image = cv.warpPerspective(self.image, matrix, (width, height))

    def reorder_cntr_corners(self, points: np.ndarray):
        """Reorder contour points as tl, tr, br, bl"""
        if points.shape != (4, 2):
            raise ValueError("Points should have (4, 2) shape")
        x_c, y_c = points.mean(axis=0)

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

        return tl, tr, br, bl

    def image_postprocess(self):
        self.image = cv.morphologyEx(self.image, cv.MORPH_CLOSE, kernel=np.ones((3, 3)))
        self.image = cv.erode(self.image, np.ones((3, 3)))
        self.image = cv.threshold(self.image, 1, 255, cv.THRESH_BINARY)[1]

    def get_sudoku_field(self):
        """Pipeline for extracting the sudoku field:

        1. Preprocess: Convert to grayscale, apply Gaussian blur, and threshold.
        2. Crop the sudoku field and correct perspective.
        3. Remove the grid using Hough line detection.
        4. Postprocess: Morphological operations to clean up the image.
        """
        steps = [
            ("_orig", logging.INFO, lambda: None),
            ("_preproc", logging.DEBUG, self.image_preprocess),
            ("_field", logging.DEBUG, self.crop_sudoku_field),
            ("_clear_field", logging.DEBUG, self.remove_grid),
            ("_postproc", logging.INFO, self.image_postprocess),
        ]

        for suffix, log_level, method in steps:
            method()
            save_debug_image(self.image, Path(self.filestem + suffix + ".jpg"), log_level)

    def __call__(self, image: np.ndarray):

        if image.size == 0:
            raise InvalidImageError(
                f"Invalid image data: shape = {image.shape}, size = {image.size}"
            )

        file_id = get_image_hash(image)
        logger.info("Apply field extractor. File id %s", file_id)

        self.filestem = f"{self.__class__.__name__}_{file_id}"
        self.image = image.copy()

        try:
            self.get_sudoku_field()
        except cv.error as e:
            raise OpenCvError(f"Image preprocessing failed: {str(e)}")

        return self.image
